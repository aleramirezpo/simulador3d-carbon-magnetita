"""Tabulacion vectorizada de la termodinamica escalar de ``simulacion_v3``.

La constante de equilibrio se almacena como ``log(K)``.  Interpolar ``K``
directamente mezcla valores que pueden diferir en muchos ordenes de magnitud;
interpolar su logaritmo conserva el error relativo y evita desbordamientos.

La rejilla base se completa con los dos lados representables de cada cambio
de tramo y con refinamiento local alrededor de cruces por cero.  Los nodos a
ambos lados impiden que una interpolacion lineal suavice una discontinuidad de
Cp o una transicion de fase.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


VERSION_TABLA = 1
PASO_PREDETERMINADO_K = 0.025
PASO_CEROS_K = 0.001

_MAGNITUDES_ESPECIE = ("cp", "h", "s")
_MAGNITUDES_REACCION = ("delta_H", "delta_G", "log_K")


def _cargar_termodinamica() -> ModuleType:
    """Localiza la fuente validada sin copiar sus datos al simulador 3-D."""

    candidata_env = os.environ.get("SIMULACION_V3_SRC")
    candidatas = []
    if candidata_env:
        candidatas.append(Path(candidata_env).expanduser())
    candidatas.append(Path(__file__).resolve().parents[2] / "simulacion_v3" / "src")
    for candidata in candidatas:
        archivo = candidata.resolve() / "termodinamica_ext.py"
        if archivo.is_file():
            if str(archivo.parent) not in sys.path:
                sys.path.insert(0, str(archivo.parent))
            modulo = importlib.import_module("termodinamica_ext")
            if Path(modulo.__file__).resolve() != archivo:
                raise ImportError(
                    f"termodinamica_ext se resolvio en {modulo.__file__}, no en {archivo}"
                )
            return modulo
    raise ImportError("No se encontro simulacion_v3/src/termodinamica_ext.py")


def _huella_origen(termodinamica: ModuleType) -> str:
    """Huella que invalida un .npz cuando cambia la implementacion de origen."""

    archivo = Path(termodinamica.__file__).resolve()
    digest = hashlib.sha256()
    digest.update(archivo.read_bytes())
    digest.update(repr(float(termodinamica.R)).encode("ascii"))
    digest.update(repr(float(termodinamica.T_REF)).encode("ascii"))
    return digest.hexdigest()


class TablaTermoquimica:
    """Tabla persistible de propiedades de especies y reacciones.

    Parameters
    ----------
    T_min, T_max:
        Intervalo tabulado en kelvin. Por defecto 298.15--2000 K.
    paso:
        Paso de la rejilla base. El valor predeterminado de 0.025 K fue
        seleccionado midiendo el error de las curvas mas exigentes, no por
        una suposicion a priori.
    fuera_de_rango:
        ``"avisar"`` avisa y recorta al extremo; ``"extrapolar"`` avisa y
        extrapola linealmente; ``"error"`` rechaza la consulta.
    """

    def __init__(
        self,
        T_min: float | None = None,
        T_max: float = 2000.0,
        paso: float = PASO_PREDETERMINADO_K,
        *,
        paso_ceros: float = PASO_CEROS_K,
        fuera_de_rango: str = "avisar",
        termodinamica: ModuleType | None = None,
    ) -> None:
        self.termodinamica = termodinamica or _cargar_termodinamica()
        self.T_min = float(self.termodinamica.T_REF if T_min is None else T_min)
        self.T_max = float(T_max)
        self.paso = float(paso)
        self.paso_ceros = float(paso_ceros)
        self.fuera_de_rango = self._validar_politica(fuera_de_rango)
        if not np.isfinite((self.T_min, self.T_max, self.paso, self.paso_ceros)).all():
            raise ValueError("Los limites y pasos de la tabla deben ser finitos")
        if self.T_min <= 0.0 or self.T_max <= self.T_min:
            raise ValueError("Se requiere 0 < T_min < T_max")
        if self.paso <= 0.0 or self.paso_ceros <= 0.0:
            raise ValueError("Los pasos de temperatura deben ser positivos")

        self.especies = tuple(self.termodinamica.BASE_TERMOQUIMICA)
        self.reacciones = tuple(self.termodinamica.REACCIONES_EXT)
        self._indice_especie = {nombre: i for i, nombre in enumerate(self.especies)}
        self._indice_reaccion = {nombre: i for i, nombre in enumerate(self.reacciones)}
        self._estequiometria = self._matriz_estequiometrica()
        self.huella_origen = _huella_origen(self.termodinamica)

        self.temperaturas = self._construir_rejilla()
        cp, h, s = self._evaluar_especies_exactas(self.temperaturas, incluir_cp=True)
        self._datos_especies = np.stack((cp, h, s))
        g = h - s * self.temperaturas[np.newaxis, :] / 1000.0
        delta_h = self._estequiometria @ h
        delta_g = self._estequiometria @ g
        log_k = np.clip(
            -delta_g * 1000.0
            / (float(self.termodinamica.R) * self.temperaturas[np.newaxis, :]),
            -700.0,
            700.0,
        )
        self._datos_reacciones = np.stack((delta_h, delta_g, log_k))

    @staticmethod
    def _validar_politica(politica: str) -> str:
        if politica not in {"avisar", "extrapolar", "error", "recortar"}:
            raise ValueError(
                "fuera_de_rango debe ser 'avisar', 'extrapolar', 'error' o 'recortar'"
            )
        return politica

    def _matriz_estequiometrica(self) -> np.ndarray:
        matriz = np.zeros((len(self.reacciones), len(self.especies)), dtype=float)
        for i, reaccion in enumerate(self.reacciones):
            for especie, coeficiente in self.termodinamica.REACCIONES_EXT[reaccion].items():
                matriz[i, self._indice_especie[especie]] = float(coeficiente)
        return matriz

    def _puntos_de_cambio(self) -> np.ndarray:
        puntos = {self.T_min, self.T_max}
        for datos in self.termodinamica.BASE_TERMOQUIMICA.values():
            for rango in datos["rangos"]:
                for borde in rango["T"]:
                    valor = float(borde)
                    if self.T_min < valor < self.T_max:
                        puntos.add(valor)
            for transicion in datos.get("transiciones", ()):
                valor = float(transicion["T"])
                if self.T_min < valor < self.T_max:
                    puntos.add(valor)
        return np.array(sorted(puntos), dtype=float)

    def _evaluar_especies_exactas(
        self, temperaturas: np.ndarray, *, incluir_cp: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        te = self.termodinamica
        forma = (len(self.especies), temperaturas.size)
        cp = np.empty(forma, dtype=float) if incluir_cp else np.empty((0, 0))
        h = np.empty(forma, dtype=float)
        s = np.empty(forma, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i, especie in enumerate(self.especies):
                if incluir_cp:
                    cp[i] = np.fromiter(
                        (te.cp_J_molK(especie, float(T)) for T in temperaturas),
                        dtype=float,
                        count=temperaturas.size,
                    )
                h[i] = np.fromiter(
                    (te.h_kJ_mol(especie, float(T)) for T in temperaturas),
                    dtype=float,
                    count=temperaturas.size,
                )
                s[i] = np.fromiter(
                    (te.s_J_molK(especie, float(T)) for T in temperaturas),
                    dtype=float,
                    count=temperaturas.size,
                )
        return cp, h, s

    @staticmethod
    def _cruces_por_cero(x: np.ndarray, curvas: np.ndarray) -> list[float]:
        """Estima cruces para decidir donde refinar; la tabla final es exacta."""

        cruces: list[float] = []
        for curva in curvas:
            finitos = np.isfinite(curva)
            indices_cero = np.flatnonzero(finitos & (curva == 0.0))
            cruces.extend(float(x[i]) for i in indices_cero)
            indices = np.flatnonzero(
                finitos[:-1]
                & finitos[1:]
                & (np.signbit(curva[:-1]) != np.signbit(curva[1:]))
            )
            for i in indices:
                x0, x1 = float(x[i]), float(x[i + 1])
                y0, y1 = float(curva[i]), float(curva[i + 1])
                if y1 != y0:
                    cruces.append(x0 - y0 * (x1 - x0) / (y1 - y0))
        return [T for T in cruces if np.isfinite(T) and x[0] <= T <= x[-1]]

    def _construir_rejilla(self) -> np.ndarray:
        cambios = self._puntos_de_cambio()
        # Una exploracion barata de 0.5 K localiza los cruces por cero. Las
        # reacciones se derivan matricialmente para no repetir sus especies.
        exploracion = np.unique(
            np.concatenate(
                (np.arange(self.T_min, self.T_max, 0.5), cambios, [self.T_max])
            )
        )
        _, h, s = self._evaluar_especies_exactas(exploracion, incluir_cp=False)
        g = h - s * exploracion[np.newaxis, :] / 1000.0
        delta_h = self._estequiometria @ h
        delta_g = self._estequiometria @ g
        cruces = self._cruces_por_cero(exploracion, h)
        cruces += self._cruces_por_cero(exploracion, delta_h)
        cruces += self._cruces_por_cero(exploracion, delta_g)

        bloques: list[np.ndarray] = [
            np.arange(self.T_min, self.T_max, self.paso),
            cambios,
            np.array([self.T_max]),
            # Las entalpias elementales parten de cero en T_REF; este bloque
            # evita que un denominador pequeno deteriore su error relativo.
            np.arange(
                self.T_min,
                min(self.T_min + 4.0, self.T_max) + 0.5 * self.paso_ceros,
                self.paso_ceros,
            ),
        ]
        # El valor exacto en un borde pertenece al tramo de alta T. Su vecino
        # representable inferior conserva por separado el limite de baja T.
        inferiores = [
            np.nextafter(T, -np.inf)
            for T in cambios[1:-1]
            if np.nextafter(T, -np.inf) >= self.T_min
        ]
        bloques.append(np.asarray(inferiores, dtype=float))
        for raiz in cruces:
            inferior = max(self.T_min, raiz - 0.35)
            superior = min(self.T_max, raiz + 0.35)
            bloques.append(
                np.arange(inferior, superior + 0.5 * self.paso_ceros, self.paso_ceros)
            )
        return np.unique(np.concatenate(bloques))

    def _controlar_temperaturas(self, T: Any) -> np.ndarray:
        temperaturas = np.asarray(T, dtype=float)
        if np.any(~np.isfinite(temperaturas)):
            raise ValueError("La temperatura debe contener solo valores finitos")
        fuera = (temperaturas < self.T_min) | (temperaturas > self.T_max)
        if np.any(fuera):
            mensaje = (
                f"Temperatura fuera de la tabla {self.T_min:g}--{self.T_max:g} K; "
                + ("se extrapola linealmente" if self.fuera_de_rango == "extrapolar" else "se recorta al extremo")
            )
            if self.fuera_de_rango == "error":
                raise ValueError(mensaje)
            if self.fuera_de_rango in {"avisar", "extrapolar"}:
                warnings.warn(mensaje, RuntimeWarning, stacklevel=3)
            if self.fuera_de_rango != "extrapolar":
                temperaturas = np.clip(temperaturas, self.T_min, self.T_max)
        return temperaturas

    def _interpolar_filas(self, filas: np.ndarray, T: Any) -> np.ndarray:
        temperaturas = self._controlar_temperaturas(T)
        planas = temperaturas.reshape(-1)
        indices = np.searchsorted(self.temperaturas, planas, side="right") - 1
        indices = np.clip(indices, 0, self.temperaturas.size - 2)
        t0 = self.temperaturas[indices]
        t1 = self.temperaturas[indices + 1]
        fraccion = (planas - t0) / (t1 - t0)
        salida = filas[:, indices] + (filas[:, indices + 1] - filas[:, indices]) * fraccion
        return salida.reshape((filas.shape[0],) + temperaturas.shape)

    @staticmethod
    def _unico(valor: np.ndarray, T: Any) -> float | np.ndarray:
        salida = np.asarray(valor)
        return float(salida) if np.asarray(T).ndim == 0 else salida

    def datos_especies(
        self, T: Any, especies: Iterable[str] | None = None
    ) -> dict[str, np.ndarray]:
        nombres = self.especies if especies is None else tuple(especies)
        try:
            indices = [self._indice_especie[nombre] for nombre in nombres]
        except KeyError as exc:
            raise KeyError(f"Especie termoquimica desconocida: {exc.args[0]!r}") from exc
        valores = self._interpolar_filas(
            self._datos_especies[:, indices, :].reshape(-1, self.temperaturas.size), T
        ).reshape((3, len(indices)) + np.asarray(T).shape)
        return {magnitud: valores[i] for i, magnitud in enumerate(_MAGNITUDES_ESPECIE)}

    def datos_reacciones(
        self, T: Any, reacciones: Iterable[str] | None = None
    ) -> dict[str, np.ndarray]:
        nombres = self.reacciones if reacciones is None else tuple(reacciones)
        try:
            indices = [self._indice_reaccion[nombre] for nombre in nombres]
        except KeyError as exc:
            raise KeyError(f"Reaccion desconocida: {exc.args[0]!r}") from exc
        valores = self._interpolar_filas(
            self._datos_reacciones[:, indices, :].reshape(-1, self.temperaturas.size), T
        ).reshape((3, len(indices)) + np.asarray(T).shape)
        return {
            "delta_H": valores[0],
            "delta_G": valores[1],
            "log_K": valores[2],
            "K_eq": np.exp(np.clip(valores[2], -700.0, 700.0)),
        }

    def cp(self, especie: str, T: Any) -> float | np.ndarray:
        return self._unico(self.datos_especies(T, (especie,))["cp"][0], T)

    def h(self, especie: str, T: Any) -> float | np.ndarray:
        return self._unico(self.datos_especies(T, (especie,))["h"][0], T)

    def s(self, especie: str, T: Any) -> float | np.ndarray:
        return self._unico(self.datos_especies(T, (especie,))["s"][0], T)

    def delta_H(self, reaccion: str, T: Any) -> float | np.ndarray:
        return self._unico(self.datos_reacciones(T, (reaccion,))["delta_H"][0], T)

    def delta_G(self, reaccion: str, T: Any) -> float | np.ndarray:
        return self._unico(self.datos_reacciones(T, (reaccion,))["delta_G"][0], T)

    def K_eq(self, reaccion: str, T: Any) -> float | np.ndarray:
        return self._unico(self.datos_reacciones(T, (reaccion,))["K_eq"][0], T)

    # Alias con las unidades explicitas de termodinamica_ext.
    cp_J_molK = cp
    h_kJ_mol = h
    s_J_molK = s
    delta_H_kj = delta_H
    delta_G_kj = delta_G

    @staticmethod
    def _maximo_relativo(tabla: np.ndarray, exacto: np.ndarray) -> float:
        # El error relativo no esta definido en un cero exacto. El piso es la
        # resolucion de coma flotante de la escala de cada curva, no una
        # tolerancia fisica que oculte error de interpolacion.
        escalas = np.maximum(np.max(np.abs(exacto), axis=1, keepdims=True), 1.0)
        denominador = np.maximum(np.abs(exacto), np.finfo(float).eps * escalas)
        return float(np.max(np.abs(tabla - exacto) / denominador))

    def error_de_tabulacion(
        self, n_muestras: int = 1000, *, semilla: int = 20260801
    ) -> dict[str, float]:
        """Compara contra la fuente escalar en temperaturas aleatorias."""

        if n_muestras <= 0:
            raise ValueError("n_muestras debe ser positivo")
        rng = np.random.default_rng(semilla)
        T = rng.uniform(self.T_min, self.T_max, int(n_muestras))
        cp, h, s = self._evaluar_especies_exactas(T, incluir_cp=True)
        g = h - s * T[np.newaxis, :] / 1000.0
        delta_h = self._estequiometria @ h
        delta_g = self._estequiometria @ g
        log_k = np.clip(
            -delta_g * 1000.0 / (float(self.termodinamica.R) * T[np.newaxis, :]),
            -700.0,
            700.0,
        )
        especies_tab = self.datos_especies(T)
        reacciones_tab = self.datos_reacciones(T)
        return {
            "cp": self._maximo_relativo(especies_tab["cp"], cp),
            "h": self._maximo_relativo(especies_tab["h"], h),
            "s": self._maximo_relativo(especies_tab["s"], s),
            "delta_H": self._maximo_relativo(reacciones_tab["delta_H"], delta_h),
            "delta_G": self._maximo_relativo(reacciones_tab["delta_G"], delta_g),
            # expm1 mide directamente el error relativo de exp(log_K).
            "K_eq": float(np.max(np.abs(np.expm1(reacciones_tab["log_K"] - log_k)))),
        }

    def _metadata(self) -> dict[str, Any]:
        return {
            "version": VERSION_TABLA,
            "huella_origen": self.huella_origen,
            "T_min": self.T_min,
            "T_max": self.T_max,
            "paso": self.paso,
            "paso_ceros": self.paso_ceros,
            "especies": list(self.especies),
            "reacciones": list(self.reacciones),
        }

    def guardar(self, ruta: str | os.PathLike[str]) -> Path:
        """Guarda todos los nodos y metadatos en un unico archivo ``.npz``."""

        destino = Path(ruta)
        if destino.suffix.lower() != ".npz":
            destino = Path(f"{destino}.npz")
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destino,
            metadata=np.asarray(json.dumps(self._metadata(), sort_keys=True)),
            temperaturas=self.temperaturas,
            datos_especies=self._datos_especies,
            datos_reacciones=self._datos_reacciones,
        )
        return destino

    @classmethod
    def cargar(
        cls,
        ruta: str | os.PathLike[str],
        *,
        termodinamica: ModuleType | None = None,
        fuera_de_rango: str = "avisar",
    ) -> "TablaTermoquimica":
        """Carga una tabla y rechaza versiones o datos de origen obsoletos."""

        te = termodinamica or _cargar_termodinamica()
        with np.load(Path(ruta), allow_pickle=False) as archivo:
            metadata = json.loads(str(archivo["metadata"]))
            if int(metadata.get("version", -1)) != VERSION_TABLA:
                raise ValueError("Version de tabla termoquimica incompatible")
            if metadata.get("huella_origen") != _huella_origen(te):
                raise ValueError("La fuente termoquimica cambio; la tabla esta invalidada")
            especies = tuple(metadata["especies"])
            reacciones = tuple(metadata["reacciones"])
            if especies != tuple(te.BASE_TERMOQUIMICA) or reacciones != tuple(te.REACCIONES_EXT):
                raise ValueError("Las especies o reacciones de origen cambiaron")
            temperaturas = np.array(archivo["temperaturas"], dtype=float, copy=True)
            datos_especies = np.array(archivo["datos_especies"], dtype=float, copy=True)
            datos_reacciones = np.array(archivo["datos_reacciones"], dtype=float, copy=True)

        objeto = cls.__new__(cls)
        objeto.termodinamica = te
        objeto.T_min = float(metadata["T_min"])
        objeto.T_max = float(metadata["T_max"])
        objeto.paso = float(metadata["paso"])
        objeto.paso_ceros = float(metadata["paso_ceros"])
        objeto.fuera_de_rango = cls._validar_politica(fuera_de_rango)
        objeto.especies = especies
        objeto.reacciones = reacciones
        objeto._indice_especie = {nombre: i for i, nombre in enumerate(especies)}
        objeto._indice_reaccion = {nombre: i for i, nombre in enumerate(reacciones)}
        objeto._estequiometria = objeto._matriz_estequiometrica()
        objeto.huella_origen = metadata["huella_origen"]
        objeto.temperaturas = temperaturas
        objeto._datos_especies = datos_especies
        objeto._datos_reacciones = datos_reacciones
        esperado_especies = (3, len(especies), temperaturas.size)
        esperado_reacciones = (3, len(reacciones), temperaturas.size)
        if datos_especies.shape != esperado_especies or datos_reacciones.shape != esperado_reacciones:
            raise ValueError("Dimensiones internas invalidas en la tabla termoquimica")
        return objeto

    @classmethod
    def cargar_o_crear(
        cls, ruta: str | os.PathLike[str], **opciones: Any
    ) -> "TablaTermoquimica":
        """Usa el cache si es valido; si no, lo invalida y lo reconstruye."""

        destino = Path(ruta)
        if destino.suffix.lower() != ".npz":
            destino = Path(f"{destino}.npz")
        try:
            return cls.cargar(
                destino,
                termodinamica=opciones.get("termodinamica"),
                fuera_de_rango=opciones.get("fuera_de_rango", "avisar"),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            tabla = cls(**opciones)
            tabla.guardar(destino)
            return tabla


__all__ = [
    "PASO_CEROS_K",
    "PASO_PREDETERMINADO_K",
    "TablaTermoquimica",
    "VERSION_TABLA",
]
