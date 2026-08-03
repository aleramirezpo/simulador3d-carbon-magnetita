"""Ventana de escritorio y servidor local efímero de la interfaz 3D."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import tempfile
import threading
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from nucleo.salida import cargar_instantanea, cargar_serie, guardar_instantanea
from nucleo.perfil import CrisolPerfilado, PERFIL_ENSAYO

from .datos_sinteticos import (
    ESPECIES,
    FORMA_DEMO,
    PARTICULAS_ENSAYO,
    TIEMPO_FINAL_S,
    estado_termico_sintetico,
    generar_instantanea_sintetica,
    generar_lineas_corriente,
    numeros_adimensionales_sinteticos,
    velocidad_centrada,
)


RAIZ_INTERFAZ = Path(__file__).resolve().parent
DIRECTORIO_WEB = RAIZ_INTERFAZ / "web"
VERSION_THREE = "0.180.0"
N_FOTOGRAMAS = 25


def _aplanar(array: np.ndarray, decimales: int = 5) -> list[float | int]:
    valores = np.asarray(array)
    if np.issubdtype(valores.dtype, np.integer):
        return valores.ravel(order="C").astype(int).tolist()
    return np.round(valores.ravel(order="C"), decimales).tolist()


def _ficha_hinchamiento() -> dict[str, Any]:
    """Modelo de hinchamiento con la carga real del ensayo.

    0,75 g de carbón y 0,25 g de concentrado: 25 % de inerte en masa. Se importa
    de forma defensiva, igual que la paleta, para que la interfaz arranque
    aunque falte el módulo de física.
    """
    try:
        from fisica.hinchamiento import resumen
        return resumen(fraccion_inerte=0.25)
    except Exception:
        return {}


def _paleta_fases() -> dict[str, Any]:
    """Paleta de fases con el color real de cada mineral.

    Se importa de forma defensiva: si el modulo de fisica no estuviera
    disponible, la interfaz debe seguir arrancando sin leyenda de fases en vez
    de fallar entera.
    """
    try:
        from fisica.fases_visuales import paleta_web
        return paleta_web()
    except Exception:
        return {"fases": {}, "orden_leyenda": [],
                "nota": "paleta de fases no disponible"}


MAGIC_S3DF = b"S3DF"
TIPO_S3DF = "application/vnd.simulador3d.float32"

# Campos densos que viajan como bloques binarios. El resto (t, forma, termico,
# numeros, metadatos) es escalar y va en la cabecera JSON.
_BLOQUES_ESCALARES = (
    "x", "y", "z", "u", "v", "w", "P", "T", "eps", "cohesion", "hinchamiento",
    "conversion", "reduccion",
)


def _codificar_s3df(estructura: dict[str, Any]) -> bytes:
    """Serializa una instantánea en el contrato binario S3DF.

    Una corrida completa son 145 fotogramas de ~9.000 celdas y 25 campos: en
    JSON eso son unos 8 MB por fotograma, casi todos gastados en escribir
    números en decimal. En Float32 crudo son 0,9 MB, y el cliente ya trae el
    decodificador (`frame-cache.js`). El formato es: magic, longitud de
    cabecera uint32 LE, cabecera JSON, relleno a 4 bytes y los bloques.
    """
    cuerpo = bytearray()
    bloques: list[dict[str, Any]] = []

    def anadir(ruta: str, valores: Any, tipo: str = "f32") -> None:
        while len(cuerpo) % 4:
            cuerpo.append(0)
        arreglo = np.ascontiguousarray(
            valores, dtype=np.uint8 if tipo == "u8" else np.float32,
        ).ravel(order="C")
        bloques.append({
            "ruta": ruta, "tipo": tipo,
            "offset": len(cuerpo), "longitud": int(arreglo.size),
        })
        cuerpo.extend(arreglo.tobytes())

    for nombre in _BLOQUES_ESCALARES:
        anadir(nombre, estructura[nombre])
    anadir("etiquetas", estructura["etiquetas"], tipo="u8")
    for nombre, valores in estructura["c_especies"].items():
        anadir(f"c_especies.{nombre}", valores)
    for nombre, valores in estructura["solido"].items():
        anadir(f"solido.{nombre}", valores)

    cabecera = {
        "fotograma": {
            "t": float(estructura["t"]),
            "forma": list(estructura["forma"]),
            "termico": estructura["termico"],
            "numeros": estructura["numeros"],
            "metadatos": estructura["metadatos"],
        },
        "bloques": bloques,
    }
    cabecera_bytes = json.dumps(cabecera, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    relleno = (-(8 + len(cabecera_bytes))) % 4
    return b"".join((
        MAGIC_S3DF,
        len(cabecera_bytes).to_bytes(4, "little"),
        cabecera_bytes,
        b"\x00" * relleno,
        bytes(cuerpo),
    ))


def _numeros_adimensionales(campos: dict[str, Any]) -> dict[str, Any]:
    """Números del solucionador si la instantánea los trae; si no, los de demo.

    `correr_simulacion` los calcula con las propiedades reales de cada celda y
    los guarda en los metadatos. La alternativa —la que se usaba siempre— es una
    función de demostración con densidad y viscosidad fijas y un Damköhler que
    es una rampa lineal en el tiempo: sirve para el modo sintético y para nada
    más. `origen` permite que la interfaz diga cuál está mostrando.
    """
    guardados = (campos.get("metadatos") or {}).get("numeros_adimensionales")
    if isinstance(guardados, dict) and guardados:
        return {
            "Re": float(guardados.get("Re_particula", 0.0)),
            "Re_celda": float(guardados.get("Re_celda", 0.0)),
            "Ra": float(guardados.get("Ra", 0.0)),
            "Pe": float(guardados.get("Pe_termico", 0.0)),
            "Pe_masa": float(guardados.get("Pe_masico", 0.0)),
            "Da": float(guardados.get("Da", 0.0)),
            "origen": "solucionador",
        }
    numeros = dict(numeros_adimensionales_sinteticos(campos))
    numeros["origen"] = "demostracion"
    return numeros


def _estructura_para_web(campos: dict[str, Any], Fe2O3_ref: Any = None) -> dict[str, Any]:
    """Campos derivados listos para el cliente, todavía como arreglos numpy.

    `Fe2O3_ref` es la hematita presente al inicio, en mol/m3. Sin ella la
    conversión no significa nada: antes se dividía por una constante de 5.000
    mol/m3 que no tiene que ver con esta carga —el valor real inicial es 126—,
    de modo que la interfaz marcaba **97,6 % de conversión en t=0**, cuando no
    se había reducido nada todavía.
    """
    uc, vc, wc = velocidad_centrada(campos)
    Fe2O3 = np.asarray(campos["solido"].get("Fe2O3", np.zeros_like(campos["T"])))
    Fe3O4 = np.asarray(campos["solido"].get("Fe3O4", np.zeros_like(campos["T"])))
    FeO = np.asarray(campos["solido"].get("FeO", np.zeros_like(campos["T"])))
    Fe = np.asarray(campos["solido"].get("Fe", np.zeros_like(campos["T"])))
    mascara = np.asarray(campos["etiquetas"]) == 3
    conversion = np.zeros_like(Fe2O3)
    referencia = np.asarray(Fe2O3_ref, dtype=float) if Fe2O3_ref is not None else None
    if referencia is None or referencia.shape != Fe2O3.shape:
        # Sin referencia de la serie se usa el máximo de la propia instantánea:
        # en t=0 da 0 y nunca produce el 97,6 % espurio de la versión anterior.
        maximo = float(np.max(Fe2O3[mascara])) if np.any(mascara) else 0.0
        referencia = np.full_like(Fe2O3, maximo)
    valida = mascara & (referencia > 1.0e-9)
    conversion[valida] = np.clip(1.0 - Fe2O3[valida] / referencia[valida], 0.0, 1.0)
    total = Fe2O3 + Fe3O4 + FeO + Fe
    reduccion = np.zeros_like(total)
    valido = total > 1e-9
    reduccion[valido] = (
        0.11 * Fe3O4[valido] + 0.50 * FeO[valido] + Fe[valido]
    ) / total[valido]
    numeros = _numeros_adimensionales(campos)
    return {
        "t": campos["t"],
        "forma": list(np.asarray(campos["P"]).shape),
        "x": np.asarray(campos["x"]),
        "y": np.asarray(campos["y"]),
        "z": np.asarray(campos["z"]),
        "etiquetas": np.asarray(campos["etiquetas"]),
        "u": uc,
        "v": vc,
        "w": wc,
        "P": np.asarray(campos["P"]),
        "T": np.asarray(campos["T"]),
        "eps": np.asarray(campos["eps"]),
        "cohesion": np.asarray(campos["cohesion"]),
        "hinchamiento": np.asarray(campos.get("hinchamiento", np.ones_like(campos["T"]))),
        "conversion": conversion,
        "reduccion": reduccion,
        "c_especies": {nombre: np.asarray(valores) for nombre, valores in campos["c_especies"].items()},
        "solido": {nombre: np.asarray(valores) for nombre, valores in campos["solido"].items()},
        "termico": campos.get("termico", {}),
        "numeros": numeros,
        "metadatos": campos["metadatos"],
    }


def _campo_para_web(campos: dict[str, Any], Fe2O3_ref: Any = None) -> dict[str, Any]:
    """Misma instantánea en JSON, para clientes que no piden el binario."""
    estructura = _estructura_para_web(campos, Fe2O3_ref)
    salida = dict(estructura)
    for nombre in (*_BLOQUES_ESCALARES, "etiquetas"):
        salida[nombre] = _aplanar(estructura[nombre])
    for grupo in ("c_especies", "solido"):
        salida[grupo] = {
            nombre: _aplanar(valores) for nombre, valores in estructura[grupo].items()
        }
    return salida


def _serie_tolerante(directorio: Path) -> list[dict[str, Any]]:
    """Índice de la serie, saltando instantáneas ilegibles.

    Ahora la interfaz abre por omisión la corrida más avanzada, que puede ser
    una que **todavía se está escribiendo**: si `cargar_serie` tropieza con un
    NPZ a medio escribir, la aplicación entera no debe caerse. Se reintenta
    archivo por archivo y se descarta el que no se pueda leer; en la siguiente
    apertura ya estará completo.
    """
    try:
        return cargar_serie(directorio)
    except Exception:
        pass
    indice: list[dict[str, Any]] = []
    for ruta in sorted(directorio.glob("*.npz")):
        try:
            instantanea = cargar_instantanea(ruta)
        except Exception:
            continue
        indice.append({
            "ruta": ruta,
            "nombre": ruta.name,
            "t": instantanea["t"],
            "forma": tuple(int(n) for n in instantanea["P"].shape),
            "metadatos": instantanea["metadatos"],
        })
    indice.sort(key=lambda elemento: (elemento["t"], elemento["nombre"]))
    return indice


class EstadoAplicacion:
    """Serie real del solucionador o almacén temporal sintético de respaldo."""

    def __init__(
        self,
        n_fotogramas: int = N_FOTOGRAMAS,
        forma: tuple[int, int, int] = FORMA_DEMO,
        datos: str | Path | None = None,
    ):
        solicitado = None if datos is None else Path(datos).resolve()
        indice_real = _serie_tolerante(solicitado) if solicitado is not None and solicitado.is_dir() else []
        # Una corrida nueva sobrescribe primero t=0. Si quedaron NPZ de una
        # corrida anterior, sólo se muestran los escritos a partir del t=0 más
        # reciente para no mezclar dos predicciones en la misma línea temporal.
        inicios = [item for item in indice_real if abs(float(item["t"])) <= 1.0e-12]
        if inicios:
            corte = max(Path(item["ruta"]).stat().st_mtime_ns for item in inicios)
            indice_real = [
                item for item in indice_real
                if Path(item["ruta"]).stat().st_mtime_ns >= corte
            ]
        self._indice_real = indice_real
        self.datos_sinteticos = not bool(indice_real)
        self.origen_datos = "sinteticos" if self.datos_sinteticos else "resultados_del_solucionador"
        self._temporal: tempfile.TemporaryDirectory[str] | None = None
        if indice_real:
            self.n_fotogramas = len(indice_real)
            self.forma = tuple(indice_real[0]["forma"])
            self.directorio = solicitado
            primera = cargar_instantanea(indice_real[0]["ruta"])
            self.especies = list(primera["c_especies"])
        else:
            self.n_fotogramas = n_fotogramas
            self.forma = forma
            self.especies = list(ESPECIES)
            self._temporal = tempfile.TemporaryDirectory(prefix="simulador3d_interfaz_")
            self.directorio = Path(self._temporal.name)
        self._bloqueo = threading.Lock()
        self.Fe2O3_inicial_mol_m3 = self._hematita_inicial()

    def _hematita_inicial(self) -> Any:
        """Campo de hematita al empezar, celda a celda, en mol/m3.

        Es la referencia de la conversión. Se toma de la PRIMERA instantánea de
        la serie, no de una constante: la carga la fija el caso. Celda a celda,
        y no el máximo del lecho, para que en t=0 la conversión sea exactamente
        cero en todas partes y no un 3 % de reparto inicial.
        """
        try:
            campos = self.cargar(0)
        except Exception:
            return None
        Fe2O3 = campos["solido"].get("Fe2O3")
        return None if Fe2O3 is None else np.asarray(Fe2O3, dtype=float).copy()

    @property
    def tiempos(self) -> list[float]:
        if self._indice_real:
            return [float(item["t"]) for item in self._indice_real]
        return np.linspace(0.0, TIEMPO_FINAL_S, self.n_fotogramas).tolist()

    def ruta_fotograma(self, indice: int) -> Path:
        if self._indice_real:
            return Path(self._indice_real[indice]["ruta"])
        return self.directorio / f"instantanea_{indice:04d}.npz"

    def cargar(self, indice: int) -> dict[str, Any]:
        if not 0 <= indice < self.n_fotogramas:
            raise IndexError("fotograma fuera de rango")
        ruta = self.ruta_fotograma(indice)
        if self.datos_sinteticos and not ruta.exists():
            with self._bloqueo:
                if not ruta.exists():
                    generado = generar_instantanea_sintetica(indice, self.n_fotogramas, self.forma)
                    guardar_instantanea(generado, ruta)
        campos = cargar_instantanea(ruta)
        if self.datos_sinteticos:
            # El contrato NPZ guarda los campos centrales, no diagnósticos escalares.
            campos["termico"] = estado_termico_sintetico(campos["t"])
        else:
            pared = np.isin(campos["etiquetas"], (1, 2))
            tapa = campos["etiquetas"] == 2
            T_pared = float(np.mean(campos["T"][pared])) if np.any(pared) else float(np.max(campos["T"]))
            T_tapa = float(np.mean(campos["T"][tapa])) if np.any(tapa) else T_pared
            # La serie 1.0 no persiste aún la curva escalar; 1143,15 K es el
            # valor medido de la mufla en t=0 y el campo real sigue siendo la
            # autoridad para pared y lecho.
            T_mufla = max(1143.15, float(np.max(campos["T"])))
            sigma = 5.670374419e-8
            campos["termico"] = {
                "T_mufla_K": T_mufla,
                "T_pared_media_K": T_pared,
                "T_tapa_K": T_tapa,
                "flujo_radiativo_W_m2": max(0.0, 0.8 * sigma * (T_mufla**4 - T_pared**4)),
            }
        return campos

    def configuracion(self) -> dict[str, Any]:
        crisol = CrisolPerfilado(PERFIL_ENSAYO)
        return {
            "aplicacion": "Simulador 3D de transporte y reacción",
            "modo": "demostración" if self.datos_sinteticos else "resultados",
            "datos_sinteticos": self.datos_sinteticos,
            "origen_datos": self.origen_datos,
            "directorio_datos": None if self.datos_sinteticos else str(self.directorio),
            "advertencia": (
                "DATOS SINTÉTICOS — predicción del modelo — sin validación experimental"
                if self.datos_sinteticos else
                "RESULTADOS DEL SOLUCIONADOR — PREDICCIÓN — sin validación experimental"
            ),
            "version_formato": "1.0.0",
            "version_three": VERSION_THREE,
            "n_fotogramas": self.n_fotogramas,
            "tiempos": self.tiempos,
            "forma": list(self.forma),
            "especies": self.especies,
            "particulas": PARTICULAS_ENSAYO,
            # Paleta con el color REAL de cada mineral, para poder ver aparecer
            # cada fase nueva con su aspecto reconocible y mostrar la leyenda.
            "fases": _paleta_fases(),
            # Índice de hinchamiento 8 del carbón, atenuado por la magnetita.
            "hinchamiento": _ficha_hinchamiento(),
            "referencias_adimensionales": {
                "Re_particula": 0.053,
                "Ra": 188.0,
                "Pe_lecho": 1.01,
                "fuente": "valores de referencia del ensayo real",
            },
            "unidades": {"longitud": "mm", "tiempo": "s", "T": "K", "P": "Pa", "c": "mol/m³"},
            "geometria": {
                "tipo": "perfil_de_revolucion_con_collar",
                "perfil_exterior_mm": [list(punto) for punto in crisol.perfil_exterior.puntos],
                "perfil_interior_mm": [list(punto) for punto in crisol.perfil_interior.puntos],
                "espesor_tapa_mm": crisol.espesor_tapa_mm,
                "tapa": {
                    "tipo": "cilindro_cerrado",
                    "radio_mm": crisol.perfil_exterior.puntos[-1][1],
                    "z_base_mm": crisol.perfil_exterior.puntos[-1][0],
                    "espesor_mm": crisol.espesor_tapa_mm,
                },
                "collar_diametro_mm": 2.0 * crisol.r_max_ext,
                "collar_altura_mm": 21.0,
                "fuente": "nucleo.perfil.PERFIL_ENSAYO",
            },
        }

    def serie(self) -> list[dict[str, float]]:
        filas: list[dict[str, float]] = []
        for indice in range(self.n_fotogramas):
            campos = self.cargar(indice)
            interior = np.isin(campos["etiquetas"], (3, 4))
            lecho = campos["etiquetas"] == 3
            Fe2O3 = campos["solido"]["Fe2O3"]
            referencia = self.Fe2O3_inicial_mol_m3
            valida = lecho & (np.asarray(referencia) > 1.0e-9) if referencia is not None else None
            conversion = (
                float(np.mean(np.clip(1.0 - Fe2O3[valida] / np.asarray(referencia)[valida], 0.0, 1.0)))
                if valida is not None and np.any(valida) else 0.0
            )
            filas.append({
                "t": float(campos["t"]),
                "T_media_K": float(np.mean(campos["T"][interior])),
                "CO_medio_mol_m3": float(np.mean(campos["c_especies"]["CO"][interior])),
                "conversion_media": conversion,
                "cohesion_max": float(np.max(campos["cohesion"])),
            })
        return filas

    def cerrar(self) -> None:
        if self._temporal is not None:
            self._temporal.cleanup()


class ServidorInterfaz(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, direccion: tuple[str, int], estado: EstadoAplicacion):
        super().__init__(direccion, ManejadorInterfaz)
        self.estado = estado


class ManejadorInterfaz(BaseHTTPRequestHandler):
    """Sirve recursos locales y la API de instantáneas, sólo en localhost."""

    server: ServidorInterfaz

    def log_message(self, formato: str, *args: Any) -> None:
        return

    def _cabeceras(self, tipo: str, longitud: int, estado: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(estado)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(longitud))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; worker-src 'self' blob:")
        self.end_headers()

    def _json(self, contenido: Any, estado: HTTPStatus = HTTPStatus.OK) -> None:
        cuerpo = json.dumps(contenido, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._cabeceras("application/json; charset=utf-8", len(cuerpo), estado)
        self.wfile.write(cuerpo)

    def _error(self, mensaje: str, estado: HTTPStatus) -> None:
        self._json({"error": mensaje}, estado)

    def do_GET(self) -> None:  # noqa: N802 - nombre exigido por BaseHTTPRequestHandler
        solicitud = urlparse(self.path)
        try:
            if solicitud.path == "/api/salud":
                self._json({
                    "estado": "ok",
                    "modo": "demostración" if self.server.estado.datos_sinteticos else "resultados",
                    "datos_sinteticos": self.server.estado.datos_sinteticos,
                    "origen_datos": self.server.estado.origen_datos,
                })
                return
            if solicitud.path == "/api/config":
                self._json(self.server.estado.configuracion())
                return
            if solicitud.path == "/api/fotograma":
                consulta = parse_qs(solicitud.query)
                indice = int(consulta.get("indice", ["0"])[0])
                campos = self.server.estado.cargar(indice)
                pide_binario = (
                    consulta.get("formato", [""])[0] == "float32"
                    or TIPO_S3DF in (self.headers.get("Accept") or "")
                )
                if pide_binario:
                    cuerpo = _codificar_s3df(_estructura_para_web(
                        campos, self.server.estado.Fe2O3_inicial_mol_m3))
                    self._cabeceras(TIPO_S3DF, len(cuerpo))
                    self.wfile.write(cuerpo)
                    return
                self._json(_campo_para_web(
                    campos, self.server.estado.Fe2O3_inicial_mol_m3))
                return
            if solicitud.path == "/api/lineas":
                indice = int(parse_qs(solicitud.query).get("indice", ["0"])[0])
                lineas = generar_lineas_corriente(self.server.estado.cargar(indice))
                self._json({"lineas": [np.round(linea, 4).tolist() for linea in lineas]})
                return
            if solicitud.path == "/api/serie":
                self._json({"serie": self.server.estado.serie()})
                return
            self._recurso_estatico(solicitud.path)
        except (IndexError, ValueError) as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # el error llega al frontend sin tumbar el servidor
            self._error(f"error interno: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def _recurso_estatico(self, ruta_url: str) -> None:
        relativa = "index.html" if ruta_url in {"", "/"} else unquote(ruta_url.lstrip("/"))
        candidata = (DIRECTORIO_WEB / relativa).resolve()
        try:
            candidata.relative_to(DIRECTORIO_WEB.resolve())
        except ValueError:
            self._error("ruta no permitida", HTTPStatus.FORBIDDEN)
            return
        if not candidata.is_file():
            self._error("recurso no encontrado", HTTPStatus.NOT_FOUND)
            return
        cuerpo = candidata.read_bytes()
        if candidata.name == "index.html" and not self.server.estado.datos_sinteticos:
            html = cuerpo.decode("utf-8")
            html = html.replace("Datos sintéticos", "Resultados del solucionador")
            html = html.replace(
                "PREDICCIÓN · DATOS SINTÉTICOS",
                "PREDICCIÓN · RESULTADOS DEL SOLUCIONADOR",
            )
            cuerpo = html.encode("utf-8")
        tipo = mimetypes.guess_type(candidata.name)[0] or "application/octet-stream"
        if tipo.startswith("text/") or tipo in {"application/javascript", "application/json"}:
            tipo += "; charset=utf-8"
        self._cabeceras(tipo, len(cuerpo))
        self.wfile.write(cuerpo)


def _tiempo_de_nombre(nombre: str) -> float | None:
    """Segundos codificados en `instantanea_<microsegundos>us.npz`."""
    coincidencia = re.fullmatch(r"instantanea_(\d+)us\.npz", nombre)
    return None if coincidencia is None else int(coincidencia.group(1)) * 1.0e-6


def directorio_predeterminado(base: str | Path = "resultados") -> Path:
    """Corrida más avanzada disponible, sin abrir los NPZ.

    Con varias corridas guardadas (una corta de prueba, otra larga, la de 720 s)
    la interfaz debe abrir la que llega más lejos en el tiempo físico. El
    tiempo está en el nombre del archivo, así que no hace falta leer decenas de
    megabytes para decidir. Si no hay ninguna se devuelve la ruta clásica y el
    programa cae en datos sintéticos, como siempre.
    """
    raiz = Path(base)
    clasico = raiz / "simulacion"
    if not raiz.is_dir():
        return clasico
    mejor: tuple[float, int, Path] | None = None
    for carpeta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        tiempos = [
            t for t in (_tiempo_de_nombre(ruta.name) for ruta in carpeta.glob("instantanea_*us.npz"))
            if t is not None
        ]
        if len(tiempos) < 2:
            continue
        candidato = (max(tiempos), len(tiempos), carpeta)
        if mejor is None or candidato[:2] > mejor[:2]:
            mejor = candidato
    return clasico if mejor is None else mejor[2]


def crear_servidor(
    puerto: int = 0, host: str = "127.0.0.1", datos: str | Path | None = None,
) -> ServidorInterfaz:
    """Crea el servidor sin arrancar hilos; útil para pruebas headless."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("la interfaz sólo puede enlazarse a localhost")
    return ServidorInterfaz((host, int(puerto)), EstadoAplicacion(datos=datos))


def iniciar_servidor(
    puerto: int = 0, datos: str | Path | None = None,
) -> tuple[ServidorInterfaz, threading.Thread, str]:
    servidor = crear_servidor(puerto, datos=datos)
    hilo = threading.Thread(target=servidor.serve_forever, name="servidor-interfaz", daemon=True)
    hilo.start()
    host, puerto_real = servidor.server_address[:2]
    return servidor, hilo, f"http://{host}:{puerto_real}/"


def detener_servidor(servidor: ServidorInterfaz, hilo: threading.Thread | None = None) -> None:
    servidor.shutdown()
    servidor.server_close()
    servidor.estado.cerrar()
    if hilo is not None:
        hilo.join(timeout=3.0)


def _importar_webview() -> Any | None:
    try:
        import webview  # type: ignore
        return webview
    except Exception:
        vendor = RAIZ_INTERFAZ / "_vendor_pywebview"
        if vendor.is_dir() and str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        try:
            import webview  # type: ignore
            return webview
        except Exception:
            return None


def ejecutar(
    comprobar: bool = False,
    forzar_navegador: bool = False,
    puerto: int = 0,
    datos: str | Path | None = None,
) -> int:
    """Abre la ventana nativa; cae al navegador si pywebview no está disponible."""
    servidor, hilo, url = iniciar_servidor(puerto, datos=datos)
    try:
        if comprobar:
            with urllib.request.urlopen(url + "api/salud", timeout=5.0) as respuesta:
                if respuesta.status != HTTPStatus.OK:
                    raise RuntimeError("el servidor local no respondió correctamente")
            print(f"Comprobación correcta: {url}")
            return 0

        webview = None if forzar_navegador else _importar_webview()
        if webview is not None:
            webview.create_window(
                "Simulador 3D — transporte y reacción",
                url,
                width=1440,
                height=920,
                min_size=(1080, 700),
                background_color="#10171d",
            )
            webview.start(debug=False)
        else:
            print("pywebview no está disponible; se abre la interfaz en el navegador predeterminado.")
            webbrowser.open(url)
            print("Pulse Ctrl+C para cerrar el servidor local.")
            try:
                hilo.join()
            except KeyboardInterrupt:
                pass
        return 0
    finally:
        detener_servidor(servidor, hilo)


def servir_sin_ventana(puerto: int = 0, datos: str | Path | None = None) -> int:
    """Mantiene el servidor local para pruebas de navegador automatizadas."""
    servidor, hilo, url = iniciar_servidor(puerto, datos=datos)
    print(f"Servidor local: {url}", flush=True)
    try:
        hilo.join()
    except KeyboardInterrupt:
        pass
    finally:
        detener_servidor(servidor, hilo)
    return 0


def main() -> None:
    analizador = argparse.ArgumentParser(description="Interfaz 3D de transporte y reacción")
    analizador.add_argument("--comprobar", action="store_true", help="arranca, comprueba y cierra sin ventana")
    analizador.add_argument("--navegador", action="store_true", help="fuerza la alternativa en navegador")
    analizador.add_argument("--headless", action="store_true", help="sinónimo de --comprobar para automatización")
    analizador.add_argument("--solo-servidor", action="store_true", help="sirve la web sin abrir una ventana")
    analizador.add_argument("--puerto", type=int, default=0, help="puerto local; 0 elige uno efímero")
    analizador.add_argument(
        "--datos", type=Path, default=None,
        help=("directorio de instantáneas reales; por defecto la corrida más "
              "avanzada de resultados/. Si está vacío se usan datos sintéticos"),
    )
    argumentos = analizador.parse_args()
    datos = directorio_predeterminado() if argumentos.datos is None else argumentos.datos
    if argumentos.solo_servidor:
        raise SystemExit(servir_sin_ventana(argumentos.puerto, datos))
    raise SystemExit(ejecutar(
        argumentos.comprobar or argumentos.headless,
        argumentos.navegador,
        argumentos.puerto,
        datos,
    ))


if __name__ == "__main__":
    main()
