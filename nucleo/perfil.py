"""Perfil de revolución del crisol real observado en la fotografía acotada.

La geometría de :mod:`nucleo.geometria` aproxima el crisol por un único tronco
de cono.  Aquí se conserva su API, pero la generatriz exterior es una polilínea
que permite representar el collar visible en ``D:\\pae\\Imagen.pdf``.

Las longitudes de este módulo están en milímetros, igual que en
``nucleo.geometria``.  No se escribe ningún archivo al importar el módulo.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # Importación como parte del paquete (pytest y simulador).
    from .geometria import CRISOL_ENSAYO, Crisol, volumen_tronco
except ImportError:  # Ejecución directa: ``python nucleo\perfil.py``.
    from geometria import CRISOL_ENSAYO, Crisol, volumen_tronco


class PerfilRevolucion:
    """Sólido de revolución definido por una generatriz poligonal ``(z, r)``.

    Los puntos se almacenan como una tupla inmutable. El radio entre dos
    vértices se interpola linealmente y el volumen se integra exactamente como
    una suma de troncos de cono.
    """

    def __init__(self, puntos: Sequence[tuple[float, float]]) -> None:
        puntos_limpios = tuple((float(z), float(r)) for z, r in puntos)
        if len(puntos_limpios) < 2:
            raise ValueError("el perfil necesita al menos dos puntos")

        z = np.asarray([p[0] for p in puntos_limpios], dtype=float)
        radios = np.asarray([p[1] for p in puntos_limpios], dtype=float)
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(radios)):
            raise ValueError("las cotas del perfil deben ser finitas")
        if not np.all(np.diff(z) > 0.0):
            raise ValueError("las cotas z deben ser estrictamente crecientes")
        if not np.all(radios > 0.0):
            raise ValueError("los radios deben ser positivos")

        self.puntos: tuple[tuple[float, float], ...] = puntos_limpios
        self._z = z
        self._radios = radios

    @property
    def z_min(self) -> float:
        return self.puntos[0][0]

    @property
    def z_max(self) -> float:
        return self.puntos[-1][0]

    @property
    def r_base(self) -> float:
        return self.puntos[0][1]

    @property
    def r_boca(self) -> float:
        return self.puntos[-1][1]

    @property
    def r_max(self) -> float:
        return float(np.max(self._radios))

    def radio(self, z: float | np.ndarray) -> float | np.ndarray:
        """Radio interpolado linealmente; acepta escalares o arrays de NumPy.

        Fuera del intervalo del perfil se prolonga el radio extremo, como hace
        :func:`numpy.interp`. La pertenencia al sólido sí restringe explícitamente
        ``z`` al intervalo cerrado ``[z_min, z_max]``.
        """
        z_array = np.asarray(z, dtype=float)
        resultado = np.interp(z_array, self._z, self._radios)
        if z_array.ndim == 0:
            return float(resultado)
        return resultado

    def dentro(self, x: np.ndarray, y: np.ndarray,
               z: np.ndarray) -> np.ndarray:
        """Indica si cada punto está dentro del sólido de revolución."""
        x_arr, y_arr, z_arr = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        r_local = np.asarray(self.radio(z_arr))
        return ((z_arr >= self.z_min) & (z_arr <= self.z_max)
                & (np.hypot(x_arr, y_arr) <= r_local))

    def volumen_mm3(self) -> float:
        """Volumen analítico exacto, sumando los troncos de cada tramo."""
        return float(sum(
            volumen_tronco(r0, r1, z1 - z0)
            for (z0, r0), (z1, r1) in zip(self.puntos[:-1], self.puntos[1:])
        ))

    def exportar_obj(self, ruta: str | Path,
                     segmentos_angulares: int = 96) -> Path:
        """Exporta este perfil como un sólido de revolución cerrado en OBJ."""
        malla = _ConstructorOBJ()
        anillos = _agregar_anillos(malla, self, segmentos_angulares)
        _agregar_superficie_lateral(malla, anillos, exterior=True)
        _agregar_disco(malla, anillos[0], self.z_min, normal_positiva=False)
        _agregar_disco(malla, anillos[-1], self.z_max, normal_positiva=True)
        return malla.escribir(ruta)


def _perfil_desde_collar(altura_collar_mm: float,
                         espesor_collar_mm: float) -> PerfilRevolucion:
    """Construye la parametrización de dos grados de libertad de la foto.

    ``altura_collar_mm`` ubica la arista de radio máximo. El espesor es la
    distancia vertical desde el hombro inferior hasta esa arista. La pequeña
    transición superior visible se mantiene en la proporción 2/3 del espesor;
    así ``(21, 3)`` reproduce exactamente el perfil nominal de CONTRATOS.md.
    """
    h = float(altura_collar_mm)
    e = float(espesor_collar_mm)
    if e <= 0.0:
        raise ValueError("el espesor del collar debe ser positivo")
    z_hombro_inferior = h - e
    z_hombro_superior = h + (2.0 / 3.0) * e
    if not 0.0 < z_hombro_inferior < h < z_hombro_superior < 32.0:
        raise ValueError("la altura y el espesor producen un collar imposible")

    return PerfilRevolucion([
        (0.0, 12.5),
        (z_hombro_inferior, 14.0),
        (h, 15.3),
        (z_hombro_superior, 14.75),
        (32.0, 14.75),
    ])


# Perfil nominal de la fotografía, también consignado como ejemplo de caso en
# docs/CONTRATOS.md. Origen de cada punto:
PERFIL_ENSAYO = PerfilRevolucion([
    (0.0, 12.5),    # Cota medida: base de diámetro 25,0 mm a z = 0.
    (18.0, 14.0),   # Hombro inferido: z = 21 - 3 mm; r=14 sigue la silueta.
    (21.0, 15.3),   # Radio medido (diámetro 30,6); z≈21 mm estimado en la foto.
    (23.0, 14.75),  # Cuello medido de diámetro 29,5; transición superior estimada.
    (32.0, 14.75),  # Cotas medidas: boca de 29,5 mm y altura total de 32 mm.
])

PERFIL_TRONCO_SIMPLE = PerfilRevolucion([
    (0.0, CRISOL_ENSAYO["diam_base_mm"] / 2.0),
    (CRISOL_ENSAYO["altura_mm"], CRISOL_ENSAYO["diam_boca_mm"] / 2.0),
])


class CrisolPerfilado:
    """Crisol hueco cuya pared exterior sigue un :class:`PerfilRevolucion`.

    El perfil interior conserva los radios de cada vértice menos el espesor de
    pared. El primer vértice interior se eleva el espesor de fondo, igual que en
    :class:`nucleo.geometria.Crisol`; de este modo el tronco simple reproduce
    exactamente sus 32,59 g y el fondo se contabiliza por separado de la pared.
    """

    def __init__(
        self,
        perfil_exterior: PerfilRevolucion = PERFIL_ENSAYO,
        espesor_pared_mm: float = CRISOL_ENSAYO["espesor_pared_mm"],
        espesor_fondo_mm: float = CRISOL_ENSAYO["espesor_fondo_mm"],
        rho_kg_m3: float = CRISOL_ENSAYO["rho_kg_m3"],
        espesor_tapa_mm: float = 2.0,
        con_tapa: bool = True,
    ) -> None:
        self.perfil_exterior = perfil_exterior
        self.espesor_pared_mm = float(espesor_pared_mm)
        self.espesor_fondo_mm = float(espesor_fondo_mm)
        self.rho_kg_m3 = float(rho_kg_m3)
        self.espesor_tapa_mm = float(espesor_tapa_mm)
        self.con_tapa = bool(con_tapa)

        if self.espesor_pared_mm <= 0.0 or self.espesor_fondo_mm <= 0.0:
            raise ValueError("los espesores deben ser positivos")
        if self.espesor_fondo_mm >= self.altura_mm:
            raise ValueError("el fondo no cabe en la altura")
        if self.espesor_pared_mm >= min(r for _, r in perfil_exterior.puntos):
            raise ValueError("la pared no cabe dentro del perfil exterior")
        if self.rho_kg_m3 <= 0.0:
            raise ValueError("la densidad debe ser positiva")
        if self.espesor_tapa_mm <= 0.0:
            raise ValueError("el espesor de tapa debe ser positivo")

        puntos_interiores = [
            (perfil_exterior.z_min + self.espesor_fondo_mm,
             perfil_exterior.r_base - self.espesor_pared_mm),
        ]
        puntos_interiores.extend(
            (z, r - self.espesor_pared_mm)
            for z, r in perfil_exterior.puntos[1:]
            if z > puntos_interiores[0][0]
        )
        self.perfil_interior = PerfilRevolucion(puntos_interiores)

    # -- Dimensiones y radios compatibles con geometria.Crisol ------------
    @property
    def altura_mm(self) -> float:
        return self.perfil_exterior.z_max - self.perfil_exterior.z_min

    @property
    def diam_base_mm(self) -> float:
        return 2.0 * self.r_base_ext

    @property
    def diam_boca_mm(self) -> float:
        return 2.0 * self.r_boca_ext

    @property
    def r_base_ext(self) -> float:
        return self.perfil_exterior.r_base

    @property
    def r_boca_ext(self) -> float:
        return self.perfil_exterior.r_boca

    @property
    def r_base_int(self) -> float:
        return self.perfil_interior.r_base

    @property
    def r_boca_int(self) -> float:
        return self.perfil_interior.r_boca

    @property
    def r_max_ext(self) -> float:
        """Máximo radio exterior, útil para encerrar el collar en una malla."""
        return self.perfil_exterior.r_max

    # -- Volúmenes y masa --------------------------------------------------
    def volumen_exterior_mm3(self) -> float:
        return self.perfil_exterior.volumen_mm3()

    def volumen_interior_mm3(self) -> float:
        """Capacidad útil de la cavidad por encima del fondo."""
        return self.perfil_interior.volumen_mm3()

    def volumen_material_mm3(self) -> float:
        return self.volumen_exterior_mm3() - self.volumen_interior_mm3()

    def masa_calculada_g(self) -> float:
        return self.volumen_material_mm3() * self.rho_kg_m3 / 1.0e6

    def verificar_masa(self, masa_declarada_g: float | None = None
                       ) -> dict[str, float]:
        declarada = (CRISOL_ENSAYO["masa_declarada_g"]
                     if masa_declarada_g is None else float(masa_declarada_g))
        calculada = self.masa_calculada_g()
        return {
            "masa_calculada_g": calculada,
            "masa_declarada_g": declarada,
            "error_relativo": (calculada - declarada) / declarada,
            "error_pct": 100.0 * (calculada - declarada) / declarada,
        }

    # -- Funciones implícitas ---------------------------------------------
    def dentro_exterior(self):
        return self.perfil_exterior.dentro

    def dentro_cavidad(self):
        return self.perfil_interior.dentro

    def dentro_tapa(self):
        if not self.con_tapa:
            return lambda x, y, z: np.zeros_like(z, dtype=bool)

        z0 = self.perfil_exterior.z_max
        z1 = z0 + self.espesor_tapa_mm
        radio_tapa = self.r_boca_ext

        def dentro(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            return ((np.asarray(z) >= z0) & (np.asarray(z) <= z1)
                    & (np.hypot(x, y) <= radio_tapa))

        return dentro

    def altura_total_mm(self) -> float:
        return self.altura_mm + (self.espesor_tapa_mm if self.con_tapa else 0.0)

    def exportar_obj(self, ruta: str | Path,
                     segmentos_angulares: int = 96) -> Path:
        """Exporta la superficie hueca del crisol, sin incluir la tapa."""
        malla = _ConstructorOBJ()
        exterior = _agregar_anillos(
            malla, self.perfil_exterior, segmentos_angulares)
        interior = _agregar_anillos(
            malla, self.perfil_interior, segmentos_angulares)

        _agregar_superficie_lateral(malla, exterior, exterior=True)
        _agregar_superficie_lateral(malla, interior, exterior=False)
        _agregar_disco(
            malla, exterior[0], self.perfil_exterior.z_min,
            normal_positiva=False)
        _agregar_disco(
            malla, interior[0], self.perfil_interior.z_min,
            normal_positiva=True)
        _agregar_anillo_plano(malla, exterior[-1], interior[-1])
        return malla.escribir(ruta)


def ajustar_collar_a_masa(
    masa_objetivo_g: float = CRISOL_ENSAYO["masa_declarada_g"],
    tolerancia_g: float = 1.0e-8,
) -> dict[str, float | bool | str]:
    """Ajusta altura y espesor del collar a la masa declarada.

    Una masa aporta una sola restricción para dos incógnitas. Para identificar
    una solución se escoge, entre las que reproducen la masa, la de menor cambio
    normalizado respecto a las estimaciones fotográficas ``21±3 mm`` y
    ``3±1,5 mm``. Primero se obtiene factibilidad con mínimos cuadrados acotados
    y luego se aplica la restricción exacta conservando ese criterio de cercanía.

    Los límites de búsqueda son deliberadamente más amplios que el intervalo
    considerado compatible con la foto. Así la función puede avisar, en vez de
    ocultarlo mediante límites, si una masa exige un collar no creíble.
    """
    if masa_objetivo_g <= 0.0 or tolerancia_g <= 0.0:
        raise ValueError("la masa objetivo y la tolerancia deben ser positivas")

    from scipy.optimize import least_squares, minimize

    nominal = np.array([21.0, 3.0], dtype=float)
    incertidumbre = np.array([3.0, 1.5], dtype=float)
    inferior = np.array([16.0, 0.5], dtype=float)
    superior = np.array([26.0, 8.0], dtype=float)

    def masa(parametros: np.ndarray) -> float:
        perfil = _perfil_desde_collar(float(parametros[0]), float(parametros[1]))
        return CrisolPerfilado(perfil).masa_calculada_g()

    def residuo_masa(parametros: np.ndarray) -> float:
        return masa(parametros) - masa_objetivo_g

    factible = least_squares(
        lambda p: np.array([residuo_masa(p)]),
        nominal,
        bounds=(inferior, superior),
        x_scale=incertidumbre,
        ftol=1.0e-14,
        xtol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=1000,
    )

    def distancia_foto(parametros: np.ndarray) -> float:
        desviacion = (parametros - nominal) / incertidumbre
        return float(np.dot(desviacion, desviacion))

    seleccionado = minimize(
        distancia_foto,
        factible.x,
        method="SLSQP",
        bounds=list(zip(inferior, superior)),
        constraints={"type": "eq", "fun": residuo_masa},
        options={"ftol": 1.0e-13, "maxiter": 500},
    )
    parametros = (seleccionado.x if seleccionado.success else factible.x)
    altura, espesor = (float(parametros[0]), float(parametros[1]))
    masa_ajustada = masa(parametros)
    residuo = masa_ajustada - masa_objetivo_g
    convergio = abs(residuo) <= tolerancia_g

    altura_plausible = 18.0 <= altura <= 24.0
    espesor_plausible = 1.0 <= espesor <= 6.0
    compatible = bool(convergio and altura_plausible and espesor_plausible)
    if not convergio:
        aviso = (
            "ADVERTENCIA: el ajuste no reproduce la masa dentro de la "
            f"tolerancia ({residuo:+.3e} g)."
        )
    elif compatible:
        aviso = (
            "Compatible con la fotografía: altura y espesor quedan dentro "
            "de los intervalos visualmente plausibles."
        )
    else:
        causas = []
        if not altura_plausible:
            causas.append("altura fuera de 18–24 mm")
        if not espesor_plausible:
            causas.append("espesor fuera de 1–6 mm")
        aviso = (
            "ADVERTENCIA: la masa puede ajustarse, pero el collar no es "
            "compatible con la fotografía (" + ", ".join(causas) + ")."
        )

    return {
        "altura_collar_mm": altura,
        "espesor_collar_mm": espesor,
        "masa_objetivo_g": float(masa_objetivo_g),
        "masa_ajustada_g": masa_ajustada,
        "residuo_g": residuo,
        "convergio": convergio,
        "compatible_fotografia": compatible,
        "aviso": aviso,
    }


def comparar_con_tronco_simple() -> list[dict[str, float | str]]:
    """Compara el collar ajustado con el tronco simple usado anteriormente."""
    ajuste = ajustar_collar_a_masa()
    perfil_ajustado = _perfil_desde_collar(
        float(ajuste["altura_collar_mm"]),
        float(ajuste["espesor_collar_mm"]),
    )
    modelos = [
        ("Tronco simple", Crisol()),
        ("Perfil con collar (ajustado)", CrisolPerfilado(perfil_ajustado)),
    ]
    declarada = CRISOL_ENSAYO["masa_declarada_g"]
    filas: list[dict[str, float | str]] = []
    for nombre, crisol in modelos:
        masa = crisol.masa_calculada_g()
        filas.append({
            "modelo": nombre,
            "masa_g": masa,
            "volumen_material_mm3": crisol.volumen_material_mm3(),
            "capacidad_util_mm3": crisol.volumen_interior_mm3(),
            "error_masa_g": masa - declarada,
            "error_masa_pct": 100.0 * (masa - declarada) / declarada,
        })
    return filas


class _ConstructorOBJ:
    """Acumulador mínimo de triángulos OBJ con una normal por cara."""

    def __init__(self) -> None:
        self.vertices: list[np.ndarray] = []
        self.normales: list[np.ndarray] = []
        self.caras: list[tuple[int, int, int, int]] = []

    def agregar_vertice(self, x: float, y: float, z: float) -> int:
        self.vertices.append(np.array([x, y, z], dtype=float))
        return len(self.vertices)

    def agregar_triangulo(self, i: int, j: int, k: int) -> None:
        a, b, c = (self.vertices[indice - 1] for indice in (i, j, k))
        normal = np.cross(b - a, c - a)
        norma = float(np.linalg.norm(normal))
        if norma <= 1.0e-12:
            raise ValueError("la triangulación produjo una cara degenerada")
        self.normales.append(normal / norma)
        self.caras.append((i, j, k, len(self.normales)))

    def escribir(self, ruta: str | Path) -> Path:
        destino = Path(ruta)
        lineas = ["# Crisol perfilado: superficie de revolución en mm"]
        lineas.extend(
            f"v {v[0]:.12g} {v[1]:.12g} {v[2]:.12g}"
            for v in self.vertices
        )
        lineas.extend(
            f"vn {n[0]:.12g} {n[1]:.12g} {n[2]:.12g}"
            for n in self.normales
        )
        lineas.extend(
            f"f {i}//{n} {j}//{n} {k}//{n}"
            for i, j, k, n in self.caras
        )
        destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        return destino


def _validar_segmentos(segmentos_angulares: int) -> int:
    segmentos = int(segmentos_angulares)
    if segmentos < 3 or segmentos != segmentos_angulares:
        raise ValueError("segmentos_angulares debe ser un entero >= 3")
    return segmentos


def _agregar_anillos(malla: _ConstructorOBJ, perfil: PerfilRevolucion,
                      segmentos_angulares: int) -> list[list[int]]:
    segmentos = _validar_segmentos(segmentos_angulares)
    angulos = 2.0 * math.pi * np.arange(segmentos) / segmentos
    anillos: list[list[int]] = []
    for z, radio in perfil.puntos:
        anillos.append([
            malla.agregar_vertice(
                radio * math.cos(float(angulo)),
                radio * math.sin(float(angulo)),
                z,
            )
            for angulo in angulos
        ])
    return anillos


def _agregar_superficie_lateral(malla: _ConstructorOBJ,
                                anillos: list[list[int]],
                                exterior: bool) -> None:
    segmentos = len(anillos[0])
    for inferior, superior in zip(anillos[:-1], anillos[1:]):
        for i in range(segmentos):
            siguiente = (i + 1) % segmentos
            a, b = inferior[i], inferior[siguiente]
            c, d = superior[siguiente], superior[i]
            if exterior:
                malla.agregar_triangulo(a, b, c)
                malla.agregar_triangulo(a, c, d)
            else:  # La normal de la cavidad apunta hacia el eje.
                malla.agregar_triangulo(a, c, b)
                malla.agregar_triangulo(a, d, c)


def _agregar_disco(malla: _ConstructorOBJ, anillo: list[int], z: float,
                   normal_positiva: bool) -> None:
    centro = malla.agregar_vertice(0.0, 0.0, z)
    segmentos = len(anillo)
    for i in range(segmentos):
        siguiente = (i + 1) % segmentos
        if normal_positiva:
            malla.agregar_triangulo(centro, anillo[i], anillo[siguiente])
        else:
            malla.agregar_triangulo(centro, anillo[siguiente], anillo[i])


def _agregar_anillo_plano(malla: _ConstructorOBJ, exterior: list[int],
                          interior: list[int]) -> None:
    """Cierra el labio superior con normales hacia ``+z``."""
    segmentos = len(exterior)
    for i in range(segmentos):
        siguiente = (i + 1) % segmentos
        malla.agregar_triangulo(
            exterior[i], exterior[siguiente], interior[siguiente])
        malla.agregar_triangulo(
            exterior[i], interior[siguiente], interior[i])


def exportar_obj(ruta: str | Path, segmentos_angulares: int = 96) -> Path:
    """Exporta el crisol nominal de la fotografía a Wavefront OBJ."""
    return CrisolPerfilado(PERFIL_ENSAYO).exportar_obj(
        ruta, segmentos_angulares=segmentos_angulares)


def _imprimir_tabla_comparativa() -> None:
    filas = comparar_con_tronco_simple()
    print("COMPARACIÓN GEOMÉTRICA DEL CRISOL")
    print("-" * 100)
    print(
        f"{'Modelo':<31} {'masa (g)':>10} {'V material (mm³)':>18} "
        f"{'capacidad (cm³)':>18} {'error masa (g)':>16}"
    )
    for fila in filas:
        print(
            f"{fila['modelo']:<31} {fila['masa_g']:10.5f} "
            f"{fila['volumen_material_mm3']:18.3f} "
            f"{fila['capacidad_util_mm3'] / 1000.0:18.3f} "
            f"{fila['error_masa_g']:+16.5f}"
        )

    ajuste = ajustar_collar_a_masa()
    print("\nAjuste del collar a 32,67 g:")
    print(f"  altura  = {ajuste['altura_collar_mm']:.4f} mm")
    print(f"  espesor = {ajuste['espesor_collar_mm']:.4f} mm")
    print(f"  residuo = {ajuste['residuo_g']:+.3e} g")
    print(f"  {ajuste['aviso']}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _imprimir_tabla_comparativa()
