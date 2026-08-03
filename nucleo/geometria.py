"""
Geometría 3D del dominio de simulación: crisol, tapa, lecho y espacio de gas.

Todo se construye a partir de las cotas reales del crisol Ni--Cr del ensayo, que
están en ``simulacion_v3/src/parametros_literatura.py``:

    diámetro base 25,0 mm · diámetro boca 29,5 mm · altura 32,0 mm
    espesor de pared 1,1 mm · espesor de fondo 2,0 mm · rho 8400 kg/m3

Estas cotas están **verificadas**: el volumen de material que implican
(3,880 cm3) multiplicado por la densidad da 32,59 g, frente a los 32,67 g
declarados para el crisol. La discrepancia es del 0,2 %, dentro de lo esperable
por los radios de acuerdo y el labio de la boca, que no se modelan.

REPRESENTACIÓN
--------------
Se usa una descripción implícita (función de distancia con signo aproximada) por
sólido primitivo, combinada con operaciones booleanas. De ahí se obtiene:

  * una máscara de vóxeles por material, para el solucionador de volúmenes
    finitos,
  * una malla de superficie por *marching cubes*, para la visualización 3D.

Se eligió CSG implícito en vez de una malla no estructurada porque el dominio es
axisimétrico y sencillo, porque los vóxeles hacen trivial el acoplamiento con un
esquema conservativo de volúmenes finitos, y porque permite importar geometrías
externas más adelante sin cambiar el solucionador.

ADVERTENCIA DE ESCALA
---------------------
El lecho de 1 g con porosidad 0,54 ocupa 1,36 cm3, lo que en el fondo del crisol
son sólo **2,77 mm de altura** frente a 25 mm de diámetro: es un disco delgado,
con relación de aspecto cercana a 9:1. Cualquier malla uniforme que resuelva bien
el espesor del lecho será innecesariamente fina en el plano horizontal. Por eso
``MallaVoxel`` admite espaciado anisótropo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Cotas del crisol del ensayo (mm). Fuente: parametros_literatura.CRISOL.
# ---------------------------------------------------------------------------
CRISOL_ENSAYO: dict[str, float] = {
    "diam_base_mm": 25.0,
    "diam_boca_mm": 29.5,
    "altura_mm": 32.0,
    "espesor_pared_mm": 1.1,
    "espesor_fondo_mm": 2.0,
    "rho_kg_m3": 8400.0,
    "masa_declarada_g": 32.67,
    "masa_tapa_declarada_g": 15.87,
    "k_W_mK": 16.0,
    "emisividad": 0.80,
    "Cp_J_gK": 0.50,
}

# Identificadores de material en la máscara de vóxeles.
VACIO = 0
PARED_CRISOL = 1
TAPA = 2
LECHO = 3
GAS = 4

NOMBRES_MATERIAL = {
    VACIO: "exterior",
    PARED_CRISOL: "pared del crisol",
    TAPA: "tapa",
    LECHO: "lecho reactivo",
    GAS: "gas interior",
}


# ---------------------------------------------------------------------------
# Primitivas implícitas
# ---------------------------------------------------------------------------
def tronco_de_cono(r_base: float, r_boca: float, z0: float, z1: float
                   ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Devuelve ``dentro(x, y, z)`` para un tronco de cono de eje Z.

    El radio interpola linealmente entre ``r_base`` en ``z0`` y ``r_boca`` en
    ``z1``. Fuera del intervalo de altura devuelve ``False``.
    """
    if z1 <= z0:
        raise ValueError("z1 debe ser mayor que z0")

    def dentro(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        t = (z - z0) / (z1 - z0)
        r_local = r_base + (r_boca - r_base) * t
        radio = np.hypot(x, y)
        return (z >= z0) & (z <= z1) & (radio <= r_local)

    return dentro


def volumen_tronco(r1: float, r2: float, altura: float) -> float:
    """Volumen exacto de un tronco de cono. Sirve de referencia de verificación."""
    return math.pi * altura * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0


# ---------------------------------------------------------------------------
# Definición del crisol
# ---------------------------------------------------------------------------
@dataclass
class Crisol:
    """Crisol troncocónico hueco, con tapa opcional."""

    diam_base_mm: float = CRISOL_ENSAYO["diam_base_mm"]
    diam_boca_mm: float = CRISOL_ENSAYO["diam_boca_mm"]
    altura_mm: float = CRISOL_ENSAYO["altura_mm"]
    espesor_pared_mm: float = CRISOL_ENSAYO["espesor_pared_mm"]
    espesor_fondo_mm: float = CRISOL_ENSAYO["espesor_fondo_mm"]
    rho_kg_m3: float = CRISOL_ENSAYO["rho_kg_m3"]
    espesor_tapa_mm: float = 2.0
    con_tapa: bool = True

    def __post_init__(self) -> None:
        if self.espesor_pared_mm <= 0 or self.espesor_fondo_mm <= 0:
            raise ValueError("los espesores deben ser positivos")
        if 2 * self.espesor_pared_mm >= self.diam_base_mm:
            raise ValueError("la pared no cabe en el diámetro de la base")
        if self.espesor_fondo_mm >= self.altura_mm:
            raise ValueError("el fondo no cabe en la altura")

    # -- radios auxiliares --------------------------------------------------
    @property
    def r_base_ext(self) -> float:
        return self.diam_base_mm / 2.0

    @property
    def r_boca_ext(self) -> float:
        return self.diam_boca_mm / 2.0

    @property
    def r_base_int(self) -> float:
        return self.r_base_ext - self.espesor_pared_mm

    @property
    def r_boca_int(self) -> float:
        return self.r_boca_ext - self.espesor_pared_mm

    # -- volúmenes analíticos ----------------------------------------------
    def volumen_exterior_mm3(self) -> float:
        return volumen_tronco(self.r_base_ext, self.r_boca_ext, self.altura_mm)

    def volumen_interior_mm3(self) -> float:
        """Capacidad útil: cavidad sobre el fondo."""
        return volumen_tronco(self.r_base_int, self.r_boca_int,
                              self.altura_mm - self.espesor_fondo_mm)

    def volumen_material_mm3(self) -> float:
        return self.volumen_exterior_mm3() - self.volumen_interior_mm3()

    def masa_calculada_g(self) -> float:
        return self.volumen_material_mm3() * self.rho_kg_m3 / 1.0e6

    def verificar_masa(self, masa_declarada_g: float | None = None) -> dict[str, float]:
        """Contrasta la masa que implica la geometría con la declarada.

        Es la comprobación que valida las cotas: si el error fuese grande, las
        dimensiones o la densidad serían incorrectas.
        """
        declarada = (CRISOL_ENSAYO["masa_declarada_g"]
                     if masa_declarada_g is None else masa_declarada_g)
        calculada = self.masa_calculada_g()
        return {
            "masa_calculada_g": calculada,
            "masa_declarada_g": declarada,
            "error_relativo": (calculada - declarada) / declarada,
            "error_pct": 100.0 * (calculada - declarada) / declarada,
        }

    # -- funciones implícitas ----------------------------------------------
    def dentro_exterior(self):
        return tronco_de_cono(self.r_base_ext, self.r_boca_ext, 0.0, self.altura_mm)

    def dentro_cavidad(self):
        return tronco_de_cono(self.r_base_int, self.r_boca_int,
                              self.espesor_fondo_mm, self.altura_mm)

    def dentro_tapa(self):
        if not self.con_tapa:
            return lambda x, y, z: np.zeros_like(z, dtype=bool)
        return tronco_de_cono(self.r_boca_ext, self.r_boca_ext,
                              self.altura_mm, self.altura_mm + self.espesor_tapa_mm)

    def altura_total_mm(self) -> float:
        return self.altura_mm + (self.espesor_tapa_mm if self.con_tapa else 0.0)


# ---------------------------------------------------------------------------
# Lecho
# ---------------------------------------------------------------------------
@dataclass
class Lecho:
    """Lecho granular en el fondo del crisol.

    La altura no se impone: se deduce de la masa, las densidades y la porosidad,
    resolviendo el volumen del tronco de cono que ocupa el lecho. Para la carga
    del ensayo (0,75 g de carbón + 0,25 g de concentrado, porosidad 0,54) salen
    unos 2,8 mm, es decir, un disco delgado.
    """

    masa_carbon_g: float = 0.75
    masa_mineral_g: float = 0.25
    rho_carbon_g_cm3: float = 1.3
    rho_mineral_g_cm3: float = 5.17
    porosidad: float = 0.54

    def volumen_solido_cm3(self) -> float:
        return (self.masa_carbon_g / self.rho_carbon_g_cm3
                + self.masa_mineral_g / self.rho_mineral_g_cm3)

    def volumen_lecho_cm3(self) -> float:
        if not 0.0 <= self.porosidad < 1.0:
            raise ValueError("la porosidad debe estar en [0, 1)")
        return self.volumen_solido_cm3() / (1.0 - self.porosidad)

    def altura_en_crisol_mm(self, crisol: Crisol) -> float:
        """Altura del lecho resolviendo el volumen del tronco de cono."""
        objetivo = self.volumen_lecho_cm3() * 1000.0  # mm3
        z0 = crisol.espesor_fondo_mm
        r0 = crisol.r_base_int
        pendiente = ((crisol.r_boca_int - crisol.r_base_int)
                     / (crisol.altura_mm - crisol.espesor_fondo_mm))

        def volumen(h: float) -> float:
            return volumen_tronco(r0, r0 + pendiente * h, h)

        h_max = crisol.altura_mm - crisol.espesor_fondo_mm
        if volumen(h_max) < objetivo:
            raise ValueError("la carga no cabe en el crisol")
        lo, hi = 0.0, h_max
        for _ in range(200):  # bisección; la función es monótona creciente
            mid = 0.5 * (lo + hi)
            if volumen(mid) < objetivo:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def dentro(self, crisol: Crisol):
        h = self.altura_en_crisol_mm(crisol)
        z0 = crisol.espesor_fondo_mm
        pendiente = ((crisol.r_boca_int - crisol.r_base_int)
                     / (crisol.altura_mm - crisol.espesor_fondo_mm))
        return tronco_de_cono(crisol.r_base_int,
                              crisol.r_base_int + pendiente * h, z0, z0 + h)


# ---------------------------------------------------------------------------
# Malla de vóxeles
# ---------------------------------------------------------------------------
@dataclass
class MallaVoxel:
    """Malla cartesiana con espaciado posiblemente anisótropo.

    El espaciado se separa en plano (dx = dy) y vertical (dz) porque el lecho es
    un disco delgado: resolverlo con una malla isótropa obligaría a un número de
    celdas mucho mayor sin ganancia física.
    """

    dx_mm: float
    dz_mm: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    z: np.ndarray = field(init=False)
    forma: tuple[int, int, int] = field(init=False)
    _origen: tuple[float, float, float] = field(init=False)

    def __init__(self, crisol: Crisol, dx_mm: float = 0.5,
                 dz_mm: float | None = None, margen_mm: float = 0.0) -> None:
        if dx_mm <= 0:
            raise ValueError("dx_mm debe ser positivo")
        self.dx_mm = float(dx_mm)
        self.dz_mm = float(dz_mm if dz_mm is not None else dx_mm)

        r_max = max(crisol.r_base_ext, crisol.r_boca_ext) + margen_mm
        z_max = crisol.altura_total_mm() + margen_mm
        nx = int(math.ceil(2.0 * r_max / self.dx_mm))
        nz = int(math.ceil(z_max / self.dz_mm))
        # centros de celda
        self.x = (np.arange(nx) + 0.5) * self.dx_mm - r_max
        self.y = self.x.copy()
        self.z = (np.arange(nz) + 0.5) * self.dz_mm
        self.forma = (nx, nx, nz)
        self._origen = (-r_max, -r_max, 0.0)

    @property
    def n_celdas(self) -> int:
        return int(np.prod(self.forma))

    @property
    def volumen_celda_mm3(self) -> float:
        return self.dx_mm * self.dx_mm * self.dz_mm

    def rejilla(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")

    def etiquetar(self, crisol: Crisol, lecho: Lecho | None = None) -> np.ndarray:
        """Máscara de materiales por vóxel.

        El orden de asignación importa: se parte del sólido del crisol y se van
        vaciando y rellenando regiones, de fuera hacia dentro.
        """
        X, Y, Z = self.rejilla()
        etiquetas = np.full(self.forma, VACIO, dtype=np.uint8)

        exterior = crisol.dentro_exterior()(X, Y, Z)
        cavidad = crisol.dentro_cavidad()(X, Y, Z)
        etiquetas[exterior & ~cavidad] = PARED_CRISOL
        etiquetas[cavidad] = GAS

        if crisol.con_tapa:
            etiquetas[crisol.dentro_tapa()(X, Y, Z)] = TAPA

        if lecho is not None:
            etiquetas[lecho.dentro(crisol)(X, Y, Z) & cavidad] = LECHO

        return etiquetas

    def volumenes_por_material(self, etiquetas: np.ndarray) -> dict[str, float]:
        """Volumen en cm3 de cada material, contando vóxeles."""
        vc = self.volumen_celda_mm3 / 1000.0
        return {NOMBRES_MATERIAL[m]: float((etiquetas == m).sum()) * vc
                for m in NOMBRES_MATERIAL}


def fraccion_volumetrica(malla: "MallaVoxel",
                         dentro: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                         submuestreo: int = 4) -> np.ndarray:
    """Fracción de cada celda ocupada por el sólido, en [0, 1].

    Una máscara binaria asigna la celda entera al material o al vacío, lo que
    convierte una superficie curva en una escalera. El error de volumen que eso
    introduce es O(h) y, lo que es peor, **no decrece de forma monótona** al
    refinar: depende de dónde caiga la frontera respecto a los centros de celda.

    Aquí cada celda se submuestrea en ``submuestreo**3`` puntos y se cuenta la
    proporción interior. El resultado es la fracción volumétrica que necesita un
    esquema conservativo de volúmenes finitos para tratar las celdas cortadas por
    la frontera, y de paso da un volumen mucho más preciso.

    El coste es ``submuestreo**3`` evaluaciones por celda, así que se calcula una
    sola vez al construir el dominio.
    """
    if submuestreo < 1:
        raise ValueError("submuestreo debe ser >= 1")
    n = submuestreo
    # desplazamientos de los subpuntos dentro de la celda, centrados
    d = (np.arange(n) + 0.5) / n - 0.5
    acumulado = np.zeros(malla.forma, dtype=np.float64)
    X, Y, Z = malla.rejilla()
    for ix in d:
        for iy in d:
            for iz in d:
                acumulado += dentro(X + ix * malla.dx_mm,
                                    Y + iy * malla.dx_mm,
                                    Z + iz * malla.dz_mm)
    return acumulado / float(n ** 3)


def convergencia_volumen(crisol: Crisol,
                         espaciados: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25),
                         submuestreo: int = 4) -> list[dict[str, Any]]:
    """Compara el volumen del material del crisol contra el valor analítico.

    Se calcula de dos formas para cada malla: contando vóxeles enteros (máscara
    binaria) y con fracciones volumétricas parciales. La primera converge como
    O(h) y de forma errática; la segunda es mucho más precisa y regular. La
    comparación documenta por qué el solucionador usará fracciones parciales.
    """
    exacto = crisol.volumen_material_mm3() / 1000.0
    f_ext = crisol.dentro_exterior()
    f_cav = crisol.dentro_cavidad()
    filas = []
    for h in espaciados:
        malla = MallaVoxel(crisol, dx_mm=h, dz_mm=h)
        vc = malla.volumen_celda_mm3 / 1000.0

        etiquetas = malla.etiquetar(crisol)
        v_bin = malla.volumenes_por_material(etiquetas)["pared del crisol"]

        frac = np.clip(fraccion_volumetrica(malla, f_ext, submuestreo)
                       - fraccion_volumetrica(malla, f_cav, submuestreo), 0.0, 1.0)
        v_frac = float(frac.sum()) * vc

        filas.append({
            "dx_mm": h,
            "n_celdas": malla.n_celdas,
            "volumen_exacto_cm3": exacto,
            "volumen_binario_cm3": v_bin,
            "error_binario": abs(v_bin - exacto) / exacto,
            "volumen_fraccion_cm3": v_frac,
            "error_fraccion": abs(v_frac - exacto) / exacto,
        })
    return filas


def orden_de_convergencia(filas: list[dict[str, Any]], clave: str) -> list[float]:
    """Orden observado entre mallas consecutivas: ``log(e1/e2)/log(h1/h2)``."""
    ordenes = []
    for a, b in zip(filas[:-1], filas[1:]):
        ea, eb = a[clave], b[clave]
        if ea > 0 and eb > 0 and a["dx_mm"] != b["dx_mm"]:
            ordenes.append(math.log(ea / eb) / math.log(a["dx_mm"] / b["dx_mm"]))
        else:
            ordenes.append(float("nan"))
    return ordenes


def dominio_del_ensayo(dx_mm: float = 0.5, dz_mm: float = 0.25
                       ) -> dict[str, Any]:
    """Construye el dominio completo del ensayo carbón--titanomagnetita."""
    crisol = Crisol()
    lecho = Lecho()
    malla = MallaVoxel(crisol, dx_mm=dx_mm, dz_mm=dz_mm)
    etiquetas = malla.etiquetar(crisol, lecho)
    return {
        "crisol": crisol,
        "lecho": lecho,
        "malla": malla,
        "etiquetas": etiquetas,
        "verificacion_masa": crisol.verificar_masa(),
        "altura_lecho_mm": lecho.altura_en_crisol_mm(crisol),
        "volumenes_cm3": malla.volumenes_por_material(etiquetas),
        "capacidad_util_cm3": crisol.volumen_interior_mm3() / 1000.0,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    d = dominio_del_ensayo()
    c, l, m = d["crisol"], d["lecho"], d["malla"]

    print("=" * 66)
    print("GEOMETRÍA DEL DOMINIO — ensayo carbón/titanomagnetita")
    print("=" * 66)
    print(f"Crisol: ⌀{c.diam_base_mm}→{c.diam_boca_mm} mm · altura {c.altura_mm} mm")
    print(f"        pared {c.espesor_pared_mm} mm · fondo {c.espesor_fondo_mm} mm")

    v = d["verificacion_masa"]
    print(f"\nVerificación de las cotas contra la masa declarada:")
    print(f"  masa desde la geometría : {v['masa_calculada_g']:.2f} g")
    print(f"  masa declarada          : {v['masa_declarada_g']:.2f} g")
    print(f"  error                   : {v['error_pct']:+.2f} %")

    print(f"\nCapacidad útil del crisol : {d['capacidad_util_cm3']:.3f} cm3")
    print(f"Volumen del lecho         : {l.volumen_lecho_cm3():.3f} cm3")
    print(f"Altura del lecho          : {d['altura_lecho_mm']:.2f} mm"
          f"   (relación de aspecto ≈ {c.diam_base_mm/d['altura_lecho_mm']:.1f}:1)")

    print(f"\nMalla: {m.forma[0]}×{m.forma[1]}×{m.forma[2]} = {m.n_celdas:,} celdas"
          f"  (dx={m.dx_mm} mm, dz={m.dz_mm} mm)")
    for nombre, vol in d["volumenes_cm3"].items():
        if vol > 0:
            print(f"  {nombre:<20s} {vol:7.3f} cm3")

    print("\nConvergencia del volumen (material del crisol), dos métodos:")
    filas = convergencia_volumen(c, espaciados=(2.0, 1.0, 0.5))
    print(f"  {'dx (mm)':>8} {'celdas':>11} | {'binario':>10} {'error':>9}"
          f" | {'fracción':>10} {'error':>9}")
    for f in filas:
        print(f"  {f['dx_mm']:8.2f} {f['n_celdas']:11,} |"
              f" {f['volumen_binario_cm3']:10.4f} {f['error_binario']:9.2e} |"
              f" {f['volumen_fraccion_cm3']:10.4f} {f['error_fraccion']:9.2e}")
    ob = orden_de_convergencia(filas, "error_binario")
    of = orden_de_convergencia(filas, "error_fraccion")
    print(f"  orden observado  binario: {[f'{v:.2f}' for v in ob]}"
          f"   fracción: {[f'{v:.2f}' for v in of]}")
    print(f"  exacto = {filas[0]['volumen_exacto_cm3']:.4f} cm3")
