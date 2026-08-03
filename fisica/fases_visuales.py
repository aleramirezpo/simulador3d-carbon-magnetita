"""
Aspecto visual de cada fase: color real del mineral y textura.

Los colores NO son decorativos: corresponden al aspecto real de cada mineral,
para que la visualización sea reconocible por alguien que haya visto las fases
en el laboratorio. Cada entrada cita la referencia del color.

Se usa para colorear las partículas y el aglomerado según su composición local,
de modo que al reproducir la simulación se vea aparecer cada fase nueva con su
color propio.

La mezcla de colores se pondera por **fracción volumétrica**, no por moles: lo
que el ojo ve de un polvo es la superficie que cada fase ocupa, y un mol de
hierro (7,09 cm³) y uno de magnetita (44,8 cm³) no ocupan lo mismo. Los campos
del solucionador están en mol/m³, así que se convierten con el volumen molar
M/rho de cada fase. Las densidades vienen del catálogo de fases de
``simulacion_v3/src/superficie.py`` (mineralogía de Anthony et al., 1990-2001;
FeO y Fe del CRC Handbook, 97.ª ed.), y son de sólidos densos a temperatura
ambiente: no son densidades bulk del aglomerado.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# color: hexadecimal aproximado del mineral en polvo (raya/apariencia macroscópica)
# metalicidad y rugosidad: parámetros de material físico para el render
# densidad_g_cm3: sólido denso, para poder pasar de mol/m³ a fracción de volumen
# grupo: a qué población de partículas pertenece la fase en la vista discreta
FASES: dict[str, dict[str, Any]] = {
    "Fe3O4": {
        "nombre": "Magnetita",
        "color": "#2B2B30", "metalicidad": 0.75, "rugosidad": 0.35,
        "descripcion": "negro con brillo metálico submetálico; raya negra",
        "referencia": "Cornell & Schwertmann, The Iron Oxides, 2003",
        "estado": "inicial", "grupo": "mineral",
        "masa_molar_g_mol": 231.531, "densidad_g_cm3": 5.17,
    },
    "Fe2O3": {
        "nombre": "Hematita",
        "color": "#8B2314", "metalicidad": 0.25, "rugosidad": 0.55,
        "descripcion": "rojo pardo característico; raya rojo ladrillo",
        "referencia": "Cornell & Schwertmann, 2003",
        "estado": "inicial", "grupo": "mineral",
        "masa_molar_g_mol": 159.687, "densidad_g_cm3": 5.26,
    },
    "FeTiO3": {
        "nombre": "Ilmenita",
        "color": "#1F1B22", "metalicidad": 0.60, "rugosidad": 0.45,
        "descripcion": "negro hierro, ligeramente violáceo; raya negra a parda",
        "referencia": "Deer, Howie & Zussman, Rock-Forming Minerals",
        "estado": "inicial", "grupo": "mineral",
        "masa_molar_g_mol": 151.709, "densidad_g_cm3": 4.72,
    },
    "SiO2": {
        "nombre": "Cuarzo",
        "color": "#DCD8CE", "metalicidad": 0.0, "rugosidad": 0.25,
        "descripcion": "incoloro a blanco lechoso, translúcido",
        "referencia": "Deer, Howie & Zussman",
        "estado": "inicial", "grupo": "mineral",
        "masa_molar_g_mol": 60.083, "densidad_g_cm3": 2.65,
    },
    "FeO": {
        "nombre": "Wüstita",
        "color": "#3A3230", "metalicidad": 0.40, "rugosidad": 0.60,
        "descripcion": "negro pardo mate; no estequiométrica (Fe(1-x)O)",
        "referencia": "Cornell & Schwertmann, 2003",
        "estado": "producto", "grupo": "mineral",
        "masa_molar_g_mol": 71.844, "densidad_g_cm3": 5.745,
    },
    "Fe": {
        "nombre": "Hierro metálico",
        "color": "#B7B2AA", "metalicidad": 1.0, "rugosidad": 0.30,
        "descripcion": "gris plateado metálico; suele aparecer como whiskers",
        "referencia": "aspecto del Fe reducido en DRI",
        "estado": "producto", "grupo": "mineral",
        "masa_molar_g_mol": 55.845, "densidad_g_cm3": 7.874,
    },
    "TiO2": {
        "nombre": "Rutilo",
        "color": "#C08A3E", "metalicidad": 0.20, "rugosidad": 0.40,
        "descripcion": "pardo rojizo a amarillento; raya parda clara",
        "referencia": "Deer, Howie & Zussman",
        "estado": "producto", "grupo": "mineral",
        "masa_molar_g_mol": 79.866, "densidad_g_cm3": 4.23,
    },
    "Fe2SiO4": {
        "nombre": "Fayalita",
        "color": "#5E6B3A", "metalicidad": 0.10, "rugosidad": 0.45,
        "descripcion": "verde oliva a pardo; funde a 1178 °C, sólida a 900 °C",
        "referencia": "Deer, Howie & Zussman",
        "estado": "producto", "grupo": "mineral",
        "masa_molar_g_mol": 203.774, "densidad_g_cm3": 4.39,
    },
    "FeS": {
        "nombre": "Troilita",
        "color": "#9A7B4F", "metalicidad": 0.65, "rugosidad": 0.40,
        "descripcion": "bronce pálido metálico",
        "referencia": "Deer, Howie & Zussman",
        "estado": "producto", "grupo": "mineral",
        "masa_molar_g_mol": 87.905, "densidad_g_cm3": 4.84,
    },
    "char": {
        "nombre": "Char / coque",
        "color": "#141312", "metalicidad": 0.0, "rugosidad": 0.95,
        "descripcion": "negro mate, poroso",
        "referencia": "aspecto del coque de carbón bituminoso",
        "estado": "producto", "grupo": "carbonoso",
        "masa_molar_g_mol": 12.011, "densidad_g_cm3": 1.35,
    },
    "carbon": {
        "nombre": "Carbón",
        "color": "#1A1815", "metalicidad": 0.05, "rugosidad": 0.85,
        "descripcion": "negro mate con brillo vítreo en las caras",
        "estado": "inicial", "grupo": "carbonoso",
        # `volatil` es una pseudoespecie de 1 g por pseudomol; se le asigna la
        # densidad de la matriz carbonosa porque los volátiles residen en ella.
        "masa_molar_g_mol": 1.0, "densidad_g_cm3": 1.35,
    },
    "cenizas": {
        "nombre": "Cenizas",
        "color": "#A9A296", "metalicidad": 0.0, "rugosidad": 0.90,
        "descripcion": "gris claro pulverulento",
        "estado": "producto", "grupo": "carbonoso",
        # `ceniza` también es pseudoespecie de 1 g por pseudomol.
        "masa_molar_g_mol": 1.0, "densidad_g_cm3": 2.30,
    },
    "aglomerado": {
        "nombre": "Aglomerado cohesionado",
        "color": "#2A2320", "metalicidad": 0.05, "rugosidad": 0.90,
        "descripcion": "matriz de coque con granos minerales embebidos",
        "estado": "producto", "grupo": "agregado",
        "masa_molar_g_mol": None, "densidad_g_cm3": None,
    },
}

FASES_INICIALES = tuple(k for k, v in FASES.items() if v["estado"] == "inicial")
FASES_PRODUCTO = tuple(k for k, v in FASES.items() if v["estado"] == "producto")

# Cómo se llama cada fase en los campos `solido__*` de las instantáneas NPZ.
# `H2O_liq` es humedad en los poros, no una fase mineral con aspecto propio:
# se deja fuera del coloreado en vez de inventarle un color.
MAPA_CAMPOS_SOLIDOS: dict[str, str | None] = {
    "H2O_liq": None,
    "volatil": "carbon",
    "C": "char",
    "ceniza": "cenizas",
    "Fe2O3": "Fe2O3",
    "Fe3O4": "Fe3O4",
    "FeO": "FeO",
    "Fe": "Fe",
    "FeTiO3": "FeTiO3",
    "TiO2": "TiO2",
    "SiO2": "SiO2",
    "Fe2SiO4": "Fe2SiO4",
    "FeS": "FeS",
}

# El aglomerado no es una fase del inventario sino el estado cohesionado de la
# mezcla: no puede aparecer como destino de ningún campo solido__*.
_FASES_CON_VOLUMEN = tuple(
    clave for clave, datos in FASES.items() if datos["densidad_g_cm3"] is not None
)


def volumenes_molares_cm3_mol() -> dict[str, float]:
    """Volumen molar M/rho de cada fase, en cm³/mol.

    Es el factor que convierte los campos del solucionador (mol/m³) en volumen
    ocupado, que es lo que determina el aspecto de la mezcla.
    """
    return {
        clave: float(FASES[clave]["masa_molar_g_mol"]) / float(FASES[clave]["densidad_g_cm3"])
        for clave in _FASES_CON_VOLUMEN
    }


def fracciones_volumetricas(
    solido: Mapping[str, float], grupo: str | None = None,
) -> dict[str, float]:
    """Fracción de volumen de cada fase a partir de concentraciones en mol/m³.

    `solido` usa los nombres de los campos NPZ (`C`, `volatil`, `ceniza`, …).
    Con `grupo` se restringe a una población: "carbonoso" para las partículas de
    carbón y "mineral" para las del concentrado. Devuelve {} si no hay sólido.
    """
    volumenes = volumenes_molares_cm3_mol()
    aporte: dict[str, float] = {}
    for campo, valor in solido.items():
        clave = MAPA_CAMPOS_SOLIDOS.get(campo)
        if clave is None or clave not in volumenes:
            continue
        if grupo is not None and FASES[clave]["grupo"] != grupo:
            continue
        cantidad = float(valor)
        if not cantidad > 0.0:
            continue
        aporte[clave] = aporte.get(clave, 0.0) + cantidad * volumenes[clave]
    total = sum(aporte.values())
    if not total > 0.0:
        return {}
    return {clave: valor / total for clave, valor in aporte.items()}


def _a_lineal(canal: float) -> float:
    return canal / 12.92 if canal <= 0.04045 else ((canal + 0.055) / 1.055) ** 2.4


def _a_srgb(canal: float) -> float:
    canal = min(max(canal, 0.0), 1.0)
    return 12.92 * canal if canal <= 0.0031308 else 1.055 * canal ** (1 / 2.4) - 0.055


def rgb_de_hex(color: str) -> tuple[float, float, float]:
    texto = color.lstrip("#")
    return tuple(int(texto[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def hex_de_rgb(rgb: Iterable[float]) -> str:
    return "#" + "".join(f"{round(min(max(c, 0.0), 1.0) * 255):02X}" for c in rgb)


def mezcla_color(fracciones: Mapping[str, float]) -> str:
    """Color de una mezcla de fases, promediado en luz lineal.

    Promediar directamente los valores sRGB oscurecería la mezcla; la
    conversión a lineal antes de promediar es la práctica estándar.
    """
    peso_total = sum(float(v) for v in fracciones.values() if float(v) > 0.0)
    if not peso_total > 0.0:
        return FASES["carbon"]["color"]
    acumulado = [0.0, 0.0, 0.0]
    for clave, peso in fracciones.items():
        if clave not in FASES or float(peso) <= 0.0:
            continue
        lineal = [_a_lineal(c) for c in rgb_de_hex(FASES[clave]["color"])]
        for i in range(3):
            acumulado[i] += float(peso) * lineal[i]
    return hex_de_rgb(_a_srgb(c / peso_total) for c in acumulado)


def paleta_web() -> dict[str, Any]:
    """Paleta lista para la interfaz, con leyenda ordenada.

    Las fases iniciales van primero y las de producto después, que es el orden
    en el que aparecen al reproducir la simulación. Se incluyen el volumen molar
    y el mapa de campos para que el cliente pueda ponderar por volumen sin
    duplicar la mineralogía en JavaScript.
    """
    volumenes = volumenes_molares_cm3_mol()
    return {
        "fases": {
            clave: {
                "nombre": d["nombre"], "color": d["color"],
                "metalicidad": d["metalicidad"], "rugosidad": d["rugosidad"],
                "descripcion": d["descripcion"], "estado": d["estado"],
                "grupo": d["grupo"],
                "densidad_g_cm3": d["densidad_g_cm3"],
                "masa_molar_g_mol": d["masa_molar_g_mol"],
                "volumen_molar_cm3_mol": volumenes.get(clave),
            }
            for clave, d in FASES.items()
        },
        "orden_leyenda": list(FASES_INICIALES) + list(FASES_PRODUCTO),
        "campos_solidos": dict(MAPA_CAMPOS_SOLIDOS),
        "ponderacion": "volumen (mol/m³ × volumen molar M/rho); mezcla en luz lineal",
        "fuente_densidades": (
            "simulacion_v3/src/superficie.py CATALOGO_FASES — Anthony et al. "
            "(1990-2001); FeO y Fe del CRC Handbook 97.ª ed."
        ),
        "nota": ("Colores aproximados al aspecto real de cada mineral. "
                 "Las fracciones de fase son PREDICCIÓN del modelo: no existe "
                 "caracterización del aglomerado después del ensayo."),
    }


__all__ = ["FASES", "FASES_INICIALES", "FASES_PRODUCTO", "MAPA_CAMPOS_SOLIDOS",
           "paleta_web", "fracciones_volumetricas",
           "volumenes_molares_cm3_mol", "mezcla_color", "rgb_de_hex", "hex_de_rgb"]
