"""Acoplamientos físicos del simulador tridimensional."""

from .adaptador_v3 import (
    estado_inicial_celda,
    fuente_de_masa_gaseosa,
    integrar_quimica_local,
    propiedades_gas,
    tasas_locales,
)

__all__ = [
    "estado_inicial_celda",
    "fuente_de_masa_gaseosa",
    "integrar_quimica_local",
    "propiedades_gas",
    "tasas_locales",
]
