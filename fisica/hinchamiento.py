"""
Hinchamiento del aglomerado durante la etapa plástica del carbón.

DATO DE LABORATORIO QUE ORIGINA ESTE MÓDULO
-------------------------------------------
El usuario aportó dos observaciones del ensayo:

1. El carbón tiene **índice de hinchamiento (IH) 8** — ensayo del botón en
   crisol, ASTM D720 / ISO 501, escala ½–9. Un IH de 8 es un carbón coquizable
   fuertemente hinchante.
2. **Mezclado con la magnetita se hincha, pero no tanto.**

Ambas cosas son cualitativas. Lo que este módulo hace es traducirlas a un
factor de expansión volumétrica con una forma funcional explícita y parámetros
declarados CALIBRABLES, no derivar una ley a partir de nada.

FÍSICA
------
El hinchamiento no es un aumento de masa: es el gas de pirólisis atrapado en la
masa plástica, que la infla mientras el carbón está fluido (350–500 °C, la misma
ventana termoplástica que usa `cohesion.py`). Al resolidificar, la estructura
queda congelada y el volumen ya no vuelve atrás. Por eso el hinchamiento sigue
al **grado de plastificación**, no a la temperatura instantánea, y es
irreversible.

Los inertes lo reducen por dos vías, ambas documentadas en la práctica de
mezclas de coquización: sólo hincha la fracción carbonosa, y los granos
minerales rompen la estructura de burbujas de la masa fluida. Aquí se agrupan
en un solo factor `(1 - f_inerte)**EXPONENTE_INERTE`.

NADA DE ESTO ESTÁ VALIDADO
--------------------------
No existe dilatometría de esta muestra ni medida del volumen del aglomerado tras
el ensayo. La única restricción experimental es la cronología aportada por el
usuario, y se usa para **falsar**, no para ajustar: véase
`tests/test_hinchamiento.py`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Dato de laboratorio aportado por el usuario. La ficha del proyecto (CLAUDE.md)
# registra IH 7,5 para la muestra; el usuario indica 8 en el ensayo. Se adopta 8
# y se deja el valor como parámetro para no enterrar la diferencia.
INDICE_HINCHAMIENTO_CARBON = 8.0
RANGO_INDICE_HINCHAMIENTO = (0.5, 9.0)

# CALIBRABLE. Expansión volumétrica del botón de carbón puro por unidad de IH.
# Con 0,125 un IH de 8 da un factor 2,0 en volumen, extremo bajo de lo que se
# atribuye a un carbón fuertemente hinchante. No se encontró una correlación
# publicada IH -> expansión volumétrica para esta muestra; el rango cubre desde
# apenas hinchar hasta triplicar.
EXPANSION_POR_UNIDAD_IH = 0.125
RANGO_EXPANSION_POR_UNIDAD_IH = (0.03, 0.30)

# CALIBRABLE. Atenuación por inertes. Con exponente 1 sólo se descuenta la
# fracción que no hincha; por encima de 1 se representa además que los granos
# minerales rompen la estructura de burbujas. Para 25 % de concentrado, 1,5 deja
# el hinchamiento en el 65 % del que tendría el carbón solo.
EXPONENTE_INERTE = 1.5
RANGO_EXPONENTE_INERTE = (1.0, 3.0)

REFERENCIA_ENSAYO_IH = (
    "ASTM D720 / ISO 501, indice de hinchamiento en crisol (free swelling "
    "index): el boton de coque se compara con perfiles patron numerados 1/2 a 9."
)
REFERENCIA_INERTES = (
    "Loison, Foch & Boyer, Coke: Quality and Production, 2.a ed., Butterworths, "
    "1989: los inertes de la mezcla reducen dilatacion y fluidez de la fase "
    "plastica."
)
VALIDACION = (
    "PREDICCION NO VALIDADA - no hay dilatometria de esta muestra ni medida del "
    "volumen del aglomerado despues del ensayo"
)


def factor_hinchamiento_maximo(
    indice_hinchamiento: float = INDICE_HINCHAMIENTO_CARBON,
    fraccion_inerte: float = 0.0,
    *,
    expansion_por_unidad: float = EXPANSION_POR_UNIDAD_IH,
    exponente_inerte: float = EXPONENTE_INERTE,
) -> float:
    """Factor de expansión volumétrica al final de la etapa plástica.

    Devuelve V_final/V_inicial ≥ 1. `fraccion_inerte` es la fracción **másica**
    de material que no hincha (el concentrado mineral), que es como se expresa
    convencionalmente el inerte en una mezcla de coquización.
    """
    ih = float(indice_hinchamiento)
    if not math.isfinite(ih) or ih < 0.0:
        raise ValueError("indice_hinchamiento debe ser finito y no negativo")
    inerte = float(fraccion_inerte)
    if not math.isfinite(inerte) or not 0.0 <= inerte <= 1.0:
        raise ValueError("fraccion_inerte debe pertenecer a [0, 1]")
    if float(expansion_por_unidad) < 0.0:
        raise ValueError("expansion_por_unidad no puede ser negativa")
    if float(exponente_inerte) <= 0.0:
        raise ValueError("exponente_inerte debe ser positivo")

    expansion_puro = float(expansion_por_unidad) * ih
    atenuacion = (1.0 - inerte) ** float(exponente_inerte)
    return 1.0 + expansion_puro * atenuacion


def hinchamiento_local(
    grado_plastificacion: Any,
    indice_hinchamiento: float = INDICE_HINCHAMIENTO_CARBON,
    fraccion_inerte: float = 0.0,
    **parametros: float,
) -> np.ndarray | float:
    """Factor de expansión celda a celda, en marcha con la plastificación.

    `grado_plastificacion` es la variable interna irreversible de
    `cohesion.HistoriaTermica`: 0 antes de entrar en la ventana termoplástica y
    1 cuando la ha recorrido entera. El hinchamiento la sigue porque es el gas
    atrapado en la masa fluida lo que infla, y queda congelado al resolidificar.
    """
    grado = np.asarray(grado_plastificacion, dtype=float)
    if np.any(~np.isfinite(grado)) or np.any((grado < 0.0) | (grado > 1.0)):
        raise ValueError("grado_plastificacion debe pertenecer a [0, 1]")
    maximo = factor_hinchamiento_maximo(
        indice_hinchamiento, fraccion_inerte, **parametros
    )
    factor = 1.0 + (maximo - 1.0) * grado
    return float(factor) if grado.ndim == 0 else factor


def volumen_hinchado_m3(
    volumen_inicial_m3: Any,
    grado_plastificacion: Any,
    indice_hinchamiento: float = INDICE_HINCHAMIENTO_CARBON,
    fraccion_inerte: float = 0.0,
    **parametros: float,
) -> np.ndarray | float:
    """Volumen que ocupa el aglomerado tras hincharse."""
    volumen = np.asarray(volumen_inicial_m3, dtype=float)
    if np.any(volumen < 0.0):
        raise ValueError("volumen_inicial_m3 no puede ser negativo")
    factor = hinchamiento_local(
        grado_plastificacion, indice_hinchamiento, fraccion_inerte, **parametros
    )
    producto = volumen * factor
    return float(producto) if producto.ndim == 0 else producto


def altura_hinchada_mm(
    altura_inicial_mm: float,
    factor_volumetrico: Any,
    *,
    confinamiento_lateral: bool = True,
) -> np.ndarray | float:
    """Altura del lecho hinchado.

    En el crisol la pared impide la expansión radial, así que el hinchamiento se
    resuelve **hacia arriba**: la altura crece con todo el factor volumétrico.
    Sin confinamiento la expansión sería isótropa y la altura crecería con la
    raíz cúbica.
    """
    factor = np.asarray(factor_volumetrico, dtype=float)
    if np.any(factor < 0.0):
        raise ValueError("factor_volumetrico no puede ser negativo")
    escala = factor if confinamiento_lateral else np.cbrt(factor)
    producto = float(altura_inicial_mm) * escala
    return float(producto) if producto.ndim == 0 else producto


def resumen(
    indice_hinchamiento: float = INDICE_HINCHAMIENTO_CARBON,
    fraccion_inerte: float = 0.0,
    **parametros: float,
) -> dict[str, Any]:
    """Ficha del modelo, para que la interfaz pueda mostrar de dónde sale."""
    maximo = factor_hinchamiento_maximo(
        indice_hinchamiento, fraccion_inerte, **parametros
    )
    puro = factor_hinchamiento_maximo(indice_hinchamiento, 0.0, **parametros)
    return {
        "indice_hinchamiento": float(indice_hinchamiento),
        "fraccion_inerte": float(fraccion_inerte),
        "factor_carbon_solo": puro,
        "factor_mezcla": maximo,
        "atenuacion_por_inertes": (maximo - 1.0) / (puro - 1.0) if puro > 1.0 else 1.0,
        "referencia_ensayo": REFERENCIA_ENSAYO_IH,
        "referencia_inertes": REFERENCIA_INERTES,
        "validacion": VALIDACION,
    }


__all__ = [
    "INDICE_HINCHAMIENTO_CARBON",
    "EXPANSION_POR_UNIDAD_IH",
    "EXPONENTE_INERTE",
    "VALIDACION",
    "factor_hinchamiento_maximo",
    "hinchamiento_local",
    "volumen_hinchado_m3",
    "altura_hinchada_mm",
    "resumen",
]
