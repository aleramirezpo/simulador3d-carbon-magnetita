"""Prediccion 3-D de cohesion y crecimiento del aglomerado.

La variable :class:`CampoCohesion` vive en los centros de celda y conserva la
memoria termica necesaria para distinguir un polvo caliente de un carbon que
ya atraveso su intervalo termoplastico.  Este modulo extiende a un campo 3-D
el diagnostico de :func:`simulacion_v3.src.superficie.sinterizacion_superficial`:
reutiliza ``carbon``, ``grano`` y ``termodinamica_ext`` mediante el puente
existente, sin copiar su quimica ni su termodinamica.

ADVERTENCIA DE VALIDACION
-------------------------
No existe caracterizacion del aglomerado despues del ensayo. Todos los campos,
aportes por mecanismo, masas, volumenes y curvas que produce este modulo son
**PREDICCIONES NO VALIDADAS**. Los parametros cineticos de coquizacion y de
puentes solidos son calibrables; no deben interpretarse como constantes
medidas para la muestra.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adaptador_v3 import carbon, grano, termodinamica_ext


T_INICIO_PLASTICA_K = 350.0 + 273.15
T_FIN_PLASTICA_K = 500.0 + 273.15
T_FUSION_FAYALITA_K = 1178.0 + 273.15

# Fuente de la envolvente termoplastica. El intervalo exacto depende del rango,
# velocidad de calentamiento y ensayo de plastometria; no se midio en la muestra.
REFERENCIA_VENTANA_PLASTICA = (
    "van Krevelen, D.W., Coal: Typology-Physics-Chemistry-Constitution, "
    "3rd ed., Elsevier, 1993, seccion sobre carbonizacion: los carbones "
    "bituminosos coquizables atraviesan aproximadamente 350-500 degC."
)

# PARAMETROS CALIBRABLES, no cinetica publicada para esta muestra de IH=7,5.
# Los rangos 10-120 s abarcan desde transformacion rapida durante el ensayo
# hasta una escala comparable con los 150 s usados en el analisis de v3.
#
# Criterio de la calibracion operativa vigente: la historia termica CALIBRADA
# del 0-D entra en 350 degC hacia 90 s y sale por 500 degC hacia 114 s. Dos
# escalas de 15 s hacen aparecer el aglomerado operativo (c >= 0,5) hacia
# 138 s en el limite espacialmente uniforme, dentro del intervalo interno
# 114-144 s exigido por esa cinetica. No hay una medicion experimental del
# instante de aglomeracion: este ajuste solo exige coherencia interna con la
# perdida de masa validada del 0-D y NO valida experimentalmente la cohesion.
TIEMPO_PLASTIFICACION_S = 15.0
RANGO_TIEMPO_PLASTIFICACION_S = (10.0, 120.0)
TIEMPO_COQUIZACION_S = 15.0
RANGO_TIEMPO_COQUIZACION_S = (10.0, 120.0)

# Inventario heredado de carbon.char(), no recalculado aqui. Para la carga de
# 0,75 g y 0,90 % de humedad residual reproduce los 0,4785 g de v3.
RENDIMIENTO_CHAR_G_G_CARBON_SECO = float(
    carbon.char()["masa_char_g_por_g_carbon_seco"]
)
MASA_CHAR_ESTIMADA_G = (
    0.75
    * (1.0 - float(carbon.ANALISIS_PROXIMO["humedad_residual_pct"]) / 100.0)
    * RENDIMIENTO_CHAR_G_G_CARBON_SECO
)

# Enclavamiento acompana la contraccion de la matriz, pero queda una decada por
# debajo de la coquizacion. Los puentes solidos son aun mas lentos y locales.
K_ENCLAVAMIENTO_S_1 = 3.0e-3
K_PUENTES_SOLIDOS_S_1 = 1.0e-3
CONCENTRACION_CONTACTO_REF_MOL_M3 = 1.0e3

MECANISMOS = (
    "coquizacion",
    "enclavamiento",
    "sinterizacion",
    "puentes_solidos",
)
VALIDACION = "PREDICCION NO VALIDADA - falta caracterizacion post-ensayo"
RUTA_CURVA = Path(__file__).resolve().parents[1] / "resultados" / "cohesion_crecimiento.csv"


def _como_array_broadcast(valor: Any, forma: tuple[int, ...], nombre: str) -> np.ndarray:
    arreglo = np.asarray(valor, dtype=float)
    try:
        salida = np.broadcast_to(arreglo, forma)
    except ValueError as exc:
        raise ValueError(
            f"{nombre} tiene forma {arreglo.shape}; no es compatible con {forma}"
        ) from exc
    if np.any(~np.isfinite(salida)):
        raise ValueError(f"{nombre} contiene valores no finitos")
    return np.asarray(salida, dtype=float)


def _escalar_si_corresponde(valor: Any, referencia: Any) -> Any:
    arreglo = np.asarray(valor)
    if np.asarray(referencia).ndim == 0:
        if arreglo.dtype == np.bool_:
            return bool(arreglo)
        return float(arreglo)
    return arreglo


def _aplicar_escalar_a_unicos(funcion: Any, valores: np.ndarray) -> np.ndarray:
    """Evalua una API escalar de v3 una vez por valor distinto del campo."""

    unicos, inversos = np.unique(valores, return_inverse=True)
    evaluados = np.fromiter(
        (float(funcion(float(valor))) for valor in unicos),
        dtype=float,
        count=unicos.size,
    )
    return evaluados[inversos].reshape(valores.shape)


@dataclass
class HistoriaTermica:
    """Variables internas irreversibles por celda.

    ``grado_plastificacion`` integra la residencia dentro de 350-500 degC.
    ``resolidificada`` se activa al superar 500 degC o al enfriar despues de
    haber entrado en la ventana. Ninguna de las dos variables vuelve a cero.
    """

    grado_plastificacion: np.ndarray
    paso_por_ventana: np.ndarray
    resolidificada: np.ndarray
    temperatura_maxima_K: np.ndarray
    temperatura_anterior_K: np.ndarray
    tiempo_en_ventana_s: np.ndarray

    @classmethod
    def vacia(cls, forma: tuple[int, ...]) -> "HistoriaTermica":
        return cls(
            grado_plastificacion=np.zeros(forma, dtype=float),
            paso_por_ventana=np.zeros(forma, dtype=bool),
            resolidificada=np.zeros(forma, dtype=bool),
            temperatura_maxima_K=np.full(forma, -np.inf, dtype=float),
            temperatura_anterior_K=np.full(forma, np.nan, dtype=float),
            tiempo_en_ventana_s=np.zeros(forma, dtype=float),
        )


@dataclass
class CampoCohesion:
    """Campo escalar de estado ``c(x,t)`` en centros de celda.

    Parameters
    ----------
    c:
        Arreglo 3-D inicial en [0, 1]. Tambien se admite una tupla de tres
        enteros, que crea un campo de ceros con esa forma.
    espaciado_m:
        ``(dx, dy, dz)`` en SI. Si se omite se adopta celda cubica unitaria;
        las propiedades geometricas solo tienen sentido fisico cuando se pasa.
    densidad_bulk_kg_m3:
        Densidad aparente escalar o por celda usada para la masa predicha.
    fraccion_volumetrica:
        Fraccion de lecho en celdas cortadas por la frontera, coherente con el
        contrato conservativo de la malla.
    """

    c: np.ndarray | tuple[int, int, int]
    espaciado_m: float | Sequence[float] = (1.0, 1.0, 1.0)
    densidad_bulk_kg_m3: float | np.ndarray = 1.0
    fraccion_volumetrica: float | np.ndarray = 1.0
    historia_termica: HistoriaTermica | None = None
    contribuciones: dict[str, np.ndarray] | None = None
    tiempo_s: float = 0.0
    es_prediccion: bool = field(default=True, init=False)
    validacion: str = field(default=VALIDACION, init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.c, tuple)
            and len(self.c) == 3
            and all(isinstance(n, (int, np.integer)) and n > 0 for n in self.c)
        ):
            self.c = np.zeros(tuple(int(n) for n in self.c), dtype=float)
        else:
            self.c = np.array(self.c, dtype=float, copy=True)
        if self.c.ndim != 3:
            raise ValueError("CampoCohesion requiere un arreglo 3-D")
        if np.any(~np.isfinite(self.c)) or np.any((self.c < 0.0) | (self.c > 1.0)):
            raise ValueError("c debe contener solo valores finitos en [0, 1]")

        espaciado = np.asarray(self.espaciado_m, dtype=float)
        if espaciado.ndim == 0:
            espaciado = np.repeat(espaciado, 3)
        if espaciado.shape != (3,) or np.any(~np.isfinite(espaciado)) or np.any(espaciado <= 0.0):
            raise ValueError("espaciado_m debe ser positivo y tener tres componentes")
        self.espaciado_m = tuple(float(x) for x in espaciado)

        self.densidad_bulk_kg_m3 = _como_array_broadcast(
            self.densidad_bulk_kg_m3, self.c.shape, "densidad_bulk_kg_m3"
        ).copy()
        if np.any(self.densidad_bulk_kg_m3 < 0.0):
            raise ValueError("densidad_bulk_kg_m3 no puede ser negativa")
        self.fraccion_volumetrica = _como_array_broadcast(
            self.fraccion_volumetrica, self.c.shape, "fraccion_volumetrica"
        ).copy()
        if np.any((self.fraccion_volumetrica < 0.0) | (self.fraccion_volumetrica > 1.0)):
            raise ValueError("fraccion_volumetrica debe pertenecer a [0, 1]")

        if self.historia_termica is None:
            self.historia_termica = HistoriaTermica.vacia(self.c.shape)
        else:
            for nombre in HistoriaTermica.__dataclass_fields__:
                if np.asarray(getattr(self.historia_termica, nombre)).shape != self.c.shape:
                    raise ValueError(f"historia_termica.{nombre} no coincide con la forma del campo")

        if self.contribuciones is None:
            self.contribuciones = {m: np.zeros_like(self.c) for m in MECANISMOS}
        else:
            recibidas = self.contribuciones
            self.contribuciones = {}
            for mecanismo in MECANISMOS:
                valor = recibidas.get(mecanismo, 0.0)
                arreglo = _como_array_broadcast(valor, self.c.shape, mecanismo).copy()
                if np.any(arreglo < 0.0):
                    raise ValueError("Las contribuciones no pueden ser negativas")
                self.contribuciones[mecanismo] = arreglo
        if not math.isfinite(float(self.tiempo_s)) or float(self.tiempo_s) < 0.0:
            raise ValueError("tiempo_s debe ser finito y no negativo")
        self.tiempo_s = float(self.tiempo_s)

    @property
    def forma(self) -> tuple[int, int, int]:
        return self.c.shape

    @property
    def volumen_celda_m3(self) -> float:
        return float(np.prod(self.espaciado_m))

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.c, dtype=dtype)


def _campo_historia(historia: HistoriaTermica | Mapping[str, Any], nombre: str) -> Any:
    if isinstance(historia, Mapping):
        if nombre in historia:
            return historia[nombre]
        alias = {
            "grado_plastificacion": "paso_ventana",
            "paso_por_ventana": "entro_ventana",
            "resolidificada": "resolidifico",
        }
        if nombre in alias and alias[nombre] in historia:
            return historia[alias[nombre]]
        raise KeyError(f"historia_termica no contiene {nombre!r}")
    return getattr(historia, nombre)


def tasa_coquizacion(
    T: Any,
    historia_termica: HistoriaTermica | Mapping[str, Any],
    fraccion_carbon: Any,
) -> float | np.ndarray:
    """Tasa irreversible de formacion de matriz de coque [s-1].

    La temperatura instantanea no basta: la tasa es exactamente cero hasta que
    la variable interna registra paso por 350-500 degC y posterior
    resolidificacion. Una vez activa, el enfriamiento no borra la memoria.

    La envolvente 350-500 degC procede de van Krevelen (1993), indicada en
    :data:`REFERENCIA_VENTANA_PLASTICA`. No se encontro una cinetica publicada
    para *esta muestra* de IH=7,5: ``TIEMPO_COQUIZACION_S=15 s`` es CALIBRABLE
    en 10-120 s. El valor actual se ajusto solo al calendario termico y de
    perdida de masa del 0-D; debe recalibrarse cuando exista una curva medida
    de crecimiento o un instante experimental de formacion del aglomerado.
    """

    temperatura, grado, resolidificada, fraccion = np.broadcast_arrays(
        np.asarray(T, dtype=float),
        np.asarray(_campo_historia(historia_termica, "grado_plastificacion"), dtype=float),
        np.asarray(_campo_historia(historia_termica, "resolidificada"), dtype=bool),
        np.asarray(fraccion_carbon, dtype=float),
    )
    if np.any(~np.isfinite(temperatura)) or np.any(temperatura <= 0.0):
        raise ValueError("T debe ser finita y positiva")
    if np.any(~np.isfinite(grado)) or np.any((grado < 0.0) | (grado > 1.0)):
        raise ValueError("grado_plastificacion debe pertenecer a [0, 1]")
    if np.any(~np.isfinite(fraccion)) or np.any((fraccion < 0.0) | (fraccion > 1.0)):
        raise ValueError("fraccion_carbon debe pertenecer a [0, 1]")

    tasa = np.where(
        resolidificada,
        fraccion
        * RENDIMIENTO_CHAR_G_G_CARBON_SECO
        * grado
        / TIEMPO_COQUIZACION_S,
        0.0,
    )
    return _escalar_si_corresponde(tasa, T)


def tasa_sinterizacion(T: Any, dt: float, porosidad: Any) -> float | np.ndarray:
    """Cierre cohesivo incremental por sinterizacion durante ``dt``.

    Llama a :func:`grano.constante_sinterizacion`, cuya ley Arrhenius y rangos
    proceden del analisis de sensibilidad basado en German, *Sintering Theory
    and Practice* (1996). Se pondera por la fraccion solida ``1-porosidad``.
    A 1173.15 K, el cierre intrinseco de v3 en 150 s es 1.109e-4 (antes de esa
    ponderacion): demasiado lento para explicar la cohesion observada.
    """

    paso = float(dt)
    if not math.isfinite(paso) or paso < 0.0:
        raise ValueError("dt debe ser finito y no negativo")
    temperatura, eps = np.broadcast_arrays(
        np.asarray(T, dtype=float), np.asarray(porosidad, dtype=float)
    )
    if np.any(~np.isfinite(temperatura)) or np.any(temperatura <= 0.0):
        raise ValueError("T debe ser finita y positiva")
    if np.any(~np.isfinite(eps)) or np.any((eps < 0.0) | (eps >= 1.0)):
        raise ValueError("porosidad debe pertenecer a [0, 1)")

    k = _aplicar_escalar_a_unicos(grano.constante_sinterizacion, temperatura)
    cierre = -np.expm1(-k * paso) * (1.0 - eps)
    cierre = np.clip(cierre, 0.0, 1.0)
    return _escalar_si_corresponde(cierre, T)


def puentes_solidos(
    T: Any,
    c_FeO: Any,
    c_SiO2: Any,
    c_Al2O3: Any,
) -> dict[str, Any]:
    """Potencial local de fayalita y hercinita formadas en estado solido.

    ``delta_G`` y ``K_eq`` se evaluan directamente con
    :mod:`termodinamica_ext`. La cantidad queda limitada por estequiometria y
    contacto local; no se declara equilibrio de fases. El ``potencial_cohesivo``
    es un factor [0,1] para la cinetica calibrable de :func:`evolucionar`.
    """

    temperatura, feo, sio2, al2o3 = np.broadcast_arrays(
        np.asarray(T, dtype=float),
        np.asarray(c_FeO, dtype=float),
        np.asarray(c_SiO2, dtype=float),
        np.asarray(c_Al2O3, dtype=float),
    )
    if np.any(~np.isfinite(temperatura)) or np.any(temperatura <= 0.0):
        raise ValueError("T debe ser finita y positiva")
    if any(np.any(~np.isfinite(x)) or np.any(x < 0.0) for x in (feo, sio2, al2o3)):
        raise ValueError("Las concentraciones deben ser finitas y no negativas")

    def termo(reaccion: str) -> tuple[np.ndarray, np.ndarray]:
        dg = _aplicar_escalar_a_unicos(
            lambda temp: termodinamica_ext.delta_G_kj(reaccion, temp), temperatura
        )
        equilibrio = _aplicar_escalar_a_unicos(
            lambda temp: termodinamica_ext.K_eq(reaccion, temp), temperatura
        )
        return dg, equilibrio

    dg_fay, keq_fay = termo("fayalita_formacion")
    dg_her, keq_her = termo("hercinita_formacion")
    max_fay = np.minimum(feo / 2.0, sio2)
    max_her = np.minimum(feo, al2o3)

    def factor(dg: np.ndarray, limite: np.ndarray) -> np.ndarray:
        fuerza = np.where(
            dg < 0.0,
            -np.expm1(dg * 1000.0 / (termodinamica_ext.R * temperatura)),
            0.0,
        )
        contacto = limite / (limite + CONCENTRACION_CONTACTO_REF_MOL_M3)
        return np.clip(fuerza * contacto, 0.0, 1.0)

    potencial_fay = factor(dg_fay, max_fay)
    potencial_her = factor(dg_her, max_her)
    forma_fay = (dg_fay < 0.0) & (max_fay > 0.0)
    forma_her = (dg_her < 0.0) & (max_her > 0.0)
    fundida_fay = temperatura >= T_FUSION_FAYALITA_K
    estado_fayalita = (
        ("líquida" if bool(fundida_fay) else "sólida")
        if np.asarray(T).ndim == 0
        else np.where(fundida_fay, "líquida", "sólida")
    )

    def salida(valor: Any) -> Any:
        return _escalar_si_corresponde(valor, T)

    return {
        "fayalita": {
            "delta_G_kJ_mol": salida(dg_fay),
            "K_eq": salida(keq_fay),
            "cantidad_maxima_mol_m3": salida(max_fay),
            "se_forma": salida(forma_fay),
            "fraccion_potencial": salida(potencial_fay),
            "punto_fusion_C": 1178.0,
            "estado_a_T": estado_fayalita,
            "puede_producir_puente_liquido": salida(fundida_fay & forma_fay),
            "referencia": (
                "Chang et al. (2021), ISIJ Int. 61, 2715-2723; "
                "fayalita solida bajo 1178 degC."
            ),
        },
        "hercinita": {
            "delta_G_kJ_mol": salida(dg_her),
            "K_eq": salida(keq_her),
            "cantidad_maxima_mol_m3": salida(max_her),
            "se_forma": salida(forma_her),
            "fraccion_potencial": salida(potencial_her),
            "estado_a_T": "sólida" if np.asarray(T).ndim == 0 else np.full(temperatura.shape, "sólida"),
            "referencia": "Robie y Hemingway (1995), datos usados por termodinamica_ext.",
        },
        "potencial_cohesivo": salida(np.clip(potencial_fay + potencial_her, 0.0, 1.0)),
        "es_prediccion": True,
        "validacion": VALIDACION,
    }


def _fraccion_tiempo_en_ventana(T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
    """Fraccion de un paso lineal T0->T1 que cae en la ventana plastica."""

    conocidos = np.isfinite(T0)
    minimo = np.minimum(T0, T1)
    maximo = np.maximum(T0, T1)
    solape = np.maximum(
        0.0,
        np.minimum(maximo, T_FIN_PLASTICA_K)
        - np.maximum(minimo, T_INICIO_PLASTICA_K),
    )
    salto = np.abs(T1 - T0)
    fraccion_salto = np.divide(solape, salto, out=np.zeros_like(T1), where=salto > 0.0)
    estacionaria = (T1 >= T_INICIO_PLASTICA_K) & (T1 <= T_FIN_PLASTICA_K)
    return np.where(conocidos & (salto > 0.0), fraccion_salto, estacionaria.astype(float))


def _actualizar_historia(campo: CampoCohesion, T: np.ndarray, dt: float) -> None:
    historia = campo.historia_termica
    assert historia is not None
    anterior = historia.temperatura_anterior_K
    fraccion = _fraccion_tiempo_en_ventana(anterior, T)
    residencia = dt * fraccion
    historia.tiempo_en_ventana_s += residencia
    historia.grado_plastificacion = 1.0 - np.exp(
        -historia.tiempo_en_ventana_s / TIEMPO_PLASTIFICACION_S
    )
    historia.paso_por_ventana |= residencia > 0.0

    supera_fin = T >= T_FIN_PLASTICA_K
    enfria_y_sale = (
        np.isfinite(anterior)
        & (anterior >= T_INICIO_PLASTICA_K)
        & (T < T_INICIO_PLASTICA_K)
    )
    historia.resolidificada |= historia.paso_por_ventana & (supera_fin | enfria_y_sale)
    historia.temperatura_maxima_K = np.maximum(historia.temperatura_maxima_K, T)
    historia.temperatura_anterior_K = T.copy()


def evolucionar(
    campo: CampoCohesion | np.ndarray,
    T: Any,
    dt: float,
    *,
    fraccion_carbon: Any = 0.75,
    porosidad: Any = 0.15,
    c_FeO: Any = 0.0,
    c_SiO2: Any = 0.0,
    c_Al2O3: Any = 0.0,
) -> CampoCohesion:
    """Avanza un paso y combina los mecanismos sin doble conteo.

    Se usa una suma de riesgos: cada mecanismo consume la misma fraccion aun
    no cohesionada y el incremento se reparte proporcionalmente entre riesgos.
    Por construccion ``0 <= c <= 1``. En 350-900 degC se considera irreversible:
    ``c`` nunca disminuye. La fractura requeriria otra variable, ley mecanica y
    datos de resistencia que no existen para el ensayo, por lo que no se modela.
    """

    if not isinstance(campo, CampoCohesion):
        campo = CampoCohesion(campo)
    paso = float(dt)
    if not math.isfinite(paso) or paso < 0.0:
        raise ValueError("dt debe ser finito y no negativo")
    temperatura = _como_array_broadcast(T, campo.forma, "T")
    if np.any(temperatura <= 0.0):
        raise ValueError("T debe ser positiva")
    carbon_local = _como_array_broadcast(fraccion_carbon, campo.forma, "fraccion_carbon")
    eps = _como_array_broadcast(porosidad, campo.forma, "porosidad")
    if np.any((carbon_local < 0.0) | (carbon_local > 1.0)):
        raise ValueError("fraccion_carbon debe pertenecer a [0, 1]")
    if np.any((eps < 0.0) | (eps >= 1.0)):
        raise ValueError("porosidad debe pertenecer a [0, 1)")
    if paso == 0.0:
        return campo

    _actualizar_historia(campo, temperatura, paso)
    historia = campo.historia_termica
    assert historia is not None
    tasa_coque = np.asarray(
        tasa_coquizacion(temperatura, historia, carbon_local), dtype=float
    )
    tasa_enclavamiento = np.where(
        historia.resolidificada,
        K_ENCLAVAMIENTO_S_1
        * carbon_local
        * historia.grado_plastificacion
        * (1.0 - eps),
        0.0,
    )
    cierre_sinter = np.asarray(tasa_sinterizacion(temperatura, paso, eps), dtype=float)
    datos_puentes = puentes_solidos(
        temperatura,
        _como_array_broadcast(c_FeO, campo.forma, "c_FeO"),
        _como_array_broadcast(c_SiO2, campo.forma, "c_SiO2"),
        _como_array_broadcast(c_Al2O3, campo.forma, "c_Al2O3"),
    )
    potencial_puentes = np.asarray(datos_puentes["potencial_cohesivo"], dtype=float)

    incrementos_libres = {
        "coquizacion": -np.expm1(-tasa_coque * paso),
        "enclavamiento": -np.expm1(-tasa_enclavamiento * paso),
        "sinterizacion": cierre_sinter,
        "puentes_solidos": -np.expm1(-K_PUENTES_SOLIDOS_S_1 * potencial_puentes * paso),
    }
    riesgos = {
        nombre: -np.log1p(-np.clip(incremento, 0.0, 1.0 - np.finfo(float).eps))
        for nombre, incremento in incrementos_libres.items()
    }
    riesgo_total = sum(riesgos.values(), np.zeros(campo.forma, dtype=float))
    incremento_total = (1.0 - campo.c) * (-np.expm1(-riesgo_total))
    incremento_total = np.maximum(incremento_total, 0.0)

    assert campo.contribuciones is not None
    for nombre, riesgo in riesgos.items():
        fraccion = np.divide(
            riesgo,
            riesgo_total,
            out=np.zeros_like(riesgo),
            where=riesgo_total > 0.0,
        )
        campo.contribuciones[nombre] += incremento_total * fraccion
    campo.c = np.clip(campo.c + incremento_total, 0.0, 1.0)
    campo.tiempo_s += paso
    return campo


def _etiquetar_componentes(mascara: np.ndarray) -> tuple[np.ndarray, int]:
    """Etiquetado 6-conexo sin imponer scipy como dependencia del modulo."""

    etiquetas = np.zeros(mascara.shape, dtype=np.int32)
    numero = 0
    nx, ny, nz = mascara.shape
    vecinos = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for semilla in zip(*np.nonzero(mascara), strict=True):
        if etiquetas[semilla] != 0:
            continue
        numero += 1
        etiquetas[semilla] = numero
        cola: deque[tuple[int, int, int]] = deque([semilla])
        while cola:
            i, j, k = cola.popleft()
            for di, dj, dk in vecinos:
                ii, jj, kk = i + di, j + dj, k + dk
                if (
                    0 <= ii < nx
                    and 0 <= jj < ny
                    and 0 <= kk < nz
                    and mascara[ii, jj, kk]
                    and etiquetas[ii, jj, kk] == 0
                ):
                    etiquetas[ii, jj, kk] = numero
                    cola.append((ii, jj, kk))
    return etiquetas, numero


def _superficie_voxelizada(mascara: np.ndarray, espaciado: Sequence[float]) -> float:
    dx, dy, dz = (float(x) for x in espaciado)
    superficie = 0.0
    for eje, area in ((0, dy * dz), (1, dx * dz), (2, dx * dy)):
        acolchada = np.pad(mascara.astype(np.int8), 1, mode="constant")
        diferencias = np.diff(acolchada, axis=eje)
        # Se quitan los acolchados de los otros dos ejes para no contar ceros extra.
        cortes = [slice(1, -1), slice(1, -1), slice(1, -1)]
        cortes[eje] = slice(None)
        superficie += float(np.count_nonzero(diferencias[tuple(cortes)])) * area
    return superficie


def aglomerado(campo: CampoCohesion | np.ndarray, umbral: float = 0.5) -> dict[str, Any]:
    """Extrae la region cohesionada y sus propiedades geometricas/conexas.

    La conectividad es por caras (6 vecinos), adecuada a celdas de volumenes
    finitos: un contacto solo por arista o esquina no constituye una pieza.
    """

    limite = float(umbral)
    if not math.isfinite(limite) or not (0.0 <= limite <= 1.0):
        raise ValueError("umbral debe pertenecer a [0, 1]")
    if isinstance(campo, CampoCohesion):
        valores = campo.c
        espaciado = campo.espaciado_m
        fraccion = campo.fraccion_volumetrica
        densidad = campo.densidad_bulk_kg_m3
    else:
        valores = np.asarray(campo, dtype=float)
        if valores.ndim != 3:
            raise ValueError("aglomerado requiere un campo 3-D")
        espaciado = (1.0, 1.0, 1.0)
        fraccion = np.ones(valores.shape, dtype=float)
        densidad = np.ones(valores.shape, dtype=float)
    if np.any(~np.isfinite(valores)) or np.any((valores < 0.0) | (valores > 1.0)):
        raise ValueError("El campo de cohesion debe estar en [0, 1]")

    mascara = (valores >= limite) & (fraccion > 0.0)
    etiquetas, numero = _etiquetar_componentes(mascara)
    volumen_celda = float(np.prod(espaciado))
    volumen = float(np.sum(fraccion[mascara], dtype=np.float64) * volumen_celda)
    masa = float(np.sum(densidad[mascara] * fraccion[mascara], dtype=np.float64) * volumen_celda)
    superficie = _superficie_voxelizada(mascara, espaciado)
    dimension = (6.0 * volumen / math.pi) ** (1.0 / 3.0) if volumen > 0.0 else 0.0
    return {
        "mascara": mascara,
        "etiquetas": etiquetas,
        "numero_componentes": numero,
        "componentes_conexas": numero,
        "es_una_sola_pieza": numero == 1,
        "una_sola_pieza": numero == 1,
        "volumen_m3": volumen,
        "masa_kg": masa,
        "superficie_m2": superficie,
        "dimension_caracteristica_m": dimension,
        "numero_celdas": int(np.count_nonzero(mascara)),
        "umbral": limite,
        "es_prediccion": True,
        "validacion": VALIDACION,
    }


def _separar_instantanea(elemento: Any) -> tuple[float, CampoCohesion | np.ndarray]:
    if isinstance(elemento, tuple) and len(elemento) == 2:
        return float(elemento[0]), elemento[1]
    if isinstance(elemento, Mapping):
        tiempo = elemento.get("tiempo_s", elemento.get("t"))
        campo = elemento.get("campo", elemento.get("cohesion"))
        if tiempo is None or campo is None:
            raise ValueError("Cada instantanea necesita tiempo y campo/cohesion")
        return float(tiempo), campo
    tiempo = getattr(elemento, "tiempo_s", getattr(elemento, "t", None))
    campo = elemento if isinstance(elemento, CampoCohesion) else getattr(elemento, "cohesion", None)
    if tiempo is None or campo is None:
        raise ValueError("Formato de instantanea temporal no reconocido")
    return float(tiempo), campo


def curva_de_crecimiento(serie_temporal: Iterable[Any] | Mapping[float, Any]) -> pd.DataFrame:
    """Calcula y exporta la curva predicha de volumen y masa frente al tiempo.

    Se usa ``umbral=0.5`` como definicion operativa fija del aglomerado. El CSV
    se escribe en ``resultados/cohesion_crecimiento.csv`` y cada fila conserva
    de forma explicita la etiqueta de prediccion no validada.
    """

    elementos: Iterable[Any]
    if isinstance(serie_temporal, Mapping):
        elementos = serie_temporal.items()
    else:
        elementos = serie_temporal
    filas: list[dict[str, Any]] = []
    for elemento in elementos:
        tiempo, campo = _separar_instantanea(elemento)
        if not math.isfinite(tiempo) or tiempo < 0.0:
            raise ValueError("Los tiempos deben ser finitos y no negativos")
        datos = aglomerado(campo, 0.5)
        filas.append(
            {
                "tiempo_s": tiempo,
                "volumen_m3": datos["volumen_m3"],
                "volumen_cm3": datos["volumen_m3"] * 1.0e6,
                "masa_kg": datos["masa_kg"],
                "masa_g": datos["masa_kg"] * 1.0e3,
                "superficie_m2": datos["superficie_m2"],
                "dimension_caracteristica_m": datos["dimension_caracteristica_m"],
                "numero_componentes": datos["numero_componentes"],
                "una_sola_pieza": datos["una_sola_pieza"],
                "es_prediccion": True,
                "validacion": VALIDACION,
            }
        )
    tabla = pd.DataFrame(filas).sort_values("tiempo_s", kind="stable", ignore_index=True)
    RUTA_CURVA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_CURVA, index=False, encoding="utf-8-sig")
    return tabla


def _contribuciones_de(objeto: CampoCohesion | Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if isinstance(objeto, CampoCohesion):
        assert objeto.contribuciones is not None
        pesos = objeto.fraccion_volumetrica * objeto.volumen_celda_m3
        return objeto.contribuciones, pesos
    if isinstance(objeto, Mapping):
        contribuciones = {
            nombre: np.asarray(objeto.get(nombre, 0.0), dtype=float) for nombre in MECANISMOS
        }
        forma = np.broadcast_shapes(*(x.shape for x in contribuciones.values()))
        contribuciones = {n: np.broadcast_to(x, forma) for n, x in contribuciones.items()}
        return contribuciones, np.ones(forma if forma else (), dtype=float)
    raise TypeError("Se esperaba CampoCohesion o un mapa de contribuciones")


def diagnostico_mecanismos(objeto: Any) -> dict[str, float] | pd.DataFrame:
    """Fraccion acumulada aportada por cada mecanismo.

    Para un campo devuelve cuatro fracciones. Para una serie temporal devuelve
    una tabla con una fila por instante; asi se puede comprobar numericamente
    que la coquizacion domina en cada momento posterior a su activacion.
    """

    if isinstance(objeto, CampoCohesion) or (
        isinstance(objeto, Mapping) and any(nombre in objeto for nombre in MECANISMOS)
    ):
        contribuciones, pesos = _contribuciones_de(objeto)
        aportes = {
            nombre: float(np.sum(valor * pesos, dtype=np.float64))
            for nombre, valor in contribuciones.items()
        }
        total = sum(aportes.values())
        if total <= 0.0:
            return {nombre: 0.0 for nombre in MECANISMOS}
        return {nombre: aporte / total for nombre, aporte in aportes.items()}

    filas: list[dict[str, float]] = []
    elementos = objeto.items() if isinstance(objeto, Mapping) else objeto
    for elemento in elementos:
        tiempo, campo = _separar_instantanea(elemento)
        if not isinstance(campo, CampoCohesion):
            raise TypeError("El diagnostico temporal requiere instancias CampoCohesion")
        fracciones = diagnostico_mecanismos(campo)
        assert isinstance(fracciones, dict)
        filas.append(
            {"tiempo_s": tiempo, **{f"fraccion_{k}": v for k, v in fracciones.items()}}
        )
    return pd.DataFrame(filas).sort_values("tiempo_s", kind="stable", ignore_index=True)


__all__ = [
    "perfil_termico_lecho",
    "hitos_crecimiento",
    "crecimiento_con_gradiente",
    "CampoCohesion",
    "HistoriaTermica",
    "MECANISMOS",
    "MASA_CHAR_ESTIMADA_G",
    "REFERENCIA_VENTANA_PLASTICA",
    "RENDIMIENTO_CHAR_G_G_CARBON_SECO",
    "RANGO_TIEMPO_COQUIZACION_S",
    "RANGO_TIEMPO_PLASTIFICACION_S",
    "TIEMPO_COQUIZACION_S",
    "TIEMPO_PLASTIFICACION_S",
    "T_FIN_PLASTICA_K",
    "T_FUSION_FAYALITA_K",
    "T_INICIO_PLASTICA_K",
    "aglomerado",
    "curva_de_crecimiento",
    "diagnostico_mecanismos",
    "evolucionar",
    "puentes_solidos",
    "tasa_coquizacion",
    "tasa_sinterizacion",
]


# ---------------------------------------------------------------------------
# Crecimiento con gradiente térmico
# ---------------------------------------------------------------------------
def perfil_termico_lecho(forma: tuple[int, int, int], espaciado_m: Sequence[float],
                         atenuacion_nucleo: float = 0.35) -> np.ndarray:
    """Peso térmico por celda: 1 en la frontera caliente, menor en el núcleo.

    El calor entra por la pared lateral y por el fondo del crisol, así que el
    centro del lecho es lo último en calentarse. Este perfil pondera la
    temperatura de cada celda entre la de la mufla (en la frontera) y la del
    lecho medio (en el núcleo).

    Sin este gradiente el modelo predice que todas las celdas cruzan el umbral
    de cohesión en el mismo instante y el aglomerado aparece de golpe, lo cual
    es un artefacto de suponer temperatura uniforme, no un resultado físico.
    """
    nx, ny, nz = forma
    dx, _, dz = (espaciado_m if len(espaciado_m) == 3
                 else (espaciado_m[0], espaciado_m[0], espaciado_m[-1]))
    xi = (np.arange(nx) + 0.5 - nx / 2.0) * dx
    yi = (np.arange(ny) + 0.5 - ny / 2.0) * dx
    zi = (np.arange(nz) + 0.5) * dz
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing="ij")
    radio = np.hypot(X, Y)
    r_max = max(float(radio.max()), 1e-30)
    dist_pared = (r_max - radio) / r_max          # 0 en la pared, 1 en el eje
    dist_fondo = Z / max(float(Z.max()), 1e-30)   # 0 en el fondo, 1 arriba
    profundidad = np.minimum(dist_pared, dist_fondo)
    profundidad /= max(float(profundidad.max()), 1e-30)
    return atenuacion_nucleo + (1.0 - atenuacion_nucleo) * profundidad


def crecimiento_con_gradiente(tiempos_s, T_muestra_K, T_mufla_K,
                              forma: tuple[int, int, int] = (46, 46, 7),
                              espaciado_m: Sequence[float] = (5e-4, 5e-4, 5e-4),
                              densidad_bulk_kg_m3: float = 736.0,
                              fraccion_volumetrica: float = 0.46,
                              fraccion_carbon: float = 0.75,
                              porosidad: float = 0.54,
                              guardar_campos_cada_s: float | None = None,
                              ) -> "pd.DataFrame":
    """Serie de crecimiento del aglomerado con el lecho térmicamente resuelto.

    Devuelve un DataFrame con volumen, masa, fracción cohesionada y número de
    componentes conexas en el tiempo. El número de componentes es el dato
    interesante: el aglomerado **nace como varios núcleos junto a la pared** y
    después coalesce en una sola pieza. Ese instante de coalescencia es el que
    corresponde a lo que el ensayo observa como "se formó el aglomerado".

    PREDICCIÓN NO VALIDADA: no existe caracterización del aglomerado después del
    ensayo, así que estos tiempos sólo pueden contrastarse con la coherencia
    interna frente a la cinética de pérdida de masa, que sí está calibrada.
    """
    tiempos = np.asarray(tiempos_s, dtype=float)
    T_m = np.asarray(T_muestra_K, dtype=float)
    T_f = np.asarray(T_mufla_K, dtype=float)
    if not (tiempos.shape == T_m.shape == T_f.shape):
        raise ValueError("tiempos, T_muestra y T_mufla deben tener la misma forma")

    peso = perfil_termico_lecho(forma, espaciado_m)
    campo = CampoCohesion(forma, espaciado_m=espaciado_m,
                          densidad_bulk_kg_m3=densidad_bulk_kg_m3,
                          fraccion_volumetrica=fraccion_volumetrica)
    filas: list[dict[str, Any]] = []
    campos_guardados: dict[float, np.ndarray] = {}
    t_prev = float(tiempos[0])
    proximo = 0.0
    for t, Tm, Tf in zip(tiempos, T_m, T_f, strict=True):
        dt = float(t) - t_prev
        if dt <= 0.0:
            continue
        # la celda ve una temperatura entre la de la mufla (frontera) y la media
        T3 = Tf + (Tm - Tf) * peso
        campo = evolucionar(campo, T3, dt, fraccion_carbon=fraccion_carbon,
                            porosidad=porosidad)
        t_prev = float(t)
        info = aglomerado(campo)
        filas.append({
            "tiempo_s": float(t),
            "T_muestra_K": float(Tm),
            "T_mufla_K": float(Tf),
            "cohesion_media": float(campo.c.mean()),
            "volumen_cm3": float(info["volumen_m3"]) * 1e6,
            "masa_g": float(info["masa_kg"]) * 1e3,
            "fraccion_lecho_cohesionada": float((campo.c >= 0.5).mean()),
            "numero_componentes": int(info["numero_componentes"]),
            "es_prediccion": True,
            "validacion": ("PREDICCION NO VALIDADA - no existe caracterizacion "
                           "del aglomerado post-ensayo"),
        })
        if guardar_campos_cada_s and float(t) >= proximo:
            campos_guardados[float(t)] = campo.c.copy()
            proximo = float(t) + guardar_campos_cada_s

    tabla = pd.DataFrame(filas)
    tabla.attrs["campos_cohesion"] = campos_guardados
    return tabla


def hitos_crecimiento(tabla: "pd.DataFrame") -> dict[str, float]:
    """Instantes en que el aglomerado alcanza el 10/50/90 % y en que cuaja."""
    V = tabla["volumen_cm3"].to_numpy()
    t = tabla["tiempo_s"].to_numpy()
    nc = tabla["numero_componentes"].to_numpy()
    if V.size == 0 or V.max() <= 0.0:
        return {k: float("nan") for k in
                ("t_10pct", "t_50pct", "t_90pct", "t_una_pieza", "n_max_fragmentos")}
    Vf = V[-1]
    def _t(frac):
        idx = np.flatnonzero(V >= frac * Vf)
        return float(t[idx[0]]) if idx.size else float("nan")
    hay_pieza = np.flatnonzero(nc == 1)
    return {
        "t_10pct": _t(0.10), "t_50pct": _t(0.50), "t_90pct": _t(0.90),
        "t_una_pieza": float(t[hay_pieza[0]]) if hay_pieza.size else float("nan"),
        "n_max_fragmentos": int(nc.max()),
        "volumen_final_cm3": float(Vf),
        "masa_final_g": float(tabla["masa_g"].iloc[-1]),
    }
