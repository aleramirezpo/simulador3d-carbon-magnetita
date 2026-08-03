"""Consistencia temporal del promedio 3-D con el modelo 0-D calibrado.

La referencia se ejecuta directamente con ``modelo_multifase`` y con la
instantanea ``parametros_calibrados_v3.json`` que acompana ese modelo. No se
duplica la quimica ni se reemplaza la curva real de mufla.

La aparicion del aglomerado usa el umbral operativo ``cohesion >= 0.5``. Es
una prediccion de coherencia interna: no existe una medicion experimental del
instante de formacion del aglomerado.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fisica import cohesion as modelo_cohesion
from fisica.adaptador_v3 import MASAS_MOLARES_SOLIDO_KG_MOL, modelo_multifase


TIEMPOS_EXPERIMENTALES_S = np.array(
    [30.0, 60.0, 90.0, 120.0, 150.0, 210.0, 360.0, 720.0],
    dtype=float,
)
RUTA_SALIDA = (
    Path(__file__).resolve().parents[1] / "resultados" / "consistencia_0d.csv"
)
UMBRAL_AGLOMERADO = 0.5

_COLUMNAS_FASE = (
    "frac_Fe_en_Fe2O3",
    "frac_Fe_en_Fe3O4",
    "frac_Fe_en_FeO",
    "frac_Fe_metalico",
    "frac_Fe_en_ilmenita",
    "frac_Fe_en_fayalita",
)


def _parametros_calibrados() -> tuple[Any, Path]:
    """Construye ``Parametros`` desde la instantanea calibrada de v3."""

    ruta = (
        Path(modelo_multifase.__file__).resolve().parents[1]
        / "resultados"
        / "parametros_calibrados_v3.json"
    )
    if not ruta.is_file():
        raise FileNotFoundError(
            "No se encontro la referencia calibrada de modelo_multifase: "
            f"{ruta}"
        )
    with ruta.open("r", encoding="utf-8") as archivo:
        documento = json.load(archivo)
    valores = documento.get("parametros")
    if not isinstance(valores, Mapping):
        raise ValueError(f"{ruta} no contiene el mapa 'parametros'")
    permitidos = modelo_multifase.Parametros.__dataclass_fields__
    recibidos = {k: valores[k] for k in valores.keys() & permitidos.keys()}
    faltantes = sorted(set(permitidos).difference(recibidos))
    if faltantes:
        raise ValueError(f"Faltan parametros calibrados en {ruta}: {faltantes}")
    return modelo_multifase.Parametros(**recibidos), ruta


def _agregar_prediccion_cohesion(tabla: pd.DataFrame) -> pd.DataFrame:
    """Recorre la historia 0-D con el mismo cierre de cohesion usado en 3-D."""

    campo = modelo_cohesion.CampoCohesion((1, 1, 1))
    cohesion = np.zeros(len(tabla), dtype=float)
    plastificacion = np.zeros(len(tabla), dtype=float)
    consolidacion = np.zeros(len(tabla), dtype=bool)
    tiempos = tabla["tiempo_s"].to_numpy(dtype=float)
    temperaturas_K = tabla["T_muestra_C"].to_numpy(dtype=float) + 273.15

    # Con concentraciones de fases puente iguales a cero, las llamadas de
    # termodinamica pueden advertir por unos pocos milikelvin bajo T_REF al
    # inicio. No afectan el cierre y se silencian solo en este recorrido.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(1, len(tabla)):
            modelo_cohesion.evolucionar(
                campo,
                temperaturas_K[i],
                tiempos[i] - tiempos[i - 1],
                fraccion_carbon=0.75,
                porosidad=0.15,
            )
            cohesion[i] = float(campo.c.item())
            historia = campo.historia_termica
            assert historia is not None
            plastificacion[i] = float(historia.grado_plastificacion.item())
            consolidacion[i] = bool(historia.resolidificada.item())

    salida = tabla.copy()
    salida["grado_plastificacion"] = plastificacion
    salida["consolidacion_iniciada"] = consolidacion
    salida["cohesion"] = cohesion
    salida["cohesion_max"] = cohesion
    salida["aglomerado"] = cohesion >= UMBRAL_AGLOMERADO
    return salida


@lru_cache(maxsize=1)
def _historia_termica_0d_cache() -> pd.DataFrame:
    parametros, ruta = _parametros_calibrados()
    # Un punto por segundo incluye exactamente los ocho tiempos experimentales;
    # solve_ivp conserva su control adaptativo interno y max_step=5 s.
    tabla = modelo_multifase.simular(
        parametros,
        t_max=float(TIEMPOS_EXPERIMENTALES_S[-1]),
        n_puntos=int(TIEMPOS_EXPERIMENTALES_S[-1]) + 1,
        estricto=True,
    )
    tabla = _agregar_prediccion_cohesion(tabla)
    tabla.attrs.update(
        {
            "modelo": "modelo_multifase_0d_calibrado",
            "parametros_calibrados": str(ruta),
            "curva_mufla": "curva real usada por modelo_multifase",
            "cohesion": (
                "prediccion no validada; calendario ajustado solo por "
                "coherencia con la cinetica 0-D"
            ),
        }
    )
    return tabla


def historia_termica_0d() -> pd.DataFrame:
    """Devuelve la trayectoria del 0-D con sus parametros calibrados.

    La tabla conserva todas las salidas de ``modelo_multifase.simular`` y
    anade el recorrido cohesivo escalar para poder fechar la consolidacion y
    el umbral operativo del aglomerado. Se devuelve una copia para que una
    comparacion no pueda contaminar la referencia cacheada.
    """

    referencia = _historia_termica_0d_cache()
    salida = referencia.copy(deep=True)
    salida.attrs = dict(referencia.attrs)
    return salida


def _obtener(objeto: Any, *nombres: str, defecto: Any = None) -> Any:
    if isinstance(objeto, Mapping):
        for nombre in nombres:
            if nombre in objeto:
                return objeto[nombre]
    else:
        for nombre in nombres:
            if hasattr(objeto, nombre):
                return getattr(objeto, nombre)
    return defecto


def _columna(tabla: pd.DataFrame, *nombres: str) -> pd.Series | None:
    for nombre in nombres:
        if nombre in tabla.columns:
            return tabla[nombre]
    return None


def _numerica(columna: pd.Series, nombre: str) -> np.ndarray:
    try:
        valores = pd.to_numeric(columna, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"La columna {nombre!r} debe ser numerica y escalar") from exc
    return valores


def _normalizar_dataframe(tabla: pd.DataFrame) -> pd.DataFrame:
    """Normaliza una serie ya promediada sin inventar magnitudes faltantes."""

    tiempo = _columna(tabla, "tiempo_s", "t", "tiempo")
    if tiempo is None:
        raise ValueError("La serie necesita una columna tiempo_s, t o tiempo")
    salida = pd.DataFrame({"tiempo_s": _numerica(tiempo, "tiempo_s")})

    temperatura_C = _columna(
        tabla,
        "T_media_C",
        "temperatura_media_C",
        "T_lecho_C",
        "T_muestra_C",
    )
    if temperatura_C is not None:
        salida["T_media_C"] = _numerica(temperatura_C, temperatura_C.name)
    else:
        temperatura_K = _columna(
            tabla, "T_media_K", "temperatura_media_K", "T_lecho_K", "T"
        )
        if temperatura_K is not None:
            salida["T_media_C"] = _numerica(
                temperatura_K, temperatura_K.name
            ) - 273.15

    perdida_pct = _columna(tabla, "perdida_masa_pct", "perdida_pct")
    if perdida_pct is not None:
        salida["perdida_masa_pct"] = _numerica(
            perdida_pct, perdida_pct.name
        )
    else:
        perdida_frac = _columna(tabla, "perdida_masa_frac", "perdida_frac")
        if perdida_frac is not None:
            salida["perdida_masa_pct"] = 100.0 * _numerica(
                perdida_frac, perdida_frac.name
            )
        else:
            masa = _columna(tabla, "masa_solida_g", "masa_solida_kg", "masa_solida")
            if masa is not None:
                valores = _numerica(masa, masa.name)
                if len(valores) and valores[0] > 0.0:
                    salida["perdida_masa_pct"] = 100.0 * (
                        1.0 - valores / valores[0]
                    )

    alias_directos = {
        "conversion_magnetita": ("conversion_magnetita", "conversion_Fe3O4"),
        "conversion_ilmenita": ("conversion_ilmenita", "conversion_FeTiO3"),
        "grado_reduccion_alpha": ("grado_reduccion_alpha", "grado_reduccion", "alpha"),
        "cohesion": ("cohesion", "cohesion_media"),
        "cohesion_max": ("cohesion_max",),
    }
    for destino, aliases in alias_directos.items():
        valores = _columna(tabla, *aliases)
        if valores is not None:
            salida[destino] = _numerica(valores, valores.name)
    for nombre in _COLUMNAS_FASE:
        valores = _columna(tabla, nombre)
        if valores is not None:
            salida[nombre] = _numerica(valores, nombre)

    aglomerado = _columna(tabla, "aglomerado", "hay_aglomerado")
    if aglomerado is not None:
        salida["aglomerado"] = aglomerado.to_numpy(dtype=bool)
    consolidacion = _columna(
        tabla, "consolidacion_iniciada", "resolidificada"
    )
    if consolidacion is not None:
        salida["consolidacion_iniciada"] = consolidacion.to_numpy(dtype=bool)
    return _ordenar_validar(salida)


def _es_dataframe_escalar(tabla: pd.DataFrame) -> bool:
    """Distingue columnas escalares de un DataFrame de instantaneas 3-D."""

    for nombre in ("T", "cohesion", "solido", "solido_fases"):
        if nombre not in tabla.columns:
            continue
        for valor in tabla[nombre]:
            if valor is None:
                continue
            if isinstance(valor, Mapping) or np.asarray(valor).ndim > 0:
                return False
            break
    return True


def _pesos_lecho(elemento: Any, forma: tuple[int, ...]) -> np.ndarray:
    fraccion = _obtener(
        elemento,
        "fraccion_lecho",
        "fraccion_volumetrica",
        "pesos_volumen",
    )
    if fraccion is not None:
        pesos = np.broadcast_to(np.asarray(fraccion, dtype=float), forma).copy()
    else:
        etiquetas = _obtener(elemento, "etiquetas")
        if etiquetas is None:
            pesos = np.ones(forma, dtype=float)
        else:
            etiquetas = np.broadcast_to(np.asarray(etiquetas), forma)
            pesos = (etiquetas == 3).astype(float)  # nucleo.geometria.LECHO
    if np.any(~np.isfinite(pesos)) or np.any(pesos < 0.0):
        raise ValueError("Los pesos volumetricos del lecho no son validos")
    if not np.any(pesos > 0.0):
        raise ValueError("La instantanea no contiene volumen de lecho")
    return pesos


def _inventarios_solidos(solido: Mapping[str, Any], pesos: np.ndarray) -> dict[str, float]:
    inventarios: dict[str, float] = {}
    for fase, valores in solido.items():
        if str(fase).startswith("_"):
            continue
        campo = np.broadcast_to(np.asarray(valores, dtype=float), pesos.shape)
        inventarios[str(fase)] = float(np.sum(campo * pesos, dtype=np.float64))
    return inventarios


def _normalizar_instantaneas(elementos: Iterable[Any]) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    for elemento_original in elementos:
        if isinstance(elemento_original, tuple) and len(elemento_original) == 2:
            tiempo_tupla, contenido = elemento_original
            if isinstance(contenido, modelo_cohesion.CampoCohesion):
                elemento: Any = {
                    "tiempo_s": tiempo_tupla,
                    "cohesion": contenido.c,
                    "fraccion_volumetrica": contenido.fraccion_volumetrica,
                }
            else:
                elemento = contenido
                if isinstance(elemento, Mapping):
                    elemento = dict(elemento)
                    elemento.setdefault("tiempo_s", tiempo_tupla)
        else:
            elemento = elemento_original

        tiempo = _obtener(elemento, "tiempo_s", "t", "tiempo")
        if tiempo is None:
            raise ValueError("Cada instantanea necesita tiempo_s, t o tiempo")
        fila: dict[str, Any] = {"tiempo_s": float(tiempo)}

        T = _obtener(elemento, "T", "T_media_K", "T_lecho_K")
        T_C = _obtener(elemento, "T_media_C", "T_lecho_C", "T_muestra_C")
        campo_base = T if T is not None else T_C
        if campo_base is None:
            campo_base = _obtener(elemento, "cohesion")
        solido = _obtener(elemento, "solido", "solido_fases")
        if campo_base is None and isinstance(solido, Mapping) and solido:
            campo_base = next(iter(solido.values()))
        if campo_base is None:
            raise ValueError("No se puede determinar la forma de la instantanea")
        forma = np.asarray(campo_base).shape
        if not forma:
            forma = (1,)
        pesos = _pesos_lecho(elemento, forma)
        suma_pesos = float(np.sum(pesos, dtype=np.float64))

        if T is not None:
            temperatura = np.broadcast_to(np.asarray(T, dtype=float), forma)
            fila["T_media_C"] = float(np.sum(temperatura * pesos) / suma_pesos - 273.15)
        elif T_C is not None:
            temperatura = np.broadcast_to(np.asarray(T_C, dtype=float), forma)
            fila["T_media_C"] = float(np.sum(temperatura * pesos) / suma_pesos)

        cohesion = _obtener(elemento, "cohesion")
        if cohesion is not None:
            valores_c = np.broadcast_to(np.asarray(cohesion, dtype=float), forma)
            activos = pesos > 0.0
            fila["cohesion"] = float(np.sum(valores_c * pesos) / suma_pesos)
            fila["cohesion_max"] = float(np.max(valores_c[activos]))
            fila["aglomerado"] = bool(
                np.any(valores_c[activos] >= UMBRAL_AGLOMERADO)
            )

        if isinstance(solido, Mapping):
            inventarios = _inventarios_solidos(solido, pesos)
            fila["_inventarios"] = inventarios
            fila["_masa_solida"] = math.fsum(
                inventarios.get(fase, 0.0) * masa_molar
                for fase, masa_molar in MASAS_MOLARES_SOLIDO_KG_MOL.items()
            )

        for nombre in (
            "perdida_masa_pct",
            "conversion_magnetita",
            "conversion_ilmenita",
            "grado_reduccion_alpha",
        ):
            valor = _obtener(elemento, nombre)
            if valor is not None and np.asarray(valor).ndim == 0:
                fila[nombre] = float(valor)
        filas.append(fila)

    if not filas:
        raise ValueError("La serie esta vacia")
    filas.sort(key=lambda fila: fila["tiempo_s"])

    primera_con_inventario = next(
        (fila for fila in filas if "_inventarios" in fila), None
    )
    if primera_con_inventario is not None:
        inicial = primera_con_inventario["_inventarios"]
        masa0 = float(primera_con_inventario["_masa_solida"])
        oxigeno0 = (
            3.0 * inicial.get("Fe2O3", 0.0)
            + 4.0 * inicial.get("Fe3O4", 0.0)
            + inicial.get("FeO", 0.0)
            + 3.0 * inicial.get("FeTiO3", 0.0)
            + 2.0 * inicial.get("TiO2", 0.0)
        )
        for fila in filas:
            inventarios = fila.get("_inventarios")
            if inventarios is None:
                continue
            if "perdida_masa_pct" not in fila and masa0 > 0.0:
                fila["perdida_masa_pct"] = 100.0 * (
                    1.0 - float(fila["_masa_solida"]) / masa0
                )
            if "conversion_magnetita" not in fila:
                n0 = inicial.get("Fe3O4", 0.0)
                fila["conversion_magnetita"] = (
                    float(np.clip(1.0 - inventarios.get("Fe3O4", 0.0) / n0, 0.0, 1.0))
                    if n0 > 0.0
                    else math.nan
                )
            if "conversion_ilmenita" not in fila:
                n0 = inicial.get("FeTiO3", 0.0)
                fila["conversion_ilmenita"] = (
                    float(np.clip(1.0 - inventarios.get("FeTiO3", 0.0) / n0, 0.0, 1.0))
                    if n0 > 0.0
                    else math.nan
                )
            oxigeno = (
                3.0 * inventarios.get("Fe2O3", 0.0)
                + 4.0 * inventarios.get("Fe3O4", 0.0)
                + inventarios.get("FeO", 0.0)
                + 3.0 * inventarios.get("FeTiO3", 0.0)
                + 2.0 * inventarios.get("TiO2", 0.0)
            )
            if "grado_reduccion_alpha" not in fila:
                fila["grado_reduccion_alpha"] = (
                    float(np.clip(1.0 - oxigeno / oxigeno0, 0.0, 1.0))
                    if oxigeno0 > 0.0
                    else math.nan
                )
            fe_total = (
                2.0 * inventarios.get("Fe2O3", 0.0)
                + 3.0 * inventarios.get("Fe3O4", 0.0)
                + inventarios.get("FeO", 0.0)
                + inventarios.get("Fe", 0.0)
                + inventarios.get("FeTiO3", 0.0)
                + 2.0 * inventarios.get("Fe2SiO4", 0.0)
            )
            if fe_total > 0.0:
                fila.update(
                    {
                        "frac_Fe_en_Fe2O3": 2.0 * inventarios.get("Fe2O3", 0.0) / fe_total,
                        "frac_Fe_en_Fe3O4": 3.0 * inventarios.get("Fe3O4", 0.0) / fe_total,
                        "frac_Fe_en_FeO": inventarios.get("FeO", 0.0) / fe_total,
                        "frac_Fe_metalico": inventarios.get("Fe", 0.0) / fe_total,
                        "frac_Fe_en_ilmenita": inventarios.get("FeTiO3", 0.0) / fe_total,
                        "frac_Fe_en_fayalita": 2.0 * inventarios.get("Fe2SiO4", 0.0) / fe_total,
                    }
                )

    for fila in filas:
        fila.pop("_inventarios", None)
        fila.pop("_masa_solida", None)
    return _ordenar_validar(pd.DataFrame(filas))


def _ordenar_validar(tabla: pd.DataFrame) -> pd.DataFrame:
    if tabla.empty:
        raise ValueError("La serie esta vacia")
    if np.any(~np.isfinite(tabla["tiempo_s"])):
        raise ValueError("Los tiempos deben ser finitos")
    if np.any(tabla["tiempo_s"] < 0.0):
        raise ValueError("Los tiempos no pueden ser negativos")
    tabla = tabla.sort_values("tiempo_s", kind="stable", ignore_index=True)
    if tabla["tiempo_s"].duplicated().any():
        raise ValueError("La serie contiene tiempos duplicados")
    return tabla


def _normalizar_serie(serie: Any) -> pd.DataFrame:
    if isinstance(serie, pd.DataFrame):
        if _es_dataframe_escalar(serie):
            return _normalizar_dataframe(serie)
        return _normalizar_instantaneas(serie.to_dict(orient="records"))
    if isinstance(serie, Mapping):
        tiempo = _obtener(serie, "tiempo_s", "t", "tiempo")
        if tiempo is not None and np.asarray(tiempo).ndim == 0:
            return _normalizar_instantaneas([serie])
        return _normalizar_dataframe(pd.DataFrame(serie))
    if isinstance(serie, (str, bytes, Path)):
        raise TypeError("Pase la serie cargada, no una ruta")
    return _normalizar_instantaneas(serie)


def _interpolar(tabla: pd.DataFrame, columna: str, tiempos: np.ndarray) -> np.ndarray:
    if columna not in tabla:
        return np.full(tiempos.shape, np.nan, dtype=float)
    x = tabla["tiempo_s"].to_numpy(dtype=float)
    y = pd.to_numeric(tabla[columna], errors="coerce").to_numpy(dtype=float)
    validos = np.isfinite(x) & np.isfinite(y)
    if not np.any(validos):
        return np.full(tiempos.shape, np.nan, dtype=float)
    x, y = x[validos], y[validos]
    salida = np.interp(tiempos, x, y)
    salida[(tiempos < x[0]) | (tiempos > x[-1])] = np.nan
    return salida


def comparar_con_0d(serie_3d: Any) -> pd.DataFrame:
    """Compara promedios 3-D con el 0-D en los ocho tiempos experimentales.

    ``serie_3d`` puede ser un DataFrame de promedios o un iterable de
    instantaneas con campos 3-D. Para estas ultimas se usa la fraccion de lecho
    si esta disponible y, en su defecto, ``etiquetas == 3``. Los inventarios
    solidos permiten reconstruir perdida de masa, conversion de magnetita e
    ilmenita, fracciones de Fe por fase y grado de reduccion.

    La tabla devuelta se exporta siempre a ``resultados/consistencia_0d.csv``.
    Una serie que no cubra un tiempo experimental produce ``NaN`` en vez de
    extrapolar silenciosamente.
    """

    es_autoconsistencia = bool(
        isinstance(serie_3d, pd.DataFrame)
        and serie_3d.attrs.get("modelo") == "modelo_multifase_0d_calibrado"
    )
    referencia = _normalizar_serie(historia_termica_0d())
    tridimensional = _normalizar_serie(serie_3d)
    tiempos = TIEMPOS_EXPERIMENTALES_S.copy()
    salida = pd.DataFrame({"tiempo_s": tiempos})
    salida["origen_serie_3d"] = (
        "limite_uniforme_0d_autoconsistencia"
        if es_autoconsistencia
        else "promedio_volumetrico_3d"
    )

    metricas = {
        "T_media_C": "T",
        "perdida_masa_pct": "perdida_masa",
        "conversion_magnetita": "conversion_magnetita",
        "conversion_ilmenita": "conversion_ilmenita",
        "grado_reduccion_alpha": "grado_reduccion",
        **{nombre: nombre for nombre in _COLUMNAS_FASE},
    }
    for columna, etiqueta in metricas.items():
        valor_0d = _interpolar(referencia, columna, tiempos)
        valor_3d = _interpolar(tridimensional, columna, tiempos)
        salida[f"{etiqueta}_0d"] = valor_0d
        salida[f"{etiqueta}_3d"] = valor_3d
        salida[f"error_{etiqueta}"] = valor_3d - valor_0d
        salida[f"error_abs_{etiqueta}"] = np.abs(valor_3d - valor_0d)

    # Alias con unidades explicitas para consumo humano; se conservan los
    # nombres breves anteriores para scripts de posproceso.
    salida["temperatura_0d_C"] = salida["T_0d"]
    salida["temperatura_3d_C"] = salida["T_3d"]
    salida["error_temperatura_C"] = salida["error_T"]

    exp = getattr(modelo_multifase, "exp", None)
    if exp is not None:
        salida["perdida_masa_experimental_pct"] = 100.0 * np.interp(
            tiempos,
            np.asarray(exp.TIEMPOS_EXP_S, dtype=float),
            np.asarray(exp.PERDIDA_MASA_EXP_FRAC, dtype=float),
        )
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(RUTA_SALIDA, index=False, float_format="%.10g")
    return salida


def _primer_cruce(
    tiempos: np.ndarray, valores: np.ndarray, umbral: float
) -> float:
    validos = np.isfinite(tiempos) & np.isfinite(valores)
    tiempos, valores = tiempos[validos], valores[validos]
    if not len(tiempos):
        return math.nan
    if valores[0] >= umbral:
        return float(tiempos[0])
    for i in range(1, len(tiempos)):
        if valores[i] >= umbral and valores[i - 1] < umbral:
            dy = valores[i] - valores[i - 1]
            if dy <= 0.0:
                return float(tiempos[i])
            fraccion = (umbral - valores[i - 1]) / dy
            return float(tiempos[i - 1] + fraccion * (tiempos[i] - tiempos[i - 1]))
    return math.nan


def tiempos_caracteristicos(serie: Any) -> dict[str, float]:
    """Extrae hitos termicos, de perdida de masa y de aglomeracion.

    Los cruces continuos se interpolan linealmente. Un hito que la serie no
    alcanza se reporta como ``NaN``; nunca se extrapola. El 50 % y el 90 % se
    refieren a la ultima perdida de masa finita de la serie.
    """

    tabla = _normalizar_serie(serie)
    tiempos = tabla["tiempo_s"].to_numpy(dtype=float)

    temperatura = (
        tabla["T_media_C"].to_numpy(dtype=float)
        if "T_media_C" in tabla
        else np.full(len(tabla), np.nan)
    )
    perdida = (
        tabla["perdida_masa_pct"].to_numpy(dtype=float)
        if "perdida_masa_pct" in tabla
        else np.full(len(tabla), np.nan)
    )
    perdida_finita = perdida[np.isfinite(perdida)]
    perdida_final = float(perdida_finita[-1]) if len(perdida_finita) else math.nan

    cohesion = None
    if "cohesion_max" in tabla:
        cohesion = tabla["cohesion_max"].to_numpy(dtype=float)
    elif "cohesion" in tabla:
        cohesion = tabla["cohesion"].to_numpy(dtype=float)

    if cohesion is not None:
        t_aglomerado = _primer_cruce(tiempos, cohesion, UMBRAL_AGLOMERADO)
    elif "aglomerado" in tabla:
        indices = np.flatnonzero(tabla["aglomerado"].to_numpy(dtype=bool))
        t_aglomerado = float(tiempos[indices[0]]) if indices.size else math.nan
    else:
        t_aglomerado = math.nan

    if "consolidacion_iniciada" in tabla:
        indices = np.flatnonzero(
            tabla["consolidacion_iniciada"].to_numpy(dtype=bool)
        )
        t_consolidacion = float(tiempos[indices[0]]) if indices.size else math.nan
    else:
        t_consolidacion = math.nan

    return {
        "t_350_C_s": _primer_cruce(tiempos, temperatura, 350.0),
        "t_500_C_s": _primer_cruce(tiempos, temperatura, 500.0),
        "t_900_C_s": _primer_cruce(tiempos, temperatura, 900.0),
        "t_perdida_50_s": (
            _primer_cruce(tiempos, perdida, 0.5 * perdida_final)
            if math.isfinite(perdida_final)
            else math.nan
        ),
        "t_perdida_90_s": (
            _primer_cruce(tiempos, perdida, 0.9 * perdida_final)
            if math.isfinite(perdida_final)
            else math.nan
        ),
        "t_consolidacion_s": t_consolidacion,
        "t_aglomerado_s": t_aglomerado,
    }


__all__ = [
    "RUTA_SALIDA",
    "TIEMPOS_EXPERIMENTALES_S",
    "UMBRAL_AGLOMERADO",
    "comparar_con_0d",
    "historia_termica_0d",
    "tiempos_caracteristicos",
]


if __name__ == "__main__":
    referencia_0d = historia_termica_0d()
    print(pd.Series(tiempos_caracteristicos(referencia_0d)).to_string())
