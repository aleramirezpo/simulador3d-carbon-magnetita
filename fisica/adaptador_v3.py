"""Puente vectorizado entre la malla 3-D y la química validada de v3.

Este módulo no define mecanismos químicos nuevos. Las constantes cinéticas,
las afinidades, las propiedades termoquímicas, la composición del carbón y la
mineralogía proceden de ``simulacion_v3/src``. El trabajo del adaptador es
convertir los inventarios molares absolutos del modelo 0-D a concentraciones y
aplicar las mismas expresiones a arreglos de celdas.

Base de los inventarios sólidos
--------------------------------
Las fases químicas se expresan en mol/m³. Para poder conservar también la masa
del análisis próximo sin inventar fórmulas moleculares, ``volatil`` y
``ceniza`` son pseudoespecies de masa molar 1 g/mol. ``volatil`` conserva los
moles de C y O que v3 libera por gramo; ``ceniza`` es inerte. ``H2O_liq`` sí es
agua real y ``C`` es el carbono fijo que usa el RHS de v3.

Importación de v3
-----------------
El repositorio hermano se localiza respecto a este archivo (o mediante
``SIMULACION_V3_SRC``) y su carpeta ``src`` se antepone explícitamente a
``sys.path``. v3 usa importaciones absolutas entre módulos (``import gases``),
por lo que esta es la única forma compatible sin copiar ni modificar v3.

Coste de funciones escalares
----------------------------
Varias funciones validadas de v3 sólo aceptan escalares. ``_aplicar_escalar``
detecta primero campos uniformes y hace una sola llamada. Para campos no
uniformes usa ``numpy.vectorize`` como último recurso: conserva exactamente el
código validado, pero sigue siendo un bucle Python interno. El benchmark debe,
por ello, distinguir campos uniformes y heterogéneos.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .tablas_termo import TablaTermoquimica


# ---------------------------------------------------------------------------
# Localización e importación robusta de simulacion_v3
# ---------------------------------------------------------------------------
_ARCHIVOS_V3 = (
    "gases.py",
    "termodinamica_ext.py",
    "grano.py",
    "mineralogia.py",
    "carbon.py",
    "modelo_multifase.py",
)


def _localizar_src_v3() -> Path:
    candidata_env = os.environ.get("SIMULACION_V3_SRC")
    candidatas = []
    if candidata_env:
        candidatas.append(Path(candidata_env).expanduser())
    # .../pae/simulador3d/fisica/adaptador_v3.py -> .../pae/simulacion_v3/src
    candidatas.append(Path(__file__).resolve().parents[2] / "simulacion_v3" / "src")

    faltantes_por_ruta: list[str] = []
    for candidata in candidatas:
        ruta = candidata.resolve()
        faltantes = [nombre for nombre in _ARCHIVOS_V3 if not (ruta / nombre).is_file()]
        if not faltantes:
            return ruta
        faltantes_por_ruta.append(f"{ruta}: faltan {', '.join(faltantes)}")
    detalle = "; ".join(faltantes_por_ruta)
    raise ImportError(
        "No se encontró la química validada de simulacion_v3. "
        "Se requiere su carpeta src con gases.py, termodinamica_ext.py, grano.py, "
        f"mineralogia.py, carbon.py y modelo_multifase.py. Rutas revisadas: {detalle}"
    )


RUTA_SRC_V3 = _localizar_src_v3()
if str(RUTA_SRC_V3) not in sys.path:
    # v3 importa sus módulos hermanos por nombre corto; debe tener precedencia.
    sys.path.insert(0, str(RUTA_SRC_V3))


def _importar_v3(nombre: str) -> Any:
    modulo = importlib.import_module(nombre)
    archivo = Path(getattr(modulo, "__file__", "")).resolve()
    if archivo.parent != RUTA_SRC_V3:
        raise ImportError(
            f"El módulo {nombre!r} se resolvió en {archivo}, no en la química v3 "
            f"esperada ({RUTA_SRC_V3}). Retire el módulo homónimo de sys.modules."
        )
    return modulo


gases = _importar_v3("gases")
termodinamica_ext = _importar_v3("termodinamica_ext")
grano = _importar_v3("grano")
mineralogia = _importar_v3("mineralogia")
carbon = _importar_v3("carbon")
modelo_multifase = _importar_v3("modelo_multifase")


# La construccion cuesta unos segundos pero se hace una sola vez. Para evitar
# incluso ese coste entre arranques se puede definir TABLA_TERMO_NPZ; la clase
# verifica la huella del archivo fuente antes de aceptar el cache.
_TABLA_TERMOQUIMICA: TablaTermoquimica | None = None
_REACCIONES_TASAS = (
    "boudouard",
    "water_gas",
    "water_gas_shift",
    "hematita_CO",
    "hematita_H2",
    "magnetita_CO",
    "magnetita_H2",
    "wustita_CO",
    "wustita_H2",
    "ilmenita_CO",
    "ilmenita_H2",
)


def obtener_tabla_termoquimica() -> TablaTermoquimica:
    """Devuelve el singleton tabulado, cargandolo de .npz cuando se configura."""

    global _TABLA_TERMOQUIMICA
    if _TABLA_TERMOQUIMICA is None:
        cache = os.environ.get("TABLA_TERMO_NPZ")
        if cache:
            _TABLA_TERMOQUIMICA = TablaTermoquimica.cargar_o_crear(
                cache, termodinamica=termodinamica_ext
            )
        else:
            _TABLA_TERMOQUIMICA = TablaTermoquimica(
                termodinamica=termodinamica_ext
            )
    return _TABLA_TERMOQUIMICA


def _usar_tablas(cfg: Any, opcion: bool | None = None) -> bool:
    if opcion is not None:
        return bool(opcion)
    return bool(_cfg(cfg, "usar_tablas", True))


def _arrhenius_vectorizado(A: Any, Ea_kJ: Any, T: np.ndarray) -> np.ndarray:
    exponente = -np.asarray(Ea_kJ, dtype=float) * 1000.0 / (
        float(modelo_multifase.R_GAS) * np.maximum(T, 1.0)
    )
    return np.asarray(A, dtype=float) * np.exp(np.minimum(exponente, 60.0))


def _sigmoide_vectorizada(x: np.ndarray, x0: float, ancho: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.minimum(-(x - x0) / ancho, 60.0)))


def _disponibilidad_vectorizada(inventario: np.ndarray, referencia: float) -> np.ndarray:
    return np.divide(
        inventario,
        inventario + referencia,
        out=np.zeros_like(inventario, dtype=float),
        where=inventario > 0.0,
    )


def _minimo_suave_vectorizado(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    positivo = (a > 0.0) & (b > 0.0)
    return np.divide(
        a * b,
        np.sqrt(a * a + b * b),
        out=np.zeros_like(a, dtype=float),
        where=positivo,
    )


# ---------------------------------------------------------------------------
# Convenciones y estequiometría del estado local
# ---------------------------------------------------------------------------
ESPECIES_GAS = tuple(gases.ESPECIES_GAS)
FASES_SOLIDAS = (
    "H2O_liq",
    "volatil",
    "C",
    "Fe2O3",
    "Fe3O4",
    "FeO",
    "Fe",
    "FeTiO3",
    "TiO2",
    "SiO2",
    "Fe2SiO4",
    "FeS",
    "ceniza",
)

# kg/mol. Las dos pseudoespecies equivalen exactamente a 1 g por pseudomol.
MASAS_MOLARES_GAS_KG_MOL = {
    especie: float(modelo_multifase.MW[especie]) * 1.0e-3
    for especie in ESPECIES_GAS
}
MASAS_MOLARES_SOLIDO_KG_MOL = {
    "H2O_liq": float(modelo_multifase.MW["H2O"]) * 1.0e-3,
    "volatil": 1.0e-3,
    "C": float(modelo_multifase.MW["C"]) * 1.0e-3,
    "Fe2O3": float(mineralogia.COMPOSICION_RIETVELD["hematita"]["MW"]) * 1.0e-3,
    "Fe3O4": float(mineralogia.COMPOSICION_RIETVELD["magnetita"]["MW"]) * 1.0e-3,
    "FeO": float(modelo_multifase.MW["FeO"]) * 1.0e-3,
    "Fe": float(modelo_multifase.MW["Fe"]) * 1.0e-3,
    "FeTiO3": float(mineralogia.COMPOSICION_RIETVELD["ilmenita"]["MW"]) * 1.0e-3,
    "TiO2": float(modelo_multifase.MW["TiO2"]) * 1.0e-3,
    "SiO2": float(mineralogia.COMPOSICION_RIETVELD["cuarzo"]["MW"]) * 1.0e-3,
    "Fe2SiO4": float(modelo_multifase.MW["Fe2SiO4"]) * 1.0e-3,
    "FeS": float(modelo_multifase.MW["FeS"]) * 1.0e-3,
    "ceniza": 1.0e-3,
}

_VOLATILES_MOL_POR_G = dict(modelo_multifase._VOLATILES_MOL_POR_G)
_C_VOLATIL_MOL_POR_G = float(modelo_multifase._C_VOLATIL_MOL_POR_G)
_O_VOLATIL_MOL_POR_G = float(
    _VOLATILES_MOL_POR_G["CO"]
    + 2.0 * _VOLATILES_MOL_POR_G["CO2"]
    + _VOLATILES_MOL_POR_G["H2O"]
)

# Coeficientes atómicos por mol o pseudomol de inventario.
COMPOSICION_ELEMENTAL: dict[str, dict[str, float]] = {
    "CO": {"C": 1.0, "O": 1.0},
    "CO2": {"C": 1.0, "O": 2.0},
    "H2": {},
    "H2O": {"O": 1.0},
    "CH4": {"C": 1.0},
    "N2": {},
    "O2": {"O": 2.0},
    "H2O_liq": {"O": 1.0},
    "volatil": {"C": _C_VOLATIL_MOL_POR_G, "O": _O_VOLATIL_MOL_POR_G},
    "C": {"C": 1.0},
    "Fe2O3": {"Fe": 2.0, "O": 3.0},
    "Fe3O4": {"Fe": 3.0, "O": 4.0},
    "FeO": {"Fe": 1.0, "O": 1.0},
    "Fe": {"Fe": 1.0},
    "FeTiO3": {"Fe": 1.0, "Ti": 1.0, "O": 3.0},
    "TiO2": {"Ti": 1.0, "O": 2.0},
    "SiO2": {"Si": 1.0, "O": 2.0},
    "Fe2SiO4": {"Fe": 2.0, "Si": 1.0, "O": 4.0},
    "FeS": {"Fe": 1.0},
    "ceniza": {},
}

_ALIAS_SOLIDO = {
    "H2O_liq": ("H2O_liq", "humedad", "m_moist"),
    "volatil": ("volatil", "m_vol"),
    "C": ("C", "char", "carbon", "m_char"),
    "Fe2O3": ("Fe2O3", "hematita", "n_Fe2O3"),
    "Fe3O4": ("Fe3O4", "magnetita", "n_Fe3O4"),
    "FeO": ("FeO", "wustita", "n_FeO"),
    "Fe": ("Fe", "hierro", "n_Fe"),
    "FeTiO3": ("FeTiO3", "ilmenita", "n_FeTiO3"),
    "TiO2": ("TiO2", "rutilo", "n_TiO2"),
    "SiO2": ("SiO2", "cuarzo", "n_SiO2"),
    "Fe2SiO4": ("Fe2SiO4", "fayalita", "n_Fe2SiO4"),
    "FeS": ("FeS", "n_FeS"),
    "ceniza": ("ceniza", "ash", "m_ash"),
}


def _cfg(cfg: Any, nombre: str, defecto: Any) -> Any:
    if cfg is None:
        return defecto
    if isinstance(cfg, Mapping):
        return cfg.get(nombre, defecto)
    return getattr(cfg, nombre, defecto)


def _parametros(cfg: Any) -> Any:
    candidato = _cfg(cfg, "parametros_v3", None)
    if candidato is not None:
        return candidato
    if isinstance(cfg, modelo_multifase.Parametros):
        return cfg
    permitidos = modelo_multifase.Parametros.__dataclass_fields__
    valores = {
        nombre: _cfg(cfg, nombre, campo.default)
        for nombre, campo in permitidos.items()
        if _cfg(cfg, nombre, None) is not None
    }
    return modelo_multifase.Parametros(**valores)


def _como_array(valor: Any, forma: tuple[int, ...], nombre: str) -> np.ndarray:
    arreglo = np.asarray(valor, dtype=float)
    try:
        salida = np.broadcast_to(arreglo, forma)
    except ValueError as exc:
        raise ValueError(
            f"{nombre} tiene forma {arreglo.shape}; no es compatible con {forma}"
        ) from exc
    if not np.all(np.isfinite(salida)):
        raise ValueError(f"{nombre} contiene valores no finitos")
    return np.asarray(salida, dtype=float)


def _valor_solido(
    solido: Mapping[str, Any], fase: str, forma: tuple[int, ...]
) -> np.ndarray:
    for alias in _ALIAS_SOLIDO[fase]:
        if alias in solido:
            valor = _como_array(solido[alias], forma, f"solido[{alias!r}]")
            if alias == "m_moist":
                # Compatibilidad: el nombre de v3 lleva gramos; en el puente la
                # unidad canónica es mol de agua.
                return valor / float(modelo_multifase.MW["H2O"])
            if alias == "m_char":
                return valor / float(modelo_multifase.MW["C"])
            return valor
    return np.zeros(forma, dtype=float)


def _es_uniforme(arreglo: np.ndarray) -> bool:
    return arreglo.size <= 1 or bool(np.all(arreglo == arreglo.flat[0]))


def _aplicar_escalar(funcion: Callable[..., float], *argumentos: Any) -> np.ndarray:
    """Aplica una función escalar de v3, con ruta rápida para campos uniformes."""

    arreglos = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in argumentos))
    forma = arreglos[0].shape
    if all(_es_uniforme(a) for a in arreglos):
        valor = float(funcion(*(float(a.flat[0]) for a in arreglos)))
        return np.full(forma, valor, dtype=float)
    # Último recurso para las APIs escalares de v3. np.vectorize no compila:
    # conserva exactitud funcional, pero su coste es un llamado Python/celda.
    vectorizada = np.vectorize(funcion, otypes=[float])
    return np.asarray(vectorizada(*arreglos), dtype=float)


def _forma_campos(T: Any, c_gas: Mapping[str, Any], solido: Mapping[str, Any]) -> tuple[int, ...]:
    formas = [np.asarray(T).shape]
    formas.extend(np.asarray(v).shape for v in c_gas.values())
    formas.extend(np.asarray(v).shape for k, v in solido.items() if not k.startswith("_"))
    try:
        return np.broadcast_shapes(*formas)
    except ValueError as exc:
        raise ValueError(f"Los campos de química no son broadcastables: {formas}") from exc


# ---------------------------------------------------------------------------
# Estado inicial conservativo
# ---------------------------------------------------------------------------
def estado_inicial_celda(
    fraccion_lecho: Any, volumen_celda_m3: Any
) -> dict[str, np.ndarray]:
    """Reparte exactamente la carga de 1,00 g entre las celdas de lecho.

    ``fraccion_lecho`` pondera celdas cortadas por la frontera. La
    concentración se define por volumen total de celda, por lo que el número
    de moles de una celda es ``c * volumen_celda_m3``. El último peso activo se
    corrige por el residuo de redondeo para que la integral discreta de cada
    componente sea exactamente su inventario 0-D a precisión de máquina.
    """

    fraccion = np.asarray(fraccion_lecho, dtype=float)
    volumen = np.asarray(volumen_celda_m3, dtype=float)
    try:
        fraccion, volumen = np.broadcast_arrays(fraccion, volumen)
    except ValueError as exc:
        raise ValueError("fraccion_lecho y volumen_celda_m3 no son compatibles") from exc
    if np.any(~np.isfinite(fraccion)) or np.any((fraccion < 0.0) | (fraccion > 1.0)):
        raise ValueError("fraccion_lecho debe contener valores finitos en [0, 1]")
    if np.any(~np.isfinite(volumen)) or np.any(volumen <= 0.0):
        raise ValueError("volumen_celda_m3 debe ser finito y positivo")
    volumen_lecho = float(np.sum(fraccion * volumen, dtype=np.float64))
    if volumen_lecho <= 0.0:
        raise ValueError("Debe existir al menos una celda con fracción de lecho positiva")

    reparto = np.asarray(fraccion / volumen_lecho, dtype=float).copy()
    activos = np.flatnonzero(fraccion.ravel() > 0.0)
    ultimo = int(activos[-1])
    integral = float(np.sum(reparto * volumen, dtype=np.float64))
    reparto.ravel()[ultimo] += (1.0 - integral) / volumen.ravel()[ultimo]

    ci = modelo_multifase.CI
    composicion = mineralogia.composicion_molar(0.25)
    n = composicion["moles_fases"]
    inventarios = {
        "H2O_liq": float(ci["m_moist0"]) / float(modelo_multifase.MW["H2O"]),
        "volatil": float(ci["m_vol0"]),  # pseudomol = g
        "C": float(ci["m_char0"]) / float(modelo_multifase.MW["C"]),
        "Fe2O3": float(n["hematita"]),
        "Fe3O4": float(n["magnetita"]),
        "FeO": 0.0,
        "Fe": 0.0,
        "FeTiO3": float(n["ilmenita"]),
        "TiO2": 0.0,
        "SiO2": float(n["cuarzo"]),
        "Fe2SiO4": 0.0,
        "FeS": 0.0,
        "ceniza": float(ci["m_ash"]),  # pseudomol = g
    }
    estado = {fase: reparto * moles for fase, moles in inventarios.items()}
    # Referencia material de la conversión magnetítica para la accesibilidad de
    # las lamas. Es metadato con la misma unidad, no otra fase química.
    estado["_Fe3O4_inicial"] = estado["Fe3O4"].copy()
    estado["_volatil_inicial"] = estado["volatil"].copy()
    return estado


# ---------------------------------------------------------------------------
# Tasas locales: misma química que modelo_multifase.rhs
# ---------------------------------------------------------------------------
def _afinidades_v3(
    T: np.ndarray,
    co: np.ndarray,
    co2: np.ndarray,
    h2: np.ndarray,
    h2o: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Devuelve los ocho canales con una sola evaluación termodinámica.

    Replica deliberadamente la ruta de :func:`modelo_multifase.rhs`: cuando
    ``factor_afinidad`` es el original, ``potencial_reductor`` se consulta una
    sola vez. Si una prueba o escenario sustituye públicamente
    ``gases.factor_afinidad``, esa compuerta se respeta par por par.
    """

    pares = ("Fe2O3/Fe3O4", "Fe3O4/FeO", "FeO/Fe", "FeTiO3/Fe")

    def escalar(
        temp: float, nco: float, nco2: float, nh2: float, nh2o: float
    ) -> tuple[float, ...]:
        inventario = {"CO": nco, "CO2": nco2, "H2": nh2, "H2O": nh2o}
        potencial = gases.potencial_reductor(inventario, temp)
        salida: list[float] = []
        for par in pares:
            if gases.factor_afinidad is modelo_multifase._FACTOR_AFINIDAD_ORIGINAL:
                compuerta = 1.0
            else:
                compuerta = gases.factor_afinidad(par, inventario, temp)
            par_normalizado = "FeTiO3/Fe+TiO2" if par == "FeTiO3/Fe" else par
            detalle = potencial["detalle"][par_normalizado]
            for canal in ("CO", "H2"):
                valor = detalle[f"afinidad_{canal}"]
                afinidad = max(0.0, float(valor)) if valor is not None else 0.0
                salida.append(min(compuerta, afinidad) if compuerta > 0.0 else 0.0)
        return tuple(salida)

    arreglos = np.broadcast_arrays(T, co, co2, h2, h2o)
    forma = arreglos[0].shape
    if all(_es_uniforme(a) for a in arreglos):
        valores = escalar(*(float(a.flat[0]) for a in arreglos))
        campos = tuple(np.full(forma, valor, dtype=float) for valor in valores)
    else:
        # La API termodinámica de v3 es escalar. Se hace una llamada/celda, no
        # ocho llamadas/celda, y se conservan exactamente sus ocho resultados.
        vectorizada = np.vectorize(escalar, otypes=[float] * 8)
        campos = tuple(np.asarray(v, dtype=float) for v in vectorizada(*arreglos))
    return {
        par: {"CO": campos[2 * i], "H2": campos[2 * i + 1]}
        for i, par in enumerate(pares)
    }


def _afinidades_tabuladas(
    co: np.ndarray,
    co2: np.ndarray,
    h2: np.ndarray,
    h2o: np.ndarray,
    K: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Forma vectorizada de ``gases.potencial_reductor`` para sus 8 canales."""

    pares = {
        "Fe2O3/Fe3O4": ("hematita_CO", "hematita_H2"),
        "Fe3O4/FeO": ("magnetita_CO", "magnetita_H2"),
        "FeO/Fe": ("wustita_CO", "wustita_H2"),
        "FeTiO3/Fe": ("ilmenita_CO", "ilmenita_H2"),
    }
    eps = float(gases._EPS_MOLES)

    def canal(
        reductor: np.ndarray, oxidado: np.ndarray, equilibrio: np.ndarray
    ) -> np.ndarray:
        cociente = np.divide(
            oxidado,
            reductor,
            out=np.full_like(reductor, np.inf),
            where=reductor > eps,
        )
        afinidad = 1.0 - cociente / equilibrio
        # Si ninguno de los dos gases existe, v3 devuelve None y luego cero.
        afinidad = np.where((reductor <= eps) & (oxidado <= eps), 0.0, afinidad)
        return np.clip(afinidad, 0.0, 1.0)

    return {
        par: {
            "CO": canal(co, co2, K[reaccion_co]),
            "H2": canal(h2, h2o, K[reaccion_h2]),
        }
        for par, (reaccion_co, reaccion_h2) in pares.items()
    }


def _calcular_tasas(
    T: Any,
    c_gas: Mapping[str, Any],
    solido: Mapping[str, Any],
    eps: Any,
    cfg: Any,
    *,
    devolver_reacciones: bool,
    usar_tablas: bool,
) -> dict[str, Any]:
    forma = _forma_campos(T, c_gas, solido)
    temperatura = _como_array(T, forma, "T")
    porosidad = _como_array(eps, forma, "eps")
    if np.any(temperatura <= 0.0):
        raise ValueError("T debe ser mayor que 0 K")
    if np.any((porosidad < 0.0) | (porosidad > 1.0)):
        raise ValueError("eps debe pertenecer a [0, 1]")

    c = {
        especie: np.maximum(
            _como_array(c_gas.get(especie, 0.0), forma, f"c_gas[{especie!r}]"), 0.0
        )
        for especie in ESPECIES_GAS
    }
    s = {fase: np.maximum(_valor_solido(solido, fase, forma), 0.0) for fase in FASES_SOLIDAS}
    # La ruta escalar ya era extraordinariamente eficiente para un campo
    # uniforme porque evalua cada funcion una sola vez. Se conserva ese atajo
    # (salvo en una celda, util para validar realmente la tabla contra el 0-D).
    if usar_tablas and temperatura.size > 1 and all(
        _es_uniforme(campo)
        for campo in (temperatura, porosidad, *c.values(), *s.values())
    ):
        usar_tablas = False
    P = _parametros(cfg)
    V_ref = float(_cfg(cfg, "volumen_referencia_m3", modelo_multifase.CI["V_libre_m3"]))
    if not np.isfinite(V_ref) or V_ref <= 0.0:
        raise ValueError("volumen_referencia_m3 debe ser positivo")

    T_termo = np.maximum(temperatura, float(termodinamica_ext.T_REF))
    p = {especie: c[especie] * termodinamica_ext.R * temperatura for especie in ESPECIES_GAS}
    n_ref = {especie: c[especie] * V_ref for especie in ESPECIES_GAS}
    m_char_ref_g = s["C"] * float(modelo_multifase.MW["C"]) * V_ref

    datos_reaccion: dict[str, dict[str, np.ndarray]] = {}
    h_combustion: dict[str, np.ndarray] = {}
    if usar_tablas:
        tabla = obtener_tabla_termoquimica()
        lote_reacciones = tabla.datos_reacciones(T_termo, _REACCIONES_TASAS)
        datos_reaccion = {
            nombre: {
                magnitud: lote_reacciones[magnitud][i]
                for magnitud in ("delta_H", "delta_G", "log_K", "K_eq")
            }
            for i, nombre in enumerate(_REACCIONES_TASAS)
        }
        lote_especies = tabla.datos_especies(T_termo, ("CO", "CO2", "C", "O2"))["h"]
        h_combustion = {
            especie: lote_especies[i]
            for i, especie in enumerate(("CO", "CO2", "C", "O2"))
        }

    # Secado y devolatilización, en las mismas bases de masa del RHS 0-D.
    m_humedad_g_m3 = s["H2O_liq"] * float(modelo_multifase.MW["H2O"])
    if usar_tablas:
        g_dry = _sigmoide_vectorizada(
            temperatura - 273.15,
            float(modelo_multifase.lit.SECADO["T_evap_C"]),
            float(modelo_multifase.lit.SECADO["ancho_C"]),
        )
    else:
        g_dry = _aplicar_escalar(
            modelo_multifase.sigmoide,
            temperatura - 273.15,
            np.full(forma, float(modelo_multifase.lit.SECADO["T_evap_C"])),
            np.full(forma, float(modelo_multifase.lit.SECADO["ancho_C"])),
        )
    r_dry_g = 0.5 * m_humedad_g_m3 * g_dry
    r_dry = r_dry_g / float(modelo_multifase.MW["H2O"])

    if usar_tablas:
        g_vol = _sigmoide_vectorizada(
            temperatura - 273.15,
            float(modelo_multifase.lit.DEVOLATILIZACION["T_inicio_C"]),
            40.0,
        )
        k_vol = _arrhenius_vectorizado(
            float(modelo_multifase.lit.DEVOLATILIZACION["A_s"] * P.A_vol_mult),
            float(modelo_multifase.lit.DEVOLATILIZACION["Ea_kJ_mol"]),
            temperatura,
        )
    else:
        g_vol = _aplicar_escalar(
            modelo_multifase.sigmoide,
            temperatura - 273.15,
            np.full(forma, float(modelo_multifase.lit.DEVOLATILIZACION["T_inicio_C"])),
            np.full(forma, 40.0),
        )
        k_vol = _aplicar_escalar(
            modelo_multifase.arrhenius,
            np.full(forma, float(modelo_multifase.lit.DEVOLATILIZACION["A_s"] * P.A_vol_mult)),
            np.full(forma, float(modelo_multifase.lit.DEVOLATILIZACION["Ea_kJ_mol"])),
            temperatura,
        )
    k_vol_ef = k_vol * g_vol
    # El residual se escala con la carga local inicial cuando está disponible.
    vol_inicial = solido.get("_volatil_inicial", None)
    if vol_inicial is None:
        vol_inicial_arr = np.full(forma, float(modelo_multifase.CI["m_vol0"]) / V_ref)
    else:
        vol_inicial_arr = _como_array(vol_inicial, forma, "solido['_volatil_inicial']")
    residual = (1.0 - np.clip(float(P.f_vol_liberable), 0.0, 1.0)) * vol_inicial_arr
    vol_reactivo = np.maximum(s["volatil"] - residual, 0.0)
    r_vol = vol_reactivo * k_vol_ef / (1.0 + 0.5 * k_vol_ef)

    # Combustión y gasificación se evalúan mediante las funciones validadas.
    escala_char = float(modelo_multifase._ESCALA_REACTIVIDAD_CHAR)
    if usar_tablas:
        actividad_o2 = np.maximum(p["O2"], 0.0) / float(gases.P_ESTANDAR_PA)
        k_comb = float(gases.A_COMBUSTION_MOL_KG_S) * np.exp(
            -float(gases.EA_COMBUSTION_J_MOL)
            / (float(termodinamica_ext.R) * T_termo)
        )
        r_comb = escala_char * k_comb * (m_char_ref_g / 1000.0) * actividad_o2 / V_ref
        disp_char = _disponibilidad_vectorizada(
            m_char_ref_g, float(modelo_multifase._M_SOLIDO_REF_G)
        )
        disp_o2 = _disponibilidad_vectorizada(
            n_ref["O2"], float(modelo_multifase._N_GAS_REF)
        )
    else:
        r_comb = escala_char * _aplicar_escalar(
            gases.r_combustion, T_termo, m_char_ref_g, p["O2"]
        ) / V_ref
        disp_char = _aplicar_escalar(
            modelo_multifase._disponibilidad,
            m_char_ref_g,
            np.full(forma, float(modelo_multifase._M_SOLIDO_REF_G)),
        )
        disp_o2 = _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["O2"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        )
    r_comb *= disp_char * disp_o2

    if usar_tablas:
        ratio_comb = 10.0**3.4 * np.exp(-12_400.0 / T_termo)
        f_co = ratio_comb / (1.0 + ratio_comb)
    else:
        f_co = _aplicar_escalar(
            lambda temp: float(gases.reparto_combustion(temp)["CO"]), T_termo
        )
    f_co2 = 1.0 - f_co

    if usar_tablas:
        a_co2 = np.maximum(p["CO2"], 0.0) / float(gases.P_ESTANDAR_PA)
        a_co = np.maximum(p["CO"], 0.0) / float(gases.P_ESTANDAR_PA)
        K_boud = datos_reaccion["boudouard"]["K_eq"]
        fuerza_boud = a_co2 - a_co * a_co / K_boud
        k_boud = float(gases.A_BOUDOUARD_MOL_KG_S) * np.exp(
            -float(gases.EA_BOUDOUARD_J_MOL)
            / (float(termodinamica_ext.R) * T_termo)
        )
        r_boud = (
            float(P.k_boudouard)
            * escala_char
            * k_boud
            * (m_char_ref_g / 1000.0)
            * fuerza_boud
            / (1.0 + float(gases.K_INHIBICION_CO_BOUDOUARD) * a_co)
            / V_ref
        )
        factor_boud_dir = disp_char * _disponibilidad_vectorizada(
            n_ref["CO2"], float(modelo_multifase._N_GAS_REF)
        )
        factor_boud_inv = _disponibilidad_vectorizada(
            n_ref["CO"], float(modelo_multifase._N_GAS_REF)
        )
    else:
        r_boud = float(P.k_boudouard) * escala_char * _aplicar_escalar(
            gases.r_boudouard, T_termo, m_char_ref_g, p["CO2"], p["CO"]
        ) / V_ref
        factor_boud_dir = disp_char * _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["CO2"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        )
        factor_boud_inv = _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["CO"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        )
    r_boud *= np.where(r_boud >= 0.0, factor_boud_dir, factor_boud_inv)

    if usar_tablas:
        a_h2o = np.maximum(p["H2O"], 0.0) / float(gases.P_ESTANDAR_PA)
        a_h2 = np.maximum(p["H2"], 0.0) / float(gases.P_ESTANDAR_PA)
        fuerza_wg = a_h2o - a_co * a_h2 / datos_reaccion["water_gas"]["K_eq"]
        k_wg = float(gases.A_WATER_GAS_MOL_KG_S) * np.exp(
            -float(gases.EA_WATER_GAS_J_MOL)
            / (float(termodinamica_ext.R) * T_termo)
        )
        r_wg = (
            float(P.k_water_gas)
            * escala_char
            * k_wg
            * (m_char_ref_g / 1000.0)
            * fuerza_wg
            / (
                1.0
                + float(gases.K_INHIBICION_H2_WATER_GAS) * a_h2
                + float(gases.K_INHIBICION_CO_WATER_GAS) * a_co
            )
            / V_ref
        )
        factor_wg_dir = disp_char * _disponibilidad_vectorizada(
            n_ref["H2O"], float(modelo_multifase._N_GAS_REF)
        )
        factor_wg_inv = _disponibilidad_vectorizada(
            n_ref["CO"], float(modelo_multifase._N_GAS_REF)
        ) * _disponibilidad_vectorizada(
            n_ref["H2"], float(modelo_multifase._N_GAS_REF)
        )
    else:
        r_wg = float(P.k_water_gas) * escala_char * _aplicar_escalar(
            gases.r_water_gas,
            T_termo,
            m_char_ref_g,
            p["H2O"],
            p["H2"],
            p["CO"],
        ) / V_ref
        factor_wg_dir = disp_char * _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["H2O"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        )
        factor_wg_inv = _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["CO"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        ) * _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["H2"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        )
    r_wg *= np.where(r_wg >= 0.0, factor_wg_dir, factor_wg_inv)

    if usar_tablas:
        fuerza_shift = a_co * a_h2o - a_co2 * a_h2 / datos_reaccion["water_gas_shift"]["K_eq"]
        n_referencia = float(gases.P_ESTANDAR_PA) * V_ref / (
            float(termodinamica_ext.R) * T_termo
        )
        k_shift = float(gases.A_WGS_S) * np.exp(
            -float(gases.EA_WGS_J_MOL)
            / (float(termodinamica_ext.R) * T_termo)
        )
        r_shift = k_shift * n_referencia * fuerza_shift / V_ref
        factor_shift_dir = _disponibilidad_vectorizada(
            n_ref["CO"], float(modelo_multifase._N_GAS_REF)
        ) * _disponibilidad_vectorizada(
            n_ref["H2O"], float(modelo_multifase._N_GAS_REF)
        )
        factor_shift_inv = _disponibilidad_vectorizada(
            n_ref["CO2"], float(modelo_multifase._N_GAS_REF)
        ) * _disponibilidad_vectorizada(
            n_ref["H2"], float(modelo_multifase._N_GAS_REF)
        )
    else:
        r_shift = _aplicar_escalar(
            lambda temp, pco, ph2o, pco2, ph2: gases.r_wgs(
                temp, pco, ph2o, pco2, ph2, V_m3=V_ref
            ),
            T_termo,
            p["CO"],
            p["H2O"],
            p["CO2"],
            p["H2"],
        ) / V_ref
        factor_shift_dir = _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["CO"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        ) * _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["H2O"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        )
        factor_shift_inv = _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["CO2"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        ) * _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["H2"],
            np.full(forma, float(modelo_multifase._N_GAS_REF)),
        )
    r_shift *= np.where(r_shift >= 0.0, factor_shift_dir, factor_shift_inv)

    # Afinidades canal por canal, idénticas a las que consume rhs.
    if usar_tablas:
        f_disp_co = _disponibilidad_vectorizada(
            n_ref["CO"], float(modelo_multifase._N_CO_REF)
        )
        f_disp_h2 = _disponibilidad_vectorizada(
            n_ref["H2"], float(modelo_multifase._N_CO_REF)
        )
    else:
        f_disp_co = _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["CO"],
            np.full(forma, float(modelo_multifase._N_CO_REF)),
        )
        f_disp_h2 = _aplicar_escalar(
            modelo_multifase._disponibilidad,
            n_ref["H2"],
            np.full(forma, float(modelo_multifase._N_CO_REF)),
        )
    if usar_tablas and gases.factor_afinidad is modelo_multifase._FACTOR_AFINIDAD_ORIGINAL:
        afinidades = _afinidades_tabuladas(
            n_ref["CO"],
            n_ref["CO2"],
            n_ref["H2"],
            n_ref["H2O"],
            {nombre: datos_reaccion[nombre]["K_eq"] for nombre in _REACCIONES_TASAS},
        )
    else:
        afinidades = _afinidades_v3(
            T_termo, n_ref["CO"], n_ref["CO2"], n_ref["H2"], n_ref["H2O"]
        )

    pares = {
        "H": ("Fe2O3/Fe3O4", "Fe2O3", "Fe2O3->Fe3O4", float(P.k_hematita)),
        "M": ("Fe3O4/FeO", "Fe3O4", "Fe3O4->FeO", float(P.k_magnetita)),
        "W": ("FeO/Fe", "FeO", "FeO->Fe", float(P.k_wustita)),
    }
    reducciones: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for etiqueta, (par, fase, clave, escala) in pares.items():
        A, Ea = modelo_multifase.CINETICA_RED[clave][:2]
        if usar_tablas:
            base = _arrhenius_vectorizado(float(A) * escala, float(Ea), temperatura) * s[fase]
        else:
            base = _aplicar_escalar(
                modelo_multifase.arrhenius,
                np.full(forma, float(A) * escala),
                np.full(forma, float(Ea)),
                temperatura,
            ) * s[fase]
        aff_co = afinidades[par]["CO"]
        aff_h2 = afinidades[par]["H2"]
        reducciones[etiqueta] = (base * aff_co * f_disp_co, base * aff_h2 * f_disp_h2)

    inicial_mag = solido.get("_Fe3O4_inicial", None)
    if inicial_mag is None:
        inicial_mag_arr = np.full(forma, float(modelo_multifase.CI["n_Fe3O4_0"]) / V_ref)
    else:
        inicial_mag_arr = _como_array(inicial_mag, forma, "solido['_Fe3O4_inicial']")
    conv_mag = np.clip(
        1.0 - s["Fe3O4"] / np.maximum(inicial_mag_arr, 1.0e-300), 0.0, 1.0
    )
    f_acc = np.asarray(
        grano.fraccion_ilmenita_accesible(
            conv_mag, espesor_lama_um=float(P.espesor_lama_um)
        ),
        dtype=float,
    )
    A_i, Ea_i = modelo_multifase._cinetica_ilmenita()
    if usar_tablas:
        base_i = _arrhenius_vectorizado(
            float(A_i) * float(P.k_ilmenita), float(Ea_i), temperatura
        ) * s["FeTiO3"] * f_acc
    else:
        base_i = _aplicar_escalar(
            modelo_multifase.arrhenius,
            np.full(forma, float(A_i) * float(P.k_ilmenita)),
            np.full(forma, float(Ea_i)),
            temperatura,
        ) * s["FeTiO3"] * f_acc
    aff_i_co = afinidades["FeTiO3/Fe"]["CO"]
    aff_i_h2 = afinidades["FeTiO3/Fe"]["H2"]
    reducciones["I"] = (base_i * aff_i_co * f_disp_co, base_i * aff_i_h2 * f_disp_h2)

    if usar_tablas:
        k_fay = _arrhenius_vectorizado(1.0e3, 250.0, temperatura)
        r_fay = k_fay * _minimo_suave_vectorizado(s["FeO"], 2.0 * s["SiO2"])
    else:
        k_fay = _aplicar_escalar(
            modelo_multifase.arrhenius,
            np.full(forma, 1.0e3),
            np.full(forma, 250.0),
            temperatura,
        )
        r_fay = k_fay * _aplicar_escalar(
            modelo_multifase._minimo_suave_positivo, s["FeO"], 2.0 * s["SiO2"]
        )

    r_H_CO, r_H_H2 = reducciones["H"]
    r_M_CO, r_M_H2 = reducciones["M"]
    r_W_CO, r_W_H2 = reducciones["W"]
    r_I_CO, r_I_H2 = reducciones["I"]
    r_H, r_M = r_H_CO + r_H_H2, r_M_CO + r_M_H2
    r_W, r_I = r_W_CO + r_W_H2, r_I_CO + r_I_H2
    o_co = r_H_CO + r_M_CO + r_W_CO + r_I_CO
    o_h2 = r_H_H2 + r_M_H2 + r_W_H2 + r_I_H2

    R_solido = {fase: np.zeros(forma, dtype=float) for fase in FASES_SOLIDAS}
    R_solido.update(
        {
            "H2O_liq": -r_dry,
            "volatil": -r_vol,
            "C": -(r_boud + r_wg + r_comb),
            "Fe2O3": -3.0 * r_H,
            "Fe3O4": 2.0 * r_H - r_M,
            "FeO": 3.0 * r_M - r_W - 2.0 * r_fay,
            "Fe": r_W + r_I,
            "FeTiO3": -r_I,
            "TiO2": r_I,
            "SiO2": -r_fay,
            "Fe2SiO4": r_fay,
        }
    )

    gen_vol = {esp: r_vol * float(_VOLATILES_MOL_POR_G.get(esp, 0.0)) for esp in ESPECIES_GAS}
    R_gas = {
        "CO": gen_vol["CO"] + 2.0 * r_boud + r_wg + f_co * r_comb - o_co - r_shift,
        "CO2": gen_vol["CO2"] - r_boud + f_co2 * r_comb + o_co + r_shift,
        "H2": gen_vol["H2"] + r_wg + r_shift - o_h2,
        "H2O": gen_vol["H2O"] + r_dry - r_wg - r_shift + o_h2,
        "CH4": gen_vol["CH4"],
        "N2": gen_vol["N2"],
        "O2": -r_comb * (0.5 * f_co + f_co2),
    }

    # Fuente térmica con el signo del solver de transporte: positiva calienta.
    def dH(nombre: str) -> np.ndarray:
        if usar_tablas:
            return datos_reaccion[nombre]["delta_H"] * 1000.0
        return _aplicar_escalar(
            lambda temp: termodinamica_ext.delta_H_kj(nombre, temp) * 1000.0,
            T_termo,
        )

    if usar_tablas:
        dH_comb = (
            f_co
            * (h_combustion["CO"] - h_combustion["C"] - 0.5 * h_combustion["O2"])
            + f_co2
            * (h_combustion["CO2"] - h_combustion["C"] - h_combustion["O2"])
        ) * 1000.0
    else:
        dH_comb = _aplicar_escalar(
            lambda temp, yco, yco2: (
                yco
                * (
                    termodinamica_ext.h_kJ_mol("CO", temp)
                    - termodinamica_ext.h_kJ_mol("C", temp)
                    - 0.5 * termodinamica_ext.h_kJ_mol("O2", temp)
                )
                + yco2
                * (
                    termodinamica_ext.h_kJ_mol("CO2", temp)
                    - termodinamica_ext.h_kJ_mol("C", temp)
                    - termodinamica_ext.h_kJ_mol("O2", temp)
                )
            )
            * 1000.0,
            T_termo,
            f_co,
            f_co2,
        )
    Q_endotermico = (
        r_H_CO * dH("hematita_CO")
        + r_H_H2 * dH("hematita_H2")
        + r_M_CO * dH("magnetita_CO")
        + r_M_H2 * dH("magnetita_H2")
        + r_W_CO * dH("wustita_CO")
        + r_W_H2 * dH("wustita_H2")
        + r_I_CO * dH("ilmenita_CO")
        + r_I_H2 * dH("ilmenita_H2")
        + r_boud * dH("boudouard")
        + r_wg * dH("water_gas")
        + r_shift * dH("water_gas_shift")
        + r_comb * dH_comb
        + r_dry_g * 2260.0
        + r_vol * float(modelo_multifase._CALOR_DEVOL_J_G)
    )

    resultado: dict[str, Any] = {
        "R_gas": R_gas,
        "R_solido": R_solido,
        "Q_reaccion": -Q_endotermico,
    }
    if devolver_reacciones:
        resultado["_R_reacciones"] = {
            "secado": r_dry,
            "devolatilizacion": r_vol,
            "combustion": r_comb,
            "boudouard": r_boud,
            "water_gas": r_wg,
            "wgs": r_shift,
            "hematita_CO": r_H_CO,
            "hematita_H2": r_H_H2,
            "magnetita_CO": r_M_CO,
            "magnetita_H2": r_M_H2,
            "wustita_CO": r_W_CO,
            "wustita_H2": r_W_H2,
            "ilmenita_CO": r_I_CO,
            "ilmenita_H2": r_I_H2,
            "fayalita": r_fay,
        }
        resultado["_fracciones_combustion"] = (f_co, f_co2)
    return resultado


def tasas_locales(
    T: Any,
    c_gas: Mapping[str, Any],
    solido: Mapping[str, Any],
    eps: Any,
    cfg: Any = None,
    *,
    usar_tablas: bool | None = None,
) -> dict[str, Any]:
    """Calcula fuentes químicas locales en mol/m³/s y W/m³.

    La salida tiene exactamente las claves públicas ``R_gas``, ``R_solido`` y
    ``Q_reaccion``. Una ``Q_reaccion`` positiva calienta el campo, coherente
    con :func:`nucleo.transporte.paso_energia`.
    """

    return _calcular_tasas(
        T,
        c_gas,
        solido,
        eps,
        cfg,
        devolver_reacciones=False,
        usar_tablas=_usar_tablas(cfg, usar_tablas),
    )


# ---------------------------------------------------------------------------
# Integrador rígido vectorizado y conservativo
# ---------------------------------------------------------------------------
_NOMBRES_ESTADO = ESPECIES_GAS + FASES_SOLIDAS


def _estequiometria_reacciones(f_co: np.ndarray, f_co2: np.ndarray) -> dict[str, dict[str, Any]]:
    vol = _VOLATILES_MOL_POR_G
    return {
        "secado": {"H2O_liq": -1.0, "H2O": 1.0},
        "devolatilizacion": {"volatil": -1.0, **{e: float(vol.get(e, 0.0)) for e in ESPECIES_GAS}},
        "combustion": {"C": -1.0, "O2": -(0.5 * f_co + f_co2), "CO": f_co, "CO2": f_co2},
        "boudouard": {"C": -1.0, "CO2": -1.0, "CO": 2.0},
        "water_gas": {"C": -1.0, "H2O": -1.0, "CO": 1.0, "H2": 1.0},
        "wgs": {"CO": -1.0, "H2O": -1.0, "CO2": 1.0, "H2": 1.0},
        "hematita_CO": {"Fe2O3": -3.0, "Fe3O4": 2.0, "CO": -1.0, "CO2": 1.0},
        "hematita_H2": {"Fe2O3": -3.0, "Fe3O4": 2.0, "H2": -1.0, "H2O": 1.0},
        "magnetita_CO": {"Fe3O4": -1.0, "FeO": 3.0, "CO": -1.0, "CO2": 1.0},
        "magnetita_H2": {"Fe3O4": -1.0, "FeO": 3.0, "H2": -1.0, "H2O": 1.0},
        "wustita_CO": {"FeO": -1.0, "Fe": 1.0, "CO": -1.0, "CO2": 1.0},
        "wustita_H2": {"FeO": -1.0, "Fe": 1.0, "H2": -1.0, "H2O": 1.0},
        "ilmenita_CO": {"FeTiO3": -1.0, "Fe": 1.0, "TiO2": 1.0, "CO": -1.0, "CO2": 1.0},
        "ilmenita_H2": {"FeTiO3": -1.0, "Fe": 1.0, "TiO2": 1.0, "H2": -1.0, "H2O": 1.0},
        "fayalita": {"FeO": -2.0, "SiO2": -1.0, "Fe2SiO4": 1.0},
    }


def _paso_patankar(
    T: np.ndarray,
    gas: dict[str, np.ndarray],
    solido: dict[str, np.ndarray],
    eps: np.ndarray,
    h: float,
    cfg: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tasas = _calcular_tasas(
        T,
        gas,
        solido,
        eps,
        cfg,
        devolver_reacciones=True,
        usar_tablas=_usar_tablas(cfg),
    )
    reacciones = tasas["_R_reacciones"]
    f_co, f_co2 = tasas["_fracciones_combustion"]
    nu = _estequiometria_reacciones(f_co, f_co2)
    estado = {**gas, **{fase: solido[fase] for fase in FASES_SOLIDAS}}

    avances: dict[str, np.ndarray] = {}
    diminuto = 1.0e-300
    for nombre, tasa in reacciones.items():
        directa = tasa >= 0.0
        frecuencia = np.zeros_like(tasa)
        for especie, coef in nu[nombre].items():
            coef_arr = np.asarray(coef, dtype=float)
            consumo = np.where(directa, np.maximum(-coef_arr, 0.0), np.maximum(coef_arr, 0.0))
            frecuencia = np.maximum(
                frecuencia,
                consumo * np.abs(tasa) / np.maximum(estado[especie], diminuto),
            )
        frecuencia = np.where(np.isfinite(frecuencia), frecuencia, 1.0 / max(h, diminuto))
        avances[nombre] = h * tasa / (1.0 + h * frecuencia)

    delta = {especie: np.zeros_like(T) for especie in _NOMBRES_ESTADO}
    for nombre, avance in avances.items():
        for especie, coef in nu[nombre].items():
            delta[especie] += np.asarray(coef) * avance

    # Si varias reacciones compiten por el mismo reactivo, un único factor por
    # celda mantiene todos los avances estequiométricos y evita negatividad.
    theta = np.ones_like(T)
    for especie in _NOMBRES_ESTADO:
        negativo = delta[especie] < 0.0
        limite = np.where(
            negativo,
            0.999999999999 * estado[especie] / np.maximum(-delta[especie], diminuto),
            1.0,
        )
        theta = np.minimum(theta, np.clip(limite, 0.0, 1.0))

    nuevos = {
        especie: np.maximum(estado[especie] + theta * delta[especie], 0.0)
        for especie in _NOMBRES_ESTADO
    }
    return (
        {especie: nuevos[especie] for especie in ESPECIES_GAS},
        {fase: nuevos[fase] for fase in FASES_SOLIDAS},
    )


def _obtener_campo(objeto: Any, nombre: str) -> Any:
    if isinstance(objeto, Mapping):
        return objeto[nombre]
    return getattr(objeto, nombre)


def integrar_quimica_local(campos: Any, dt: float, cfg: Any = None) -> Any:
    """Integra la química con Patankar-Euler implícito sobre toda la malla.

    El método trata implícitamente cada frecuencia de destrucción
    ``r/n``: el avance es ``dt*r/(1+dt*r/n)``. Es L-estable para una
    destrucción lineal, positivo para pasos arbitrarios y no requiere un
    ``solve_ivp`` por celda. Las reacciones se actualizan por avances comunes,
    de modo que sus invariantes elementales se conservan por construcción.
    Sólo hay un bucle sobre subpasos y otro sobre las 15 reacciones; nunca un
    bucle sobre celdas.
    """

    paso = float(dt)
    if not np.isfinite(paso) or paso < 0.0:
        raise ValueError("dt debe ser finito y no negativo")
    if paso == 0.0:
        return copy.copy(campos)

    T = np.asarray(_obtener_campo(campos, "T"), dtype=float)
    forma = T.shape
    gas_original = _obtener_campo(campos, "c")
    solido_original = _obtener_campo(campos, "solido")
    eps = _como_array(_obtener_campo(campos, "eps"), forma, "campos.eps")
    gas = {
        especie: np.maximum(
            _como_array(gas_original.get(especie, 0.0), forma, f"campos.c[{especie!r}]").copy(),
            0.0,
        )
        for especie in ESPECIES_GAS
    }
    solido = {
        fase: np.maximum(_valor_solido(solido_original, fase, forma).copy(), 0.0)
        for fase in FASES_SOLIDAS
    }
    for metadato in ("_Fe3O4_inicial", "_volatil_inicial"):
        if metadato in solido_original:
            solido[metadato] = _como_array(
                solido_original[metadato], forma, f"campos.solido[{metadato!r}]"
            ).copy()
    if "_Fe3O4_inicial" not in solido:
        solido["_Fe3O4_inicial"] = solido["Fe3O4"].copy()
    if "_volatil_inicial" not in solido:
        solido["_volatil_inicial"] = solido["volatil"].copy()

    dt_max = float(_cfg(cfg, "dt_quimica_max_s", 0.05))
    if not np.isfinite(dt_max) or dt_max <= 0.0:
        raise ValueError("dt_quimica_max_s debe ser positivo")
    n_subpasos = max(1, int(np.ceil(paso / dt_max)))
    max_subpasos = int(_cfg(cfg, "max_subpasos_quimica", 256))
    if n_subpasos > max_subpasos:
        raise RuntimeError(
            f"La integración química requiere {n_subpasos} subpasos, por encima de "
            f"max_subpasos_quimica={max_subpasos}; reduzca dt o aumente el límite."
        )
    h = paso / n_subpasos
    metadatos = {k: v for k, v in solido.items() if k.startswith("_")}
    for _ in range(n_subpasos):
        gas, nuevas_fases = _paso_patankar(T, gas, solido, eps, h, cfg)
        solido = {**nuevas_fases, **metadatos}

    gas_salida = dict(gas_original)
    gas_salida.update(gas)
    solido_salida = dict(solido_original)
    solido_salida.update(solido)

    if isinstance(campos, Mapping):
        salida = dict(campos)
        salida["c"] = gas_salida
        salida["solido"] = solido_salida
        return salida
    salida = copy.copy(campos)
    salida.c = gas_salida
    salida.solido = solido_salida
    return salida


# ---------------------------------------------------------------------------
# Acoplamiento de masa y propiedades de mezcla
# ---------------------------------------------------------------------------
def fuente_de_masa_gaseosa(
    tasas_o_T: Any,
    c_gas: Mapping[str, Any] | None = None,
    solido: Mapping[str, Any] | None = None,
    eps: Any = None,
    cfg: Any = None,
) -> np.ndarray:
    """Devuelve la fuente neta de masa gaseosa, en kg/m³/s.

    Puede recibir directamente el resultado de :func:`tasas_locales` o los
    cinco argumentos de esa función. La suma incluye la masa transferida desde
    humedad, volátiles, char y oxígeno mineral; las reacciones exclusivamente
    gaseosas (por ejemplo WGS) se cancelan por su masa molecular.
    """

    if isinstance(tasas_o_T, Mapping) and "R_gas" in tasas_o_T:
        tasas = tasas_o_T
    else:
        if c_gas is None or solido is None or eps is None:
            raise TypeError("Faltan c_gas, solido o eps para calcular las tasas")
        tasas = tasas_locales(tasas_o_T, c_gas, solido, eps, cfg)
    R_gas = tasas["R_gas"]
    forma = np.asarray(next(iter(R_gas.values()))).shape
    fuente = np.zeros(forma, dtype=float)
    for especie, masa_molar in MASAS_MOLARES_GAS_KG_MOL.items():
        fuente += np.asarray(R_gas.get(especie, 0.0), dtype=float) * masa_molar
    return fuente


# Sutherland: mu=mu0(T/T0)^3/2(T0+S)/(T+S). Valores de referencia de NIST y
# Reid, Prausnitz & Poling, The Properties of Gases and Liquids, 4a ed.
_SUTHERLAND = {
    "CO": (1.65e-5, 300.0, 118.0),
    "CO2": (1.48e-5, 300.0, 240.0),
    "H2": (8.76e-6, 300.0, 72.0),
    "H2O": (9.00e-6, 300.0, 1110.0),
    "CH4": (1.10e-5, 300.0, 199.0),
    "N2": (1.663e-5, 300.0, 107.0),
    "O2": (2.07e-5, 300.0, 127.0),
}


def _mezcla_wilke(
    x: dict[str, np.ndarray], propiedad: dict[str, np.ndarray]
) -> np.ndarray:
    """Regla de Wilke; los únicos bucles son sobre las siete especies."""

    forma = next(iter(x.values())).shape
    mezcla = np.zeros(forma, dtype=float)
    for i in ESPECIES_GAS:
        denominador = np.zeros(forma, dtype=float)
        Mi = MASAS_MOLARES_GAS_KG_MOL[i]
        for j in ESPECIES_GAS:
            Mj = MASAS_MOLARES_GAS_KG_MOL[j]
            razon = np.divide(
                propiedad[i], propiedad[j], out=np.ones(forma), where=propiedad[j] > 0.0
            )
            phi = (1.0 + np.sqrt(razon) * (Mj / Mi) ** 0.25) ** 2
            phi /= np.sqrt(8.0 * (1.0 + Mi / Mj))
            denominador += x[j] * phi
        mezcla += np.divide(
            x[i] * propiedad[i],
            denominador,
            out=np.zeros(forma),
            where=denominador > 0.0,
        )
    return mezcla


def propiedades_gas(
    T: Any, c_gas: Mapping[str, Any], *, usar_tablas: bool = True
) -> dict[str, np.ndarray]:
    """Densidad, viscosidad y conductividad de la mezcla gaseosa.

    La densidad usa gas ideal, ``rho=sum(c_i M_i)``. Las viscosidades puras
    usan Sutherland y se mezclan con Wilke (Wilke, J. Chem. Phys. 18, 517,
    1950). La conductividad pura usa Eucken modificado,
    ``k=mu/M*(Cp+1.25R)``, y la mezcla la forma Wassiljewa--Mason--Saxena
    (Poling et al., *The Properties of Gases and Liquids*, 5a ed., cap. 10).
    """

    formas = [np.asarray(T).shape] + [np.asarray(v).shape for v in c_gas.values()]
    forma = np.broadcast_shapes(*formas)
    temperatura = _como_array(T, forma, "T")
    if np.any(temperatura <= 0.0):
        raise ValueError("T debe ser mayor que 0 K")
    c = {
        e: np.maximum(_como_array(c_gas.get(e, 0.0), forma, f"c_gas[{e!r}]"), 0.0)
        for e in ESPECIES_GAS
    }
    c_total = sum(c.values(), np.zeros(forma, dtype=float))
    x = {
        e: np.divide(c[e], c_total, out=np.zeros(forma), where=c_total > 0.0)
        for e in ESPECIES_GAS
    }
    rho = sum(
        (c[e] * MASAS_MOLARES_GAS_KG_MOL[e] for e in ESPECIES_GAS),
        np.zeros(forma, dtype=float),
    )
    masa_molar = sum(
        (x[e] * MASAS_MOLARES_GAS_KG_MOL[e] for e in ESPECIES_GAS),
        np.zeros(forma, dtype=float),
    )

    mu_puro: dict[str, np.ndarray] = {}
    k_puro: dict[str, np.ndarray] = {}
    cp_puro: dict[str, np.ndarray] = {}
    cp_tabulado = None
    if usar_tablas:
        cp_tabulado = obtener_tabla_termoquimica().datos_especies(
            temperatura, ESPECIES_GAS
        )["cp"]
    for indice, especie in enumerate(ESPECIES_GAS):
        mu0, T0, S = _SUTHERLAND[especie]
        mu = mu0 * (temperatura / T0) ** 1.5 * (T0 + S) / (temperatura + S)
        if cp_tabulado is not None:
            cp = cp_tabulado[indice]
        else:
            cp = _aplicar_escalar(
                lambda temp, esp=especie: termodinamica_ext.cp_J_molK(esp, temp),
                temperatura,
            )
        mu_puro[especie] = mu
        cp_puro[especie] = cp
        k_puro[especie] = (
            mu / MASAS_MOLARES_GAS_KG_MOL[especie]
            * (cp + 1.25 * termodinamica_ext.R)
        )
    mu = _mezcla_wilke(x, mu_puro)
    conductividad = _mezcla_wilke(x, k_puro)
    cp_molar = sum(
        (x[e] * cp_puro[e] for e in ESPECIES_GAS), np.zeros(forma, dtype=float)
    )
    presion = c_total * termodinamica_ext.R * temperatura
    return {
        "rho": rho,
        "mu": mu,
        "k": conductividad,
        "cp_molar": cp_molar,
        "masa_molar": masa_molar,
        "P": presion,
    }


__all__ = [
    "COMPOSICION_ELEMENTAL",
    "ESPECIES_GAS",
    "FASES_SOLIDAS",
    "MASAS_MOLARES_GAS_KG_MOL",
    "MASAS_MOLARES_SOLIDO_KG_MOL",
    "RUTA_SRC_V3",
    "TablaTermoquimica",
    "estado_inicial_celda",
    "fuente_de_masa_gaseosa",
    "integrar_quimica_local",
    "obtener_tabla_termoquimica",
    "propiedades_gas",
    "tasas_locales",
]
