"""Carga y valida casos declarativos del simulador 3-D.

El YAML es la única fuente de configuración del ensayo. Este módulo traduce
sus magnitudes a los objetos del núcleo y falla antes de construir la malla si
falta un dato, una unidad no coincide o la composición no es la que consume la
química validada de :mod:`fisica.adaptador_v3`.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from fisica import adaptador_v3

from .geometria import (
    GAS,
    LECHO,
    PARED_CRISOL,
    TAPA,
    VACIO,
    Lecho,
    MallaVoxel,
    fraccion_volumetrica,
)
from .perfil import CrisolPerfilado, PERFIL_ENSAYO
from . import momentum as _momentum_paquete
from . import transporte as _transporte_paquete

# ``acople.py`` mantiene importaciones históricas por nombre corto. Se registran
# los módulos del paquete antes de importarlo para que ``nucleo.caso`` funcione
# desde la raíz sin pedir al usuario que modifique PYTHONPATH.
sys.modules.setdefault("momentum", _momentum_paquete)
sys.modules.setdefault("transporte", _transporte_paquete)
from . import acople  # noqa: E402  (debe ocurrir después de los alias anteriores)

momentum = acople.mom


# Cambio mínimo de porosidad que justifica refactorizar el precondicionador del
# solucionador viscoso. Véase `actualizar_porosidad`.
TOLERANCIA_POROSIDAD = 5.0e-3


class ErrorCaso(ValueError):
    """Configuración declarativa incompleta o físicamente incoherente."""


@dataclass(frozen=True)
class CurvaMufla:
    """Curva experimental mediana de temperatura de la mufla."""

    t_s: np.ndarray
    T_K: np.ndarray
    fuente: Path
    emisividad: float

    def __call__(self, t_s: float) -> float:
        return float(np.interp(float(t_s), self.t_s, self.T_K))


@dataclass
class CasoSimulacion:
    """Objetos completamente construidos que necesita el bucle temporal."""

    nombre: str
    ruta: Path
    datos: dict[str, Any]
    geometria: CrisolPerfilado
    lecho: Lecho
    malla: MallaVoxel
    etiquetas: np.ndarray
    fraccion_lecho: np.ndarray
    propiedades: Any
    propiedades_termicas: dict[str, Any]
    estado_inicial: Any
    config_acople: Any
    config_quimica: dict[str, Any]
    curva_mufla: CurvaMufla
    preajuste_malla: str
    mascara_venteo: np.ndarray

    @property
    def crisol(self) -> CrisolPerfilado:
        """Alias explícito usado por consumidores geométricos."""
        return self.geometria

    @property
    def props(self) -> Any:
        """Alias breve compatible con la nomenclatura del núcleo."""
        return self.propiedades

    @property
    def capacidad_util_cm3(self) -> float:
        return self.geometria.volumen_interior_mm3() / 1000.0

    @property
    def volumen_celda_m3(self) -> float:
        return self.malla.volumen_celda_mm3 * 1.0e-9

    @property
    def t_final_s(self) -> float:
        return float(self.datos["tiempo"]["t_final_s"])

    @property
    def intervalo_guardado_s(self) -> float:
        return float(self.datos["resultados"]["intervalo_guardado_s"])

    def masa_solida_kg(self, estado: Any | None = None) -> float:
        est = self.estado_inicial if estado is None else estado
        solido = getattr(est, "solido_fases", getattr(est, "solido", {}))
        return float(sum(
            np.sum(np.asarray(solido[fase]), dtype=np.float64)
            * self.volumen_celda_m3 * masa_molar
            for fase, masa_molar in adaptador_v3.MASAS_MOLARES_SOLIDO_KG_MOL.items()
        ))

    def masa_gas_kg(self, estado: Any | None = None) -> float:
        est = self.estado_inicial if estado is None else estado
        return float(sum(
            np.sum(np.asarray(est.c[especie]), dtype=np.float64)
            * self.volumen_celda_m3 * masa_molar
            for especie, masa_molar in adaptador_v3.MASAS_MOLARES_GAS_KG_MOL.items()
        ))

    def inventario_elemental_mol(
        self, estado: Any | None = None,
    ) -> dict[str, float]:
        est = self.estado_inicial if estado is None else estado
        solido = getattr(est, "solido_fases", getattr(est, "solido", {}))
        volumen = self.volumen_celda_m3
        resultado = {elemento: 0.0 for elemento in ("C", "O", "Fe", "Ti", "Si")}
        for nombre, campo in est.c.items():
            composicion = adaptador_v3.COMPOSICION_ELEMENTAL.get(nombre, {})
            moles = float(np.sum(np.asarray(campo)) * volumen)
            for elemento in resultado:
                resultado[elemento] += composicion.get(elemento, 0.0) * moles
        for nombre, campo in solido.items():
            if nombre.startswith("_"):
                continue
            composicion = adaptador_v3.COMPOSICION_ELEMENTAL.get(nombre, {})
            moles = float(np.sum(np.asarray(campo)) * volumen)
            for elemento in resultado:
                resultado[elemento] += composicion.get(elemento, 0.0) * moles
        return resultado


def _mapa(valor: Any, ruta: str) -> Mapping[str, Any]:
    if not isinstance(valor, Mapping):
        raise ErrorCaso(f"el campo '{ruta}' debe ser un mapa YAML")
    return valor


def _campo(mapa: Mapping[str, Any], nombre: str, ruta: str = "") -> Any:
    completa = f"{ruta}.{nombre}" if ruta else nombre
    if nombre not in mapa:
        raise ErrorCaso(f"falta el campo obligatorio '{completa}'")
    return mapa[nombre]


def _numero(
    mapa: Mapping[str, Any], nombre: str, ruta: str, unidad: str,
    *, minimo: float | None = None, maximo: float | None = None,
) -> float:
    valor = _campo(mapa, nombre, ruta)
    completa = f"{ruta}.{nombre}"
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErrorCaso(
            f"unidad incoherente en '{completa}': se esperaba un número en {unidad}"
        )
    numero = float(valor)
    if not math.isfinite(numero):
        raise ErrorCaso(f"'{completa}' debe ser finito ({unidad})")
    if minimo is not None and numero < minimo:
        raise ErrorCaso(f"'{completa}' debe ser >= {minimo} {unidad}")
    if maximo is not None and numero > maximo:
        raise ErrorCaso(f"'{completa}' debe ser <= {maximo} {unidad}")
    return numero


def _texto(mapa: Mapping[str, Any], nombre: str, ruta: str) -> str:
    valor = _campo(mapa, nombre, ruta)
    if not isinstance(valor, str) or not valor.strip():
        raise ErrorCaso(f"'{ruta}.{nombre}' debe ser texto no vacío")
    return valor.strip()


def _validar_unidad(mapa: Mapping[str, Any], nombre: str, ruta: str, esperada: str) -> None:
    recibida = _texto(mapa, nombre, ruta)
    if recibida != esperada:
        raise ErrorCaso(
            f"unidad incoherente en '{ruta}.{nombre}': "
            f"se esperaba '{esperada}' y se recibió '{recibida}'"
        )


def _validar_estructura(datos: Mapping[str, Any]) -> None:
    for seccion in (
        "nombre", "descripcion", "geometria", "malla", "fisica", "especies",
        "condiciones_frontera", "propiedades_termicas", "carga", "estado_inicial",
        "medio", "quimica", "tiempo", "resultados",
    ):
        _campo(datos, seccion)

    geometria = _mapa(datos["geometria"], "geometria")
    for nombre in (
        "tipo", "perfil", "perfil_mm", "espesor_pared_mm", "espesor_fondo_mm",
        "espesor_tapa_mm", "con_tapa", "densidad_crisol_kg_m3",
    ):
        _campo(geometria, nombre, "geometria")
    if _texto(geometria, "tipo", "geometria") != "crisol_perfil":
        raise ErrorCaso("'geometria.tipo' debe ser 'crisol_perfil'")
    if _texto(geometria, "perfil", "geometria") != "nucleo.perfil.PERFIL_ENSAYO":
        raise ErrorCaso("'geometria.perfil' debe referir a nucleo.perfil.PERFIL_ENSAYO")
    try:
        puntos = np.asarray(geometria["perfil_mm"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ErrorCaso("'geometria.perfil_mm' debe contener pares [z_mm, r_mm]") from exc
    if puntos.shape != (5, 2) or not np.all(np.isfinite(puntos)):
        raise ErrorCaso("'geometria.perfil_mm' debe contener cinco pares finitos [z_mm, r_mm]")
    if not np.array_equal(puntos, np.asarray(PERFIL_ENSAYO.puntos)):
        raise ErrorCaso("'geometria.perfil_mm' no coincide con PERFIL_ENSAYO (collar obligatorio)")
    for nombre in ("espesor_pared_mm", "espesor_fondo_mm", "espesor_tapa_mm"):
        _numero(geometria, nombre, "geometria", "mm", minimo=np.finfo(float).tiny)
    _numero(geometria, "densidad_crisol_kg_m3", "geometria", "kg/m3", minimo=1.0)
    if not isinstance(geometria["con_tapa"], bool):
        raise ErrorCaso("'geometria.con_tapa' debe ser booleano")

    malla = _mapa(datos["malla"], "malla")
    _texto(malla, "preajuste_predeterminado", "malla")
    preajustes = _mapa(_campo(malla, "preajustes", "malla"), "malla.preajustes")
    for nombre in ("gruesa", "media", "fina"):
        ajuste = _mapa(_campo(preajustes, nombre, "malla.preajustes"), f"malla.preajustes.{nombre}")
        _numero(ajuste, "dx_mm", f"malla.preajustes.{nombre}", "mm", minimo=np.finfo(float).tiny)
        _numero(ajuste, "dz_mm", f"malla.preajustes.{nombre}", "mm", minimo=np.finfo(float).tiny)
    submuestreo = _campo(malla, "submuestreo_fracciones", "malla")
    if isinstance(submuestreo, bool) or not isinstance(submuestreo, int) or submuestreo < 1:
        raise ErrorCaso("'malla.submuestreo_fracciones' debe ser un entero >= 1")

    fisica = _campo(datos, "fisica")
    requeridos_fisica = {"momentum", "energia", "especies", "quimica", "cohesion"}
    if not isinstance(fisica, list) or set(fisica) != requeridos_fisica:
        raise ErrorCaso(f"'fisica' debe contener exactamente {sorted(requeridos_fisica)}")
    especies = _campo(datos, "especies")
    if not isinstance(especies, list) or tuple(especies) != adaptador_v3.ESPECIES_GAS:
        raise ErrorCaso(f"'especies' debe ser exactamente {list(adaptador_v3.ESPECIES_GAS)}")

    fronteras = _mapa(datos["condiciones_frontera"], "condiciones_frontera")
    mufla = _mapa(_campo(fronteras, "mufla", "condiciones_frontera"), "condiciones_frontera.mufla")
    tapa = _mapa(_campo(fronteras, "tapa", "condiciones_frontera"), "condiciones_frontera.tapa")
    if _texto(mufla, "tipo", "condiciones_frontera.mufla") != "radiacion":
        raise ErrorCaso("'condiciones_frontera.mufla.tipo' debe ser 'radiacion'")
    for nombre in ("curva", "hoja", "estadistico"):
        _texto(mufla, nombre, "condiciones_frontera.mufla")
    if mufla["estadistico"] != "mediana":
        raise ErrorCaso("'condiciones_frontera.mufla.estadistico' debe ser 'mediana'")
    _validar_unidad(mufla, "unidad_tiempo", "condiciones_frontera.mufla", "s")
    _validar_unidad(mufla, "unidad_temperatura", "condiciones_frontera.mufla", "degC")
    _numero(mufla, "emisividad", "condiciones_frontera.mufla", "1", minimo=0.0, maximo=1.0)
    if _texto(tapa, "tipo", "condiciones_frontera.tapa") != "venteo":
        raise ErrorCaso("'condiciones_frontera.tapa.tipo' debe ser 'venteo'")
    if _texto(tapa, "conductancia", "condiciones_frontera.tapa") != "calculada":
        raise ErrorCaso("'condiciones_frontera.tapa.conductancia' debe ser 'calculada'")
    _numero(tapa, "presion_exterior_Pa", "condiciones_frontera.tapa", "Pa", minimo=1.0)
    if _texto(tapa, "modelo", "condiciones_frontera.tapa") != "sumidero_masa_junta_tapa":
        raise ErrorCaso(
            "'condiciones_frontera.tapa.modelo' debe ser 'sumidero_masa_junta_tapa'"
        )
    _numero(tapa, "caudal_referencia_cm3_s", "condiciones_frontera.tapa",
            "cm3/s", minimo=0.0)
    _texto(tapa, "descripcion", "condiciones_frontera.tapa")

    carga = _mapa(datos["carga"], "carga")
    masa_carbon = _numero(carga, "masa_carbon_g", "carga", "g", minimo=0.0)
    masa_mineral = _numero(carga, "masa_concentrado_g", "carga", "g", minimo=0.0)
    if not math.isclose(masa_carbon, 0.75, abs_tol=1.0e-15) or not math.isclose(masa_mineral, 0.25, abs_tol=1.0e-15):
        raise ErrorCaso("la química adaptador_v3 exige exactamente 0,75 g de carbón y 0,25 g de concentrado")
    if not math.isclose(masa_carbon + masa_mineral, 1.0, abs_tol=1.0e-15):
        raise ErrorCaso("'carga' debe sumar exactamente 1,00 g")
    _numero(carga, "porosidad_inicial", "carga", "1", minimo=0.0, maximo=0.999)
    _numero(carga, "diametro_particula_m", "carga", "m", minimo=np.finfo(float).tiny)
    _numero(carga, "densidad_carbon_g_cm3", "carga", "g/cm3", minimo=np.finfo(float).tiny)
    _numero(carga, "densidad_concentrado_g_cm3", "carga", "g/cm3", minimo=np.finfo(float).tiny)

    carbon = _mapa(_campo(carga, "carbon", "carga"), "carga.carbon")
    if _texto(carbon, "base_analisis_proximo", "carga.carbon") != "humedad_residual":
        raise ErrorCaso("'carga.carbon.base_analisis_proximo' debe ser 'humedad_residual'")
    for nombre in ("humedad_residual_pct", "cenizas_pct", "materia_volatil_pct", "carbono_fijo_pct"):
        _numero(carbon, nombre, "carga.carbon", "%", minimo=0.0, maximo=100.0)

    rietveld = _mapa(_campo(carga, "mineralogia_rietveld", "carga"), "carga.mineralogia_rietveld")
    suma = 0.0
    for mineral, referencia in adaptador_v3.mineralogia.COMPOSICION_RIETVELD.items():
        fase = _mapa(_campo(rietveld, mineral, "carga.mineralogia_rietveld"), f"carga.mineralogia_rietveld.{mineral}")
        formula = _texto(fase, "formula", f"carga.mineralogia_rietveld.{mineral}")
        fraccion = _numero(fase, "fraccion_masica_pct", f"carga.mineralogia_rietveld.{mineral}", "%", minimo=0.0, maximo=100.0)
        densidad = _numero(fase, "densidad_g_cm3", f"carga.mineralogia_rietveld.{mineral}", "g/cm3", minimo=np.finfo(float).tiny)
        if formula != referencia["formula"] or not math.isclose(fraccion, referencia["w_pct"], abs_tol=1.0e-12) or not math.isclose(densidad, referencia["densidad_g_cm3"], abs_tol=1.0e-12):
            raise ErrorCaso(f"'carga.mineralogia_rietveld.{mineral}' no coincide con la caracterización Rietveld validada")
        suma += fraccion
    if not math.isclose(suma, 100.0, abs_tol=1.0e-12):
        raise ErrorCaso("las fracciones de 'carga.mineralogia_rietveld' deben sumar 100 %")

    inicial = _mapa(datos["estado_inicial"], "estado_inicial")
    _numero(inicial, "temperatura_K", "estado_inicial", "K", minimo=np.finfo(float).tiny)
    _numero(inicial, "presion_Pa", "estado_inicial", "Pa", minimo=1.0)
    gas = _mapa(_campo(inicial, "gas_fracciones_molares", "estado_inicial"), "estado_inicial.gas_fracciones_molares")
    fracciones = [_numero(gas, nombre, "estado_inicial.gas_fracciones_molares", "1", minimo=0.0, maximo=1.0) for nombre in gas]
    if not math.isclose(sum(fracciones), 1.0, abs_tol=1.0e-12):
        raise ErrorCaso("'estado_inicial.gas_fracciones_molares' debe sumar 1")
    if any(nombre not in adaptador_v3.ESPECIES_GAS for nombre in gas):
        raise ErrorCaso("'estado_inicial.gas_fracciones_molares' contiene una especie desconocida")

    medio = _mapa(datos["medio"], "medio")
    _numero(medio, "coeficiente_forchheimer", "medio", "1", minimo=0.0)
    _numero(medio, "expansion_termica_1_K", "medio", "1/K", minimo=0.0)
    _numero(medio, "viscosidad_referencia_Pa_s", "medio", "Pa s", minimo=np.finfo(float).tiny)
    _numero(medio, "permeabilidad_solido_m2", "medio", "m2", minimo=np.finfo(float).tiny)
    if not isinstance(_campo(medio, "activar_boyancia", "medio"), bool):
        raise ErrorCaso("'medio.activar_boyancia' debe ser booleano")
    _texto(medio, "justificacion_boyancia", "medio")

    quimica = _mapa(datos["quimica"], "quimica")
    if not isinstance(_campo(quimica, "usar_tablas", "quimica"), bool):
        raise ErrorCaso("'quimica.usar_tablas' debe ser booleano")
    _numero(quimica, "dt_quimica_max_s", "quimica", "s", minimo=np.finfo(float).tiny)
    max_subpasos = _campo(quimica, "max_subpasos_quimica", "quimica")
    if isinstance(max_subpasos, bool) or not isinstance(max_subpasos, int) or max_subpasos < 1:
        raise ErrorCaso("'quimica.max_subpasos_quimica' debe ser un entero >= 1")

    tiempo = _mapa(datos["tiempo"], "tiempo")
    for nombre in ("t_final_s", "dt_inicial_s", "dt_min_s", "dt_max_s"):
        _numero(tiempo, nombre, "tiempo", "s", minimo=np.finfo(float).tiny)
    if not isinstance(_campo(tiempo, "adaptativo", "tiempo"), bool):
        raise ErrorCaso("'tiempo.adaptativo' debe ser booleano")
    _numero(tiempo, "cfl", "tiempo", "1", minimo=np.finfo(float).tiny, maximo=1.0)
    if tiempo["dt_min_s"] > tiempo["dt_inicial_s"] or tiempo["dt_inicial_s"] > tiempo["dt_max_s"]:
        raise ErrorCaso("se requiere dt_min_s <= dt_inicial_s <= dt_max_s")

    resultados = _mapa(datos["resultados"], "resultados")
    _numero(resultados, "intervalo_guardado_s", "resultados", "s", minimo=np.finfo(float).tiny)
    _texto(resultados, "fuente", "resultados")
    if _campo(resultados, "datos_sinteticos", "resultados") is not False:
        raise ErrorCaso("'resultados.datos_sinteticos' debe ser false para una corrida real")
    _texto(resultados, "validacion_global", "resultados")

    termicas = _mapa(
        _campo(datos, "propiedades_termicas"), "propiedades_termicas"
    )
    pared = _mapa(_campo(termicas, "pared_nicr", "propiedades_termicas"),
                  "propiedades_termicas.pared_nicr")
    lecho_t = _mapa(_campo(termicas, "lecho", "propiedades_termicas"),
                    "propiedades_termicas.lecho")
    gas_t = _mapa(_campo(termicas, "gas", "propiedades_termicas"),
                  "propiedades_termicas.gas")
    difusion = _mapa(_campo(termicas, "difusion_especies", "propiedades_termicas"),
                     "propiedades_termicas.difusion_especies")
    for nombre in ("conductividad_W_m_K", "densidad_kg_m3", "calor_especifico_J_kg_K"):
        _numero(pared, nombre, "propiedades_termicas.pared_nicr", "SI", minimo=np.finfo(float).tiny)
    for nombre in (
        "conductividad_efectiva_inicial_W_m_K",
        "conductividad_efectiva_reducida_W_m_K",
        "calor_especifico_carbon_J_kg_K",
        "calor_especifico_concentrado_J_kg_K",
    ):
        _numero(lecho_t, nombre, "propiedades_termicas.lecho", "SI", minimo=np.finfo(float).tiny)
    for nombre in (
        "temperatura_referencia_K", "presion_referencia_Pa",
        "conductividad_W_m_K", "calor_especifico_J_kg_K",
    ):
        _numero(gas_t, nombre, "propiedades_termicas.gas", "SI", minimo=np.finfo(float).tiny)
    for seccion, ruta_fuente in (
        (pared, "propiedades_termicas.pared_nicr"),
        (lecho_t, "propiedades_termicas.lecho"),
        (gas_t, "propiedades_termicas.gas"),
        (difusion, "propiedades_termicas.difusion_especies"),
    ):
        _texto(seccion, "fuente", ruta_fuente)
    _numero(difusion, "temperatura_referencia_K",
            "propiedades_termicas.difusion_especies", "K", minimo=np.finfo(float).tiny)
    _numero(difusion, "presion_referencia_Pa",
            "propiedades_termicas.difusion_especies", "Pa", minimo=np.finfo(float).tiny)
    _numero(difusion, "exponente_temperatura",
            "propiedades_termicas.difusion_especies", "1", minimo=0.0)
    _numero(difusion, "exponente_tortuosidad_lecho",
            "propiedades_termicas.difusion_especies", "1", minimo=0.0)
    d_ref = _mapa(_campo(difusion, "D_referencia_m2_s",
                         "propiedades_termicas.difusion_especies"),
                  "propiedades_termicas.difusion_especies.D_referencia_m2_s")
    for especie in adaptador_v3.ESPECIES_GAS:
        _numero(d_ref, especie,
                "propiedades_termicas.difusion_especies.D_referencia_m2_s",
                "m2/s", minimo=np.finfo(float).tiny)


def _cargar_curva(ruta_yaml: Path, datos: Mapping[str, Any]) -> CurvaMufla:
    mufla = datos["condiciones_frontera"]["mufla"]
    origen = (ruta_yaml.parent / mufla["curva"]).resolve()
    if not origen.is_file():
        raise ErrorCaso(
            "no existe la curva experimental obligatoria "
            f"'condiciones_frontera.mufla.curva': {origen}"
        )
    try:
        modulo = importlib.import_module("curvas_temperatura")
        curva = modulo.curva_media(
            origen, dt_s=1.0, t_max=float(datos["tiempo"]["t_final_s"])
        )
        t_s = curva["time_s"].to_numpy(dtype=float)
        T_K = curva["T_median_C"].to_numpy(dtype=float) + 273.15
    except Exception as exc:
        raise ErrorCaso(f"no se pudo leer la curva experimental '{origen}': {exc}") from exc
    if t_s.size < 2 or np.any(~np.isfinite(t_s)) or np.any(~np.isfinite(T_K)):
        raise ErrorCaso(f"la curva experimental '{origen}' no contiene datos finitos suficientes")
    # Emisividad EFECTIVA: la del material por el factor de vista del crisol
    # dentro de la mufla. El factor va calibrado contra la curva de pérdida de
    # masa medida y agrupa lo que la geometría del montaje no resuelve (crisol
    # tapado, apoyado en un soporte, sin vista plena a los resistores). Si el
    # caso no lo declara, vale 1 y la emisividad efectiva es la del material.
    factor_vista = float(mufla.get("factor_vista", 1.0))
    if not 0.0 < factor_vista <= 1.0:
        raise ErrorCaso(
            "'condiciones_frontera.mufla.factor_vista' debe pertenecer a (0, 1]")
    return CurvaMufla(t_s, T_K, origen, float(mufla["emisividad"]) * factor_vista)


def cargar_caso(
    ruta: str | Path, preajuste_malla: str | None = None, *, malla: str | None = None,
) -> CasoSimulacion:
    """Valida ``ruta`` y construye geometría, propiedades y estado inicial."""
    if preajuste_malla is not None and malla is not None:
        raise ErrorCaso("indique sólo uno de 'preajuste_malla' o 'malla'")
    seleccion = malla if malla is not None else preajuste_malla
    origen = Path(ruta).resolve()
    if not origen.is_file():
        raise ErrorCaso(f"no existe el archivo de caso: {origen}")
    try:
        datos_cargados = yaml.safe_load(origen.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ErrorCaso(f"YAML inválido en '{origen}': {exc}") from exc
    datos = dict(_mapa(datos_cargados, "raíz"))
    _validar_estructura(datos)

    nombre_malla = seleccion or str(datos["malla"]["preajuste_predeterminado"])
    preajustes = datos["malla"]["preajustes"]
    if nombre_malla not in preajustes:
        raise ErrorCaso(
            f"preajuste de malla desconocido '{nombre_malla}'; "
            f"opciones: {', '.join(preajustes)}"
        )
    ajuste = preajustes[nombre_malla]

    g = datos["geometria"]
    crisol = CrisolPerfilado(
        PERFIL_ENSAYO,
        espesor_pared_mm=float(g["espesor_pared_mm"]),
        espesor_fondo_mm=float(g["espesor_fondo_mm"]),
        rho_kg_m3=float(g["densidad_crisol_kg_m3"]),
        espesor_tapa_mm=float(g["espesor_tapa_mm"]),
        con_tapa=bool(g["con_tapa"]),
    )
    carga = datos["carga"]
    lecho = Lecho(
        masa_carbon_g=float(carga["masa_carbon_g"]),
        masa_mineral_g=float(carga["masa_concentrado_g"]),
        rho_carbon_g_cm3=float(carga["densidad_carbon_g_cm3"]),
        rho_mineral_g_cm3=float(carga["densidad_concentrado_g_cm3"]),
        porosidad=float(carga["porosidad_inicial"]),
    )
    # MallaVoxel encierra base y boca; este margen incorpora además el collar.
    margen = crisol.r_max_ext - max(crisol.r_base_ext, crisol.r_boca_ext)
    malla_obj = MallaVoxel(
        crisol, dx_mm=float(ajuste["dx_mm"]), dz_mm=float(ajuste["dz_mm"]),
        margen_mm=max(0.0, margen),
    )
    etiquetas = malla_obj.etiquetar(crisol, lecho)
    fraccion = fraccion_volumetrica(
        malla_obj, lecho.dentro(crisol),
        submuestreo=int(datos["malla"]["submuestreo_fracciones"]),
    )

    forma = malla_obj.forma
    porosidad = float(carga["porosidad_inicial"])
    eps = np.full(forma, 1.0e-6, dtype=float)
    eps[etiquetas == GAS] = 1.0
    eps[etiquetas == LECHO] = porosidad
    K = np.full(forma, float(datos["medio"]["permeabilidad_solido_m2"]), dtype=float)
    K[etiquetas == GAS] = np.inf
    K[etiquetas == LECHO] = momentum.PropiedadesMedio.permeabilidad_kozeny_carman(
        float(carga["diametro_particula_m"]), np.full(np.count_nonzero(etiquetas == LECHO), porosidad)
    )

    inicial = datos["estado_inicial"]
    T0 = float(inicial["temperatura_K"])
    P0 = float(inicial["presion_Pa"])
    fracciones_gas = inicial["gas_fracciones_molares"]
    c_total = P0 / (adaptador_v3.termodinamica_ext.R * T0)
    interior = np.isin(etiquetas, (LECHO, GAS))
    c = {especie: np.zeros(forma, dtype=float) for especie in adaptador_v3.ESPECIES_GAS}
    for especie, fraccion_molar in fracciones_gas.items():
        # Las concentraciones del contrato son por volumen total de celda. En
        # el lecho se incorpora por ello la fracción de volumen poroso.
        c[especie][interior] = float(fraccion_molar) * c_total * eps[interior]
    solido = adaptador_v3.estado_inicial_celda(fraccion, malla_obj.volumen_celda_mm3 * 1.0e-9)
    nx, ny, nz = forma
    estado = acople.Estado(
        t=0.0,
        u=np.zeros((nx + 1, ny, nz), dtype=float),
        v=np.zeros((nx, ny + 1, nz), dtype=float),
        w=np.zeros((nx, ny, nz + 1), dtype=float),
        P=np.full(forma, P0, dtype=float),
        T=np.full(forma, T0, dtype=float),
        c=c,
        eps=eps,
        solido_fases=solido,
        cohesion=np.zeros(forma, dtype=float),
    )
    # El contrato público llama ``solido`` al inventario; ``acople.Estado``
    # conserva aún el nombre histórico ``solido_fases``.
    estado.solido = estado.solido_fases

    masa_molar_aire = sum(
        float(fracciones_gas.get(especie, 0.0)) * masa
        for especie, masa in adaptador_v3.MASAS_MOLARES_GAS_KG_MOL.items()
    )
    rho_aire = P0 * masa_molar_aire / (adaptador_v3.termodinamica_ext.R * T0)
    medio = datos["medio"]
    propiedades = momentum.PropiedadesMedio(
        rho=np.full(forma, rho_aire, dtype=float),
        mu=float(medio["viscosidad_referencia_Pa_s"]),
        eps=eps,
        K=K,
        C_F=float(medio["coeficiente_forchheimer"]),
        beta=float(medio["expansion_termica_1_K"]),
        T_ref=T0,
    )
    tiempo = datos["tiempo"]
    cfg_momentum = momentum.ConfigMomentum(
        con_adveccion=True, con_viscoso=True, con_darcy=True,
        con_forchheimer=True, con_boyancia=bool(medio["activar_boyancia"]),
        viscoso_implicito=True,
        solucionador_directo=True, con_proyeccion=True,
        paredes_en_el_borde=True,
    )
    cfg_acople = acople.ConfigAcople(
        dt_inicial=float(tiempo["dt_inicial_s"]),
        dt_min=float(tiempo["dt_min_s"]),
        dt_max=float(tiempo["dt_max_s"]),
        cfl=float(tiempo["cfl"]),
        con_momentum=True,
        con_transporte=True,
        con_quimica=True,
        cfg_momentum=cfg_momentum,
    )
    config_quimica = {
        "usar_tablas": bool(datos["quimica"]["usar_tablas"]),
        "dt_quimica_max_s": float(datos["quimica"]["dt_quimica_max_s"]),
        "max_subpasos_quimica": int(datos["quimica"]["max_subpasos_quimica"]),
    }
    # Parámetros del modelo de v3 que el caso quiera fijar. `adaptador_v3`
    # construye con ellos su objeto `Parametros`, así que basta con nombrarlos
    # igual. El 3-D venía usando los valores por omisión (todos 1,0) aunque v3
    # los trata como ajustables.
    for nombre in modelo_multifase_campos():
        if nombre in datos["quimica"]:
            config_quimica[nombre] = float(datos["quimica"][nombre])
    curva = _cargar_curva(origen, datos)
    # La compuerta de devolatilización se recoloca antes de construir el estado
    # inicial: la química la lee de un diccionario compartido con el 0-D.
    ajustar_devolatilizacion(datos)

    caso = CasoSimulacion(
        nombre=str(datos["nombre"]), ruta=origen, datos=datos,
        geometria=crisol, lecho=lecho, malla=malla_obj,
        etiquetas=etiquetas, fraccion_lecho=fraccion,
        propiedades=propiedades, propiedades_termicas={}, estado_inicial=estado,
        config_acople=cfg_acople, config_quimica=config_quimica,
        curva_mufla=curva, preajuste_malla=nombre_malla,
        mascara_venteo=_construir_mascara_venteo(malla_obj, etiquetas),
    )
    actualizar_propiedades_termicas(caso, estado)
    if not math.isclose(caso.masa_solida_kg(), 1.0e-3, rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError("el reparto discreto de la carga no conserva 1,00 g")
    return caso


def crear_integrador_quimico(caso: CasoSimulacion) -> Callable[[Any, float], Any]:
    """Adapta ``Estado.solido_fases`` al nombre ``solido`` del puente v3."""
    def integrar(estado: Any, dt: float) -> Any:
        estado.solido = estado.solido_fases
        salida = adaptador_v3.integrar_quimica_local(estado, dt, caso.config_quimica)
        salida.solido_fases = salida.solido
        return salida
    return integrar


def _construir_mascara_venteo(malla: MallaVoxel, etiquetas: np.ndarray) -> np.ndarray:
    """Corona de gas inmediatamente bajo la tapa, representación de la junta.

    La malla gruesa no resuelve una holgura submilimétrica. Se usa por ello la
    corona exterior de la última capa de gas conectada a la cavidad; así el
    sumidero pertenece al mismo componente fluido que la fuente del lecho.
    """
    X, Y, Z = malla.rejilla()
    gas = etiquetas == GAS
    if not np.any(gas):
        raise RuntimeError("el caso con venteo no contiene celdas de gas")
    z_superior = float(np.max(Z[gas]))
    capa = gas & np.isclose(Z, z_superior, rtol=0.0, atol=0.25 * malla.dz_mm)
    radios = np.hypot(X, Y)
    if np.any(capa):
        umbral = float(np.quantile(radios[capa], 0.60))
        corona = capa & (radios >= umbral)
    else:  # pragma: no cover - la existencia de gas garantiza una capa
        corona = capa
    if not np.any(corona):
        corona = capa
    return corona


def _grado_reduccion_local(caso: CasoSimulacion, estado: Any) -> np.ndarray:
    """Fracción local de oxígeno reducible que abandonó los óxidos de hierro."""
    actual = getattr(estado, "solido_fases", getattr(estado, "solido", {}))
    inicial = caso.estado_inicial.solido_fases

    def oxigeno_reducible(fases: Mapping[str, Any]) -> np.ndarray:
        cero = np.zeros(caso.malla.forma, dtype=float)
        return (
            3.0 * np.asarray(fases.get("Fe2O3", cero), dtype=float)
            + 4.0 * np.asarray(fases.get("Fe3O4", cero), dtype=float)
            + np.asarray(fases.get("FeO", cero), dtype=float)
            + np.asarray(fases.get("FeTiO3", cero), dtype=float)
        )

    o0 = oxigeno_reducible(inicial)
    o = oxigeno_reducible(actual)
    return np.clip(
        np.divide(o0 - o, o0, out=np.zeros_like(o0), where=o0 > 0.0),
        0.0, 1.0,
    )


def modelo_multifase_campos() -> tuple[str, ...]:
    """Nombres de los parámetros ajustables del modelo de `simulacion_v3`."""
    return tuple(
        adaptador_v3.modelo_multifase.Parametros.__dataclass_fields__
    )


def ajustar_devolatilizacion(datos: Mapping[str, Any]) -> float | None:
    """Recoloca la compuerta de devolatilización si el caso lo pide.

    `modelo_multifase` la evalúa como ``sigmoide(T-273,15, T_inicio_C, 40)``, con
    ``T_inicio_C = 200`` degC en `simulacion_v3`. Ese valor es el asomo de las
    primeras trazas de gas, pero al usarse como **centro** de una sigmoide de
    40 degC la compuerta está abierta al 98 % ya a 360 degC: el modelo suelta el
    volátil unos 100 degC por debajo de donde lo suelta un carbón bituminoso.

    En el 0-D de v3 el desfase no se notaba porque su historia térmica estaba
    calibrada y era mucho más lenta que la que resuelve el 3-D. Aquí, con el
    calentamiento resuelto, sí se nota: la devolatilización ocurría 30 s antes
    de lo medido.

    Se modifica el diccionario compartido, de modo que 0-D y 3-D siguen usando
    el mismo valor y su consistencia se mantiene. Devuelve el valor anterior.
    """
    quimica = datos.get("quimica") or {}
    solicitado = quimica.get("T_devolatilizacion_C")
    if solicitado is None:
        return None
    valor = float(solicitado)
    if not 100.0 <= valor <= 700.0:
        raise ErrorCaso(
            "'quimica.T_devolatilizacion_C' debe estar entre 100 y 700 degC")
    devol = adaptador_v3.modelo_multifase.lit.DEVOLATILIZACION
    previo = float(devol["T_inicio_C"])
    devol["T_inicio_C"] = valor
    return previo


def actualizar_propiedades_termicas(
    caso: CasoSimulacion, estado: Any | None = None,
) -> dict[str, Any]:
    """Actualiza ``k``, ``rho``, ``cp`` y ``D_i`` por celda, siempre en SI.

    La densidad del gas sigue ``rho=P*M/(R*T)``. Las difusividades usan la
    correlación de Fuller--Schettler--Giddings ``D ~ T**1.75/P`` y, dentro del
    lecho, la corrección de tortuosidad ``D_ef=eps**1.5 D``. La conductividad
    del lecho interpola los extremos de Kiamehr con el grado de reducción y se
    mezcla con el gas usando la porosidad local.
    """
    est = caso.estado_inicial if estado is None else estado
    cfg = caso.datos["propiedades_termicas"]
    pared_cfg = cfg["pared_nicr"]
    lecho_cfg = cfg["lecho"]
    gas_cfg = cfg["gas"]
    dif_cfg = cfg["difusion_especies"]
    forma = caso.malla.forma
    etiquetas = caso.etiquetas
    mascara_pared = np.isin(etiquetas, (PARED_CRISOL, TAPA))
    mascara_lecho = etiquetas == LECHO
    mascara_gas = np.isin(etiquetas, (GAS, VACIO))

    T = np.maximum(np.asarray(est.T, dtype=float), 1.0)
    P = np.maximum(np.asarray(est.P, dtype=float), 1.0)
    concentracion_total = np.zeros(forma, dtype=float)
    masa_gas_vol = np.zeros(forma, dtype=float)
    for especie, masa_molar in adaptador_v3.MASAS_MOLARES_GAS_KG_MOL.items():
        campo = np.maximum(np.asarray(est.c.get(especie, 0.0), dtype=float), 0.0)
        concentracion_total += campo
        masa_gas_vol += campo * masa_molar
    fracciones_iniciales = caso.datos["estado_inicial"]["gas_fracciones_molares"]
    masa_molar_aire = sum(
        float(fracciones_iniciales.get(especie, 0.0)) * masa
        for especie, masa in adaptador_v3.MASAS_MOLARES_GAS_KG_MOL.items()
    )
    masa_molar = np.divide(
        masa_gas_vol, concentracion_total,
        out=np.full(forma, masa_molar_aire, dtype=float),
        where=concentracion_total > 0.0,
    )
    rho_gas = P * masa_molar / (adaptador_v3.termodinamica_ext.R * T)

    k_gas = float(gas_cfg["conductividad_W_m_K"])
    cp_gas = float(gas_cfg["calor_especifico_J_kg_K"])
    k = np.full(forma, k_gas, dtype=float)
    rho = np.asarray(rho_gas, dtype=float).copy()
    cp = np.full(forma, cp_gas, dtype=float)

    k[mascara_pared] = float(pared_cfg["conductividad_W_m_K"])
    rho[mascara_pared] = float(pared_cfg["densidad_kg_m3"])
    cp[mascara_pared] = float(pared_cfg["calor_especifico_J_kg_K"])

    carga = caso.datos["carga"]
    m_carbon = float(carga["masa_carbon_g"])
    m_mineral = float(carga["masa_concentrado_g"])
    volumen_solido_cm3 = (
        m_carbon / float(carga["densidad_carbon_g_cm3"])
        + m_mineral / float(carga["densidad_concentrado_g_cm3"])
    )
    rho_grano = (m_carbon + m_mineral) / volumen_solido_cm3 * 1000.0
    eps0 = float(carga["porosidad_inicial"])
    rho_lecho = rho_grano * np.maximum(1.0 - np.asarray(est.eps), 1.0e-6)
    cp_lecho = (
        m_carbon * float(lecho_cfg["calor_especifico_carbon_J_kg_K"])
        + m_mineral * float(lecho_cfg["calor_especifico_concentrado_J_kg_K"])
    ) / (m_carbon + m_mineral)
    reduccion = _grado_reduccion_local(caso, est)
    k_objetivo = (
        float(lecho_cfg["conductividad_efectiva_inicial_W_m_K"])
        + reduccion * (
            float(lecho_cfg["conductividad_efectiva_reducida_W_m_K"])
            - float(lecho_cfg["conductividad_efectiva_inicial_W_m_K"])
        )
    )
    k_esqueleto = (k_objetivo - eps0 * k_gas) / (1.0 - eps0)
    k_lecho = (
        (1.0 - np.asarray(est.eps, dtype=float)) * k_esqueleto
        + np.asarray(est.eps, dtype=float) * k_gas
    )
    k[mascara_lecho] = k_lecho[mascara_lecho]
    rho[mascara_lecho] = rho_lecho[mascara_lecho]
    cp[mascara_lecho] = cp_lecho

    T_ref = float(dif_cfg["temperatura_referencia_K"])
    P_ref = float(dif_cfg["presion_referencia_Pa"])
    exponente_T = float(dif_cfg["exponente_temperatura"])
    exponente_eps = float(dif_cfg["exponente_tortuosidad_lecho"])
    factor = (T / T_ref) ** exponente_T * P_ref / P
    difusividades: dict[str, np.ndarray] = {}
    for especie in adaptador_v3.ESPECIES_GAS:
        D = float(dif_cfg["D_referencia_m2_s"][especie]) * factor
        D = np.asarray(D, dtype=float)
        D[mascara_pared] = 0.0
        D[mascara_lecho] *= np.maximum(est.eps[mascara_lecho], 0.0) ** exponente_eps
        difusividades[especie] = D

    propiedades = caso.propiedades_termicas
    propiedades.update({
        "k": k,
        "rho": rho,
        "cp": cp,
        # Capacidad volumétrica DEL GAS, no la efectiva de la celda. Sólo el gas
        # advecta: en el lecho, `rho*cp` es el del bulto (6,9e5 J/m3K) y el del
        # gas es 500 veces menor. Usar la efectiva para el término advectivo
        # equivale a que el gas arrastre la entalpía del sólido, y ése era el
        # origen del enfriamiento espurio del lecho (véase transporte.py).
        "rho_cp_fluido": np.asarray(rho_gas, dtype=float) * cp_gas,
        "D_especies": difusividades,
        "esquema": "tvd_superbee",
        "fuentes": {
            "pared_nicr": pared_cfg["fuente"],
            "lecho": lecho_cfg["fuente"],
            "gas": gas_cfg["fuente"],
            "difusion": dif_cfg["fuente"],
        },
    })
    # Se resuelve también el gas exterior a la pared. Esto permite aplicar la
    # condición radiativa disponible en las seis caras del bloque y obtener la
    # cadena mufla -> gas exterior -> pared -> lecho sin una fuente interna.
    propiedades["condiciones_frontera"] = {
        "T": {"todas": {
            "tipo": "radiacion",
            "T_mufla": caso.curva_mufla(float(est.t)),
            "emisividad": caso.curva_mufla.emisividad,
        }}
    }
    assert np.all(mascara_pared | mascara_lecho | mascara_gas)
    return propiedades


def crear_fuentes_transporte(
    caso: CasoSimulacion, estado: Any, dt: float = 0.0,
) -> dict[str, Any]:
    """Frontera radiativa evaluada en el paso actual.

    La química validada actualiza inventarios, pero su ``Q_reaccion`` explícito
    es rígido y no forma parte del cierre térmico solicitado aquí. El calor del
    caso entra exclusivamente desde la curva experimental de mufla.
    """
    t_bc = float(estado.t) + 0.5 * max(float(dt), 0.0)
    radiacion = {
        "tipo": "radiacion",
        "T_mufla": caso.curva_mufla(t_bc),
        "emisividad": caso.curva_mufla.emisividad,
    }
    return {
        "energia": {
            "condiciones_frontera": {"T": {"todas": radiacion}},
            "Q": calor_pirolisis(caso, estado, dt),
        },
        "especies": {},
    }


def calor_pirolisis(caso: CasoSimulacion, estado: Any, dt: float) -> np.ndarray:
    """Sumidero endotérmico de la devolatilización, en W/m3.

    La pirólisis **consume** energía y el modelo no lo contabilizaba: la carga
    se calentaba como si liberar los volátiles fuese gratis. Se calcula del
    descenso del inventario de volátil entre pasos, con un paso de retraso —la
    escala de la devolatilización es de segundos y el paso de milisegundos, así
    que el desfase es despreciable.

    El calor por kg de volátil lo declara el caso y está marcado CALIBRABLE: lo
    reportado para carbones bituminosos abarca 200-1400 kJ/kg de volátil.
    """
    forma = tuple(caso.malla.forma)
    volatil = np.asarray(estado.solido_fases.get("volatil", np.zeros(forma)), dtype=float)
    previo = getattr(caso, "_volatil_previo", None)
    caso._volatil_previo = np.array(volatil, copy=True)
    paso = float(dt)
    if previo is None or previo.shape != volatil.shape or not paso > 0.0:
        return np.zeros(forma)
    calor_kJ_kg = float(
        caso.datos["propiedades_termicas"]["lecho"].get(
            "calor_pirolisis_kJ_kg_volatil", 0.0)
    )
    if not calor_kJ_kg > 0.0:
        return np.zeros(forma)
    # `volatil` es pseudoespecie de 1 g por pseudomol: mol/m3 -> kg/m3 es 1e-3.
    liberado_kg_m3 = np.maximum(previo - volatil, 0.0) * 1.0e-3
    # Endotérmico: el signo es negativo en el balance de energía.
    return -liberado_kg_m3 * (calor_kJ_kg * 1.0e3) / paso


class _FuenteMasaConVenteo:
    """Fuente química compensada por un sumidero en la junta de la tapa."""

    def __init__(self, caso: CasoSimulacion):
        self.caso = caso
        self.ultimo_caudal_salida_kg_s = 0.0
        self.ultimo_caudal_volumetrico_cm3_s = 0.0
        self.ultimo_balance_kg_s = 0.0

    def __call__(self, estado: Any) -> np.ndarray:
        fuente = adaptador_v3.fuente_de_masa_gaseosa(
            estado.T, estado.c, estado.solido_fases, estado.eps,
            self.caso.config_quimica,
        )
        fluido = np.isin(self.caso.etiquetas, (LECHO, GAS))
        # Momentum sólo resuelve el componente fluido; fuentes químicas en una
        # celda cortada cuyo centro quedó etiquetado como pared no pertenecen al
        # RHS de ese Poisson y, por tanto, tampoco deben entrar en su compensación.
        compensada = np.where(fluido, np.asarray(fuente, dtype=float), 0.0)
        integral_celdas = float(np.sum(compensada, dtype=np.float64))
        n_venteo = int(np.count_nonzero(self.caso.mascara_venteo))
        compensada[self.caso.mascara_venteo] -= integral_celdas / n_venteo
        caudal = integral_celdas * self.caso.volumen_celda_m3
        self.ultimo_caudal_salida_kg_s = max(caudal, 0.0)
        rho_venteo = float(np.mean(self.caso.propiedades.rho[self.caso.mascara_venteo]))
        self.ultimo_caudal_volumetrico_cm3_s = (
            self.ultimo_caudal_salida_kg_s / max(rho_venteo, 1.0e-30) * 1.0e6
        )
        self.ultimo_balance_kg_s = float(
            np.sum(compensada, dtype=np.float64) * self.caso.volumen_celda_m3
        )
        return compensada


def crear_fuente_masa(caso: CasoSimulacion) -> Callable[[Any], np.ndarray]:
    """Fuente química con venteo conservativo en la junta de la tapa.

    El sumidero hace nula la integral discreta de la fuente que llega al Poisson.
    Es la condición de compatibilidad de un recinto con paredes Neumann y evita
    que el solucionador tenga que descartar artificialmente la media del RHS.
    """
    return _FuenteMasaConVenteo(caso)


def _retirar_inventario_venteado(
    caso: CasoSimulacion, estado: Any, masa_objetivo_kg: float,
) -> tuple[float, dict[str, float]]:
    """Retira del gas conectado la masa que cruzó la junta durante el paso.

    La celda de junta impone el sumidero a momentum. En una malla de 2 mm no es
    posible resolver el chorro submilimétrico ni su capa límite de composición;
    para el inventario se adopta mezcla perfecta del volumen gaseoso conectado.
    Así la salida permanece positiva y conserva exactamente masa y especies.
    """
    objetivo = max(float(masa_objetivo_kg), 0.0)
    interior = np.isin(caso.etiquetas, (LECHO, GAS))
    volumen = caso.volumen_celda_m3
    masa_disponible = 0.0
    for especie, masa_molar in adaptador_v3.MASAS_MOLARES_GAS_KG_MOL.items():
        masa_disponible += float(np.sum(estado.c[especie][interior])) * volumen * masa_molar
    retirada = min(objetivo, masa_disponible)
    if retirada <= 0.0 or masa_disponible <= 0.0:
        return 0.0, {especie: 0.0 for especie in adaptador_v3.ESPECIES_GAS}
    factor = max(0.0, 1.0 - retirada / masa_disponible)
    moles_retirados: dict[str, float] = {}
    for especie in adaptador_v3.ESPECIES_GAS:
        antes = float(np.sum(estado.c[especie][interior], dtype=np.float64)) * volumen
        estado.c[especie][interior] *= factor
        despues = float(np.sum(estado.c[especie][interior], dtype=np.float64)) * volumen
        moles_retirados[especie] = antes - despues
    retirada_real = sum(
        moles_retirados[especie] * adaptador_v3.MASAS_MOLARES_GAS_KG_MOL[especie]
        for especie in adaptador_v3.ESPECIES_GAS
    )
    return float(retirada_real), moles_retirados


def _volumen_solido_por_celda(estado: Any) -> np.ndarray:
    """Volumen que ocupa el sólido en cada celda, en m3 de sólido por m3.

    Suma ``c_i * Vm_i`` sobre las fases del inventario. Los volúmenes molares
    ``M/rho`` vienen de :mod:`fisica.fases_visuales`, que es donde vive la tabla
    de densidades minerales con su procedencia (catálogo de fases de
    ``simulacion_v3``); una prueba ata sus masas molares a las del modelo 0-D.
    """
    from fisica.fases_visuales import MAPA_CAMPOS_SOLIDOS, volumenes_molares_cm3_mol

    volumenes = volumenes_molares_cm3_mol()
    fases = estado.solido_fases
    total = None
    for campo, clave in MAPA_CAMPOS_SOLIDOS.items():
        if clave is None or clave not in volumenes or campo not in fases:
            continue
        aporte = np.asarray(fases[campo], dtype=float) * (volumenes[clave] * 1.0e-6)
        total = aporte if total is None else total + aporte
    return np.zeros_like(np.asarray(estado.T, dtype=float)) if total is None else total


def actualizar_porosidad(caso: CasoSimulacion, estado: Any) -> np.ndarray:
    """Actualiza la porosidad del lecho con el inventario sólido que queda.

    La porosidad estaba **congelada** en su valor inicial durante toda la
    corrida, mientras el sólido perdía el 28 % de su masa por devolatilización.
    Con ella quedaban congeladas la permeabilidad de Kozeny--Carman, la
    conductividad efectiva y la corrección de tortuosidad de las difusividades,
    que dependen de la porosidad.

    Se aplica el cambio **relativo** del volumen de sólido sobre la porosidad
    inicial calibrada, no el valor absoluto del inventario: la fracción sólida
    que dan los volúmenes molares (0,413) no coincide con la que declara el caso
    (0,46, de las densidades aparentes del YAML), y la calibración de partida
    debe respetarse. Con la razón, la discrepancia se cancela y en t=0 sale
    exactamente la porosidad del caso.

    El hinchamiento NO entra aquí. Sobre malla fija el lecho no puede crecer de
    volumen; lo que el inventario describe es el hueco que dejan los volátiles
    al marcharse. La expansión del aglomerado se representa aparte, como campo
    de diagnóstico y en la visualización.
    """
    lecho = caso.etiquetas == LECHO
    if not np.any(lecho):
        return np.asarray(estado.eps, dtype=float)
    volumen = _volumen_solido_por_celda(estado)
    inicial = getattr(caso, "_volumen_solido_inicial", None)
    if inicial is None:
        caso._volumen_solido_inicial = np.array(volumen, copy=True)
        return np.asarray(estado.eps, dtype=float)

    referencia = inicial[lecho]
    razon = np.divide(
        volumen[lecho], referencia,
        out=np.ones_like(referencia), where=referencia > 1.0e-12,
    )
    porosidad_inicial = float(caso.datos["carga"]["porosidad_inicial"])
    solido_inicial = 1.0 - porosidad_inicial
    # La porosidad no puede pasar de 1 ni bajar de la inicial: el sólido sólo se
    # va, no vuelve. El tope de 0,95 evita que una celda casi vacía degenere la
    # permeabilidad de Kozeny--Carman.
    nueva = np.clip(1.0 - solido_inicial * razon, porosidad_inicial, 0.95)

    # UMBRAL DE ACTUALIZACIÓN.
    #
    # El solucionador viscoso guarda en caché la factorización ILU indexada por
    # (dt, nu, darcy), y darcy depende de la permeabilidad. Reescribir K en cada
    # paso invalida la caché **siempre** y obliga a refactorizar: medido, la
    # corrida se volvió diez veces más lenta. La porosidad evoluciona en la
    # escala de la devolatilización, decenas de segundos, así que actualizarla
    # cuando ha cambiado más de un 0,5 % conserva la física y devuelve la caché.
    previa = getattr(caso, "_porosidad_aplicada", None)
    if previa is not None and previa.shape == nueva.shape:
        if float(np.max(np.abs(nueva - previa))) < TOLERANCIA_POROSIDAD:
            return np.asarray(estado.eps, dtype=float)
    caso._porosidad_aplicada = np.array(nueva, copy=True)
    estado.eps[lecho] = nueva
    # `caso.propiedades.eps` es el mismo arreglo que `estado.eps`, pero la
    # permeabilidad hay que recalcularla: K = d^2 eps^3 / (150 (1-eps)^2).
    caso.propiedades.K[lecho] = momentum.PropiedadesMedio.permeabilidad_kozeny_carman(
        float(caso.datos["carga"]["diametro_particula_m"]), nueva,
    )
    return np.asarray(estado.eps, dtype=float)


def _diagnosticos_termicos(caso: CasoSimulacion, estado: Any) -> dict[str, float]:
    pared = np.isin(caso.etiquetas, (PARED_CRISOL, TAPA))
    lecho = caso.etiquetas == LECHO
    X, Y, _ = caso.malla.rejilla()
    radio = np.hypot(X, Y)
    if np.any(lecho):
        r_min = float(np.min(radio[lecho]))
        centro = lecho & (radio <= r_min + 0.51 * caso.malla.dx_mm)
    else:  # pragma: no cover
        centro = lecho
    return {
        "T_pared_media_K": float(np.mean(estado.T[pared])),
        "T_lecho_media_K": float(np.mean(estado.T[lecho])),
        "T_centro_lecho_K": float(np.mean(estado.T[centro])),
        "T_mufla_K": caso.curva_mufla(float(estado.t)),
    }


def integrar_caso(
    caso: CasoSimulacion,
    t_final: float,
    *,
    estado: Any | None = None,
    cfg: Any | None = None,
    quimica: Callable[[Any, float], Any] | None = None,
    fuente_masa: Callable[[Any], np.ndarray] | None = None,
    al_guardar: Callable[[Any, dict[str, Any]], None] | None = None,
    intervalo_guardado: float | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Integra el caso pasando explícitamente térmica, radiación y venteo.

    Esta envoltura existe porque el contrato público de ``paso_global`` ya
    admite ``propiedades_termicas`` y ``fuentes_transporte``. Mantiene los
    solucionadores lineales entre pasos y añade al diagnóstico el inventario
    acumulado que salió por la junta.
    """
    configuracion = caso.config_acople if cfg is None else cfg
    est = (caso.estado_inicial if estado is None else estado).copia()
    est.solido = est.solido_fases
    quimica_local = crear_integrador_quimico(caso) if quimica is None else quimica
    fuente_venteo = crear_fuente_masa(caso) if fuente_masa is None else fuente_masa
    mascara_solida = np.isin(caso.etiquetas, (VACIO, PARED_CRISOL, TAPA))
    solucionador = momentum.SolucionadorPresion(
        est.P.shape,
        caso.malla.dx_mm * 1.0e-3,
        getattr(caso.malla, "dy_mm", caso.malla.dx_mm) * 1.0e-3,
        caso.malla.dz_mm * 1.0e-3,
    )
    solucionador_viscoso = momentum.SolucionadorViscoso(
        caso.malla.dx_mm * 1.0e-3,
        getattr(caso.malla, "dy_mm", caso.malla.dx_mm) * 1.0e-3,
        caso.malla.dz_mm * 1.0e-3,
    )
    dt = float(configuracion.dt_inicial)
    historial: list[dict[str, Any]] = []
    masa_venteada = float(getattr(estado, "masa_venteada_kg", 0.0)) if estado is not None else 0.0
    moles_venteados = {
        especie: 0.0 for especie in adaptador_v3.ESPECIES_GAS
    }
    intervalo = caso.intervalo_guardado_s if intervalo_guardado is None else float(intervalo_guardado)
    proximo_guardado = float(est.t) + intervalo
    n_paso = 0

    while est.t < float(t_final) - 1.0e-15:
        limites = acople.dt_estable(est, caso.propiedades, caso.malla, configuracion)
        restante = float(t_final) - float(est.t)
        dt = max(
            min(dt * configuracion.factor_crecimiento, configuracion.dt_max),
            configuracion.dt_min,
        )
        limite_global = float(limites["global"])
        if np.isfinite(limite_global) and limite_global < configuracion.dt_max:
            # Cuantizar el CFL en niveles binarios conserva siempre dt<=límite
            # y permite reutilizar el precondicionador viscoso durante muchos
            # pasos consecutivos. Con un dt distinto en cada iteración, aun por
            # cambios minúsculos del CFL, se reconstruía una factorización cara.
            razon = max(configuracion.dt_max / limite_global, 1.0)
            nivel = int(math.ceil(math.log2(razon)))
            dt = min(dt, configuracion.dt_max / (2.0 ** nivel))
        dt = min(dt, limite_global, restante)
        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError(f"paso temporal invalido: dt={dt!r}")

        propiedades_t = actualizar_propiedades_termicas(caso, est)
        if configuracion.con_quimica:
            fuentes_t = crear_fuentes_transporte(caso, est, dt)
        else:
            radiacion = {
                "tipo": "radiacion",
                "T_mufla": caso.curva_mufla(float(est.t) + 0.5 * dt),
                "emisividad": caso.curva_mufla.emisividad,
            }
            fuentes_t = {
                "energia": {"condiciones_frontera": {"T": {"todas": radiacion}}},
                "especies": {},
            }
        # SciPy BiCGSTAB informa breakdown (-10) cuando tanto la condición
        # inicial como el RHS de una especie agotada son exactamente cero. Una
        # semilla muy por debajo de precisión física evita ese caso algebraico;
        # su masa total es <1e-30 kg y no altera ningún balance reportable.
        fluido = np.isin(caso.etiquetas, (LECHO, GAS))
        for campo in est.c.values():
            campo[fluido] = np.maximum(campo[fluido], 1.0e-30)
        est, diag = acople.paso_global(
            est, caso.propiedades, caso.malla, dt, configuracion,
            quimica_local, fuente_venteo, solucionador, solucionador_viscoso,
            propiedades_termicas=propiedades_t,
            fuentes_transporte=fuentes_t,
            solido=mascara_solida,
        )
        caudal_salida = max(
            float(getattr(fuente_venteo, "ultimo_caudal_salida_kg_s", 0.0)), 0.0
        )
        retirada, retirada_especies = _retirar_inventario_venteado(
            caso, est, caudal_salida * dt,
        )
        masa_venteada += retirada
        for especie, moles in retirada_especies.items():
            moles_venteados[especie] += moles
        est.masa_venteada_kg = masa_venteada
        est.moles_venteados = dict(moles_venteados)
        est.solido = est.solido_fases
        # La porosidad sigue al inventario sólido: los volátiles que se van
        # dejan hueco, y de ella dependen permeabilidad, conductividad efectiva
        # y tortuosidad.
        actualizar_porosidad(caso, est)

        diag.update({
            "restringe": limites["restringe"],
            "limite_estabilidad": float(limites["global"]),
            "n_paso": n_paso,
            "caudal_venteo_kg_s": caudal_salida,
            "caudal_venteo_cm3_s": float(getattr(
                fuente_venteo, "ultimo_caudal_volumetrico_cm3_s", 0.0)),
            "balance_fuente_venteo_kg_s": float(getattr(
                fuente_venteo, "ultimo_balance_kg_s", 0.0)),
            "masa_venteada_paso_kg": retirada,
            "masa_venteada_acumulada_kg": masa_venteada,
            **_diagnosticos_termicos(caso, est),
        })
        historial.append(diag)
        n_paso += 1
        if al_guardar is not None and est.t + 16.0 * np.finfo(float).eps >= proximo_guardado:
            al_guardar(est, diag)
            proximo_guardado += intervalo

    if abs(est.t - float(t_final)) <= 8.0 * np.finfo(float).eps * max(1.0, abs(float(t_final))):
        est.t = float(t_final)
        if historial:
            historial[-1]["t"] = float(t_final)
    actualizar_propiedades_termicas(caso, est)
    return est, historial


def instantanea_desde_estado(caso: CasoSimulacion, estado: Any) -> dict[str, Any]:
    """Vista del estado que satisface exactamente el contrato de ``salida``."""
    solido = getattr(estado, "solido_fases", getattr(estado, "solido", {}))
    return {
        "t": float(estado.t),
        "x": caso.malla.x,
        "y": caso.malla.y,
        "z": caso.malla.z,
        "etiquetas": caso.etiquetas,
        "u": estado.u,
        "v": estado.v,
        "w": estado.w,
        "P": estado.P,
        "T": estado.T,
        "c_especies": estado.c,
        "eps": estado.eps,
        "solido": {fase: solido[fase] for fase in adaptador_v3.FASES_SOLIDAS},
        "cohesion": estado.cohesion,
        "metadatos": {
            "fuente": caso.datos["resultados"]["fuente"],
            "datos_sinteticos": False,
        },
    }


__all__ = [
    "CasoSimulacion", "CurvaMufla", "ErrorCaso", "acople", "momentum",
    "actualizar_propiedades_termicas", "cargar_caso", "crear_fuente_masa",
    "crear_fuentes_transporte", "crear_integrador_quimico", "integrar_caso",
    "instantanea_desde_estado",
]
