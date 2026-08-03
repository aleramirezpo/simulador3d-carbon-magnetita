"""Pruebas del caso declarativo y de su recorrido hasta las instantáneas."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from fisica import adaptador_v3
from nucleo.caso import (
    ErrorCaso,
    acople,
    cargar_caso,
    crear_fuente_masa,
    crear_integrador_quimico,
    integrar_caso,
    instantanea_desde_estado,
)
from nucleo.geometria import GAS, LECHO, PARED_CRISOL, TAPA
from nucleo.perfil import PERFIL_ENSAYO
from nucleo.salida import cargar_instantanea, cargar_serie, guardar_instantanea


RUTA_CASO = Path(__file__).resolve().parents[1] / "casos" / "carbon_magnetita.yaml"


@pytest.fixture(scope="module")
def caso():
    return cargar_caso(RUTA_CASO, malla="gruesa")


@pytest.fixture(scope="module")
def corrida_termica_10s(caso):
    cfg = replace(
        caso.config_acople,
        dt_inicial=10.0,
        dt_max=10.0,
        con_momentum=False,
        con_quimica=False,
    )
    return integrar_caso(caso, 10.0, cfg=cfg)


@pytest.fixture(scope="module")
def corrida_venteo_corta(caso):
    cfg = replace(caso.config_acople, dt_inicial=0.05, dt_max=0.05)
    return integrar_caso(caso, 0.1, cfg=cfg)


def test_yaml_carga_y_produce_malla_y_estado_coherentes(caso):
    nx, ny, nz = caso.malla.forma
    estado = caso.estado_inicial
    assert caso.malla.n_celdas > 0
    assert caso.etiquetas.shape == caso.malla.forma
    assert estado.P.shape == caso.malla.forma
    assert estado.T.shape == caso.malla.forma
    assert estado.u.shape == (nx + 1, ny, nz)
    assert estado.v.shape == (nx, ny + 1, nz)
    assert estado.w.shape == (nx, ny, nz + 1)
    assert tuple(estado.c) == adaptador_v3.ESPECIES_GAS
    assert all(np.all(np.isfinite(campo)) for campo in estado.c.values())


def test_masa_inicial_suma_un_gramo(caso):
    assert caso.masa_solida_kg() == pytest.approx(1.0e-3, rel=0.0, abs=1.0e-15)


def test_geometria_usa_perfil_con_collar(caso):
    assert caso.geometria.perfil_exterior is PERFIL_ENSAYO
    assert caso.geometria.r_max_ext == pytest.approx(15.3)
    assert caso.capacidad_util_cm3 == pytest.approx(15.6481290179, rel=2.0e-10)
    assert 15.5 < caso.capacidad_util_cm3 < 15.8


def test_permeabilidad_finita_en_lecho_e_infinita_en_gas(caso):
    assert np.any(caso.etiquetas == LECHO)
    assert np.any(caso.etiquetas == GAS)
    assert np.all(np.isfinite(caso.propiedades.K[caso.etiquetas == LECHO]))
    assert np.all(caso.propiedades.K[caso.etiquetas == LECHO] > 0.0)
    assert np.all(np.isinf(caso.propiedades.K[caso.etiquetas == GAS]))


def test_caso_proporciona_propiedades_termicas_por_material(caso):
    p = caso.propiedades_termicas
    pared = np.isin(caso.etiquetas, (PARED_CRISOL, TAPA))
    lecho = caso.etiquetas == LECHO
    gas = caso.etiquetas == GAS
    assert np.all(p["k"][pared] == pytest.approx(16.0))
    assert np.all(p["rho"][pared] == pytest.approx(8400.0))
    assert np.all(p["cp"][pared] == pytest.approx(500.0))
    assert np.all(p["k"][lecho] == pytest.approx(1.2))
    assert np.all(p["rho"][gas] > 0.0)
    assert np.all(p["cp"][gas] == pytest.approx(1150.0))
    assert set(p["D_especies"]) == set(adaptador_v3.ESPECIES_GAS)
    assert all(np.all(np.asarray(D)[gas] > 0.0) for D in p["D_especies"].values())
    assert "Fuller" in p["fuentes"]["difusion"]


def test_corrida_corta_estable_sin_nan_y_conservativa(caso, corrida_venteo_corta):
    inicial = caso.estado_inicial
    masa_inicial = caso.masa_solida_kg(inicial) + caso.masa_gas_kg(inicial)
    final, historial = corrida_venteo_corta
    assert final.t == 0.1
    assert historial
    for campo in (final.u, final.v, final.w, final.P, final.T, final.eps, final.cohesion):
        assert np.all(np.isfinite(campo))
    assert all(np.all(np.isfinite(campo)) for campo in final.c.values())
    assert all(
        np.all(np.isfinite(campo))
        for nombre, campo in final.solido_fases.items()
        if not nombre.startswith("_")
    )
    masa_final = caso.masa_solida_kg(final) + caso.masa_gas_kg(final)
    assert abs(masa_final + final.masa_venteada_kg - masa_inicial) < 1.0e-7


def test_tras_10_s_el_conjunto_se_calienta_sin_retrasos_espurios(caso, corrida_termica_10s):
    """El crisol y su carga se calientan JUNTOS, no con 500 K de diferencia.

    Esta prueba exigía antes ``T_pared_media > T_centro_lecho`` en todo paso.
    Se cumplía por un defecto, no por física: el término advectivo de la
    energía usaba la forma en divergencia (metía ``-T*div(u)``, cientos de K/s
    de enfriamiento donde la devolatilización crea gas) y la difusión promediaba
    difusividades en vez de conductividades, lo que creaba energía en la
    interfaz gas/metal. El lecho quedaba clavado 500 K por debajo de la pared.
    Corregidos ambos, el conjunto —pared, gas y lecho— sube junto, y el orden
    entre dos medias que se diferencian en 10 K deja de estar garantizado.

    Lo que sí debe cumplirse, y es lo que ahora se comprueba, es que nada supere
    a la mufla y que no reaparezca un desacople grande entre pared y carga.
    """
    final, historial = corrida_termica_10s
    T0 = float(caso.datos["estado_inicial"]["temperatura_K"])
    assert historial[-1]["T_lecho_media_K"] > T0
    assert historial[-1]["T_pared_media_K"] > T0
    for paso in historial:
        T_mufla = float(paso["T_mufla_K"])
        assert paso["T_pared_media_K"] <= T_mufla + 1.0
        assert paso["T_lecho_media_K"] <= T_mufla + 1.0
        desacople = abs(paso["T_pared_media_K"] - paso["T_lecho_media_K"])
        assert desacople < 150.0, (
            f"pared y lecho difieren {desacople:.1f} K en t={paso.get('t', '?')}: "
            "es el síntoma del sumidero advectivo espurio que se corrigió")
    assert np.all(np.isfinite(final.T))


def test_venteo_compensa_la_fuente_y_contabiliza_la_masa(caso, corrida_venteo_corta):
    final, historial = corrida_venteo_corta
    assert np.any(caso.mascara_venteo)
    assert final.masa_venteada_kg > 0.0
    assert max(abs(paso["balance_fuente_venteo_kg_s"]) for paso in historial) < 1.0e-18
    assert max(float(paso.get("divergencia_residual", 0.0)) for paso in historial) < 1.0e-8
    assert max(abs(float(paso.get("incompatibilidad_divergencia", 0.0))) for paso in historial) < 1.0e-10


def test_instantaneas_se_cargan_con_campos_del_contrato(caso, tmp_path):
    ruta = guardar_instantanea(
        instantanea_desde_estado(caso, caso.estado_inicial),
        tmp_path / "instantanea_0000.npz",
    )
    serie = cargar_serie(tmp_path)
    cargada = cargar_instantanea(ruta)
    assert len(serie) == 1
    assert serie[0]["forma"] == caso.malla.forma
    assert cargada["metadatos"]["datos_sinteticos"] is False
    assert cargada["metadatos"]["fuente"].startswith("solucionador 3D")
    assert set((
        "t", "x", "y", "z", "etiquetas", "u", "v", "w", "P", "T",
        "c_especies", "eps", "solido", "cohesion", "metadatos",
    )).issubset(cargada)
    assert tuple(cargada["c_especies"]) == adaptador_v3.ESPECIES_GAS
    assert set(adaptador_v3.FASES_SOLIDAS).issubset(cargada["solido"])


def test_campo_obligatorio_ausente_da_error_especifico(tmp_path):
    datos = yaml.safe_load(RUTA_CASO.read_text(encoding="utf-8"))
    incompleto = copy.deepcopy(datos)
    del incompleto["carga"]["porosidad_inicial"]
    ruta = tmp_path / "incompleto.yaml"
    ruta.write_text(yaml.safe_dump(incompleto, sort_keys=False), encoding="utf-8")
    with pytest.raises(ErrorCaso, match=r"falta el campo obligatorio 'carga\.porosidad_inicial'"):
        cargar_caso(ruta)


def test_la_porosidad_sigue_al_inventario_solido(caso):
    """No puede quedarse congelada mientras el sólido pierde el 28 % de su masa.

    De la porosidad dependen la permeabilidad de Kozeny--Carman, la conductividad
    efectiva del lecho y la corrección de tortuosidad de las difusividades.
    """
    from nucleo.caso import actualizar_porosidad

    est = caso.estado_inicial
    lecho = caso.etiquetas == 3
    porosidad_inicial = float(caso.datos["carga"]["porosidad_inicial"])

    actualizar_porosidad(caso, est)          # fija la referencia
    actualizar_porosidad(caso, est)          # sin cambios: sigue igual
    assert float(np.mean(est.eps[lecho])) == pytest.approx(porosidad_inicial, abs=1e-6)

    K_inicial = np.array(caso.propiedades.K[lecho], copy=True)
    # Se va la mitad del volátil: el hueco que deja sube la porosidad.
    est.solido_fases["volatil"] = est.solido_fases["volatil"] * 0.5
    actualizar_porosidad(caso, est)
    porosidad = float(np.mean(est.eps[lecho]))
    assert porosidad > porosidad_inicial + 0.05, (
        f"la porosidad apenas se movió: {porosidad:.4f}")
    assert porosidad < 0.95
    # Y la permeabilidad debe haber subido con ella (Kozeny--Carman).
    assert float(np.mean(caso.propiedades.K[lecho])) > float(np.mean(K_inicial))


def test_la_porosidad_no_refactoriza_por_cambios_minusculos(caso):
    """El umbral existe para no invalidar la caché del ILU en cada paso."""
    from nucleo.caso import TOLERANCIA_POROSIDAD, actualizar_porosidad

    est = caso.estado_inicial
    lecho = caso.etiquetas == 3
    actualizar_porosidad(caso, est)
    actualizar_porosidad(caso, est)
    antes = np.array(est.eps[lecho], copy=True)

    # Un cambio muy por debajo del umbral no debe tocar el campo.
    est.solido_fases["volatil"] = est.solido_fases["volatil"] * 0.9995
    actualizar_porosidad(caso, est)
    assert np.array_equal(est.eps[lecho], antes)

    # Uno por encima, sí.
    est.solido_fases["volatil"] = est.solido_fases["volatil"] * 0.8
    actualizar_porosidad(caso, est)
    assert float(np.max(np.abs(est.eps[lecho] - antes))) > TOLERANCIA_POROSIDAD
