"""Verificación del puente 3-D contra la química validada de simulacion_v3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fisica import adaptador_v3 as ad


T_900_C = 1173.15
V_REF = float(ad.modelo_multifase.CI["V_libre_m3"])


def _gas_binario(especie: str, forma: tuple[int, ...], T: float = T_900_C):
    c_total = 101_325.0 / (ad.termodinamica_ext.R * T)
    gas = {e: np.zeros(forma) for e in ad.ESPECIES_GAS}
    gas[especie][...] = c_total
    return gas


def _masa_inicial_kg(solido: dict[str, np.ndarray], volumen: np.ndarray) -> float:
    return float(
        sum(
            np.sum(solido[fase] * volumen, dtype=np.float64) * masa_molar
            for fase, masa_molar in ad.MASAS_MOLARES_SOLIDO_KG_MOL.items()
        )
    )


def test_masa_inicial_repartida_suma_exactamente_un_gramo():
    fraccion = np.array(
        [[[1.0, 0.25], [0.0, 0.8]], [[0.5, 1.0], [0.1, 0.0]]], dtype=float
    )
    volumen = np.array(
        [[[1.0, 1.2], [0.7, 0.8]], [[1.1, 0.9], [1.3, 0.6]]], dtype=float
    ) * 1.0e-8
    solido = ad.estado_inicial_celda(fraccion, volumen)
    assert _masa_inicial_kg(solido, volumen) == pytest.approx(1.0e-3, abs=1.0e-18)


def test_mineralogia_por_celda_respeta_rietveld():
    forma = (3, 2, 2)
    volumen = np.full(forma, 2.0e-9)
    solido = ad.estado_inicial_celda(np.ones(forma), volumen)
    fases = ("Fe3O4", "FeTiO3", "Fe2O3", "SiO2")
    masas = np.array(
        [
            np.sum(solido[f] * volumen) * ad.MASAS_MOLARES_SOLIDO_KG_MOL[f]
            for f in fases
        ]
    )
    np.testing.assert_allclose(masas / masas.sum(), [0.707, 0.173, 0.109, 0.011], rtol=0, atol=5e-16)


def test_tasas_solidas_conservan_fe_ti_si():
    forma = (2, 2, 2)
    solido = ad.estado_inicial_celda(np.ones(forma), V_REF / np.prod(forma))
    # Activa accesibilidad de ilmenita, reducción de wüstita y fayalita.
    solido["_Fe3O4_inicial"] = 1.5 * solido["Fe3O4"]
    solido["FeO"] = np.full(forma, 10.0)
    gas = _gas_binario("CO", forma)
    tasas = ad.tasas_locales(np.full(forma, T_900_C), gas, solido, np.full(forma, 0.54))
    R = tasas["R_solido"]
    fe = (
        2.0 * R["Fe2O3"]
        + 3.0 * R["Fe3O4"]
        + R["FeO"]
        + R["Fe"]
        + R["FeTiO3"]
        + 2.0 * R["Fe2SiO4"]
        + R["FeS"]
    )
    ti = R["FeTiO3"] + R["TiO2"]
    si = R["SiO2"] + R["Fe2SiO4"]
    np.testing.assert_allclose(fe, 0.0, atol=2e-12)
    np.testing.assert_allclose(ti, 0.0, atol=2e-12)
    np.testing.assert_allclose(si, 0.0, atol=2e-12)


def test_co2_puro_a_900_c_no_reduce():
    forma = (2, 2, 2)
    solido = ad.estado_inicial_celda(np.ones(forma), V_REF / np.prod(forma))
    solido["_Fe3O4_inicial"] = 2.0 * solido["Fe3O4"]
    tasas = ad.tasas_locales(
        np.full(forma, T_900_C), _gas_binario("CO2", forma), solido, np.ones(forma)
    )
    R = tasas["R_solido"]
    # Productos que sólo pueden aparecer por reducción.
    for producto in ("FeO", "Fe", "TiO2"):
        assert np.all(R[producto] <= 0.0)


def test_co_puro_a_900_c_reduce_magnetita():
    forma = (1, 1, 1)
    solido = ad.estado_inicial_celda(np.ones(forma), V_REF)
    tasas = ad.tasas_locales(
        np.full(forma, T_900_C), _gas_binario("CO", forma), solido, np.ones(forma)
    )
    assert tasas["R_solido"]["FeO"].item() > 0.0
    assert tasas["R_solido"]["Fe3O4"].item() < 0.0


def test_consistencia_rhs_0d_relativa_1e_6():
    """Una celda reproduce todas las derivadas químicas del RHS de v3."""

    m = ad.modelo_multifase
    P = m.Parametros(C_venteo=0.0)
    y = m.estado_inicial(P).copy()
    y[m.INDICES["T_s"]] = T_900_C
    y[m.INDICES["T_cru"]] = T_900_C
    y[m.INDICES["T_lid"]] = T_900_C
    n_total = 101_325.0 * V_REF / (m.R_GAS * T_900_C)
    composicion = {"CO": 0.70, "CO2": 0.10, "H2": 0.10, "H2O": 0.10}
    for especie, indice in m.ESPECIES_GAS_IDX.items():
        y[indice] = composicion.get(especie, 0.0) * n_total

    forma = (1, 1, 1)
    solido = ad.estado_inicial_celda(np.ones(forma), V_REF)
    gas = {
        especie: np.full(forma, y[indice] / V_REF)
        for especie, indice in m.ESPECIES_GAS_IDX.items()
    }
    tasas = ad.tasas_locales(
        np.full(forma, T_900_C),
        gas,
        solido,
        np.ones(forma),
        {"parametros_v3": P},
    )
    rhs = np.asarray(m.rhs(0.0, y, P), dtype=float)

    for especie, indice in m.ESPECIES_GAS_IDX.items():
        np.testing.assert_allclose(
            tasas["R_gas"][especie].item() * V_REF, rhs[indice], rtol=1e-6, atol=1e-14
        )
    for fase in ("Fe2O3", "Fe3O4", "FeO", "Fe", "FeTiO3", "TiO2", "Fe2SiO4", "FeS"):
        np.testing.assert_allclose(
            tasas["R_solido"][fase].item() * V_REF,
            rhs[m.INDICES[f"n_{fase}"]],
            rtol=1e-6,
            atol=1e-14,
        )
    # SiO2 es implícita en v3: n_SiO2=n_SiO2,0-n_Fe2SiO4.
    np.testing.assert_allclose(
        tasas["R_solido"]["SiO2"].item() * V_REF,
        -rhs[m.INDICES["n_Fe2SiO4"]],
        rtol=1e-6,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        tasas["R_solido"]["H2O_liq"].item() * V_REF * m.MW["H2O"],
        rhs[m.INDICES["m_moist"]],
        rtol=1e-6,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        tasas["R_solido"]["volatil"].item() * V_REF,
        rhs[m.INDICES["m_vol"]],
        rtol=1e-6,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        tasas["R_solido"]["C"].item() * V_REF * m.MW["C"],
        rhs[m.INDICES["m_char"]],
        rtol=1e-6,
        atol=1e-14,
    )


def test_propiedades_aire_a_900_c():
    forma = (2, 1, 1)
    c_total = 101_325.0 / (ad.termodinamica_ext.R * T_900_C)
    gas = {e: np.zeros(forma) for e in ad.ESPECIES_GAS}
    gas["N2"][...] = 0.79 * c_total
    gas["O2"][...] = 0.21 * c_total
    props = ad.propiedades_gas(np.full(forma, T_900_C), gas)
    np.testing.assert_allclose(props["rho"], 0.2997, rtol=0.02)
    assert np.all(props["mu"] > 0.0)
    assert np.all(props["k"] > 0.0)


def test_vectorizacion_equivale_a_celda_por_celda():
    forma = (2, 2, 2)
    volumen = V_REF / np.prod(forma)
    solido = ad.estado_inicial_celda(np.ones(forma), volumen)
    solido["_Fe3O4_inicial"] = 1.1 * solido["Fe3O4"]
    T = np.linspace(900.0, 1173.15, np.prod(forma)).reshape(forma)
    c_total = 101_325.0 / (ad.termodinamica_ext.R * T)
    gas = {e: np.zeros(forma) for e in ad.ESPECIES_GAS}
    gas["CO"] = c_total * np.linspace(0.25, 0.85, np.prod(forma)).reshape(forma)
    gas["CO2"] = c_total - gas["CO"]
    eps = np.linspace(0.4, 1.0, np.prod(forma)).reshape(forma)
    lote = ad.tasas_locales(T, gas, solido, eps)

    for indice in np.ndindex(forma):
        una = (1, 1, 1)
        gas_i = {e: np.full(una, gas[e][indice]) for e in ad.ESPECIES_GAS}
        solido_i = {
            e: np.full(una, np.asarray(valor)[indice])
            for e, valor in solido.items()
        }
        celda = ad.tasas_locales(
            np.full(una, T[indice]), gas_i, solido_i, np.full(una, eps[indice])
        )
        for especie in ad.ESPECIES_GAS:
            np.testing.assert_allclose(lote["R_gas"][especie][indice], celda["R_gas"][especie].item(), rtol=2e-13, atol=1e-13)
        for fase in ad.FASES_SOLIDAS:
            np.testing.assert_allclose(lote["R_solido"][fase][indice], celda["R_solido"][fase].item(), rtol=2e-13, atol=1e-13)
        np.testing.assert_allclose(lote["Q_reaccion"][indice], celda["Q_reaccion"].item(), rtol=2e-13, atol=1e-9)


@dataclass
class _Campos:
    T: np.ndarray
    c: dict[str, np.ndarray]
    solido: dict[str, np.ndarray]
    eps: np.ndarray


def _inventario_elemental(campos: _Campos, elemento: str) -> np.ndarray:
    total = np.zeros_like(campos.T)
    for especie, valor in campos.c.items():
        total += ad.COMPOSICION_ELEMENTAL[especie].get(elemento, 0.0) * valor
    for fase, valor in campos.solido.items():
        if fase in ad.COMPOSICION_ELEMENTAL:
            total += ad.COMPOSICION_ELEMENTAL[fase].get(elemento, 0.0) * valor
    return total


def test_integrador_rigido_conserva_elementos_por_celda():
    forma = (2, 2, 2)
    solido = ad.estado_inicial_celda(np.ones(forma), V_REF / np.prod(forma))
    gas = _gas_binario("CO", forma)
    gas["CO"] *= 0.8
    gas["CO2"] += 0.1 * 101_325.0 / (ad.termodinamica_ext.R * T_900_C)
    gas["H2"] += 0.1 * 101_325.0 / (ad.termodinamica_ext.R * T_900_C)
    campos = _Campos(np.full(forma, T_900_C), gas, solido, np.full(forma, 0.54))
    elementos = ("Fe", "Ti", "Si", "C", "O")
    antes = {e: _inventario_elemental(campos, e) for e in elementos}
    despues = ad.integrar_quimica_local(
        campos, 0.02, {"dt_quimica_max_s": 0.01, "max_subpasos_quimica": 8}
    )
    for elemento in elementos:
        np.testing.assert_allclose(
            _inventario_elemental(despues, elemento), antes[elemento], rtol=3e-15, atol=2e-13
        )
    assert all(np.all(v >= 0.0) for v in despues.c.values())
    assert all(
        np.all(v >= 0.0)
        for nombre, v in despues.solido.items()
        if not nombre.startswith("_")
    )


def test_fuente_masica_es_suma_de_las_especies():
    forma = (1, 1, 1)
    solido = ad.estado_inicial_celda(np.ones(forma), V_REF)
    tasas = ad.tasas_locales(
        np.full(forma, T_900_C), _gas_binario("CO", forma), solido, np.ones(forma)
    )
    esperada = sum(
        tasas["R_gas"][e] * ad.MASAS_MOLARES_GAS_KG_MOL[e]
        for e in ad.ESPECIES_GAS
    )
    np.testing.assert_allclose(ad.fuente_de_masa_gaseosa(tasas), esperada, rtol=0, atol=0)
