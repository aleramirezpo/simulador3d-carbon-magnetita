"""Exactitud, persistencia y rendimiento de la tabulacion termoquimica."""

from __future__ import annotations

import time

import numpy as np
import pytest

from fisica import adaptador_v3 as ad
from fisica.tablas_termo import TablaTermoquimica


T_900_C = 1173.15
V_REF = float(ad.modelo_multifase.CI["V_libre_m3"])


@pytest.fixture(scope="session")
def tabla() -> TablaTermoquimica:
    resultado = TablaTermoquimica(termodinamica=ad.termodinamica_ext)
    # Comparte la misma construccion costosa con los benchmarks del adaptador.
    ad._TABLA_TERMOQUIMICA = resultado
    return resultado


def test_1000_temperaturas_reproducen_fuente_a_1e_6(tabla: TablaTermoquimica):
    errores = tabla.error_de_tabulacion(1000, semilla=20260801)
    print(f"errores relativos maximos de tabulacion: {errores}")
    assert max(errores.values()) < 1.0e-6


def test_interpolacion_3d_equivale_a_consulta_celda_por_celda(
    tabla: TablaTermoquimica,
):
    rng = np.random.default_rng(731)
    T = rng.uniform(tabla.T_min, tabla.T_max, size=(5, 4, 3))
    especies = ("CO", "Fe2O3", "FeS")
    reacciones = ("boudouard", "hematita_CO", "FeS_oxidacion")
    lote_e = tabla.datos_especies(T, especies)
    lote_r = tabla.datos_reacciones(T, reacciones)
    metodos_e = {"cp": tabla.cp, "h": tabla.h, "s": tabla.s}
    metodos_r = {
        "delta_H": tabla.delta_H,
        "delta_G": tabla.delta_G,
        "K_eq": tabla.K_eq,
    }
    for magnitud, metodo in metodos_e.items():
        for i, nombre in enumerate(especies):
            esperado = np.array([metodo(nombre, float(t)) for t in T.flat]).reshape(T.shape)
            np.testing.assert_array_equal(lote_e[magnitud][i], esperado)
    for magnitud, metodo in metodos_r.items():
        for i, nombre in enumerate(reacciones):
            esperado = np.array([metodo(nombre, float(t)) for t in T.flat]).reshape(T.shape)
            np.testing.assert_array_equal(lote_r[magnitud][i], esperado)


def test_fuera_de_rango_avisa_y_recorta_sin_basura(tabla: TablaTermoquimica):
    temperaturas = np.array([tabla.T_min - 25.0, tabla.T_max + 25.0])
    with pytest.warns(RuntimeWarning, match="fuera de la tabla"):
        valores = tabla.K_eq("boudouard", temperaturas)
    extremos = tabla.K_eq("boudouard", np.array([tabla.T_min, tabla.T_max]))
    np.testing.assert_array_equal(valores, extremos)
    assert np.all(np.isfinite(valores)) and np.all(valores > 0.0)


def test_guardar_y_cargar_es_identico(tabla: TablaTermoquimica, tmp_path):
    ruta = tabla.guardar(tmp_path / "tabla_termo.npz")
    cargada = TablaTermoquimica.cargar(
        ruta, termodinamica=ad.termodinamica_ext
    )
    assert cargada.huella_origen == tabla.huella_origen
    assert np.array_equal(cargada.temperaturas, tabla.temperaturas)
    assert np.array_equal(cargada._datos_especies, tabla._datos_especies)
    assert np.array_equal(cargada._datos_reacciones, tabla._datos_reacciones)


def _campo_heterogeneo(n: int):
    forma = (n, n, n)
    z = np.linspace(0.0, 1.0, n**3).reshape(forma)
    T = 850.0 + 500.0 * z
    c_total = 101_325.0 / (ad.termodinamica_ext.R * T)
    gas = {especie: np.zeros(forma) for especie in ad.ESPECIES_GAS}
    gas["CO"] = c_total * (0.20 + 0.65 * z)
    gas["CO2"] = c_total - gas["CO"]
    solido = ad.estado_inicial_celda(np.ones(forma), V_REF / n**3)
    solido["_Fe3O4_inicial"] = 1.1 * solido["Fe3O4"]
    return T, gas, solido, np.full(forma, 0.54)


def test_rendimiento_heterogeneo_32_cubica_supera_200000_celdas_s(
    tabla: TablaTermoquimica,
):
    campos = _campo_heterogeneo(32)
    ad.tasas_locales(*campos, usar_tablas=True)  # calentamiento fuera de la medida
    medidas = []
    for _ in range(3):
        inicio = time.perf_counter()
        ad.tasas_locales(*campos, usar_tablas=True)
        medidas.append(32**3 / (time.perf_counter() - inicio))
    rendimiento = float(np.median(medidas))
    print(f"rendimiento tabulado heterogeneo 32^3: {rendimiento:.0f} celdas/s")
    assert rendimiento > 200_000.0


def test_consistencia_con_rhs_0d_con_tablas_activadas(tabla: TablaTermoquimica):
    m = ad.modelo_multifase
    parametros = m.Parametros(C_venteo=0.0)
    y = m.estado_inicial(parametros).copy()
    for nombre in ("T_s", "T_cru", "T_lid"):
        y[m.INDICES[nombre]] = T_900_C
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
        {"parametros_v3": parametros},
        usar_tablas=True,
    )
    rhs = np.asarray(m.rhs(0.0, y, parametros), dtype=float)
    for especie, indice in m.ESPECIES_GAS_IDX.items():
        np.testing.assert_allclose(
            tasas["R_gas"][especie].item() * V_REF,
            rhs[indice],
            rtol=1.0e-6,
            atol=1.0e-14,
        )
    for fase in ("Fe2O3", "Fe3O4", "FeO", "Fe", "FeTiO3", "TiO2", "Fe2SiO4", "FeS"):
        np.testing.assert_allclose(
            tasas["R_solido"][fase].item() * V_REF,
            rhs[m.INDICES[f"n_{fase}"]],
            rtol=1.0e-6,
            atol=1.0e-14,
        )

