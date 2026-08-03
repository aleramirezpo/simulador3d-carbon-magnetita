"""Verificación por soluciones manufacturadas y órdenes observados."""

from __future__ import annotations

import numpy as np
import pytest

from nucleo.momentum import divergencia
from verificacion.convergencia import (
    estudio_convergencia_espacial,
    estudio_convergencia_temporal,
    indice_de_convergencia_de_malla,
)
from verificacion.mms import (
    VERIFICACION_SIMBOLICA,
    fuente_mms_momentum,
    solucion_manufacturada_escalar,
    solucion_manufacturada_velocidad,
)


@pytest.fixture(scope="module")
def espacial() -> dict:
    return estudio_convergencia_espacial(generar_archivos=False)


@pytest.fixture(scope="module")
def temporal() -> dict:
    return estudio_convergencia_temporal(generar_archivos=False)


def _derivada_cuarto_orden(funcion, punto, eje, h=1.0e-3):
    desplazados = []
    for factor in (-2.0, -1.0, 1.0, 2.0):
        q = list(punto)
        q[eje] += factor * h
        desplazados.append(funcion(*q))
    return (desplazados[0] - 8.0 * desplazados[1]
            + 8.0 * desplazados[2] - desplazados[3]) / (12.0 * h)


def _segunda_cuarto_orden(funcion, punto, eje, h=1.0e-3):
    valores = []
    for factor in (2.0, 1.0, 0.0, -1.0, -2.0):
        q = list(punto)
        q[eje] += factor * h
        valores.append(funcion(*q))
    return (-valores[0] + 16.0 * valores[1] - 30.0 * valores[2]
            + 16.0 * valores[3] - valores[4]) / (12.0 * h * h)


def test_velocidad_manufacturada_es_solenoidal_a_1e_12() -> None:
    rng = np.random.default_rng(2026)
    x, y, z = (rng.random((7, 5, 3)) for _ in range(3))
    sol = solucion_manufacturada_velocidad(x, y, z, 0.37)
    assert np.max(np.abs(sol["divergencia"])) < 1.0e-12
    assert VERIFICACION_SIMBOLICA["divergencia_velocidad"] is True

    # La propiedad crítica también debe sobrevivir al muestreo escalonado MAC.
    n, h = 17, 1.0 / 17.0
    c = (np.arange(n) + 0.5) * h
    f = np.arange(n + 1) * h
    u = solucion_manufacturada_velocidad(
        f[:, None, None], c[None, :, None], c[None, None, :], 0.37
    )["u"]
    v = solucion_manufacturada_velocidad(
        c[:, None, None], f[None, :, None], c[None, None, :], 0.37
    )["v"]
    w = solucion_manufacturada_velocidad(
        c[:, None, None], c[None, :, None], f[None, None, :], 0.37
    )["w"]
    assert np.max(np.abs(divergencia(u, v, w, h, h, h))) < 1.0e-12


def test_derivadas_analiticas_coinciden_con_diferencias_de_orden_alto() -> None:
    punto = (0.173, 0.287, 0.391, 0.23)
    escalar = solucion_manufacturada_escalar(*punto)
    funcion_escalar = lambda x, y, z, t: solucion_manufacturada_escalar(  # noqa: E731
        x, y, z, t
    )["valor"]
    for eje, nombre in enumerate(("dx", "dy", "dz", "dt")):
        fd = _derivada_cuarto_orden(funcion_escalar, punto, eje)
        assert abs(fd - escalar[nombre]) < 1.0e-8
    for eje, nombre in enumerate(("dxx", "dyy", "dzz")):
        fd = _segunda_cuarto_orden(funcion_escalar, punto, eje)
        assert abs(fd - escalar[nombre]) < 1.0e-8

    velocidad = solucion_manufacturada_velocidad(*punto)
    for componente, nombre in enumerate(("u", "v", "w")):
        funcion = lambda x, y, z, t, n=nombre: (  # noqa: E731
            solucion_manufacturada_velocidad(x, y, z, t)[n]
        )
        for eje in range(4):
            fd = _derivada_cuarto_orden(funcion, punto, eje)
            exacta = (velocidad["jacobiano"][componente][eje]
                      if eje < 3 else velocidad["dt"][componente])
            assert abs(fd - exacta) < 1.0e-8


def test_fuentes_fueron_verificadas_y_momentum_incluye_todos_los_terminos() -> None:
    assert VERIFICACION_SIMBOLICA["fuente_escalar"] is True
    terminos = fuente_mms_momentum(
        0.17, 0.29, 0.41, 0.23, devolver_terminos=True
    )
    esperados = {
        "transitorio", "adveccion", "gradiente_presion", "menos_viscoso",
        "darcy", "forchheimer", "menos_boyancia",
    }
    assert esperados <= set(terminos)
    assert all(np.any(np.abs(terminos[nombre]) > 0.0) for nombre in esperados)


def test_orden_transporte_escalar_es_dos(espacial: dict) -> None:
    resultado = espacial["casos"]["transporte_central"]
    for variable in ("T", "c"):
        for norma in ("L1", "L2", "Linf"):
            observado = resultado["ordenes"][variable][norma]
            assert observado == pytest.approx(2.0, abs=0.15)


def test_orden_momentum_sin_adveccion_es_al_menos_1_8(espacial: dict) -> None:
    resultado = espacial["casos"]["momentum_sin_adveccion"]
    for variable in ("u", "v", "w"):
        for norma in ("L1", "L2", "Linf"):
            assert resultado["ordenes"][variable][norma] >= 1.8


def test_upwind_de_momentum_degrada_a_primer_orden(espacial: dict) -> None:
    resultado = espacial["casos"]["momentum_upwind"]
    for variable in ("u", "v", "w"):
        for norma in ("L1", "L2", "Linf"):
            observado = resultado["ordenes"][variable][norma]
            assert observado == pytest.approx(1.0, abs=0.15)


def test_orden_temporal_momentum_es_uno(temporal: dict) -> None:
    resultado = temporal["resultado"]
    for variable in ("u", "v", "w"):
        for norma in ("L1", "L2", "Linf"):
            observado = resultado["ordenes"][variable][norma]
            assert observado == pytest.approx(1.0, abs=0.15)


def test_gci_es_positivo() -> None:
    gci = indice_de_convergencia_de_malla(1.12, 1.03, 1.007)
    assert gci["orden_aparente"] > 0.0
    assert gci["GCI_fino"] > 0.0
    assert gci["GCI_fino_porcentaje"] > 0.0


def test_error_decrece_monotonamente_con_fuente_mms(espacial: dict) -> None:
    for resultado in espacial["casos"].values():
        for variable in resultado["componentes"]:
            for norma in resultado["normas"]:
                assert resultado["monotono"][variable][norma], (
                    f"error no monótono: {variable} {norma} "
                    f"{resultado['errores'][variable][norma]}"
                )
