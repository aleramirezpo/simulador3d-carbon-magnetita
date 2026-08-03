"""Pruebas del perfil de revolución y del crisol con collar."""

from __future__ import annotations

import inspect
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nucleo.geometria import Crisol, MallaVoxel, fraccion_volumetrica
from nucleo.perfil import (
    PERFIL_ENSAYO,
    CrisolPerfilado,
    PerfilRevolucion,
    ajustar_collar_a_masa,
    exportar_obj,
)


def test_volumen_analitico_cilindro() -> None:
    radio = 4.25
    altura = 17.3
    perfil = PerfilRevolucion([(0.0, radio), (altura, radio)])
    exacto = math.pi * radio ** 2 * altura
    assert perfil.volumen_mm3() == pytest.approx(exacto, rel=1.0e-12)


def test_volumen_analitico_tronco_de_cono() -> None:
    r_base, r_boca, altura = 3.2, 7.1, 12.4
    perfil = PerfilRevolucion([(0.0, r_base), (altura, r_boca)])
    exacto = math.pi * altura * (
        r_base ** 2 + r_base * r_boca + r_boca ** 2
    ) / 3.0
    assert perfil.volumen_mm3() == pytest.approx(exacto, rel=1.0e-12)


def test_volumen_analitico_contra_voxelizado() -> None:
    crisol = CrisolPerfilado(PERFIL_ENSAYO, con_tapa=False)
    # El margen es necesario porque el collar (r=15,3 mm) excede la boca.
    malla = MallaVoxel(crisol, dx_mm=0.5, dz_mm=0.5, margen_mm=1.0)
    fraccion = fraccion_volumetrica(
        malla, PERFIL_ENSAYO.dentro, submuestreo=4)
    numerico = float(fraccion.sum()) * malla.volumen_celda_mm3
    exacto = PERFIL_ENSAYO.volumen_mm3()
    assert abs(numerico - exacto) / exacto < 0.01


def test_radio_continuo_y_reproduce_vertices() -> None:
    z = np.asarray([p[0] for p in PERFIL_ENSAYO.puntos])
    radios = np.asarray([p[1] for p in PERFIL_ENSAYO.puntos])
    np.testing.assert_allclose(PERFIL_ENSAYO.radio(z), radios, rtol=0.0, atol=0.0)

    epsilon = 1.0e-8
    for z_vertice, _ in PERFIL_ENSAYO.puntos[1:-1]:
        izquierdo = PERFIL_ENSAYO.radio(z_vertice - epsilon)
        derecho = PERFIL_ENSAYO.radio(z_vertice + epsilon)
        assert abs(izquierdo - derecho) < 1.0e-6


def test_collar_aumenta_masa_y_reduce_error() -> None:
    simple = Crisol()
    perfilado = CrisolPerfilado(PERFIL_ENSAYO)
    objetivo = 32.67
    assert perfilado.masa_calculada_g() > simple.masa_calculada_g()
    assert (abs(perfilado.masa_calculada_g() - objetivo)
            < abs(simple.masa_calculada_g() - objetivo))


def test_ajuste_collar_converge_en_rango_plausible() -> None:
    resultado = ajustar_collar_a_masa()
    assert resultado["convergio"] is True
    assert resultado["compatible_fotografia"] is True
    assert abs(float(resultado["residuo_g"])) <= 1.0e-8
    assert 18.0 <= float(resultado["altura_collar_mm"]) <= 24.0
    assert 1.0 <= float(resultado["espesor_collar_mm"]) <= 6.0
    assert "Compatible" in str(resultado["aviso"])


def test_obj_valido_coherente_y_sin_caras_degeneradas() -> None:
    # El directorio temporal se elimina dentro de la propia prueba.
    with tempfile.TemporaryDirectory() as directorio:
        ruta = Path(directorio) / "crisol.obj"
        exportar_obj(ruta, segmentos_angulares=32)
        lineas = ruta.read_text(encoding="utf-8").splitlines()

    vertices = np.asarray([
        [float(valor) for valor in linea.split()[1:4]]
        for linea in lineas if linea.startswith("v ")
    ])
    normales = np.asarray([
        [float(valor) for valor in linea.split()[1:4]]
        for linea in lineas if linea.startswith("vn ")
    ])
    caras = [linea.split()[1:] for linea in lineas if linea.startswith("f ")]

    assert len(vertices) > 0
    assert len(caras) > 0
    assert len(normales) == len(caras)
    for cara in caras:
        assert len(cara) == 3
        indices_vertices = [int(token.split("//")[0]) for token in cara]
        indices_normales = [int(token.split("//")[1]) for token in cara]
        assert len(set(indices_vertices)) == 3
        assert all(1 <= indice <= len(vertices) for indice in indices_vertices)
        assert all(1 <= indice <= len(normales) for indice in indices_normales)
        assert len(set(indices_normales)) == 1

        a, b, c = vertices[np.asarray(indices_vertices) - 1]
        cruz = np.cross(b - a, c - a)
        area_doble = np.linalg.norm(cruz)
        assert area_doble > 1.0e-10
        normal_geometrica = cruz / area_doble
        normal_obj = normales[indices_normales[0] - 1]
        assert np.dot(normal_geometrica, normal_obj) > 1.0 - 1.0e-9


def test_crisol_perfilado_conserva_api_publica() -> None:
    metodos_referencia = {
        nombre for nombre, miembro in inspect.getmembers(Crisol, inspect.isfunction)
        if not nombre.startswith("_")
    }
    metodos_perfil = {
        nombre for nombre, miembro in inspect.getmembers(
            CrisolPerfilado, inspect.isfunction)
        if not nombre.startswith("_")
    }
    assert metodos_referencia <= metodos_perfil

    crisol = CrisolPerfilado(PERFIL_ENSAYO)
    for propiedad in (
        "r_base_ext", "r_boca_ext", "r_base_int", "r_boca_int",
        "diam_base_mm", "diam_boca_mm", "altura_mm",
    ):
        assert hasattr(crisol, propiedad)
