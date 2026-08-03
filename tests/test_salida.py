"""Pruebas del contrato de instantáneas y exportación de posproceso."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from interfaz.datos_sinteticos import generar_instantanea_sintetica
from nucleo.salida import (
    VERSION_FORMATO,
    cargar_instantanea,
    cargar_serie,
    exportar_vtk,
    guardar_instantanea,
)


def test_ida_y_vuelta_npz_sin_perdida(tmp_path):
    original = generar_instantanea_sintetica(3, n_fotogramas=7, forma=(8, 8, 10))
    ruta = guardar_instantanea(original, tmp_path / "estado")
    cargado = cargar_instantanea(ruta)

    assert ruta.suffix == ".npz"
    assert cargado["t"] == original["t"]
    for nombre in ("x", "y", "z", "etiquetas", "u", "v", "w", "P", "T", "eps", "cohesion"):
        np.testing.assert_array_equal(cargado[nombre], original[nombre])
    assert cargado["c_especies"].keys() == original["c_especies"].keys()
    assert cargado["solido"].keys() == original["solido"].keys()
    for grupo in ("c_especies", "solido"):
        for nombre in original[grupo]:
            np.testing.assert_array_equal(cargado[grupo][nombre], original[grupo][nombre])


def test_metadatos_incluyen_unidades_version_y_validacion(tmp_path):
    ruta = guardar_instantanea(
        generar_instantanea_sintetica(0, forma=(6, 6, 8)),
        tmp_path / "metadatos.npz",
    )
    cargado = cargar_instantanea(ruta)
    meta = cargado["metadatos"]

    assert meta["version_formato"] == VERSION_FORMATO
    assert meta["unidades_geometria"] == "mm"
    assert meta["datos_sinteticos"] is True
    assert meta["campos"]
    for nombre, descripcion in meta["campos"].items():
        assert descripcion["unidades"], nombre
        assert descripcion["validado"] is False, nombre

    # El JSON de metadatos es texto, no un objeto que requiera pickle.
    with np.load(ruta, allow_pickle=False) as archivo:
        json.loads(str(archivo["_metadatos_json"].item()))


def test_cargar_serie_ordena_por_tiempo(tmp_path):
    for indice in (5, 0, 3):
        campos = generar_instantanea_sintetica(indice, n_fotogramas=6, forma=(6, 6, 8))
        guardar_instantanea(campos, tmp_path / f"captura_{indice}.npz")
    indice = cargar_serie(tmp_path)
    assert [elemento["t"] for elemento in indice] == sorted(elemento["t"] for elemento in indice)
    assert all(elemento["forma"] == (6, 6, 8) for elemento in indice)


def test_vtk_legacy_es_parseable(tmp_path):
    campos = generar_instantanea_sintetica(2, n_fotogramas=5, forma=(6, 6, 8))
    ruta = exportar_vtk(campos, tmp_path / "estado")
    texto = ruta.read_text(encoding="ascii")
    lineas = texto.splitlines()

    assert lineas[0] == "# vtk DataFile Version 3.0"
    assert lineas[2] == "ASCII"
    assert "DATASET RECTILINEAR_GRID" in lineas
    dimensiones = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", texto)
    assert dimensiones and tuple(map(int, dimensiones.groups())) == campos["P"].shape
    puntos = int(re.search(r"POINT_DATA\s+(\d+)", texto).group(1))
    assert puntos == int(np.prod(campos["P"].shape))
    assert texto.count("VECTORS velocidad_m_s double") == 1
    assert "SCALARS temperatura_K double 1" in texto
    assert "SCALARS concentracion_CO_mol_m3 double 1" in texto
    assert texto.rstrip().endswith(tuple("0123456789"))



def test_el_hinchamiento_es_opcional_y_va_y_vuelve_sin_perdida(tmp_path):
    """Campo nuevo del contrato: las series anteriores deben seguir cargando."""
    campos = generar_instantanea_sintetica(2, n_fotogramas=5, forma=(6, 6, 8))
    ruta = guardar_instantanea(campos, tmp_path / "sin_hinchamiento.npz")
    recuperado = cargar_instantanea(ruta)
    # Sin el campo, se rellena con unos: "no ha hinchado".
    assert recuperado["hinchamiento"].shape == recuperado["T"].shape
    assert np.allclose(recuperado["hinchamiento"], 1.0)

    campos["hinchamiento"] = np.full(recuperado["T"].shape, 1.65)
    ruta2 = guardar_instantanea(campos, tmp_path / "con_hinchamiento.npz")
    recuperado2 = cargar_instantanea(ruta2)
    assert np.allclose(recuperado2["hinchamiento"], 1.65)


def test_un_hinchamiento_menor_que_uno_se_rechaza(tmp_path):
    """Hinchar es expandir: un factor por debajo de 1 sería una contracción."""
    campos = generar_instantanea_sintetica(1, n_fotogramas=5, forma=(5, 5, 6))
    campos["hinchamiento"] = np.full((5, 5, 6), 0.8)
    with pytest.raises(ValueError):
        guardar_instantanea(campos, tmp_path / "invalido.npz")
