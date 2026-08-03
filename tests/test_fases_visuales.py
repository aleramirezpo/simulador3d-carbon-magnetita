"""Pruebas de la paleta de fases: mineralogía coherente y mezcla por volumen."""

from __future__ import annotations

import numpy as np

from fisica import adaptador_v3
from fisica import fases_visuales as fv


def test_masas_molares_coinciden_con_las_del_modelo_validado():
    """La paleta no puede inventar mineralogía: debe usar la del modelo 0-D."""
    for campo, masa_kg_mol in adaptador_v3.MASAS_MOLARES_SOLIDO_KG_MOL.items():
        clave = fv.MAPA_CAMPOS_SOLIDOS[campo]
        if clave is None:
            continue
        assert fv.FASES[clave]["masa_molar_g_mol"] == float(f"{masa_kg_mol * 1000.0:.6g}"), campo


def test_el_mapa_cubre_todas_las_fases_solidas_del_solucionador():
    assert set(fv.MAPA_CAMPOS_SOLIDOS) == set(adaptador_v3.MASAS_MOLARES_SOLIDO_KG_MOL)
    destinos = {clave for clave in fv.MAPA_CAMPOS_SOLIDOS.values() if clave is not None}
    assert destinos.issubset(set(fv.FASES))
    # El aglomerado es un estado de la mezcla, no una fase del inventario.
    assert "aglomerado" not in destinos
    # Cada fase referida tiene densidad y masa molar para poder pesar volumen.
    for clave in destinos:
        assert fv.FASES[clave]["densidad_g_cm3"] > 0
        assert fv.FASES[clave]["masa_molar_g_mol"] > 0


def test_volumen_molar_reproduce_valores_conocidos():
    volumenes = fv.volumenes_molares_cm3_mol()
    # M/rho: hierro 55,845/7,874 = 7,09 cm3/mol; magnetita 231,531/5,17 = 44,79.
    assert abs(volumenes["Fe"] - 7.093) < 0.01
    assert abs(volumenes["Fe3O4"] - 44.79) < 0.01
    assert "aglomerado" not in volumenes


def test_fracciones_volumetricas_ponderan_por_volumen_y_no_por_moles():
    """Un mol de Fe ocupa 7,09 cm3 y uno de magnetita 44,79: no pesan igual."""
    fracciones = fv.fracciones_volumetricas({"Fe": 1.0, "Fe3O4": 1.0})
    assert abs(sum(fracciones.values()) - 1.0) < 1e-12
    assert fracciones["Fe3O4"] > fracciones["Fe"]
    esperado = 44.79 / (44.79 + 7.093)
    assert abs(fracciones["Fe3O4"] - esperado) < 1e-3


def test_fracciones_por_grupo_separan_carbon_de_mineral():
    solido = {"C": 20_000.0, "volatil": 5_000.0, "Fe3O4": 600.0, "Fe": 100.0, "H2O_liq": 3.0}
    carbonoso = fv.fracciones_volumetricas(solido, grupo="carbonoso")
    mineral = fv.fracciones_volumetricas(solido, grupo="mineral")
    assert set(carbonoso) <= {"carbon", "char", "cenizas"}
    assert set(mineral) <= {"Fe3O4", "Fe"}
    # La humedad de los poros no tiene aspecto propio y no entra en ningún grupo.
    assert "H2O_liq" not in carbonoso and "H2O_liq" not in mineral
    assert abs(sum(carbonoso.values()) - 1.0) < 1e-12
    assert abs(sum(mineral.values()) - 1.0) < 1e-12


def test_mezcla_de_colores_promedia_en_luz_lineal_y_no_oscurece():
    """Promediar sRGB directamente daría una mezcla más oscura que la real."""
    blanco_negro = fv.mezcla_color({"SiO2": 0.5, "char": 0.5})
    canal = fv.rgb_de_hex(blanco_negro)[0]
    promedio_srgb = (fv.rgb_de_hex(fv.FASES["SiO2"]["color"])[0]
                     + fv.rgb_de_hex(fv.FASES["char"]["color"])[0]) / 2.0
    assert canal > promedio_srgb
    # Y una mezcla de una sola fase devuelve exactamente esa fase.
    assert fv.mezcla_color({"Fe": 1.0}).upper() == fv.FASES["Fe"]["color"].upper()
    assert fv.mezcla_color({}) == fv.FASES["carbon"]["color"]


def test_paleta_web_lleva_todo_lo_que_el_cliente_necesita():
    paleta = fv.paleta_web()
    assert set(paleta["orden_leyenda"]) == set(fv.FASES)
    # Las iniciales van antes que las de producto: es el orden de aparición.
    posicion = {clave: i for i, clave in enumerate(paleta["orden_leyenda"])}
    assert max(posicion[c] for c in fv.FASES_INICIALES) < min(posicion[c] for c in fv.FASES_PRODUCTO)
    assert paleta["campos_solidos"] == fv.MAPA_CAMPOS_SOLIDOS
    for clave, fase in paleta["fases"].items():
        assert fase["color"].startswith("#") and len(fase["color"]) == 7
        assert fase["grupo"] in {"mineral", "carbonoso", "agregado"}
        if clave != "aglomerado":
            assert fase["volumen_molar_cm3_mol"] > 0
    assert "PREDICCIÓN" in paleta["nota"]
    assert "volumen" in paleta["ponderacion"]


def test_composicion_inicial_del_concentrado_da_un_mineral_oscuro():
    """70,7 % magnetita + 17,3 % ilmenita + 10,9 % hematita: casi negro.

    Sirve de control de que el coloreado no produce un lecho rojo brillante por
    la hematita, que es minoritaria.
    """
    # Fracciones másicas Rietveld convertidas a mol/m3 con base 1 kg/m3.
    masas = {"Fe3O4": 0.707, "FeTiO3": 0.173, "Fe2O3": 0.109, "SiO2": 0.011}
    solido = {
        campo: masa * 1000.0 / fv.FASES[campo]["masa_molar_g_mol"]
        for campo, masa in masas.items()
    }
    fracciones = fv.fracciones_volumetricas(solido, grupo="mineral")
    assert max(fracciones, key=fracciones.get) == "Fe3O4"
    canales = fv.rgb_de_hex(fv.mezcla_color(fracciones))
    assert max(canales) < 0.30, f"el concentrado debe verse oscuro, no {canales}"


def test_hierro_reducido_aclara_la_mezcla_mineral():
    """Ver aparecer el Fe metálico es el fenómeno que la paleta debe mostrar."""
    inicial = fv.mezcla_color(fv.fracciones_volumetricas({"Fe3O4": 100.0}, grupo="mineral"))
    reducido = fv.mezcla_color(fv.fracciones_volumetricas({"Fe": 300.0, "Fe3O4": 5.0}, grupo="mineral"))
    assert max(fv.rgb_de_hex(reducido)) > 2.0 * max(fv.rgb_de_hex(inicial))


def test_fracciones_sobre_una_instantanea_real_del_solucionador(tmp_path):
    """El cálculo debe funcionar tal cual sobre los campos NPZ del solucionador."""
    from nucleo.salida import cargar_instantanea
    from pathlib import Path

    directorio = Path(__file__).resolve().parents[1] / "resultados" / "simulacion_larga"
    rutas = sorted(directorio.glob("*.npz")) if directorio.is_dir() else []
    if not rutas:
        import pytest
        pytest.skip("no hay instantáneas reales en resultados/simulacion_larga")
    campos = cargar_instantanea(rutas[-1])
    lecho = np.asarray(campos["etiquetas"]) == 3
    assert lecho.any()
    solido = {
        nombre: float(np.sum(np.asarray(valores)[lecho]))
        for nombre, valores in campos["solido"].items()
    }
    fracciones = fv.fracciones_volumetricas(solido)
    assert abs(sum(fracciones.values()) - 1.0) < 1e-9
    # El lecho es mayoritariamente carbonoso: 0,75 g de carbón frente a 0,25 g
    # de concentrado, y además el carbón es mucho menos denso.
    assert fracciones["char"] > 0.5
    assert set(fracciones).issubset(set(fv.FASES))
