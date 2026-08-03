"""Pruebas del modelo predictivo de cohesion y crecimiento."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from fisica.cohesion import (
    CampoCohesion,
    MASA_CHAR_ESTIMADA_G,
    aglomerado,
    curva_de_crecimiento,
    diagnostico_mecanismos,
    evolucionar,
    puentes_solidos,
    tasa_sinterizacion,
)


T_900_C = 1173.15


def _activar_coquizacion(campo: CampoCohesion) -> CampoCohesion:
    evolucionar(campo, 700.0, 30.0, fraccion_carbon=0.75)
    evolucionar(
        campo,
        T_900_C,
        120.0,
        fraccion_carbon=0.75,
        c_FeO=3000.0,
        c_SiO2=1000.0,
        c_Al2O3=1000.0,
    )
    return campo


def test_campo_permanece_en_intervalo_cerrado():
    campo = CampoCohesion((3, 2, 2))
    evolucionar(campo, 700.0, 60.0)
    evolucionar(
        campo,
        np.full(campo.forma, T_900_C),
        1.0e6,
        c_FeO=1.0e5,
        c_SiO2=1.0e5,
        c_Al2O3=1.0e5,
    )
    assert np.all(campo.c >= 0.0)
    assert np.all(campo.c <= 1.0)


def test_cohesion_es_monotona_no_decreciente():
    campo = CampoCohesion((2, 2, 2))
    anterior = campo.c.copy()
    for temperatura in (300.0, 650.0, 720.0, 800.0, T_900_C, 500.0):
        evolucionar(campo, temperatura, 20.0)
        assert np.all(campo.c >= anterior)
        anterior = campo.c.copy()


def test_sin_pasar_ventana_plastica_no_hay_coquizacion():
    # Empezar directamente a 900 C no inventa una historia termoplastica.
    campo = CampoCohesion((1, 1, 1))
    evolucionar(campo, T_900_C, 150.0)
    assert campo.contribuciones is not None
    assert campo.contribuciones["coquizacion"].item() == 0.0


def test_paso_por_ventana_enfriamiento_y_persistencia_irreversible():
    campo = CampoCohesion((1, 1, 1))
    evolucionar(campo, 700.0, 30.0)
    evolucionar(campo, 800.0, 30.0)
    cohesion_caliente = campo.c.item()
    assert cohesion_caliente > 0.0

    evolucionar(campo, 300.0, 60.0)
    assert campo.c.item() >= cohesion_caliente
    assert campo.historia_termica is not None
    assert campo.historia_termica.resolidificada.item()


def test_sinterizacion_a_900_c_es_menor_que_un_por_ciento_en_150_s():
    cierre = tasa_sinterizacion(T_900_C, 150.0, porosidad=0.15)
    assert cierre == pytest.approx(9.428e-5, rel=5.0e-4)
    assert cierre < 0.01

    reparto = diagnostico_mecanismos(_activar_coquizacion(CampoCohesion((1, 1, 1))))
    assert reparto["sinterizacion"] < 0.01


def test_fayalita_es_favorable_pero_no_funde_a_900_c():
    puentes = puentes_solidos(
        T_900_C, c_FeO=3000.0, c_SiO2=1000.0, c_Al2O3=1000.0
    )
    fayalita = puentes["fayalita"]
    assert fayalita["delta_G_kJ_mol"] == pytest.approx(-8.2722, abs=1.0e-3)
    assert fayalita["se_forma"]
    assert fayalita["punto_fusion_C"] == 1178.0
    assert fayalita["estado_a_T"] == "sólida"
    assert not fayalita["puede_producir_puente_liquido"]


def test_componentes_conexas_distinguen_dos_fragmentos_y_una_pieza():
    dos = np.zeros((5, 5, 3))
    dos[0:2, 0:2, :] = 0.8
    dos[3:5, 3:5, :] = 0.9
    assert aglomerado(dos, 0.5)["numero_componentes"] == 2

    una = dos.copy()
    una[1:4, 1:4, 1] = 0.8
    resultado = aglomerado(una, 0.5)
    assert resultado["numero_componentes"] == 1
    assert resultado["es_una_sola_pieza"]


def test_volumen_crece_monotonamente_en_historia_termica_creciente():
    campo = CampoCohesion(
        (3, 1, 1),
        espaciado_m=(1.0e-3, 1.0e-3, 1.0e-3),
        densidad_bulk_kg_m3=800.0,
    )
    temperaturas = (
        np.array([700.0, 300.0, 300.0]).reshape(3, 1, 1),
        np.array([800.0, 700.0, 300.0]).reshape(3, 1, 1),
        np.array([900.0, 800.0, 700.0]).reshape(3, 1, 1),
        np.array([1000.0, 900.0, 800.0]).reshape(3, 1, 1),
        np.array([T_900_C, 1000.0, 900.0]).reshape(3, 1, 1),
        np.full((3, 1, 1), T_900_C),
    )
    serie = [(0.0, copy.deepcopy(campo))]
    for temperatura in temperaturas:
        evolucionar(campo, temperatura, 30.0)
        serie.append((campo.tiempo_s, copy.deepcopy(campo)))

    curva = curva_de_crecimiento(serie)
    diferencias = np.diff(curva["volumen_m3"].to_numpy())
    assert np.all(diferencias >= 0.0)
    assert curva["volumen_m3"].iloc[-1] > curva["volumen_m3"].iloc[0]
    assert curva["es_prediccion"].all()


def test_diagnostico_suma_uno_y_coquizacion_domina_en_ensayo():
    assert MASA_CHAR_ESTIMADA_G == pytest.approx(0.4785, abs=5.0e-5)
    campo = _activar_coquizacion(CampoCohesion((2, 2, 2)))
    reparto = diagnostico_mecanismos(campo)
    assert math.fsum(reparto.values()) == pytest.approx(1.0, abs=1.0e-14)
    assert reparto["coquizacion"] == max(reparto.values())
    assert reparto["coquizacion"] > 0.85



def test_la_cohesion_solo_avanza_donde_hay_carbon():
    """Sin carga no hay coquización: la pared del crisol no puede aglomerar.

    `correr_simulacion` pasa la fracción de carbón enmascarada al lecho. Con un
    0,75 uniforme, la pared Ni-Cr —que es lo primero en calentarse— terminaba
    con c=1 en 185 celdas de la corrida de 720 s, contaminando todo escalar
    calculado sobre el campo completo.
    """
    forma = (4, 4, 6)
    lecho = np.zeros(forma, dtype=bool)
    lecho[:, :, :2] = True                      # disco de carga en el fondo
    fraccion_carbon = np.where(lecho, 0.75, 0.0)
    campo = CampoCohesion(
        forma, espaciado_m=(2.0e-3, 2.0e-3, 1.0e-3),
        densidad_bulk_kg_m3=736.0, fraccion_volumetrica=0.46,
    )
    # La pared se calienta antes que el lecho, como en el ensayo.
    t = 0.0
    while t < 720.0 - 1.0e-9:
        t += 5.0
        T = np.where(lecho, min(298.15 + 1.2 * t, 1173.15),
                     min(298.15 + 2.5 * t, 1173.15))
        campo = evolucionar(campo, T, 5.0,
                                     fraccion_carbon=fraccion_carbon,
                                     porosidad=0.54)
    c = np.asarray(campo.c)
    assert float(np.min(c[lecho])) > 0.9, "el lecho debe cohesionar"
    assert float(np.max(c[~lecho])) < 1.0e-3, (
        f"fuera del lecho no puede haber aglomerado: max={float(np.max(c[~lecho]))}")
    assert int(np.sum(c[~lecho] > 0.5)) == 0
