"""Comprobaciones del documento de fenomenología.

El documento afirma cosas concretas —que la ilmenita no se mueve, que el gas
queda entre dos fronteras, que el aglomerado es un disco del ancho del crisol—
y las afirma con cifras que salen de la corrida. Lo que se comprueba aquí no es
el texto sino esas afirmaciones: si una corrida futura las desmintiera, el
documento estaría mintiendo y nadie se enteraría al recompilarlo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from informe.datos_fenomenologia import (  # noqa: E402
    FASES_ESPECTADORAS,
    diagnosticos,
    fronteras_co,
)


@pytest.fixture(scope="module")
def datos():
    try:
        return diagnosticos()
    except SystemExit as exc:
        pytest.skip(str(exc))


def test_las_fronteras_de_co_estan_ordenadas_por_dificultad(datos):
    """La cascada del hierro tiene un orden, y sale de la termodinámica.

    Cada paso exige un gas más reductor que el anterior, y la ilmenita más que
    ninguno. No se comprueban valores concretos —dependen de las tablas— sino
    el orden, que es de lo que depende el argumento del documento.
    """
    f = fronteras_co()
    assert f["hematita"] < f["magnetita"] < f["wustita"] < f["ilmenita"]
    assert f["ilmenita"] > 0.9, "la ilmenita debe exigir un gas casi de CO puro"


def test_el_gas_queda_entre_la_magnetita_y_la_wustita(datos):
    """El eje de todo el documento, en una sola comprobación."""
    fronteras = fronteras_co()
    x_gas = float(datos["serie"]["CO_sobre_COx"][-1])
    assert x_gas > fronteras["magnetita"], (
        "sin superar la frontera de la magnetita no habría wüstita")
    assert x_gas < fronteras["wustita"], (
        "por encima de la frontera de la wüstita aparecería hierro metálico")
    assert x_gas < fronteras["ilmenita"]


def test_las_fases_espectadoras_no_se_mueven(datos):
    """Ilmenita, rutilo y cuarzo: el documento dice que no cambian."""
    s = datos["serie"]
    for fase in FASES_ESPECTADORAS:
        clave = f"m_{fase}"
        if clave not in s:
            continue
        inicial, final = s[clave][0], s[clave][-1]
        if inicial <= 0.0:
            assert final < 1.0e-7, f"{fase} aparece de la nada"
            continue
        assert abs(final - inicial) / inicial < 1.0e-5, (
            f"{fase} cambia: {inicial:.6g} -> {final:.6g} g")


def test_la_reduccion_se_detiene_en_wustita(datos):
    """Se consumen los óxidos superiores y el hierro metálico queda en trazas."""
    s = datos["serie"]
    assert s["m_Fe2O3"][-1] < 1.0e-6, "debía consumirse toda la hematita"
    assert s["m_Fe3O4"][-1] < 1.0e-6, "debía consumirse toda la magnetita"
    assert s["m_FeO"][-1] > 0.1, "la wüstita es el producto mayoritario"
    hierro_total = sum(s[f"m_{f}"][-1] for f in ("Fe", "FeO", "Fe3O4", "Fe2O3"))
    assert s["m_Fe"][-1] / hierro_total < 0.02, (
        "el hierro metálico debe quedar en trazas, no ser un producto")


def test_la_adveccion_solo_cuenta_durante_el_estallido_de_gas(datos):
    """La primera versión de esta prueba exigía Pe < 1 en TODA la corrida.

    Falló, y tenía razón en fallar: en el pico de devolatilización los Péclet
    llegan a valores de unidades, es decir que durante esos segundos el gas
    generado arrastra calor y especies más rápido de lo que difunden. El
    documento afirmaba lo contrario y hubo que corregirlo. Lo que se comprueba
    ahora es la forma real del fenómeno: un pico breve, y valores despreciables
    en el estado final.
    """
    s = datos["serie"]
    t = np.asarray(s["t"])

    for clave in ("Re_p", "Pe_termico", "Pe_masico"):
        serie = np.abs(np.asarray(s[clave]))
        assert serie[-1] < 1.0e-3, (
            f"{clave} vale {serie[-1]:.3g} al final: debería ser despreciable")
        t_pico = t[int(np.nanargmax(serie))]
        assert 40.0 < t_pico < 140.0, (
            f"el pico de {clave} cae en t={t_pico:.0f} s, fuera de la "
            "devolatilización")

    # El Damköhler sí es despreciable siempre: no hay control por transporte en
    # ningún instante, y de eso depende el argumento sobre la ilmenita.
    assert float(np.nanmax(np.abs(np.asarray(s["Da"])))) < 1.0e-6


def test_el_aglomerado_es_un_disco_del_ancho_del_crisol(datos):
    """No es una esfera: la pared le fija el diámetro y no crece con el tiempo."""
    s = datos["serie"]
    diametros = np.asarray(s["D_huella_mm"])
    formado = diametros > 0.0
    assert formado.any(), "no llegó a formarse aglomerado"
    anchos = diametros[formado]
    # El cuerpo tarda un par de fotogramas en cuajar de pared a pared: cuando se
    # cruza el umbral de cohesión todavía hay corona sin cohesionar. A partir
    # del tercero ya no se mueve, y eso es lo que dice el documento.
    estables = anchos[2:]
    assert estables.max() - estables.min() < 0.01, (
        "una vez cuajado, el diámetro lo fija la pared y no debería moverse")
    assert 20.0 < anchos[-1] < 26.0, (
        f"el disco mide {anchos[-1]:.1f} mm; el interior del crisol ronda 23 mm")
    # Y es un disco, no una bola: mucho más ancho que alto.
    assert anchos[-1] > 4.0 * s["altura_aglomerado_mm"][-1]


def test_el_volumen_aparente_es_coherente_con_el_inventario_de_fases(datos):
    """Envolvente por el complemento de la porosidad = volumen de materia.

    Son dos caminos independientes: uno cuenta celdas y lee la porosidad, el
    otro suma moles por su volumen molar. Si se separan, o la porosidad no
    representa lo que dice o el inventario no cuadra.
    """
    s = datos["serie"]
    envolvente = s["volumen_aglomerado_mm3"][-1]
    porosidad = s["eps_media"][-1]
    solido = s["volumen_solido_mm3"][-1]
    estimado = envolvente * (1.0 - porosidad)
    assert abs(estimado - solido) / solido < 0.05, (
        f"envolvente*(1-eps)={estimado:.1f} frente a inventario={solido:.1f} mm³")


def test_la_perdida_de_masa_se_detiene_antes_del_final(datos):
    """El documento afirma que la meseta llega mucho antes de los 720 s."""
    s = datos["serie"]
    t = np.asarray(s["t"])
    perdida = np.asarray(s["perdida_pct"])
    if t[-1] < 300.0:
        pytest.skip("la corrida no llega a la meseta")
    alcanzado = t[perdida >= 0.99 * perdida[-1]]
    assert alcanzado[0] < 200.0, (
        f"el 99 % de la pérdida se alcanza a los {alcanzado[0]:.0f} s")
    tardios = perdida[t >= 300.0]
    assert tardios.max() - tardios.min() < 0.05, "la meseta no es plana"
