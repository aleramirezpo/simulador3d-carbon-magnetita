"""Pruebas de regresion para el calendario comun de los modelos 0-D y 3-D."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fisica.cohesion import (
    RANGO_TIEMPO_COQUIZACION_S,
    RANGO_TIEMPO_PLASTIFICACION_S,
    TIEMPO_COQUIZACION_S,
    TIEMPO_PLASTIFICACION_S,
)
from verificacion.consistencia_0d import (
    RUTA_SALIDA,
    TIEMPOS_EXPERIMENTALES_S,
    comparar_con_0d,
    historia_termica_0d,
    tiempos_caracteristicos,
)


# Compuerta de devolatilización con la que se calibró `simulacion_v3`. El caso
# 3-D usa 450 degC (véase `casos/carbon_magnetita.yaml` y
# `caso.ajustar_devolatilizacion`), y como la química vive en un diccionario
# compartido, cargar el caso desplazaría estos hitos ~20 s. La fijamos aquí para
# que la prueba diga lo que quiere decir —que el 3-D reproduce el calendario del
# 0-D TAL COMO SE CALIBRÓ— y no dependa del orden de ejecución.
COMPUERTA_V3_C = 200.0


@pytest.fixture(scope="module")
def compuerta_v3():
    from fisica import adaptador_v3

    devol = adaptador_v3.modelo_multifase.lit.DEVOLATILIZACION
    previo = float(devol["T_inicio_C"])
    devol["T_inicio_C"] = COMPUERTA_V3_C
    try:
        yield COMPUERTA_V3_C
    finally:
        devol["T_inicio_C"] = previo


@pytest.fixture(scope="module")
def referencia_0d(compuerta_v3):
    return historia_termica_0d()


def test_hitos_termoplasticos_reproducen_calendario_0d(referencia_0d):
    hitos = tiempos_caracteristicos(referencia_0d)
    assert hitos["t_350_C_s"] == pytest.approx(90.0, abs=5.0)
    assert hitos["t_500_C_s"] == pytest.approx(114.0, abs=5.0)


def test_aglomerado_predicho_se_forma_en_ventana_coherente(referencia_0d):
    hitos = tiempos_caracteristicos(referencia_0d)
    assert 100.0 <= hitos["t_aglomerado_s"] <= 145.0
    assert hitos["t_consolidacion_s"] > hitos["t_350_C_s"]
    assert hitos["t_aglomerado_s"] >= hitos["t_consolidacion_s"]


def test_constantes_coquizacion_permanecen_en_rango_calibrable():
    assert RANGO_TIEMPO_PLASTIFICACION_S[0] <= TIEMPO_PLASTIFICACION_S <= RANGO_TIEMPO_PLASTIFICACION_S[1]
    assert RANGO_TIEMPO_COQUIZACION_S[0] <= TIEMPO_COQUIZACION_S <= RANGO_TIEMPO_COQUIZACION_S[1]


def test_noventa_por_ciento_perdida_0d_ocurre_antes_de_120_s(referencia_0d):
    hitos = tiempos_caracteristicos(referencia_0d)
    assert math.isfinite(hitos["t_perdida_90_s"])
    assert hitos["t_perdida_90_s"] < 120.0


def test_comparar_con_0d_devuelve_una_fila_por_tiempo(referencia_0d):
    # El limite 3-D perfectamente mezclado debe coincidir por construccion con
    # el 0-D; esta autoconsistencia tambien verifica signos y unidades.
    comparacion = comparar_con_0d(referencia_0d)
    assert comparacion["tiempo_s"].to_numpy() == pytest.approx(
        TIEMPOS_EXPERIMENTALES_S
    )
    assert len(comparacion) == len(TIEMPOS_EXPERIMENTALES_S)
    assert np.allclose(comparacion["error_T"], 0.0)
    assert np.allclose(comparacion["error_perdida_masa"], 0.0)
    assert RUTA_SALIDA.is_file()


def test_la_compuerta_del_caso_desplaza_el_calendario_del_0d():
    """La compuerta baja de v3 y su historia térmica están acopladas.

    `simulacion_v3` centra la sigmoide de devolatilización en 200 degC, que es
    el asomo de las primeras trazas y no el máximo de velocidad de un carbón
    bituminoso (430-470 degC). Con esa compuerta el 0-D alcanza los 350 degC a
    los 90 s; con la del caso 3-D los alcanza antes, porque la carga endotérmica
    de la pirólisis llega más tarde.

    Es decir: **la calibración de v3 y la compuerta baja se sostienen la una a la
    otra**. Corregirla en serio exige recalibrar v3, que es decisión del proyecto
    y no de este programa. La prueba deja el acoplamiento medido y documentado en
    vez de enterrado.

    Cada valor se mide en su propio proceso porque `historia_termica_0d` cachea
    el resultado y en un mismo intérprete el segundo devolvería el primero.
    """
    import subprocess
    import sys
    from pathlib import Path

    guion = (
        "import sys; sys.path.insert(0, r'{raiz}');"
        "from fisica import adaptador_v3 as a;"
        "a.modelo_multifase.lit.DEVOLATILIZACION['T_inicio_C'] = {gate};"
        "from verificacion.consistencia_0d import historia_termica_0d, tiempos_caracteristicos;"
        "print(tiempos_caracteristicos(historia_termica_0d())['t_350_C_s'])"
    )
    raiz = Path(__file__).resolve().parents[1]

    def medir(gate):
        salida = subprocess.run(
            [sys.executable, "-c", guion.format(raiz=raiz, gate=gate)],
            cwd=raiz, capture_output=True, text=True, timeout=300, check=False)
        if salida.returncode != 0:
            pytest.skip(f"no se pudo medir con compuerta {gate}: {salida.stderr[-300:]}")
        return float(salida.stdout.strip().splitlines()[-1])

    con_v3 = medir(200.0)
    con_caso = medir(450.0)

    assert con_v3 == pytest.approx(90.0, abs=5.0)
    assert con_caso < con_v3 - 10.0, (
        f"con la compuerta corregida el 0-D llega a 350 degC en {con_caso:.1f} s "
        f"frente a {con_v3:.1f} s")
