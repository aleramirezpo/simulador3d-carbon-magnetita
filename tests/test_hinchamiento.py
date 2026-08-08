"""Pruebas del modelo de hinchamiento y de la cronología observada en el ensayo."""

from __future__ import annotations

import numpy as np
import pytest

from fisica import hinchamiento as hin


def test_el_carbon_solo_hincha_mas_que_mezclado_con_magnetita():
    """La observación del laboratorio: «se hincha, pero no tanto»."""
    solo = hin.factor_hinchamiento_maximo(8.0, fraccion_inerte=0.0)
    mezcla = hin.factor_hinchamiento_maximo(8.0, fraccion_inerte=0.25)
    assert solo > mezcla > 1.0
    # Sigue hinchando de forma apreciable: no queda anulado por el inerte.
    assert (mezcla - 1.0) > 0.4 * (solo - 1.0)


def test_el_hinchamiento_crece_con_el_indice():
    factores = [hin.factor_hinchamiento_maximo(ih, 0.25) for ih in (1, 2, 4, 6, 8, 9)]
    assert all(b > a for a, b in zip(factores, factores[1:]))
    # Un carbón que no hincha (IH 0) no puede expandirse.
    assert hin.factor_hinchamiento_maximo(0.0, 0.25) == pytest.approx(1.0)


def test_un_inerte_total_anula_el_hinchamiento():
    assert hin.factor_hinchamiento_maximo(8.0, fraccion_inerte=1.0) == pytest.approx(1.0)


def test_el_hinchamiento_sigue_a_la_plastificacion_y_es_irreversible():
    """No depende de la temperatura instantánea sino de la historia."""
    grados = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    factores = hin.hinchamiento_local(grados, 8.0, 0.25)
    assert factores[0] == pytest.approx(1.0)
    assert np.all(np.diff(factores) > 0.0)
    assert factores[-1] == pytest.approx(hin.factor_hinchamiento_maximo(8.0, 0.25))


def test_el_confinamiento_lateral_manda_todo_el_hinchamiento_a_la_altura():
    factor = hin.factor_hinchamiento_maximo(8.0, 0.25)
    confinado = hin.altura_hinchada_mm(3.26, factor, confinamiento_lateral=True)
    libre = hin.altura_hinchada_mm(3.26, factor, confinamiento_lateral=False)
    assert confinado > libre > 3.26
    assert confinado == pytest.approx(3.26 * factor)
    assert libre == pytest.approx(3.26 * factor ** (1.0 / 3.0))


def test_entradas_invalidas_se_rechazan():
    with pytest.raises(ValueError):
        hin.factor_hinchamiento_maximo(-1.0)
    with pytest.raises(ValueError):
        hin.factor_hinchamiento_maximo(8.0, fraccion_inerte=1.5)
    with pytest.raises(ValueError):
        hin.hinchamiento_local(np.array([1.5]))
    with pytest.raises(ValueError):
        hin.altura_hinchada_mm(3.26, -1.0)


def test_la_ficha_declara_que_no_esta_validado():
    ficha = hin.resumen(8.0, 0.25)
    assert "NO VALIDADA" in ficha["validacion"]
    assert "D720" in ficha["referencia_ensayo"]
    assert ficha["factor_mezcla"] < ficha["factor_carbon_solo"]


def test_el_volumen_del_lecho_hinchado_es_el_esperado_del_ensayo():
    """1,36 cm3 de lecho con IH 8 y 25 % de inerte."""
    factor = hin.factor_hinchamiento_maximo(8.0, 0.25)
    volumen = hin.volumen_hinchado_m3(1.36e-6, 1.0, 8.0, 0.25)
    assert volumen == pytest.approx(1.36e-6 * factor)
    # El lecho crece de 3,26 mm a algo que sigue cabiendo en el crisol (32 mm).
    altura = hin.altura_hinchada_mm(3.26, factor)
    assert 3.26 < altura < 32.0


# --- Cronología observada en el laboratorio -------------------------------
#
# Datos aportados por el usuario, cualitativos pero reales:
#   t =  30 s  no ha pasado nada: al sacarlo, polvo suelto
#   t =  90 s  ya es más grande, está creciendo y se hincha
#   t = 120 s  ya está bien formado
#
# Se usan para FALSAR: cualquier ajuste que forme el aglomerado antes de los
# 30 s, o que no lo tenga formado hacia los 120 s, queda refutado.

CRONOLOGIA_OBSERVADA = ((30.0, "nada"), (90.0, "creciendo"), (120.0, "formado"))


def _corrida_de_referencia():
    """Instantáneas de la corrida vigente, la misma que abre el visor.

    El recorte a la corrida vigente y la elección de carpeta no se reimplementan
    aquí: los hacen `nucleo.salida` e `interfaz.app`, que es de donde los toman
    también el visor y el informe. Antes esta función apuntaba a una carpeta
    fija, `resultados/simulacion_720s`, que llegó a contener dos corridas
    mezcladas mientras el sitio publicaba otra distinta.
    """
    from pathlib import Path

    from interfaz.app import directorio_predeterminado
    from nucleo.salida import cargar_instantanea, serie_vigente

    raiz = Path(__file__).resolve().parents[1]
    directorio = directorio_predeterminado(raiz / "resultados")
    indice = serie_vigente(directorio) if Path(directorio).is_dir() else []
    if len(indice) < 5:
        pytest.skip(f"no hay corrida utilizable en {directorio}")
    return {
        float(elemento["t"]): cargar_instantanea(elemento["ruta"])
        for elemento in indice
    }


def _estado_en(tiempos, objetivo):
    clave = min(tiempos, key=lambda t: abs(t - objetivo))
    if abs(clave - objetivo) > 6.0:
        pytest.skip(f"la serie no tiene instantánea cerca de t={objetivo} s")
    campos = tiempos[clave]
    lecho = np.asarray(campos["etiquetas"]) == 3
    cohesion = np.asarray(campos["cohesion"])[lecho]
    return {
        "t": clave,
        "fraccion_cohesionada": float(np.mean(cohesion > 0.5)),
        "cohesion_media": float(np.mean(cohesion)),
        "hinchamiento": float(np.mean(np.asarray(campos["hinchamiento"])[lecho])),
        "T_lecho": float(np.mean(np.asarray(campos["T"])[lecho])),
    }


def test_a_los_30_segundos_no_hay_aglomerado():
    """«A los 30 segundos lo sacábamos y no había pasado nada.»"""
    estado = _estado_en(_corrida_de_referencia(), 30.0)
    assert estado["fraccion_cohesionada"] == 0.0, (
        f"a t={estado['t']:.1f} s ya hay {100 * estado['fraccion_cohesionada']:.1f} % "
        "del lecho cohesionado; el ensayo dio polvo suelto")
    assert estado["hinchamiento"] < 1.02, "tampoco puede haber hinchado todavía"


def test_a_los_90_segundos_el_aglomerado_ya_crece_y_se_hincha():
    """«Cuando llega a 90 segundos ya es más grande, crece, se hincha.»

    Lo observado a los 90 s es el CRECIMIENTO, no la pieza terminada: el carbón
    está en la ventana plástica y el gas atrapado lo infla. La coquización, que
    es la que da la pieza cohesionada, viene después, al resolidificar. Por eso
    aquí se exige hinchamiento y en la prueba de los 120 s, cohesión.
    """
    tiempos = _corrida_de_referencia()
    estado = _estado_en(tiempos, 90.0)
    temprano = _estado_en(tiempos, 30.0)
    assert estado["hinchamiento"] > 1.2, (
        f"a t={estado['t']:.1f} s el hinchamiento es {estado['hinchamiento']:.3f}; "
        "el ensayo ya lo veía hincharse")
    assert estado["hinchamiento"] > temprano["hinchamiento"] + 0.1, (
        "el hinchamiento debe estar creciendo entre los 30 y los 90 s")
    # Y no puede pasarse del máximo que permite el modelo con IH 8 y 25 % de inerte.
    assert estado["hinchamiento"] <= hin.factor_hinchamiento_maximo(8.0, 0.25) + 1e-6


def test_a_los_120_segundos_el_aglomerado_esta_formado():
    """«A los 120 segundos ya está formado bien.»"""
    estado = _estado_en(_corrida_de_referencia(), 120.0)
    assert estado["fraccion_cohesionada"] > 0.60, (
        f"a t={estado['t']:.1f} s sólo hay {100 * estado['fraccion_cohesionada']:.1f} % "
        "del lecho formado")
