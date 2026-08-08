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


def test_el_gas_se_tampona_sobre_la_frontera_de_la_magnetita(datos):
    """El eje de todo el documento, en una sola comprobación.

    La versión anterior exigía que el gas FINAL estuviera estrictamente por
    encima de la frontera de la magnetita. Eso describía el modelo con la
    termodinámica defectuosa, que ponía esa frontera en 0,0076 en vez de 0,3222
    y por tanto la superaba con enorme margen (véase
    `simulacion_v3/tests/test_fronteras_redox.py`).

    El comportamiento real es un tamponamiento: el gas sube por encima de la
    frontera mientras hay volátil, forma wüstita, y al agotarse el reductor cae
    hasta apoyarse en ella y ahí se queda. Lo que hay que comprobar son las tres
    cosas que sostienen el argumento, no el signo de una desigualdad en el
    último instante.
    """
    import numpy as np

    fronteras = fronteras_co()
    x = np.asarray(datos["serie"]["CO_sobre_COx"], dtype=float)
    x = x[np.isfinite(x)]

    # 1. En algún momento superó la frontera de la magnetita: por eso hay wüstita.
    assert float(np.max(x)) > fronteras["magnetita"], (
        "sin superar nunca la frontera de la magnetita no habría wüstita")

    # 2. El gas final se apoya en esa frontera, dentro de una banda declarada.
    #    No se pide igualdad: el venteo sigue sacando gas después de que la
    #    reacción se pare, así que queda ligeramente por debajo.
    x_final = float(x[-1])
    assert abs(x_final - fronteras["magnetita"]) < 0.25 * fronteras["magnetita"], (
        f"el gas final es {x_final:.4f} y la frontera de la magnetita "
        f"{fronteras['magnetita']:.4f}: ya no se puede hablar de tamponamiento")

    # 3. Nunca se instala por encima de la frontera de la wüstita, que es la que
    #    metalizaría el hierro en masa, ni se acerca a la de la ilmenita.
    assert float(np.median(x)) < fronteras["wustita"]
    assert x_final < fronteras["ilmenita"]


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


def test_la_reduccion_se_detiene_en_la_magnetita(datos):
    """La cascada se para en la magnetita, y por eso el aglomerado sigue magnético.

    La versión anterior exigía que se consumiese TODA la magnetita. Eso era el
    modelo con la frontera Fe3O4/FeO equivocada por un factor 42, y quedó
    refutado por la observación del laboratorio: al final del ensayo el
    aglomerado todavía se pega al imán (véase `tests/test_magnetismo.py`).
    """
    s = datos["serie"]

    # La hematita sí se agota: su frontera está órdenes por debajo del gas.
    assert s["m_Fe2O3"][-1] < 1.0e-6, "debía consumirse toda la hematita"

    # La magnetita se conserva. De hecho GANA masa: hereda el hierro de la
    # hematita, y sólo pierde una parte al pasar a wüstita.
    assert s["m_Fe3O4"][-1] > 0.9 * s["m_Fe3O4"][0], (
        f"la magnetita cae de {s['m_Fe3O4'][0]:.4f} a {s['m_Fe3O4'][-1]:.4f} g; "
        "el ensayo dice que al final el aglomerado sigue siendo magnético")

    # Y aparece wüstita, pero como fase minoritaria frente a la magnetita.
    assert s["m_FeO"][-1] > 1.0e-3, "tiene que haberse formado algo de wüstita"
    assert s["m_FeO"][-1] < s["m_Fe3O4"][-1], (
        "la wüstita no puede desbancar a la magnetita: el gas se tampona en su "
        "frontera común")

    # Hierro metálico: se mide en ÁTOMOS de Fe, que es lo que significa
    # "cuánto del hierro se ha metalizado". El umbral sube de 2 % a 5 % y
    # conviene explicar por qué, porque es un cambio real de física y no un
    # ajuste de tolerancia: con la frontera FeO/Fe corregida (0,709 en fracción
    # de CO) el gas del lecho la SUPERA durante el estallido de devolatilización,
    # cuando llega a 0,84. Antes el modelo decía 0,004 % de hierro metálico
    # porque su frontera FeO/Fe estaba en 0,834 y no se cruzaba nunca.
    atomos = {"Fe": 1.0, "FeO": 1.0, "Fe3O4": 3.0, "Fe2O3": 2.0}
    masas_molares = {"Fe": 55.845, "FeO": 71.844, "Fe3O4": 231.531, "Fe2O3": 159.687}
    moles_fe = {
        f: s[f"m_{f}"][-1] / masas_molares[f] * atomos[f] for f in atomos
    }
    fraccion_metalico = moles_fe["Fe"] / sum(moles_fe.values())
    assert fraccion_metalico < 0.05, (
        f"el hierro metálico llega al {100 * fraccion_metalico:.2f} % del hierro: "
        "deja de ser una traza y pasa a ser un producto, lo que contradice el "
        "argumento del documento")


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
