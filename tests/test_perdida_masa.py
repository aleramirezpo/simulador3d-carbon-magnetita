"""Contraste con la curva de pérdida de masa medida (Tabla 3 del artículo).

Ocho puntos entre 30 y 720 s. Es el único dato experimental **cuantitativo**
disponible sobre la evolución del ensayo, frente a las tres observaciones
cualitativas de la cronología. Estas pruebas no ajustan nada: comprueban que el
modelo no se aleje de lo medido más de lo que se ha declarado aceptable, y sirven
para detectar una regresión de la física.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fisica.adaptador_v3 import MASAS_MOLARES_SOLIDO_KG_MOL
from nucleo.salida import cargar_instantanea

RUTA_V3 = Path(__file__).resolve().parents[2] / "simulacion_v3" / "src"


def _tabla_experimental():
    if str(RUTA_V3) not in sys.path:
        sys.path.insert(0, str(RUTA_V3))
    try:
        from datos_experimentales import TABLA_PERDIDA_MASA
    except Exception:  # pragma: no cover - depende del árbol del proyecto
        pytest.skip("no se encuentra simulacion_v3/src/datos_experimentales.py")
    return np.asarray(TABLA_PERDIDA_MASA, dtype=float)


def _serie_vigente():
    """La misma corrida que abre la interfaz, con el mismo recorte.

    Antes esta función fijaba a mano el directorio `simulacion_720s` y
    reimplementaba por su cuenta el descarte de los NPZ de corridas anteriores.
    Era la TERCERA copia de ese criterio, y quedó anclada a una carpeta donde se
    había relanzado la simulación: la serie vigente allí se detiene a los 265 s,
    así que estas pruebas dejaron de ver la cola de la curva —la meseta que
    comprueban— y fallaban por falta de datos aunque en `resultados/` hubiera
    una corrida completa de 720 s.

    Se pasa a `directorio_predeterminado`, que ya elige la corrida más avanzada,
    y al recorte compartido. Así la física que se contrasta aquí es exactamente
    la que se publica en el visor, y no hay un cuarto criterio que se desvíe.
    """
    from interfaz.app import _recortar_a_la_corrida_vigente, directorio_predeterminado

    raiz = Path(__file__).resolve().parents[1] / "resultados"
    directorio = directorio_predeterminado(raiz)
    rutas = sorted(directorio.glob("*.npz")) if directorio.is_dir() else []
    if not rutas:
        pytest.skip(f"no hay ninguna corrida en {raiz}")
    cargadas = [(ruta, cargar_instantanea(ruta)) for ruta in rutas]
    vigentes = _recortar_a_la_corrida_vigente(
        cargadas,
        lambda par: par[1]["t"],
        lambda par: par[0].stat().st_mtime_ns,
    )
    return sorted(((float(c["t"]), c) for _, c in vigentes), key=lambda e: e[0])


def _masa_solida_g(campos, volumen_celda_m3):
    return 1000.0 * sum(
        float(np.sum(np.asarray(campos["solido"][fase]))) * volumen_celda_m3 * masa_molar
        for fase, masa_molar in MASAS_MOLARES_SOLIDO_KG_MOL.items()
    )


def _perdidas_del_modelo():
    serie = _serie_vigente()
    primera = serie[0][1]
    dx = (float(primera["x"][1]) - float(primera["x"][0])) * 1.0e-3
    dy = (float(primera["y"][1]) - float(primera["y"][0])) * 1.0e-3
    dz = (float(primera["z"][1]) - float(primera["z"][0])) * 1.0e-3
    volumen = dx * dy * dz
    m0 = _masa_solida_g(primera, volumen)
    assert m0 > 0.0
    return serie, volumen, m0


def _perdida_en(serie, volumen, m0, objetivo):
    t, campos = min(serie, key=lambda e: abs(e[0] - objetivo))
    if abs(t - objetivo) > 6.0:
        pytest.skip(f"la serie no llega a t={objetivo} s")
    return 100.0 * (m0 - _masa_solida_g(campos, volumen)) / m0


def test_a_los_30_s_la_perdida_coincide_con_la_medida():
    """Medida: 1,50 %. Es el punto de la curva antes de devolatilizar."""
    serie, volumen, m0 = _perdidas_del_modelo()
    tabla = _tabla_experimental()
    medida = float(tabla[tabla[:, 0] == 30][0, 3])
    modelo = _perdida_en(serie, volumen, m0, 30.0)
    assert abs(modelo - medida) < 1.0, (
        f"a 30 s el modelo pierde {modelo:.2f} % y se midió {medida:.2f} %")


def test_la_perdida_final_no_excede_lo_medido_en_mas_de_cinco_puntos():
    """Medida: 25,20 % de meseta. El modelo devolatiliza algo de más.

    La diferencia (~3 puntos) es que el modelo libera toda la materia volátil del
    análisis próximo y el ensayo no. Se acota para que una regresión que la
    dispare se detecte, no para dar por bueno el valor.
    """
    serie, volumen, m0 = _perdidas_del_modelo()
    tabla = _tabla_experimental()
    medida = float(tabla[tabla[:, 0] == 720][0, 3])
    modelo = _perdida_en(serie, volumen, m0, 720.0)
    assert modelo > medida - 2.0, (
        f"el modelo pierde {modelo:.2f} %, muy por debajo de los {medida:.2f} % medidos")
    assert modelo - medida < 5.0, (
        f"el modelo pierde {modelo:.2f} % frente a {medida:.2f} % medidos: "
        "se ha ido más de cinco puntos por encima")


def test_la_perdida_de_masa_es_monotona_y_termina_en_meseta():
    """El sólido sólo pierde masa, y al agotarse el volátil la curva se aplana.

    La meseta también está medida: 25,80 % a 210 s y 25,20 % a 360 y 720 s.
    """
    serie, volumen, m0 = _perdidas_del_modelo()
    perdidas = [(t, 100.0 * (m0 - _masa_solida_g(c, volumen)) / m0) for t, c in serie]
    valores = [p for _, p in perdidas]
    assert all(b >= a - 1.0e-6 for a, b in zip(valores, valores[1:])), "la pérdida no es monótona"
    tardios = [p for t, p in perdidas if t >= 300.0]
    assert tardios and max(tardios) - min(tardios) < 0.05, (
        "pasados los 300 s la pérdida debe estar en meseta")
