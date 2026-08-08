"""Pruebas del modelo magnetico y de la prueba del iman observada en el ensayo.

La segunda mitad de este archivo son FALSADORES. La observacion del laboratorio
es cualitativa -- un iman contra la muestra -- y por eso no se le pide al modelo
que reproduzca un numero, sino que respete un signo y un orden:

    * al final del ensayo el aglomerado TODAVIA responde al iman,
    * y responde MENOS que al principio.

Cualquier ajuste futuro que apague el iman antes de los 720 s, o que lo haga
mas fuerte con el tiempo, queda refutado por el ensayo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisica import magnetismo as mag
from fisica.adaptador_v3 import MASAS_MOLARES_SOLIDO_KG_MOL
from nucleo.salida import cargar_instantanea, serie_vigente

LECHO = 3
V = 1.0e-6  # volumen de celda de juguete, m3


def _solido(**moles_por_m3: float) -> dict[str, np.ndarray]:
    return {fase: np.array([valor]) for fase, valor in moles_por_m3.items()}


# --- Modelo puro -----------------------------------------------------------


def test_la_magnetita_pura_da_su_magnetizacion_de_tabla() -> None:
    """92 A m2/kg, Hunt, Moskowitz y Banerjee (1995), Tabla 3."""

    solido = _solido(Fe3O4=1000.0)
    assert mag.magnetizacion_Am2_kg(solido, V) == pytest.approx(92.0, rel=1e-12)


def test_la_wustita_pura_no_responde_al_iman() -> None:
    """T_Neel = 198 K: a temperatura ambiente es paramagnetica."""

    solido = _solido(FeO=1000.0)
    assert mag.magnetizacion_Am2_kg(solido, V) == 0.0


def test_la_mezcla_es_lineal_en_masa() -> None:
    """Los momentos magneticos son aditivos: no hay regla que calibrar."""

    a = _solido(Fe3O4=700.0)
    b = _solido(Fe=300.0)
    juntos = _solido(Fe3O4=700.0, Fe=300.0)
    momento = mag.momento_magnetico_Am2
    assert momento(juntos, V) == pytest.approx(momento(a, V) + momento(b, V), rel=1e-12)


def test_reducir_magnetita_a_wustita_apaga_el_iman() -> None:
    """El mecanismo que explica lo observado, en su forma mas simple.

    Se conserva el hierro: 1 mol de Fe3O4 da 3 de FeO. La magnetizacion cae
    monotonamente segun avanza la conversion, y al final es practicamente nula.
    """

    valores = []
    for conversion in (0.0, 0.25, 0.5, 0.75, 1.0):
        solido = _solido(
            Fe3O4=1000.0 * (1.0 - conversion),
            FeO=3000.0 * conversion,
        )
        valores.append(mag.magnetizacion_Am2_kg(solido, V))
    assert np.all(np.diff(valores) < 0.0)
    assert valores[0] > 90.0
    assert valores[-1] == 0.0


def test_el_eutectoide_devuelve_mas_magnetismo_del_que_habia() -> None:
    """La cuenta que obliga a que el enfriamiento sea rapido.

    4 FeO -> Fe3O4 + Fe da 8,37 A m2 por mol de hierro, frente a los 7,10 de la
    magnetita de partida. Si el enfriamiento fuese lento, el aglomerado saldria
    MAS magnetico que al empezar. Como se observa lo contrario, la wustita tiene
    que conservarse en buena parte.
    """

    por_mol_de_Fe = {
        "magnetita": 92.0 * MASAS_MOLARES_SOLIDO_KG_MOL["Fe3O4"] / 3.0,
        "eutectoide": (
            92.0 * MASAS_MOLARES_SOLIDO_KG_MOL["Fe3O4"]
            + 218.0 * MASAS_MOLARES_SOLIDO_KG_MOL["Fe"]
        ) / 4.0,
    }
    assert por_mol_de_Fe["magnetita"] == pytest.approx(7.10, abs=0.02)
    assert por_mol_de_Fe["eutectoide"] == pytest.approx(8.37, abs=0.02)
    assert por_mol_de_Fe["eutectoide"] > por_mol_de_Fe["magnetita"]

    # Y lo mismo a traves del modulo: la cota de enfriamiento lento esta por
    # encima de la de temple siempre que quede wustita.
    solido = _solido(FeO=3000.0)
    cotas = mag.cotas_magnetizacion_Am2_kg(solido, V)
    assert cotas["temple"] == 0.0
    assert cotas["enfriamiento_lento"] > 20.0


def test_las_dos_cotas_encierran_cualquier_enfriamiento_intermedio() -> None:
    solido = _solido(Fe3O4=200.0, FeO=2400.0, C=5000.0)
    cotas = mag.cotas_magnetizacion_Am2_kg(solido, V)
    for fraccion in (0.0, 0.3, 0.7, 1.0):
        valor = mag.magnetizacion_Am2_kg(solido, V, wustita_descompuesta=fraccion)
        assert cotas["temple"] - 1e-12 <= valor <= cotas["enfriamiento_lento"] + 1e-12


def test_entradas_invalidas_se_rechazan() -> None:
    solido = _solido(Fe3O4=1000.0)
    with pytest.raises(ValueError):
        mag.magnetizacion_Am2_kg(solido, V, wustita_descompuesta=1.5)
    with pytest.raises(ValueError):
        mag.magnetizacion_Am2_kg(solido, V, wustita_descompuesta=-0.1)
    with pytest.raises(ValueError):
        mag.magnetizacion_Am2_kg(solido, V, ms_titanohematita=-1.0)
    with pytest.raises(ValueError):
        mag.flujo_de_enfriamiento_W_m2(-5.0)


def test_la_ficha_declara_que_no_esta_validado() -> None:
    ficha = mag.resumen()
    assert "NO VALIDADA" in ficha["validacion"]
    assert "Hunt" in ficha["referencia_ms"]
    assert "198 K" in ficha["referencia_wustita"]


# --- El enfriamiento: la pregunta que hizo el laboratorio ------------------


def test_la_capacidad_concentrada_es_licita_para_el_crisol() -> None:
    """Antes de usar la curva hay que comprobar que se puede usar."""

    assert mag.numero_de_biot(1173.15, cuerpo=mag.CRISOL_ENSAYO) < 0.1


def test_sacar_el_crisol_al_aire_es_un_temple_solo_a_medias() -> None:
    """«Se saca y se deja a temperatura ambiente; no se si es un temple o que.»

    Con crisol y tapa el conjunto pesa 48,5 g de Ni-Cr y cruza el eutectoide a
    unos 300 grados C/min, tres veces por debajo del umbral publicado de 1000
    para suprimir la transformacion. Es rapido para un horno y lento para un
    temple: la wustita se conserva en parte, y por eso el observable se reporta
    entre dos cotas y no como un numero.
    """

    v = mag.veredicto_de_temple(cuerpo=mag.CRISOL_ENSAYO)
    assert 100.0 < v["velocidad_en_eutectoide_C_min"] < mag.VELOCIDAD_SUPRESION_C_MIN
    assert v["veredicto"].startswith("TEMPLE PARCIAL")
    # Y aun asi pasa poco tiempo en la ventana en que el eutectoide puede operar.
    assert v["tiempo_en_ventana_eutectoide_s"] < 120.0


def test_volcar_el_aglomerado_solo_si_seria_un_temple() -> None:
    """Consecuencia practica para el laboratorio, no un resultado del modelo.

    Los 0,72 g del aglomerado sin el crisol se enfrian dos ordenes de magnitud
    mas rapido. Si interesa que el iman lea el estado que habia a 900 grados C,
    basta con volcarlo fuera del crisol al sacarlo.
    """

    solo = mag.veredicto_de_temple(cuerpo=mag.AGLOMERADO_SOLO)
    con_crisol = mag.veredicto_de_temple(cuerpo=mag.CRISOL_ENSAYO)
    assert solo["velocidad_en_eutectoide_C_min"] > 10.0 * con_crisol["velocidad_en_eutectoide_C_min"]
    assert solo["veredicto"].startswith("TEMPLE:")


# --- Lo que se recupera a temperatura ambiente ------------------------------


def test_el_eutectoide_conserva_el_hierro() -> None:
    """4 FeO -> Fe3O4 + Fe: ni un atomo de mas ni de menos."""

    solido = _solido(FeO=4000.0, Fe3O4=100.0)
    for fraccion in (0.0, 0.35, 1.0):
        moles = mag.fases_tras_enfriar(solido, V, fraccion)
        fe = moles["FeO"] + 3.0 * moles["Fe3O4"] + moles["Fe"]
        assert fe == pytest.approx(4.0e-3 + 3.0 * 1.0e-4, rel=1e-12)
    # Con descomposicion total no queda wustita, y aparecen las dos fases.
    completo = mag.fases_tras_enfriar(solido, V, 1.0)
    assert completo["FeO"] == pytest.approx(0.0, abs=1e-18)
    assert completo["Fe3O4"] == pytest.approx(1.0e-4 + 1.0e-3, rel=1e-12)
    assert completo["Fe"] == pytest.approx(1.0e-3, rel=1e-12)


def test_lo_que_no_es_hierro_baja_congelado() -> None:
    """Ilmenita, char y ceniza no tienen ruta accesible entre 900 C y el ambiente."""

    solido = _solido(FeO=1000.0, FeTiO3=500.0, C=8000.0, ceniza=300.0)
    antes = mag.fases_tras_enfriar(solido, V, 0.0)
    despues = mag.fases_tras_enfriar(solido, V, 1.0)
    for fase in ("FeTiO3", "C", "ceniza"):
        assert despues[fase] == pytest.approx(antes[fase], rel=1e-12)


def test_la_fraccion_de_eutectoide_decrece_al_enfriar_mas_rapido() -> None:
    """Menos tiempo en la ventana, menos descomposicion. Y sus dos extremos."""

    tiempos = [1.0, 5.0, 20.0, 50.0, 200.0, 10_000.0]
    fracciones = [mag.fraccion_eutectoide(t) for t in tiempos]
    assert np.all(np.diff(fracciones) > 0.0)
    assert mag.fraccion_eutectoide(0.0) == 0.0
    assert fracciones[-1] > 0.99, "un enfriamiento de horas la descompone entera"
    with pytest.raises(ValueError):
        mag.fraccion_eutectoide(-1.0)


def test_el_enfriado_de_este_ensayo_cae_entre_las_dos_cotas() -> None:
    """No es temple ni es lento: el numero central existe y esta en medio."""

    v = mag.veredicto_de_temple(cuerpo=mag.CRISOL_ENSAYO)
    fraccion = mag.fraccion_eutectoide(float(v["tiempo_en_ventana_eutectoide_s"]))
    assert 0.15 < fraccion < 0.85, (
        f"la fraccion estimada es {fraccion:.2f}; si se pegase a 0 o a 1 no "
        "haria falta reportar dos cotas"
    )
    # Y volcando el aglomerado solo, se acerca mucho mas al temple.
    solo = mag.veredicto_de_temple(cuerpo=mag.AGLOMERADO_SOLO)
    fraccion_solo = mag.fraccion_eutectoide(
        float(solo["tiempo_en_ventana_eutectoide_s"])
    )
    assert fraccion_solo < 0.5 * fraccion


def test_la_ficha_declara_lo_que_el_enfriado_no_modela() -> None:
    """Reoxidacion, combustion del char y gradientes: no estan, y se dice."""

    texto = " ".join(mag.NO_MODELADO_AL_ENFRIAR).lower()
    assert "reoxidacion" in texto
    assert "char" in texto
    assert "gradientes" in texto


# --- Falsadores contra la corrida real -------------------------------------
#
# Observacion del usuario, textual:
#   «si le pasamos el iman sigue siendo magnetico, pero a medida que lo dejamos
#    mas tiempo en la mufla sigue perdiendo su capacidad magnetica»
#   «ya al final, si se pone el iman, pero se siente mas debil el magnetismo»


def _serie_de_la_corrida() -> list[tuple[float, float]]:
    """(t, magnetizacion) de la corrida vigente, medido sobre el lecho."""

    from interfaz.app import directorio_predeterminado

    raiz = Path(__file__).resolve().parents[1]
    directorio = directorio_predeterminado(raiz / "resultados")
    indice = serie_vigente(directorio) if Path(directorio).is_dir() else []
    if len(indice) < 5:
        pytest.skip(f"no hay corrida utilizable en {directorio}")

    salida: list[tuple[float, float]] = []
    for elemento in indice:
        campos = cargar_instantanea(elemento["ruta"])
        x, y, z = (np.asarray(campos[k], dtype=float) for k in ("x", "y", "z"))
        volumen = float(
            np.diff(x).mean() * np.diff(y).mean() * np.diff(z).mean()
        )
        # Todo el solido: el 12 % de la carga esta en celdas cortadas por la
        # frontera del lecho, que llevan otra etiqueta.
        solido = {
            fase: np.asarray(valores)
            for fase, valores in campos["solido"].items()
        }
        salida.append(
            (float(campos["t"]), mag.magnetizacion_Am2_kg(solido, volumen))
        )
    return sorted(salida)


def test_al_final_del_ensayo_el_aglomerado_todavia_responde_al_iman() -> None:
    """«Ya al final, si se pone el iman, pero se siente mas debil.»"""

    serie = _serie_de_la_corrida()
    t_final, m_final = serie[-1]
    m_inicial = serie[0][1]
    assert t_final > 600.0, "la corrida no llega al final del ensayo"
    assert m_final > 0.05 * m_inicial, (
        f"a t={t_final:.0f} s el modelo deja el aglomerado con "
        f"{m_final:.2f} A m2/kg frente a los {m_inicial:.2f} iniciales, es decir "
        f"el {100 * m_final / m_inicial:.1f} %: practicamente no magnetico. "
        "En el ensayo todavia se pega al iman."
    )


def test_el_magnetismo_decrece_con_el_tiempo_en_la_mufla() -> None:
    """«A medida que lo dejamos mas tiempo, sigue perdiendo capacidad magnetica.»

    Se compara el final contra el maximo, no contra t=0: al principio la
    reduccion de la titanohematita hacia magnetita puede subir un poco la
    magnetizacion antes de que empiece a bajar, y eso es fisica, no un fallo.
    Lo que la observacion exige es que a tiempos largos vaya a menos.
    """

    serie = _serie_de_la_corrida()
    valores = np.array([m for _, m in serie])
    tiempos = np.array([t for t, _ in serie])

    indice_maximo = int(np.argmax(valores))
    assert tiempos[indice_maximo] < 300.0, "el maximo deberia caer en el transitorio"
    assert valores[-1] < valores[indice_maximo], (
        "la magnetizacion final no es menor que el maximo: el ensayo dice que "
        "pierde capacidad magnetica con el tiempo"
    )
    # Y despues del maximo no puede volver a subir de forma apreciable.
    posteriores = valores[indice_maximo:]
    subidas = np.diff(posteriores)
    assert float(np.max(subidas, initial=0.0)) < 0.02 * valores[indice_maximo]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DISCREPANCIA ABIERTA, no un fallo de la prueba. El modelo se congela a "
        "los ~150 s: agotado el volatil no queda reductor, la gasificacion del "
        "char esta practicamente apagada (Da ~ 1,6e-17) y ninguna fase se mueve "
        "mas. El laboratorio, en cambio, ve que el magnetismo SIGUE bajando con "
        "el tiempo en la mufla. Se deja como xfail estricto para que avise el "
        "dia en que el modelo lo reproduzca."
    ),
)
def test_el_magnetismo_sigue_bajando_despues_de_los_150_segundos() -> None:
    """«A medida que lo dejamos mas tiempo, sigue perdiendo capacidad magnetica.»

    Lo que este falsador mide, y por que importa: si el magnetismo sigue
    cayendo despues de los 150 s, algo tiene que seguir reduciendo la magnetita,
    y el unico candidato con carbono disponible es la gasificacion del char
    (Boudouard), que regeneraria CO a partir del CO2. `k_boudouard` es uno de
    los parametros que la calibracion declara NO identificables con la curva de
    perdida de masa. La prueba del iman si podria identificarlo: seria el primer
    observable que restringe esa constante.

    No se ajusta aqui nada. Se deja constancia de que el dato existe y de que el
    modelo todavia no lo explica.
    """

    serie = _serie_de_la_corrida()
    tardios = [(t, m) for t, m in serie if t >= 150.0]
    assert len(tardios) >= 5, "hacen falta instantaneas tardias para medirlo"
    inicial = tardios[0][1]
    final = tardios[-1][1]
    assert final < 0.95 * inicial, (
        f"entre t={tardios[0][0]:.0f} s y t={tardios[-1][0]:.0f} s la "
        f"magnetizacion pasa de {inicial:.2f} a {final:.2f} A m2/kg: el modelo "
        "la deja congelada y el ensayo la ve seguir bajando"
    )
