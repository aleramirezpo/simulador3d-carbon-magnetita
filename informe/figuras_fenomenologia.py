"""Figuras del documento de fenomenología. Todas salen de la corrida.

Uso:  python informe/figuras_fenomenologia.py [directorio_de_resultados]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ.parent / "simulacion_v3" / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ.parent / "simulacion_v3" / "src"))

from informe.datos_fenomenologia import (  # noqa: E402
    diagnosticos,
    fronteras_co,
    serie_vigente,
)

SALIDA = RAIZ / "informe"

AZUL = "#1f5c8b"
ROJO = "#b3402f"
VERDE = "#5e6b3a"
NARANJA = "#c0842f"
GRIS = "#5a6570"
MORADO = "#6b4a7a"
CIAN = "#2f8b8b"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# Cronología cualitativa del laboratorio. Son fechas, no ajustes.
HITOS = ((30.0, "30 s\npolvo suelto"), (90.0, "90 s\nhincha"), (120.0, "120 s\nformado"))


def _guardar(fig, nombre):
    ruta = SALIDA / nombre
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  {nombre}")


def _marcar_hitos(eje, con_texto=True, y=0.97):
    # Los tres hitos caen en 30, 90 y 120 s: muy juntos en un eje de 720 s. Se
    # escalonan en altura y se anclan por el lado que no invade al vecino.
    posiciones = ((y, "right"), (y - 0.13, "right"), (y, "left"))
    for (t, etiqueta), (altura, lado) in zip(HITOS, posiciones):
        eje.axvline(t, color=GRIS, lw=0.7, ls=":", zorder=0)
        if con_texto:
            eje.annotate(f" {etiqueta} ", xy=(t, altura),
                         xycoords=("data", "axes fraction"),
                         fontsize=6.5, color=GRIS, ha=lado, va="top")


def _tabla_experimental():
    from datos_experimentales import TABLA_PERDIDA_MASA
    return np.asarray(TABLA_PERDIDA_MASA, dtype=float)


# --------------------------------------------------------------------- calor

def figura_termica(s):
    t = np.asarray(s["t"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(t, np.asarray(s["T_mufla"]) - 273.15, color=ROJO, lw=1.6, label="mufla")
    a1.plot(t, np.asarray(s["T_fondo_crisol"]) - 273.15, color=NARANJA, lw=1.4,
            label="fondo del crisol")
    a1.plot(t, np.asarray(s["T_lecho"]) - 273.15, color=AZUL, lw=1.6, label="lecho (media)")
    a1.fill_between(t, np.asarray(s["T_lecho_min"]) - 273.15,
                    np.asarray(s["T_lecho_max"]) - 273.15,
                    color=AZUL, alpha=0.18, lw=0, label="lecho (mín-máx)")
    a1.axhspan(350, 500, color=VERDE, alpha=0.13, lw=0, zorder=0)
    a1.annotate("ventana termoplástica 350–500 °C", xy=(430, 425), fontsize=6.5,
                color=VERDE, ha="left", va="center")
    a1.set_ylabel("temperatura [°C]")
    a1.legend(loc="lower right", ncol=2)
    _marcar_hitos(a1, y=0.62)
    a1.set_title("Historia térmica: la mufla radia, el crisol conduce, el lecho sigue")

    a2.plot(t, np.asarray(s["T_fondo_crisol"]) - np.asarray(s["T_lecho_base"]),
            color=NARANJA, lw=1.4, label="fondo del crisol − base del lecho")
    a2.plot(t, np.asarray(s["T_lecho_base"]) - np.asarray(s["T_lecho_techo"]),
            color=MORADO, lw=1.4, label="base − techo del lecho")
    a2.axhline(0.0, color=GRIS, lw=0.7)
    a2.set_ylabel("salto [K]")
    a2.set_xlabel("tiempo [s]")
    a2.set_xlim(0, t[-1])
    a2.legend(loc="upper right")
    _marcar_hitos(a2, con_texto=False)
    a2.set_title("El calor entra por abajo: los dos saltos son positivos siempre",
                 fontsize=9)
    _guardar(fig, "fig_fen_termico.pdf")


# --------------------------------------------------------------------- fases

def figura_fases(s, fases):
    t = np.asarray(s["t"])
    estilo = {
        "volatil": ("volátil del carbón", ROJO, "-"),
        "C": ("carbono fijo → char", GRIS, "-"),
        "Fe3O4": ("magnetita Fe$_3$O$_4$", AZUL, "-"),
        "Fe2O3": ("hematita Fe$_2$O$_3$", NARANJA, "-"),
        "FeO": ("wüstita FeO", VERDE, "-"),
        "Fe": ("hierro metálico", MORADO, "-"),
        "FeTiO3": ("ilmenita FeTiO$_3$", CIAN, "--"),
        "SiO2": ("cuarzo SiO$_2$", "#8b6b2f", "--"),
        "ceniza": ("cenizas", "#9a9a9a", "--"),
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for fase, (etiqueta, color, ls) in estilo.items():
        clave = f"m_{fase}"
        if clave not in s:
            continue
        y = 1000.0 * np.asarray(s[clave])
        eje = a1 if y.max() > 60.0 else a2
        eje.plot(t, y, color=color, ls=ls, lw=1.5, label=etiqueta)
    for eje, titulo in ((a1, "Fases mayoritarias"), (a2, "Fases minoritarias")):
        eje.set_xlabel("tiempo [s]")
        eje.set_ylabel("masa [mg]")
        eje.set_xlim(0, t[-1])
        eje.legend(loc="center right")
        eje.set_title(titulo, fontsize=9)
        _marcar_hitos(eje, con_texto=False)
    a2.annotate("la ilmenita y el cuarzo\nno se mueven en 720 s",
                xy=(210, 43.25), xytext=(60, 30), fontsize=7, color=CIAN,
                arrowprops=dict(arrowstyle="->", color=CIAN, lw=0.8))
    a2.legend(loc="center right", framealpha=0.92)
    fig.suptitle("Qué cambia de fase y qué no", fontsize=10)
    _guardar(fig, "fig_fen_fases.pdf")


def figura_cascada_hierro(s):
    t = np.asarray(s["t"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for fase, etiqueta, color in (
        ("Fe2O3", "hematita Fe$_2$O$_3$", NARANJA),
        ("Fe3O4", "magnetita Fe$_3$O$_4$", AZUL),
        ("FeO", "wüstita FeO", VERDE),
        ("FeTiO3", "ilmenita FeTiO$_3$", CIAN),
    ):
        a1.plot(t, 1000.0 * np.asarray(s[f"m_{fase}"]), color=color, lw=1.6, label=etiqueta)
    a1.set_ylabel("masa [mg]")
    a1.set_xlabel("tiempo [s]")
    a1.set_xlim(0, t[-1])
    a1.legend(loc="center right")
    a1.set_title("La cascada se detiene en wüstita", fontsize=9)
    _marcar_hitos(a1, con_texto=False)

    a2.plot(t, 1000.0 * np.asarray(s["m_Fe"]), color=MORADO, lw=1.6)
    a2.set_ylabel("hierro metálico [mg]")
    a2.set_xlabel("tiempo [s]")
    a2.set_xlim(0, t[-1])
    a2.set_title("El hierro metálico se queda en trazas", fontsize=9)
    total_fe = 1000.0 * (np.asarray(s["m_Fe"])[-1])
    a2.annotate(f"{total_fe:.2f} mg al final", xy=(0.96, 0.12), xycoords="axes fraction",
                ha="right", fontsize=7.5, color=MORADO)
    _marcar_hitos(a2, con_texto=False)
    fig.suptitle("Reducción del hierro: hasta dónde llega", fontsize=10)
    _guardar(fig, "fig_fen_hierro.pdf")


# --------------------------------------------------------- pérdida de masa

def figura_perdida(s):
    t = np.asarray(s["t"])
    perdida = np.asarray(s["perdida_pct"])
    fig, eje = plt.subplots(figsize=(6.4, 3.6))
    eje.plot(t, perdida, color=AZUL, lw=1.8, label="modelo 3-D", zorder=3)
    try:
        tabla = _tabla_experimental()
        eje.plot(tabla[:, 0], tabla[:, 3], "o", color=ROJO, ms=5,
                 label="medido (8 puntos, Tabla 3)", zorder=4)
    except Exception:
        pass
    eje.set_xlabel("tiempo [s]")
    eje.set_ylabel("pérdida de masa sólida [%]")
    eje.set_xlim(0, t[-1])
    eje.legend(loc="lower right")
    _marcar_hitos(eje, y=0.55)
    meseta = perdida[t >= 300.0]
    eje.annotate(
        f"meseta: {meseta.min():.2f}–{meseta.max():.2f} %\n"
        f"(el volátil ya se agotó)",
        xy=(600, perdida[-1]), xytext=(430, perdida[-1] - 9), fontsize=7.5, color=GRIS,
        arrowprops=dict(arrowstyle="->", color=GRIS, lw=0.8))
    eje.set_title("Casi toda la pérdida ocurre entre 60 y 120 s")
    _guardar(fig, "fig_fen_perdida.pdf")


# --------------------------------------------------------------------- gases

def figura_gases(s, especies):
    t = np.asarray(s["t"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    colores = {"CO": AZUL, "CO2": ROJO, "H2": VERDE, "H2O": CIAN,
               "CH4": NARANJA, "N2": GRIS, "O2": MORADO}
    for especie in especies:
        y = np.asarray(s[f"c_{especie}"])
        if y.max() <= 0.0:
            continue
        a1.plot(t, y, color=colores.get(especie, GRIS), lw=1.4,
                label=especie.replace("2", "$_2$").replace("4", "$_4$"))
    a1.set_yscale("log")
    a1.set_xlabel("tiempo [s]")
    a1.set_ylabel("concentración media [mol/m$^3$]")
    a1.set_xlim(0, t[-1])
    a1.legend(loc="lower right", ncol=2)
    a1.set_title("Gases en el interior", fontsize=9)
    _marcar_hitos(a1, con_texto=False)

    a2.plot(t, np.asarray(s["CO_sobre_COx"]), color=AZUL, lw=1.8, zorder=5,
            label="gas de la corrida")
    fronteras = fronteras_co()
    estilo_frontera = (
        # La frontera de la magnetita cae casi sobre el cero de este eje: su
        # rótulo se ancla a la izquierda para no chocar con la leyenda.
        ("magnetita", "Fe$_3$O$_4$→FeO", VERDE, 0.02, "left"),
        ("wustita", "FeO→Fe", NARANJA, 0.98, "right"),
        ("ilmenita", "FeTiO$_3$→Fe", CIAN, 0.98, "right"),
    )
    for clave, etiqueta, color, x, lado in estilo_frontera:
        umbral = fronteras[clave]
        a2.axhline(umbral, color=color, lw=1.1, ls="--")
        a2.annotate(f"{etiqueta}  {100 * umbral:.2f} %",
                    xy=(x, umbral), xycoords=("axes fraction", "data"),
                    ha=lado, va="bottom", fontsize=6.5, color=color)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("tiempo [s]")
    a2.set_ylabel(r"CO/(CO+CO$_2$)")
    a2.set_xlim(0, t[-1])
    a2.legend(loc="lower right", fontsize=6.5)
    a2.set_title("El gas queda entre dos fronteras", fontsize=9)
    _marcar_hitos(a2, con_texto=False)
    _guardar(fig, "fig_fen_gases.pdf")


# ---------------------------------------------------------------- aglomerado

def figura_aglomerado(s):
    t = np.asarray(s["t"])
    fig, ejes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    fig.subplots_adjust(hspace=0.42, wspace=0.30)
    (a1, a2), (a3, a4) = ejes

    a1.plot(t, np.asarray(s["cohesion_max"]), color=ROJO, lw=1.6, label="máxima")
    a1.plot(t, np.asarray(s["cohesion_media"]), color=AZUL, lw=1.4, label="media del lecho")
    a1.axhline(0.5, color=GRIS, lw=0.8, ls="--")
    a1.annotate("umbral de aglomerado", xy=(430, 0.53), fontsize=6.5, color=GRIS)
    a1.set_ylabel("cohesión [-]")
    a1.legend(loc="lower right")
    a1.set_title("Cohesión: el polvo se vuelve cuerpo", fontsize=9)

    a2.plot(t, np.asarray(s["hinchamiento_medio"]), color=VERDE, lw=1.6)
    a2.set_ylabel("factor de hinchamiento [-]")
    a2.set_title("Hinchamiento (irreversible)", fontsize=9)

    a3.plot(t, np.asarray(s["eps_media"]), color=MORADO, lw=1.6)
    a3.set_ylabel("porosidad del lecho [-]")
    a3.set_xlabel("tiempo [s]")
    a3.set_title("El hinchamiento entra como porosidad", fontsize=9)

    a4.plot(t, np.asarray(s["D_huella_mm"]), color=AZUL, lw=1.6, label="Ø de la huella")
    a4.plot(t, np.asarray(s["D_aglomerado_mm"]), color=NARANJA, lw=1.4,
            label="Ø de esfera equivalente")
    a4.plot(t, np.asarray(s["altura_aglomerado_mm"]), color=VERDE, lw=1.4, label="altura")
    a4.plot(t, np.asarray(s["altura_libre_mm"]), color=VERDE, lw=1.2, ls="--",
            label="altura libre predicha")
    a4.set_ylabel("dimensión [mm]")
    a4.set_xlabel("tiempo [s]")
    a4.legend(loc="center right")
    a4.set_title("Tamaño del cuerpo formado", fontsize=9)

    for eje in (a1, a2, a3, a4):
        eje.set_xlim(0, t[-1])
        _marcar_hitos(eje, con_texto=False)
    fig.suptitle("Formación del aglomerado", fontsize=10, y=0.98)
    _guardar(fig, "fig_fen_aglomerado.pdf")


# ----------------------------------------------------------- adimensionales

def figura_adimensionales(s, Ra_critico):
    t = np.asarray(s["t"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for clave, etiqueta, color in (
        ("Re_p", "Re de partícula", AZUL),
        ("Pe_termico", "Pe térmico", ROJO),
        ("Pe_masico", "Pe másico", VERDE),
        ("Da", "Da", MORADO),
    ):
        y = np.abs(np.asarray(s[clave]))
        y[y <= 0] = np.nan
        a1.plot(t, y, color=color, lw=1.4, label=etiqueta)
    a1.axhline(1.0, color=GRIS, lw=0.9, ls="--")
    a1.annotate("advección = difusión", xy=(0.98, 1.4),
                xycoords=("axes fraction", "data"), ha="right",
                fontsize=6.5, color=GRIS)
    a1.set_yscale("log")
    a1.set_xlabel("tiempo [s]")
    a1.set_ylabel("valor [-]")
    a1.set_xlim(0, t[-1])
    a1.legend(loc="lower left", ncol=2)
    a1.set_title("Sólo el estallido de gas cruza el 1", fontsize=9)

    a2.plot(t, np.asarray(s["Ra"]), color=NARANJA, lw=1.6, label="Ra")
    if np.isfinite(Ra_critico):
        a2.axhline(Ra_critico, color=GRIS, lw=0.9, ls="--",
                   label=f"Ra crítico = {Ra_critico:.0f}")
    a2.set_xlabel("tiempo [s]")
    a2.set_ylabel("Rayleigh [-]")
    a2.set_xlim(0, t[-1])
    a2.legend(loc="lower right")
    a2.set_title("Rayleigh: el único que roza su umbral", fontsize=9)
    for eje in (a1, a2):
        _marcar_hitos(eje, con_texto=False)
    fig.suptitle("Números adimensionales del solucionador", fontsize=10)
    _guardar(fig, "fig_fen_adimensionales.pdf")


# ------------------------------------------------------------------- campos

def figura_campos(instantaneas):
    """Corte vertical con el crisol incluido.

    Enmascarar las paredes dejaba sólo la cavidad y escondía justo lo que
    explica la historia térmica: el calor entra por el cuerpo del crisol. Aquí
    se oculta únicamente el exterior del dominio y se marca el contorno del
    lecho para poder situarlo.
    """
    tiempos = [30.0, 90.0, 120.0, 720.0]
    fig, ejes = plt.subplots(2, len(tiempos), figsize=(7.6, 4.6))
    fig.subplots_adjust(wspace=0.35, hspace=0.10)
    for columna, objetivo in enumerate(tiempos):
        t, campos = min(instantaneas, key=lambda e: abs(e[0] - objetivo))
        etiquetas = np.asarray(campos["etiquetas"])
        T = np.asarray(campos["T"]) - 273.15
        CO = np.asarray(campos["c_especies"]["CO"])
        j = T.shape[1] // 2
        x = np.asarray(campos["x"])
        z = np.asarray(campos["z"])
        exterior = etiquetas == 0
        lecho = (etiquetas == 3).astype(float)
        # El solucionador no transporta especies dentro del metal: lo que queda
        # en esas celdas no es una concentración, son residuos del término
        # fuente. Pintarlos hacía que la escala del CO llegara a 665 mol/m³,
        # sesenta veces la densidad molar del gas a 1 atm y 900 °C.
        sin_gas = ~np.isin(etiquetas, (3, 4))

        for fila, (campo, mapa, titulo, mascara) in enumerate((
            (T, "inferno", "T [°C]", exterior),
            (CO, "viridis", "CO [mol/m³]", sin_gas),
        )):
            eje = ejes[fila, columna]
            corte = np.ma.masked_array(campo[:, j, :], mask=mascara[:, j, :])
            imagen = eje.pcolormesh(x, z, corte.T, cmap=mapa, shading="nearest")
            eje.contour(x, z, lecho[:, j, :].T, levels=[0.5],
                        colors="#59f0ff", linewidths=0.8)
            eje.set_aspect("equal")
            eje.set_xticks([])
            eje.grid(False)
            if columna == 0:
                eje.set_ylabel(f"{titulo}\nz [mm]", fontsize=8)
            else:
                eje.set_yticks([])
            if fila == 0:
                eje.set_title(f"t = {t:.0f} s", fontsize=9)
            fig.colorbar(imagen, ax=eje, fraction=0.05, pad=0.04).ax.tick_params(labelsize=6)
    fig.suptitle("Corte vertical por el eje: temperatura y CO "
                 "(contorno azul = lecho)", fontsize=10)
    _guardar(fig, "fig_fen_campos.pdf")


# ---------------------------------------------------------------- magnetismo

def figura_magnetismo(s, enfriamiento):
    """La prueba del iman y el enfriado, que es lo que decide como leerla.

    Tres paneles y no dos: la magnetizacion especifica y el momento total dicen
    cosas distintas y hay que poder verlas por separado. El lecho pierde una
    cuarta parte de su masa al devolatilizarse, y eso sube la magnetizacion
    especifica sin que cambie ninguna fase de hierro; el momento no se entera de
    esa dilucion y sigue solo a la quimica. Un eje gemelo los superponia, pero
    con dos escalas distintas en el mismo dibujo se leia mal y ademas chocaba
    con el panel contiguo.
    """
    t = np.asarray(s["t"])
    m = np.asarray(s["magnetizacion_Am2_kg"])
    m_lenta = np.asarray(s["magnetizacion_lenta_Am2_kg"])
    momento = np.asarray(s["momento_magnetico_Am2"]) * 1e3

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.8, 3.1))

    a1.fill_between(t, m, m_lenta, color=AZUL, alpha=0.15,
                    label="banda segun el enfriado")
    a1.plot(t, m, color=AZUL, lw=1.8, label="wustita conservada", zorder=3)
    a1.plot(t, m_lenta, color=ROJO, lw=1.2, ls="--",
            label="wustita descompuesta", zorder=3)
    a1.set_xlabel("tiempo en la mufla [s]")
    a1.set_ylabel("magnetizacion en frio [A m$^2$/kg]")
    a1.set_xlim(0, t[-1])
    a1.set_ylim(bottom=0)
    a1.legend(loc="lower right", fontsize=6.5)
    _marcar_hitos(a1, y=0.30)
    a1.set_title("Sigue respondiendo al iman", fontsize=9)

    a2.plot(t, momento, color=MORADO, lw=1.8)
    a2.set_xlabel("tiempo en la mufla [s]")
    a2.set_ylabel("momento total [mA m$^2$]")
    a2.set_xlim(0, t[-1])
    a2.set_ylim(bottom=0)
    _marcar_hitos(a2, y=0.30)
    a2.set_title("Sin la dilucion: solo la quimica", fontsize=9)

    T = np.asarray(enfriamiento["temperatura_K"]) - 273.15
    tiempo = np.asarray(enfriamiento["tiempo_s"])
    a3.plot(tiempo, T, color=NARANJA, lw=1.8)
    a3.axhline(570.0, color=GRIS, lw=0.9, ls=":")
    a3.annotate("570 °C · eutectoide", xy=(tiempo[-1], 570.0),
                xytext=(0.30 * tiempo[-1], 620.0), fontsize=7, color=GRIS)
    a3.set_xlabel("tiempo fuera de la mufla [s]")
    a3.set_ylabel("temperatura del crisol [°C]")
    a3.set_xlim(0, tiempo[-1])
    a3.set_title(
        f"{enfriamiento['velocidad_en_eutectoide_C_min']:.0f} °C/min al cruzarlo "
        f"(umbral {enfriamiento['umbral_supresion_C_min']:.0f})", fontsize=9)
    _guardar(fig, "fig_fen_magnetismo.pdf")


def figura_fases_tres_estados(datos):
    """La misma fase en los tres estados en que puede encontrarse el sólido.

    El cuadro de enfriado da esto mismo en nueve instantes; la figura lo da en
    los 145 y deja ver una cosa que la tabla esconde: las tres curvas nacen
    juntas y sólo se separan cuando aparece wüstita, porque **todo** lo que
    distingue a las tres rutas actúa sobre la wüstita. Sin wüstita, enfriar como
    se quiera da el mismo sólido.
    """
    from fisica import magnetismo as mag

    s = datos["serie"]
    volumen = float(np.prod(datos["paso_malla_mm"])) * 1.0e-9
    fases = ("volatil", "C", "ceniza", "Fe2O3", "Fe3O4", "FeO", "Fe", "FeTiO3")

    t, estados = [], {"mufla": [], "tapa": [], "sin": []}
    # La ventana del eutectoide sólo depende del cuerpo que se enfría, no de la
    # instantánea: se calcula una vez y no 145 veces.
    for instante, campos in datos["instantaneas"]:
        i = _indice_mas_cercano_t(s, float(instante))
        solido = {f: np.asarray(v) for f, v in campos["solido"].items()}
        total_mg = 1000.0 * sum(s[f"m_{f}"][i] for f in fases if f"m_{f}" in s)
        n_FeO = mag.fases_tras_enfriar(solido, volumen, 0.0)["FeO"]
        t.append(float(instante))
        estados["mufla"].append(
            (mag.composicion_tras_ruta(solido, volumen, f_eutectoide=0.0), total_mg))
        for etiqueta, ruta in (("tapa", mag.RUTA_CON_TAPA), ("sin", mag.RUTA_SIN_TAPA)):
            p = mag.parametros_de_ruta(ruta, n_FeO, float(s["T_lecho"][i]))
            estados[etiqueta].append((mag.composicion_tras_ruta(
                solido, volumen,
                f_eutectoide=p["f_eutectoide"],
                f_reoxidacion_gas=p["f_reoxidacion_gas"],
                f_oxidacion_aire=p["f_oxidacion_aire"]), total_mg))

    t = np.asarray(t)
    estilo = (
        ("mufla", "en la mufla (900 °C)", GRIS, ":"),
        ("tapa", "recuperado con tapa", AZUL, "-"),
        ("sin", "recuperado sin tapa", ROJO, "--"),
    )
    paneles = (
        ("Fe3O4", "magnetita Fe$_3$O$_4$"),
        ("FeO", "wüstita FeO"),
        ("Fe", "hierro metálico Fe"),
        ("Fe2O3", "hematita Fe$_2$O$_3$"),
    )
    fig, ejes = plt.subplots(1, 4, figsize=(10.4, 2.9))
    for eje, (fase, titulo) in zip(ejes, paneles):
        for clave, etiqueta, color, ls in estilo:
            y = np.asarray([100.0 * c[f"{fase}_mg"] / m if m > 0 else np.nan
                            for c, m in estados[clave]])
            eje.plot(t, y, color=color, ls=ls, lw=1.6, label=etiqueta)
        eje.set_xlabel("tiempo en la mufla [s]")
        eje.set_xlim(0, t[-1])
        eje.set_ylim(bottom=0)
        eje.set_title(titulo, fontsize=9)
        _marcar_hitos(eje, con_texto=False)
    ejes[0].set_ylabel("% en masa del sólido")
    ejes[0].legend(loc="lower left", fontsize=6.5)
    fig.suptitle(
        "Las tres curvas se separan sólo donde hay wüstita que transformar",
        fontsize=10, y=1.02)
    _guardar(fig, "fig_fen_tres_estados.pdf")


def _indice_mas_cercano_t(s, objetivo):
    return int(np.argmin(np.abs(np.asarray(s["t"], dtype=float) - float(objetivo))))


def figura_calentamiento(s):
    """Cuánto tarda la carga en enterarse de que la mufla está a 900 °C.

    La figura térmica general enseña la cadena mufla → crisol → lecho; ésta
    responde a una pregunta distinta y concreta: **cuándo llega el lecho a los
    900 °C**. Se marca la fracción recorrida del salto térmico, que es lo que
    permite decir «a los 30 s no ha pasado nada» sin apelar a la química.
    """
    t = np.asarray(s["t"])
    T = np.asarray(s["T_lecho"]) - 273.15
    T_mufla = np.asarray(s["T_mufla"]) - 273.15
    T0, T_obj = float(T[0]), float(T_mufla[-1])
    salto = T_obj - T0

    def cuando(fraccion):
        """Instante a partir del cual el lecho ya NO baja de ese umbral.

        No vale el primer cruce: al arrancar, mufla y lecho están los dos a
        \\SI{25}{\\celsius} y cualquier criterio de cercanía se cumple de forma
        trivial. Lo que interesa es el último cruce.
        """
        objetivo = T0 + fraccion * salto
        pendientes = np.nonzero(T < objetivo)[0]
        if not pendientes.size:
            return float(t[0]), objetivo
        i = int(pendientes[-1]) + 1
        return (float(t[i]), objetivo) if i < t.size else (np.nan, objetivo)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.2),
                                 gridspec_kw={"wspace": 0.30})

    a1.plot(t, T_mufla, color=ROJO, lw=1.3, ls="--", label="mufla")
    a1.fill_between(t, np.asarray(s["T_lecho_min"]) - 273.15,
                    np.asarray(s["T_lecho_max"]) - 273.15,
                    color=AZUL, alpha=0.16, lw=0, label="lecho (mín–máx)")
    a1.plot(t, T, color=AZUL, lw=1.9, label="lecho (media)", zorder=3)
    # Los tres puntos caen sobre la curva a alturas muy distintas: las etiquetas
    # van al margen derecho, escalonadas, cada una con su guía.
    for fraccion, texto, altura in ((0.5, "50 %", 0.28),
                                    (0.9, "90 %", 0.50),
                                    (0.99, "99 %", 0.72)):
        t_h, T_h = cuando(fraccion)
        if not np.isfinite(t_h):
            continue
        a1.plot([t_h], [T_h], "o", ms=4.5, color=MORADO, zorder=5)
        a1.annotate(f"{texto} del salto\na los {t_h:.0f} s", xy=(t_h, T_h),
                    xytext=(0.985, altura), textcoords="axes fraction",
                    fontsize=6.5, color=MORADO, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color=MORADO, lw=0.6,
                                    shrinkA=0, shrinkB=3))
    a1.set_xlabel("tiempo [s]")
    a1.set_ylabel("temperatura [°C]")
    a1.set_xlim(0, t[-1])
    a1.legend(loc="lower right")
    _marcar_hitos(a1, con_texto=False)
    a1.set_title(f"El lecho acaba en {T[-1]:.0f} °C", fontsize=9)

    # Lo que falta para la mufla, en escala logarítmica: una recta querría decir
    # una sola constante de tiempo, y no lo es --- la reacción y el volátil que
    # sale se llevan calor mientras duran.
    falta = np.maximum(T_mufla - T, 1e-2)
    a2.semilogy(t, falta, color=AZUL, lw=1.8)
    for grados in (100.0, 10.0, 1.0):
        pendientes = np.nonzero(falta > grados)[0]
        if not pendientes.size or int(pendientes[-1]) + 1 >= t.size:
            continue
        a2.axhline(grados, color=GRIS, lw=0.6, ls=":")
        a2.annotate(f"{grados:g} K a los {t[int(pendientes[-1]) + 1]:.0f} s",
                    xy=(0.98 * t[-1], grados * 1.25), fontsize=6.5,
                    color=GRIS, ha="right")
    a2.set_xlabel("tiempo [s]")
    a2.set_ylabel("mufla − lecho [K]")
    a2.set_xlim(0, t[-1])
    _marcar_hitos(a2, con_texto=False)
    a2.set_title("Lo que falta para los 900 °C", fontsize=9)
    fig.suptitle("Transmisión del calor a la carga", fontsize=10)
    _guardar(fig, "fig_fen_calentamiento.pdf")


def main() -> int:
    directorio = sys.argv[1] if len(sys.argv) > 1 else None
    datos = diagnosticos(directorio)
    s = datos["serie"]
    print("figuras de fenomenología:")
    figura_termica(s)
    figura_calentamiento(s)
    figura_fases(s, datos["fases"])
    figura_fases_tres_estados(datos)
    figura_cascada_hierro(s)
    figura_perdida(s)
    figura_gases(s, datos["especies"])
    figura_aglomerado(s)
    figura_adimensionales(s, datos["Ra_critico"])
    figura_magnetismo(s, datos["enfriamiento"])
    figura_campos(datos["instantaneas"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
