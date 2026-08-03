"""Figuras del informe. Se generan de la corrida real, no se dibujan a mano.

Uso:  python informe/figuras_informe.py [directorio_de_resultados]
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent / "simulacion_v3" / "src"))

from fisica.adaptador_v3 import MASAS_MOLARES_SOLIDO_KG_MOL  # noqa: E402
from fisica.fases_visuales import FASES, MAPA_CAMPOS_SOLIDOS, volumenes_molares_cm3_mol  # noqa: E402
from nucleo.salida import cargar_instantanea  # noqa: E402

SALIDA = RAIZ / "informe"
LECHO, GAS, PARED, TAPA = 3, 4, 1, 2

AZUL = "#1f5c8b"
ROJO = "#b3402f"
VERDE = "#5e6b3a"
NARANJA = "#c0842f"
GRIS = "#5a6570"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------- utilidades

def serie_vigente(directorio):
    rutas = sorted(glob.glob(f"{directorio}/*.npz"))
    if not rutas:
        raise SystemExit(f"no hay instantáneas en {directorio}")
    cargadas = [(r, cargar_instantanea(r)) for r in rutas]
    inicios = [r for r, c in cargadas if abs(float(c["t"])) <= 1e-12]
    if inicios:
        corte = max(os.stat(r).st_mtime_ns for r in inicios)
        cargadas = [(r, c) for r, c in cargadas if os.stat(r).st_mtime_ns >= corte]
    return sorted(((float(c["t"]), c) for _, c in cargadas), key=lambda e: e[0])


def masa_solida_g(campos, volumen):
    return 1000.0 * sum(
        float(np.sum(np.asarray(campos["solido"][f]))) * volumen * M
        for f, M in MASAS_MOLARES_SOLIDO_KG_MOL.items()
    )


def volumen_celda(campos):
    return (
        (float(campos["x"][1]) - float(campos["x"][0]))
        * (float(campos["y"][1]) - float(campos["y"][0]))
        * (float(campos["z"][1]) - float(campos["z"][0]))
    ) * 1.0e-9


def _estilo(ax, xlabel="", ylabel=""):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, bbox_inches="tight")
    plt.close(fig)


def fracciones_solidas(campos):
    """% de volumen de cada fase sobre el sólido del lecho."""
    vm = volumenes_molares_cm3_mol()
    lecho = np.asarray(campos["etiquetas"]) == LECHO
    aporte, total = {}, 0.0
    for campo, clave in MAPA_CAMPOS_SOLIDOS.items():
        if clave is None or clave not in vm:
            continue
        v = float(np.sum(np.asarray(campos["solido"][campo])[lecho])) * vm[clave]
        if v > 0:
            aporte[clave] = aporte.get(clave, 0.0) + v
            total += v
    return {k: 100.0 * v / total for k, v in aporte.items()} if total else {}


# ------------------------------------------------------------------ figuras

def fig_cronologia(serie):
    """Temperatura, hinchamiento y cohesión con las tres observaciones."""
    t = np.array([p[0] for p in serie])
    T, hinch, coh = [], [], []
    for _, c in serie:
        lecho = np.asarray(c["etiquetas"]) == LECHO
        T.append(float(np.asarray(c["T"])[lecho].mean()))
        hinch.append(float(np.asarray(c["hinchamiento"])[lecho].mean()))
        coh.append(100.0 * float(np.mean(np.asarray(c["cohesion"])[lecho] > 0.5)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 5.0), sharex=True)
    limite = 300.0

    ax1.axhspan(623.15, 773.15, color=VERDE, alpha=0.15, zorder=0)
    ax1.plot(t, T, color=AZUL, linewidth=2.0, zorder=3)
    ax1.set_ylim(280, 1200)
    ax1.annotate("ventana termoplástica\n350–500 °C", xy=(0.62, 0.42),
                 xycoords="axes fraction", fontsize=8, color="#3f4a24", ha="left")
    _estilo(ax1, "", "T media del lecho (K)")

    ax2.plot(t, hinch, color=VERDE, linewidth=2.0, label="hinchamiento (×)")
    ax2.set_ylim(0.95, 1.75)
    _estilo(ax2, "tiempo (s)", "hinchamiento (×)")
    gemelo = ax2.twinx()
    gemelo.plot(t, coh, color=ROJO, linewidth=2.0, linestyle="--",
                label="lecho cohesionado (%)")
    gemelo.set_ylabel("lecho cohesionado (%)", color=ROJO)
    gemelo.tick_params(axis="y", colors=ROJO)
    gemelo.set_ylim(-3, 105)
    for lado in ("top",):
        gemelo.spines[lado].set_visible(False)

    for instante, texto, altura in ((30, "nada", 0.90), (90, "crece y\nse hincha", 0.62),
                                    (120, "formado", 0.34)):
        if instante <= limite:
            ax2.axvline(instante, color="#666", linewidth=0.9, linestyle=":", zorder=0)
            ax2.annotate(texto, xy=(instante / limite, altura), xycoords="axes fraction",
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=7.5, color="#333", va="center")
    lineas = ax2.get_lines()[:1] + gemelo.get_lines()[:1]
    ax2.legend(lineas, [l.get_label() for l in lineas], frameon=False,
               loc="center right", fontsize=7.5)
    ax2.set_xlim(0, limite)
    fig.tight_layout()
    _guardar(fig, "fig_cronologia.pdf")


def fig_perdida_masa(serie, volumen, m0):
    from datos_experimentales import ANALISIS_PROXIMO_CARBON, TABLA_PERDIDA_MASA

    tabla = np.asarray(TABLA_PERDIDA_MASA, dtype=float)
    t = np.array([p[0] for p in serie])
    perdida = np.array([100.0 * (m0 - masa_solida_g(c, volumen)) / m0 for _, c in serie])

    volatil_max = 0.75 * float(ANALISIS_PROXIMO_CARBON["materia_volatil_pct"])
    humedad = 0.75 * float(ANALISIS_PROXIMO_CARBON["humedad_residual_pct"])

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.3, 5.0), sharex=True,
                                  gridspec_kw={"height_ratios": [2.6, 1]})
    ax.axhline(volatil_max + humedad, color=GRIS, linewidth=1.0, linestyle="--")
    ax.annotate(f"volátil + humedad del análisis próximo = {volatil_max + humedad:.1f} %",
                xy=(340, volatil_max + humedad + 0.7), fontsize=7.5, color=GRIS)
    ax.plot(t, perdida, color=AZUL, linewidth=2.0, label="modelo 3-D")
    sd = tabla[:, 4]
    ax.errorbar(tabla[:, 0], tabla[:, 3], yerr=np.where(np.isnan(sd), 0.0, sd),
                fmt="o", color=ROJO, markersize=6, capsize=3, linewidth=1.2,
                label="medido (Tabla 3)", zorder=5)
    _estilo(ax, "", "pérdida de masa (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 740)

    residuos = []
    for t_exp, _, _, pct, _ in tabla:
        i = int(np.argmin(np.abs(t - t_exp)))
        residuos.append(perdida[i] - pct)
    axr.axhline(0, color="#888", linewidth=0.9)
    axr.bar(tabla[:, 0], residuos, width=18, color=NARANJA, edgecolor="none")
    _estilo(axr, "tiempo (s)", "modelo − medido\n(puntos %)")
    fig.tight_layout()
    _guardar(fig, "fig_perdida_masa.pdf")


def fig_fases_solidas(serie):
    """Composición del sólido del lecho, % de volumen, en cada instante."""
    t = np.array([p[0] for p in serie])
    claves = ["char", "carbon", "cenizas", "Fe3O4", "FeO", "Fe2O3", "FeTiO3", "SiO2", "Fe", "TiO2"]
    datos = {k: [] for k in claves}
    for _, c in serie:
        fr = fracciones_solidas(c)
        for k in claves:
            datos[k].append(fr.get(k, 0.0))

    presentes = [k for k in claves if max(datos[k]) > 0.02]
    # Paleta LEGIBLE, no la mineral real: en un apilado, char, carbón, magnetita
    # e ilmenita son todos casi negros y no se distinguirían. La paleta con el
    # color real de cada mineral se usa en la vista 3-D, donde cada grano se
    # dibuja aparte y ahí sí tiene sentido.
    legible = {"char": "#2f3640", "carbon": "#8c7b5c", "cenizas": "#d9d2c5",
               "Fe3O4": "#3d6b8f", "FeO": "#c0764a", "Fe2O3": "#a8342a",
               "FeTiO3": "#6d5b8e", "SiO2": "#b9c4a8", "Fe": "#9aa5ad",
               "TiO2": "#d2a44e"}
    colores = [legible[k] for k in presentes]
    etiquetas = [FASES[k]["nombre"] for k in presentes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 5.9))
    ax1.stackplot(t, *[datos[k] for k in presentes], colors=colores,
                  labels=etiquetas, edgecolor="white", linewidth=0.2)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, 740)
    _estilo(ax1, "", "% del volumen de sólido")
    ax1.legend(frameon=False, fontsize=7, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.10))
    ax1.set_title("composición del lecho (apilada)", fontsize=9)

    minoritarias = [k for k in presentes if max(datos[k]) < 15.0]
    for k in minoritarias:
        ax2.plot(t, datos[k], linewidth=1.8, color=legible[k],
                 label=FASES[k]["nombre"])
    _estilo(ax2, "tiempo (s)", "% del volumen de sólido")
    ax2.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.22))
    ax2.set_xlim(0, 740)
    ax2.set_title("fases minerales, ampliadas", fontsize=9)
    fig.tight_layout()
    _guardar(fig, "fig_fases_solidas.pdf")


def fig_hierro(serie):
    """Reparto del hierro entre sus óxidos, en % del inventario de Fe."""
    t = np.array([p[0] for p in serie])
    atomos = {"Fe2O3": 2.0, "Fe3O4": 3.0, "FeO": 1.0, "Fe": 1.0, "FeTiO3": 1.0}
    series = {k: [] for k in atomos}
    for _, c in serie:
        lecho = np.asarray(c["etiquetas"]) == LECHO
        moles = {k: float(np.sum(np.asarray(c["solido"][k])[lecho])) * n
                 for k, n in atomos.items()}
        total = sum(moles.values()) or 1.0
        for k in atomos:
            series[k].append(100.0 * moles[k] / total)

    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for k, color, etiqueta in (("Fe2O3", "#8B2314", "hematita Fe$_2$O$_3$"),
                               ("Fe3O4", "#2B2B30", "magnetita Fe$_3$O$_4$"),
                               ("FeO", "#a8613a", "wüstita FeO"),
                               ("FeTiO3", "#5b5170", "ilmenita FeTiO$_3$"),
                               ("Fe", "#B7B2AA", "hierro metálico")):
        ax.plot(t, series[k], linewidth=2.0, color=color, label=etiqueta)
    _estilo(ax, "tiempo (s)", "% del hierro total")
    ax.legend(frameon=False, ncol=2, fontsize=7.5)
    ax.set_xlim(0, 740)
    ax.set_ylim(-2, 100)
    fig.tight_layout()
    _guardar(fig, "fig_hierro.pdf")


def fig_gases(serie):
    t = np.array([p[0] for p in serie])
    especies = ["CO", "CO2", "H2", "H2O", "CH4", "N2"]
    colores = {"CO": AZUL, "CO2": GRIS, "H2": ROJO, "H2O": "#5aa0c8",
               "CH4": VERDE, "N2": "#9c8b6e"}
    series = {e: [] for e in especies}
    for _, c in serie:
        interior = np.isin(np.asarray(c["etiquetas"]), (LECHO, GAS))
        for e in especies:
            series[e].append(float(np.asarray(c["c_especies"][e])[interior].mean()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 4.8), sharex=True)
    for e in especies:
        ax1.plot(t, series[e], linewidth=1.8, color=colores[e], label=e)
    _estilo(ax1, "", "concentración media (mol/m³)")
    ax1.legend(frameon=False, ncol=3, fontsize=7.5)
    ax1.set_xlim(0, 300)

    razon = []
    for _, c in serie:
        interior = np.isin(np.asarray(c["etiquetas"]), (LECHO, GAS))
        co = float(np.asarray(c["c_especies"]["CO"])[interior].mean())
        co2 = float(np.asarray(c["c_especies"]["CO2"])[interior].mean())
        razon.append(co / (co + co2) if (co + co2) > 0 else np.nan)
    ax2.plot(t, razon, color=AZUL, linewidth=2.0)
    ax2.axhline(0.00757, color=ROJO, linewidth=1.2, linestyle="--")
    ax2.annotate("frontera Fe$_3$O$_4$/FeO = 0,00757", xy=(150, 0.0090),
                 fontsize=7.5, color=ROJO)
    ax2.set_yscale("log")
    _estilo(ax2, "tiempo (s)", "CO/(CO+CO$_2$)")
    ax2.set_xlim(0, 740)
    fig.tight_layout()
    _guardar(fig, "fig_gases.pdf")


def fig_porosidad(serie):
    t = np.array([p[0] for p in serie])
    eps, K = [], []
    for _, c in serie:
        lecho = np.asarray(c["etiquetas"]) == LECHO
        e = float(np.asarray(c["eps"])[lecho].mean())
        eps.append(e)
        # Kozeny-Carman con el diámetro nominal de 175 um
        K.append((175e-6) ** 2 * e ** 3 / (150.0 * (1 - e) ** 2))

    fig, ax1 = plt.subplots(figsize=(6.3, 3.2))
    ax1.plot(t, eps, color=AZUL, linewidth=2.0)
    _estilo(ax1, "tiempo (s)", "porosidad del lecho")
    ax1.set_xlim(0, 740)
    ax2 = ax1.twinx()
    ax2.plot(t, np.array(K) * 1e12, color=NARANJA, linewidth=1.8, linestyle="--")
    ax2.set_ylabel("permeabilidad (10$^{-12}$ m²)", color=NARANJA)
    ax2.tick_params(axis="y", colors=NARANJA)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    _guardar(fig, "fig_porosidad.pdf")


def fig_termico(serie):
    t = np.array([p[0] for p in serie])
    pared, lecho_T, gas_T, exterior = [], [], [], []
    for _, c in serie:
        et = np.asarray(c["etiquetas"])
        T = np.asarray(c["T"])
        pared.append(float(T[np.isin(et, (PARED, TAPA))].mean()))
        lecho_T.append(float(T[et == LECHO].mean()))
        gas_T.append(float(T[et == GAS].mean()))
        exterior.append(float(T[et == 0].max()) if np.any(et == 0) else np.nan)

    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.plot(t, exterior, color=GRIS, linewidth=1.5, linestyle=":", label="mufla")
    ax.plot(t, pared, color=NARANJA, linewidth=2.0, label="crisol")
    ax.plot(t, gas_T, color=VERDE, linewidth=1.8, label="gas interior")
    ax.plot(t, lecho_T, color=AZUL, linewidth=2.0, label="lecho")
    ax.axhspan(623.15, 773.15, color=VERDE, alpha=0.12, zorder=0)
    _estilo(ax, "tiempo (s)", "temperatura (K)")
    ax.legend(frameon=False, ncol=2)
    ax.set_xlim(0, 740)
    fig.tight_layout()
    _guardar(fig, "fig_termico.pdf")


def fig_adimensionales(serie):
    t, Re, Ra, Pe, Da = [], [], [], [], []
    for tiempo, c in serie:
        n = (c["metadatos"] or {}).get("numeros_adimensionales")
        if not n:
            continue
        t.append(tiempo)
        Re.append(abs(float(n.get("Re_particula", np.nan))))
        Ra.append(abs(float(n.get("Ra", np.nan))))
        Pe.append(abs(float(n.get("Pe_termico", np.nan))))
        Da.append(abs(float(n.get("Da", np.nan))))
    if not t:
        return
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for valores, color, etiqueta in ((Re, AZUL, "Re$_p$"), (Ra, ROJO, "Ra"),
                                     (Pe, VERDE, "Pe térmico"), (Da, NARANJA, "Da")):
        v = np.array(valores)
        ax.semilogy(t, np.where(v > 0, v, np.nan), linewidth=1.8, color=color, label=etiqueta)
    ax.axhline(1708, color=ROJO, linewidth=0.9, linestyle=":")
    ax.annotate("Ra crítico = 1708", xy=(430, 2600), fontsize=7.5, color=ROJO)
    _estilo(ax, "tiempo (s)", "número adimensional")
    ax.legend(frameon=False, ncol=4, loc="lower left")
    ax.set_xlim(0, 740)
    fig.tight_layout()
    _guardar(fig, "fig_adimensionales.pdf")


def fig_aglomerado(serie):
    t = np.array([p[0] for p in serie])
    volumen_cm3, celdas, hinchado = [], [], []
    for _, c in serie:
        et = np.asarray(c["etiquetas"])
        lecho = et == LECHO
        vcelda = volumen_celda(c) * 1e6
        n = int(np.sum(np.asarray(c["cohesion"])[lecho] > 0.5))
        h = float(np.asarray(c["hinchamiento"])[lecho].mean())
        celdas.append(100.0 * n / max(int(lecho.sum()), 1))
        volumen_cm3.append(n * vcelda * h)
        hinchado.append(h)

    fig, ax1 = plt.subplots(figsize=(6.3, 3.2))
    ax1.plot(t, volumen_cm3, color=AZUL, linewidth=2.0)
    ax1.axhline(1.3593, color=GRIS, linewidth=1.0, linestyle="--")
    ax1.annotate("volumen inicial del lecho 1,36 cm³", xy=(300, 1.45),
                 fontsize=7.5, color=GRIS)
    _estilo(ax1, "tiempo (s)", "volumen del aglomerado (cm³)")
    ax1.set_xlim(0, 740)
    ax2 = ax1.twinx()
    ax2.plot(t, celdas, color=ROJO, linewidth=1.6, linestyle="--")
    ax2.set_ylabel("% del lecho cohesionado", color=ROJO)
    ax2.tick_params(axis="y", colors=ROJO)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    _guardar(fig, "fig_aglomerado.pdf")


def fig_campos(serie):
    """Cortes verticales de temperatura y cohesión en cuatro instantes."""
    objetivos = [30, 90, 120, 720]
    fig, ejes = plt.subplots(2, len(objetivos), figsize=(7.2, 4.2))
    for columna, objetivo in enumerate(objetivos):
        t, c = min(serie, key=lambda e: abs(e[0] - objetivo))
        T = np.asarray(c["T"])
        coh = np.asarray(c["cohesion"])
        et = np.asarray(c["etiquetas"])
        j = T.shape[1] // 2
        x = np.asarray(c["x"])
        z = np.asarray(c["z"])
        fuera = ~np.isin(et[:, j, :], (LECHO, GAS, PARED, TAPA))

        campo_T = np.ma.masked_where(fuera, T[:, j, :])
        im = ejes[0, columna].pcolormesh(x, z, campo_T.T, cmap="inferno",
                                         vmin=298, vmax=1180, shading="nearest")
        ejes[0, columna].set_title(f"t = {t:.0f} s", fontsize=8.5)
        campo_c = np.ma.masked_where(~(et[:, j, :] == LECHO), coh[:, j, :])
        imc = ejes[1, columna].pcolormesh(x, z, campo_c.T, cmap="YlGnBu",
                                          vmin=0, vmax=1, shading="nearest",
                                          rasterized=True)
        for fila in (0, 1):
            ejes[fila, columna].set_aspect("equal")
            ejes[fila, columna].set_xticks([])
            if columna:
                ejes[fila, columna].set_yticks([])
        ejes[1, columna].set_ylim(0, 12)
    ejes[0, 0].set_ylabel("temperatura\nz (mm)")
    ejes[1, 0].set_ylabel("cohesión\nz (mm)")
    fig.colorbar(im, ax=ejes[0, :], shrink=0.85, label="T (K)", pad=0.02)
    fig.colorbar(imc, ax=ejes[1, :], shrink=0.85, label="cohesión", pad=0.02)
    fig.savefig(SALIDA / "fig_campos.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    directorio = sys.argv[1] if len(sys.argv) > 1 else str(RAIZ / "resultados" / "simulacion_720s")
    serie = serie_vigente(directorio)
    volumen = volumen_celda(serie[0][1])
    m0 = masa_solida_g(serie[0][1], volumen)
    SALIDA.mkdir(exist_ok=True)

    fig_cronologia(serie)
    fig_perdida_masa(serie, volumen, m0)
    fig_fases_solidas(serie)
    fig_hierro(serie)
    fig_gases(serie)
    fig_porosidad(serie)
    fig_termico(serie)
    fig_adimensionales(serie)
    fig_aglomerado(serie)
    fig_campos(serie)
    print(f"10 figuras escritas en {SALIDA} de {len(serie)} instantáneas "
          f"(t = {serie[0][0]:.0f}..{serie[-1][0]:.0f} s)")


if __name__ == "__main__":
    main()
