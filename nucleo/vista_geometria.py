"""Vista rápida de la geometría generada, para inspección visual.

No es la interfaz definitiva: es una comprobación de que el dominio construido
por ``geometria.py`` corresponde al crisol real. La visualización interactiva 3D
será la aplicación web.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import geometria as g

SALIDA = Path(__file__).resolve().parents[1] / "resultados"

# Paleta segura para daltónicos, en el orden de los identificadores de material.
COLORES = ["#ffffff", "#4c6b8a", "#8a6d3b", "#c8553d", "#dfe6ec"]
ETIQUETAS = ["exterior", "pared del crisol", "tapa", "lecho reactivo", "gas interior"]


def figura_geometria(dx_mm: float = 0.4, dz_mm: float = 0.2,
                     archivo: str = "geometria_crisol.png") -> Path:
    d = g.dominio_del_ensayo(dx_mm=dx_mm, dz_mm=dz_mm)
    malla, etq, crisol, lecho = d["malla"], d["etiquetas"], d["crisol"], d["lecho"]
    cmap = ListedColormap(COLORES)

    fig = plt.figure(figsize=(12.5, 5.2))
    fig.suptitle("Geometría generada a partir de las cotas del crisol del ensayo",
                 fontsize=12, y=0.98)

    # --- corte vertical por el eje ---
    ax1 = fig.add_subplot(1, 3, 1)
    j = malla.forma[1] // 2
    corte = etq[:, j, :].T
    ax1.imshow(corte, origin="lower", cmap=cmap, vmin=0, vmax=4, aspect="equal",
               extent=[malla.x[0], malla.x[-1], malla.z[0], malla.z[-1]],
               interpolation="nearest")
    ax1.set(xlabel="x (mm)", ylabel="z (mm)", title="Corte vertical por el eje")

    # --- detalle del lecho ---
    ax2 = fig.add_subplot(1, 3, 2)
    z_lim = crisol.espesor_fondo_mm + d["altura_lecho_mm"] * 2.2
    k = np.searchsorted(malla.z, z_lim)
    ax2.imshow(etq[:, j, :k].T, origin="lower", cmap=cmap, vmin=0, vmax=4,
               aspect="equal", interpolation="nearest",
               extent=[malla.x[0], malla.x[-1], malla.z[0], malla.z[k - 1]])
    ax2.axhline(crisol.espesor_fondo_mm, color="k", lw=0.7, ls=":")
    ax2.axhline(crisol.espesor_fondo_mm + d["altura_lecho_mm"], color="k", lw=0.7, ls=":")
    ax2.annotate(f"lecho: {d['altura_lecho_mm']:.2f} mm",
                 xy=(0, crisol.espesor_fondo_mm + d["altura_lecho_mm"] / 2),
                 ha="center", fontsize=9)
    ax2.set(xlabel="x (mm)", ylabel="z (mm)",
            title="Detalle del lecho (disco delgado)")

    # --- planta a la altura del lecho ---
    ax3 = fig.add_subplot(1, 3, 3)
    k_l = np.searchsorted(malla.z, crisol.espesor_fondo_mm + d["altura_lecho_mm"] / 2)
    ax3.imshow(etq[:, :, k_l].T, origin="lower", cmap=cmap, vmin=0, vmax=4,
               aspect="equal", interpolation="nearest",
               extent=[malla.x[0], malla.x[-1], malla.y[0], malla.y[-1]])
    ax3.set(xlabel="x (mm)", ylabel="y (mm)", title="Planta a media altura del lecho")

    fig.legend(handles=[Patch(facecolor=c, edgecolor="#666", label=l)
                        for c, l in zip(COLORES, ETIQUETAS)],
               loc="lower center", ncol=5, frameon=False, fontsize=9)

    v = d["verificacion_masa"]
    fig.text(0.5, 0.075,
             f"Cotas verificadas: masa desde la geometría {v['masa_calculada_g']:.2f} g "
             f"frente a {v['masa_declarada_g']:.2f} g declarados ({v['error_pct']:+.2f} %)   ·   "
             f"malla {malla.forma[0]}×{malla.forma[1]}×{malla.forma[2]} = {malla.n_celdas:,} celdas",
             ha="center", fontsize=8.5, color="#444")

    fig.tight_layout(rect=[0, 0.11, 1, 0.95])
    SALIDA.mkdir(exist_ok=True)
    ruta = SALIDA / archivo
    fig.savefig(ruta, dpi=170)
    plt.close(fig)
    return ruta


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("generado:", figura_geometria())
