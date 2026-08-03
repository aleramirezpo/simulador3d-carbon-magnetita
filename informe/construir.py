"""Construye el informe en PDF de principio a fin.

Genera las figuras y las tablas de la corrida indicada y compila el LaTeX. No hay
ninguna cifra escrita a mano en `informe.tex`: todo entra por \\input.

Uso:  python informe/construir.py [directorio_de_resultados]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent


def ejecutar(orden, descripcion, cwd=RAIZ):
    print(f"→ {descripcion}")
    resultado = subprocess.run(orden, cwd=cwd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    if resultado.returncode != 0:
        cola = (resultado.stdout or "")[-2500:] + (resultado.stderr or "")[-1500:]
        raise SystemExit(f"falló: {descripcion}\n{cola}")
    return resultado.stdout


def main():
    datos = sys.argv[1] if len(sys.argv) > 1 else str(RAIZ / "resultados" / "simulacion_720s")
    ejecutar([sys.executable, str(AQUI / "figuras_informe.py"), datos], "figuras")
    ejecutar([sys.executable, str(AQUI / "tablas_informe.py"), datos], "tablas y párrafos")

    latex = shutil.which("pdflatex") or shutil.which("xelatex")
    if latex is None:
        raise SystemExit("no se encontró pdflatex ni xelatex en el PATH")
    # Dos pasadas: la primera resuelve las referencias, la segunda las coloca.
    for pasada in (1, 2):
        ejecutar([latex, "-interaction=nonstopmode", "-halt-on-error", "informe.tex"],
                 f"LaTeX, pasada {pasada}", cwd=AQUI)

    pdf = AQUI / "informe.pdf"
    print(f"\nlisto: {pdf}  ({pdf.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
