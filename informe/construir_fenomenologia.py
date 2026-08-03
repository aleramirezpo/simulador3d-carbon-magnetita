"""Construye el documento de fenomenología en PDF, de principio a fin.

Genera las figuras y las tablas de la corrida indicada y compila el LaTeX. No
hay ninguna cifra escrita a mano en `fenomenologia.tex`: todo entra por \\input.

Uso:  python informe/construir_fenomenologia.py [directorio_de_resultados]
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
    resultado = subprocess.run(
        orden, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if resultado.returncode != 0:
        cola = (resultado.stdout or "")[-3000:] + (resultado.stderr or "")[-1500:]
        raise SystemExit(f"falló: {descripcion}\n{cola}")
    return resultado.stdout


def main() -> None:
    argumentos = sys.argv[1:]
    ejecutar([sys.executable, str(AQUI / "figuras_fenomenologia.py"), *argumentos],
             "figuras")
    ejecutar([sys.executable, str(AQUI / "tablas_fenomenologia.py"), *argumentos],
             "tablas y datos")

    latex = shutil.which("pdflatex") or shutil.which("xelatex")
    if latex is None:
        raise SystemExit("no se encontró pdflatex ni xelatex en el PATH")
    # Dos pasadas: la primera resuelve el índice, la segunda lo coloca.
    for pasada in (1, 2):
        ejecutar([latex, "-interaction=nonstopmode", "-halt-on-error",
                  "fenomenologia.tex"], f"LaTeX, pasada {pasada}", cwd=AQUI)

    pdf = AQUI / "fenomenologia.pdf"
    print(f"\nlisto: {pdf}  ({pdf.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
