"""Exporta la interfaz 3D como sitio estático, sin servidor.

`interfaz.app` calcula cada respuesta al vuelo: necesita Python, numpy y los
NPZ del solucionador en disco, así que sólo se puede abrir en el equipo donde
se corrió la simulación. Este módulo congela esa misma interfaz en archivos
planos —una corrida concreta, ya elegida— para poder publicarla en cualquier
alojamiento estático (GitHub Pages) y abrirla en cualquier PC.

La equivalencia es exacta por construcción: el sitio se genera llamando a las
MISMAS funciones que sirven la API local (`_estructura_para_web`,
`_codificar_s3df`, `EstadoAplicacion.configuracion`, `generar_lineas_corriente`),
de modo que no hay una segunda implementación que se pueda desviar. Lo único
que cambia es el transporte:

    /api/config              ->  datos/config.json
    /api/fotograma?indice=N  ->  datos/fotograma_NNNN.bin   (S3DF, idéntico)
    /api/lineas?indice=N     ->  datos/lineas_NNNN.json
    /api/serie               ->  datos/serie.json

y el cliente resuelve unas u otras a través de `web/js/rutas.js`, según el
`index.html` declare o no `window.SIMULADOR3D_ESTATICO`.

Uso:

    python -m interfaz.exportar_estatico                 # corrida predeterminada
    python -m interfaz.exportar_estatico --datos resultados/simulacion_larga
    python -m interfaz.exportar_estatico --salida sitio --limpiar
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .app import (
    DIRECTORIO_WEB,
    EstadoAplicacion,
    _codificar_s3df,
    _estructura_para_web,
    directorio_predeterminado,
)
from .datos_sinteticos import generar_lineas_corriente


SUBDIRECTORIO_DATOS = "datos"


def _config_publicable(estado: EstadoAplicacion, nombre_corrida: str) -> dict[str, Any]:
    """Configuración de la corrida, saneada para publicarse.

    `directorio_datos` es la ruta absoluta de la máquina donde se simuló
    (`D:\\pae\\simulador3d\\resultados\\...`). En el servidor local eso es
    información útil; en un sitio público es ruido y una fuga innecesaria del
    árbol de archivos de quien lo generó. Se sustituye por el nombre de la
    corrida, que es lo único que el lector necesita para saber qué está viendo.
    """
    config = estado.configuracion()
    config["directorio_datos"] = nombre_corrida
    config["corrida"] = nombre_corrida
    config["despliegue"] = "estatico"
    config["tiempo_final_s"] = float(estado.tiempos[-1]) if estado.tiempos else 0.0
    return config


def _index_estatico(html: str, sinteticos: bool) -> str:
    """`index.html` con el origen de datos declarado.

    El servidor local reescribe estos mismos rótulos al vuelo en
    `_recurso_estatico`; aquí hay que hacerlo una vez, al exportar, porque
    después ya no hay nadie que reescriba nada.
    """
    if not sinteticos:
        html = html.replace("Datos sintéticos", "Resultados del solucionador")
        html = html.replace(
            "PREDICCIÓN · DATOS SINTÉTICOS",
            "PREDICCIÓN · RESULTADOS DEL SOLUCIONADOR",
        )
    declaracion = (
        f'  <script>window.SIMULADOR3D_ESTATICO = "{SUBDIRECTORIO_DATOS}";</script>\n'
    )
    marca = '  <script type="module" src="js/app.js"></script>'
    if marca not in html:
        raise RuntimeError("no se encontró la etiqueta de arranque en index.html")
    return html.replace(marca, declaracion + marca)


def _copiar_web(salida: Path, sinteticos: bool) -> None:
    for sub in ("css", "js"):
        destino = salida / sub
        if destino.exists():
            shutil.rmtree(destino)
        shutil.copytree(DIRECTORIO_WEB / sub, destino)
    html = (DIRECTORIO_WEB / "index.html").read_text(encoding="utf-8")
    (salida / "index.html").write_text(_index_estatico(html, sinteticos), encoding="utf-8")
    # GitHub Pages pasa el sitio por Jekyll si no se le dice lo contrario, y
    # Jekyll descarta los archivos y carpetas que empiezan por `_`.
    (salida / ".nojekyll").write_text("", encoding="utf-8")


def exportar(
    destino: str | Path,
    datos: str | Path | None = None,
    limpiar: bool = False,
    registrar: Any = print,
) -> dict[str, Any]:
    """Escribe el sitio completo en `destino` y devuelve su manifiesto."""
    origen = Path(datos) if datos is not None else directorio_predeterminado()
    salida = Path(destino)
    if limpiar and salida.exists():
        shutil.rmtree(salida)
    directorio_datos = salida / SUBDIRECTORIO_DATOS
    directorio_datos.mkdir(parents=True, exist_ok=True)

    estado = EstadoAplicacion(datos=origen)
    nombre_corrida = Path(origen).name
    registrar(
        f"Corrida: {nombre_corrida} · {estado.n_fotogramas} fotogramas · "
        f"malla {'x'.join(str(n) for n in estado.forma)} · "
        f"origen {estado.origen_datos}"
    )

    config = _config_publicable(estado, nombre_corrida)
    (directorio_datos / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, separators=(",", ":")), encoding="utf-8",
    )

    bytes_totales = 0
    for indice in range(estado.n_fotogramas):
        campos = estado.cargar(indice)
        binario = _codificar_s3df(
            _estructura_para_web(campos, estado.Fe2O3_inicial_mol_m3))
        (directorio_datos / f"fotograma_{indice:04d}.bin").write_bytes(binario)
        lineas = generar_lineas_corriente(campos)
        (directorio_datos / f"lineas_{indice:04d}.json").write_text(
            json.dumps(
                {"lineas": [np.round(linea, 4).tolist() for linea in lineas]},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        bytes_totales += len(binario)
        if indice % 10 == 0 or indice == estado.n_fotogramas - 1:
            registrar(f"  fotograma {indice + 1}/{estado.n_fotogramas}")

    (directorio_datos / "serie.json").write_text(
        json.dumps({"serie": estado.serie()}, separators=(",", ":")), encoding="utf-8",
    )
    (directorio_datos / "salud.json").write_text(
        json.dumps({
            "estado": "ok",
            "modo": "demostración" if estado.datos_sinteticos else "resultados",
            "datos_sinteticos": estado.datos_sinteticos,
            "origen_datos": estado.origen_datos,
            "despliegue": "estatico",
        }, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    _copiar_web(salida, estado.datos_sinteticos)
    estado.cerrar()

    manifiesto = {
        "corrida": nombre_corrida,
        "n_fotogramas": estado.n_fotogramas,
        "forma": list(estado.forma),
        "t_final_s": config["tiempo_final_s"],
        "MB_fotogramas": round(bytes_totales / 1.0e6, 1),
        "MB_sitio": round(
            sum(p.stat().st_size for p in salida.rglob("*") if p.is_file()) / 1.0e6, 1,
        ),
    }
    (salida / "manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    registrar(
        f"Sitio en {salida} · {manifiesto['MB_sitio']} MB "
        f"({manifiesto['MB_fotogramas']} MB de fotogramas)"
    )
    return manifiesto


def main() -> int:
    analizador = argparse.ArgumentParser(
        description="Exporta la interfaz 3D como sitio estático publicable")
    analizador.add_argument(
        "--datos", type=Path, default=None,
        help="directorio de instantáneas; por defecto la corrida más avanzada de resultados/")
    analizador.add_argument(
        "--salida", type=Path, default=Path("sitio"),
        help="directorio del sitio generado (por omisión: sitio/)")
    analizador.add_argument(
        "--limpiar", action="store_true",
        help="borra el directorio de salida antes de escribir")
    argumentos = analizador.parse_args()
    exportar(argumentos.salida, argumentos.datos, argumentos.limpiar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
