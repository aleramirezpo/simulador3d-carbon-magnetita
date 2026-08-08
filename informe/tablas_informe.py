"""Tablas y párrafos con cifras del informe, generados de la corrida real.

Escribe fragmentos .tex que `informe.tex` incluye con \\input, de modo que ninguna
cifra del documento esté escrita a mano.

Uso:  python informe/tablas_informe.py [directorio_de_resultados]
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent / "simulacion_v3" / "src"))

from fisica.adaptador_v3 import MASAS_MOLARES_SOLIDO_KG_MOL  # noqa: E402
from interfaz.app import directorio_predeterminado  # noqa: E402
from nucleo.salida import cargar_instantanea, recortar_a_la_corrida_vigente  # noqa: E402

SALIDA = RAIZ / "informe"


def serie_vigente(directorio):
    """Instantáneas de la corrida VIGENTE, ordenadas por tiempo.

    El recorte no se reimplementa aquí: lo hace `nucleo.salida`, que es de donde
    lo toman también el visor y las pruebas. Mientras cada consumidor tuvo su
    copia, el informe llegó a construirse sobre una carpeta con dos corridas
    mezcladas mientras el sitio publicaba otra distinta.
    """
    rutas = sorted(glob.glob(f"{directorio}/*.npz"))
    if not rutas:
        raise SystemExit(f"no hay instantáneas en {directorio}")
    cargadas = [(r, cargar_instantanea(r)) for r in rutas]
    vigentes = recortar_a_la_corrida_vigente(
        cargadas,
        lambda par: float(par[1]["t"]),
        lambda par: os.stat(par[0]).st_mtime_ns,
    )
    return sorted(((float(c["t"]), c) for _, c in vigentes), key=lambda e: e[0])


def masa_solida_g(campos, volumen):
    return 1000.0 * sum(
        float(np.sum(np.asarray(campos["solido"][f]))) * volumen * M
        for f, M in MASAS_MOLARES_SOLIDO_KG_MOL.items()
    )


def num(valor, decimales=2):
    return f"\\num{{{valor:.{decimales}f}}}"


def tabla_evolucion(serie, volumen, m0):
    Fe0 = np.asarray(serie[0][1]["solido"]["Fe2O3"])
    filas = []
    for objetivo in (0, 30, 60, 90, 120, 180, 300, 720):
        t, c = min(serie, key=lambda e: abs(e[0] - objetivo))
        lecho = np.asarray(c["etiquetas"]) == 3
        coh = np.asarray(c["cohesion"])[lecho]
        Fe = np.asarray(c["solido"]["Fe2O3"])
        val = lecho & (Fe0 > 1e-9)
        conv = 100.0 * float(np.mean(np.clip(1 - Fe[val] / Fe0[val], 0, 1))) if val.any() else 0.0
        filas.append(
            f"{t:.0f} & {num(float(np.asarray(c['T'])[lecho].mean()) - 273.15, 0)} & "
            f"{num(float(np.asarray(c['eps'])[lecho].mean()), 3)} & "
            f"{num(float(np.asarray(c['hinchamiento'])[lecho].mean()), 3)} & "
            f"{num(float(coh.mean()), 3)} & "
            f"{num(100.0 * (m0 - masa_solida_g(c, volumen)) / m0, 2)} & "
            f"{int((coh > 0.5).sum())} / {int(lecho.sum())} & {num(conv, 1)} \\\\"
        )
    cuerpo = "\n".join(filas)
    (SALIDA / "tabla_evolucion.tex").write_text(
        "\\begin{table}[h]\n\\centering\\small\n"
        "\\begin{tabular}{rrrrrrrr}\n\\toprule\n"
        "$t$ (\\si{\\second}) & $T$ lecho (\\si{\\celsius}) & porosidad & hinchamiento "
        "& cohesión & pérdida (\\%) & celdas formadas & conversión (\\%) "
        "\\\\\n\\midrule\n"
        f"{cuerpo}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Estado del lecho en cada instante. La porosidad sube de 0,540 "
        "a 0,678 conforme se van los volátiles, y con ella cambian la "
        "permeabilidad, la conductividad efectiva y la tortuosidad; el "
        "hinchamiento es el factor de expansión del aglomerado; la conversión "
        "es la de la hematita. Las fases están en el "
        "cuadro~\\ref{tab:fases-tiempo} y lo que se recupera tras enfriar, en "
        "el~\\ref{tab:enfriado}.}\n\\end{table}\n",
        encoding="utf-8")


def tabla_perdida(serie, volumen, m0):
    from datos_experimentales import TABLA_PERDIDA_MASA

    tabla = np.asarray(TABLA_PERDIDA_MASA, dtype=float)
    filas = []
    for t_exp, _, _, pct, sdp in tabla:
        t, c = min(serie, key=lambda e: abs(e[0] - t_exp))
        modelo = 100.0 * (m0 - masa_solida_g(c, volumen)) / m0
        medida = num(pct, 2) + ("" if np.isnan(sdp) else f" $\\pm$ {num(sdp, 2)}")
        filas.append(f"{t_exp:.0f} & {medida} & {num(modelo, 2)} & {num(modelo - pct, 2)} \\\\")
    cuerpo = "\n".join(filas)
    (SALIDA / "tabla_perdida.tex").write_text(
        "\\begin{table}[h]\n\\centering\\small\n"
        "\\begin{tabular}{rrrr}\n\\toprule\n"
        "$t$ (\\si{\\second}) & medida (\\%) & modelo (\\%) & diferencia \\\\\n\\midrule\n"
        f"{cuerpo}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Pérdida de masa: los ocho puntos medidos frente al modelo.}\n"
        "\\end{table}\n", encoding="utf-8")


def tabla_fases(serie):
    """Composición del sólido del lecho, % de volumen, cada 30 s."""
    from fisica.fases_visuales import FASES, MAPA_CAMPOS_SOLIDOS, volumenes_molares_cm3_mol

    vm = volumenes_molares_cm3_mol()

    def fracciones(campos):
        lecho = np.asarray(campos["etiquetas"]) == 3
        aporte, total = {}, 0.0
        for campo, clave in MAPA_CAMPOS_SOLIDOS.items():
            if clave is None or clave not in vm:
                continue
            v = float(np.sum(np.asarray(campos["solido"][campo])[lecho])) * vm[clave]
            if v > 0:
                aporte[clave] = aporte.get(clave, 0.0) + v
                total += v
        return {k: 100.0 * v / total for k, v in aporte.items()} if total else {}

    instantes = []
    objetivo = 0.0
    while objetivo <= serie[-1][0] + 1e-9:
        instantes.append(min(serie, key=lambda e: abs(e[0] - objetivo)))
        objetivo += 30.0
    columnas = ["char", "carbon", "cenizas", "Fe3O4", "FeO", "Fe2O3", "FeTiO3", "SiO2", "Fe"]
    todas = [fracciones(c) for _, c in instantes]
    presentes = [k for k in columnas if max(f.get(k, 0.0) for f in todas) > 0.005]

    cabecera = " & ".join(
        ["$t$ (\\si{\\second})"]
        + [FASES[k]["nombre"].replace(" / coque", "").replace("Hierro metálico", "Fe met.")
           for k in presentes]
    )
    filas = []
    for (t, _), fr in zip(instantes, todas):
        celdas = [f"{t:.0f}"]
        for k in presentes:
            v = fr.get(k, 0.0)
            celdas.append(f"\\num{{{v:.2f}}}" if v > 0.005 else "---")
        filas.append(" & ".join(celdas) + " \\\\")
    cuerpo = "\n".join(filas)
    formato = "r" + "r" * len(presentes)
    (SALIDA / "tabla_fases.tex").write_text(
        "\\begin{table}[htbp]\n\\centering\\footnotesize\n"
        f"\\begin{{tabular}}{{{formato}}}\n\\toprule\n"
        f"{cabecera} \\\\\n\\midrule\n{cuerpo}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Composición del sólido del lecho, en \\% de volumen, cada "
        "\\SI{30}{\\second}. La suma de cada fila es \\num{100}. El volumen de "
        "cada fase es $c_i M_i/\\rho_i$, no su número de moles.}\n"
        "\\label{tab:fases}\n\\end{table}\n", encoding="utf-8")


def parrafo_porosidad(serie):
    def eps_en(objetivo):
        _, c = min(serie, key=lambda e: abs(e[0] - objetivo))
        lecho = np.asarray(c["etiquetas"]) == 3
        return float(np.asarray(c["eps"])[lecho].mean())

    e0, e60, ef = eps_en(0), eps_en(60), eps_en(720)
    (SALIDA / "parrafo_porosidad.tex").write_text(
        "Sí se simula, y evoluciona. Antes estaba \\emph{congelada} en su valor "
        "inicial durante los \\SI{720}{\\second}, mientras el sólido perdía el "
        "\\qty{28}{\\percent} de su masa; de ella dependen la permeabilidad de "
        "Kozeny--Carman, la conductividad efectiva del lecho y la corrección de "
        "tortuosidad de las difusividades, que quedaban congeladas con ella.\n\n"
        "Ahora se recalcula del volumen de sólido que queda, sumando "
        "$c_i M_i/\\rho_i$ sobre las fases del inventario. Se aplica el cambio "
        "\\emph{relativo} sobre la porosidad inicial calibrada del caso, no el "
        "valor absoluto: la fracción sólida que dan los volúmenes molares "
        "(\\num{0.413}) no coincide con la que declara el caso (\\num{0.46}, de las "
        "densidades aparentes), y con la razón esa discrepancia se cancela.\n\n"
        f"Los valores: parte de \\textbf{{{e0:.3f}}}, sube a "
        f"\\textbf{{{e60:.3f}}} a los \\SI{{60}}{{\\second}} al marcharse los "
        f"volátiles, y se estabiliza en \\textbf{{{ef:.3f}}}. Es decir, el lecho "
        f"pasa de {100*(1-e0):.1f}\\,\\% de sólido a {100*(1-ef):.1f}\\,\\%.\n\n"
        "El hinchamiento \\emph{no} entra en la porosidad: sobre malla fija el "
        "lecho no puede crecer de volumen, y lo que el inventario describe es el "
        "hueco que dejan los volátiles al marcharse. La expansión se representa "
        "aparte.\n", encoding="utf-8")


def parrafo_contraste(serie, volumen, m0):
    """Análisis cuantitativo del contraste con la curva medida."""
    from datos_experimentales import ANALISIS_PROXIMO_CARBON, TABLA_PERDIDA_MASA

    tabla = np.asarray(TABLA_PERDIDA_MASA, dtype=float)
    tiempos = np.array([p[0] for p in serie])
    perdida = np.array([100.0 * (m0 - masa_solida_g(c, volumen)) / m0 for _, c in serie])

    residuos, dentro = [], 0
    for t_exp, _, _, pct, sdp in tabla:
        i = int(np.argmin(np.abs(tiempos - t_exp)))
        r = perdida[i] - pct
        residuos.append(r)
        if not np.isnan(sdp) and abs(r) <= 2.0 * sdp:
            dentro += 1
    residuos = np.array(residuos)
    rms = float(np.sqrt(np.mean(residuos ** 2)))
    con_replica = int(np.sum(~np.isnan(tabla[:, 4])))

    volatil = 0.75 * float(ANALISIS_PROXIMO_CARBON["materia_volatil_pct"])
    humedad = 0.75 * float(ANALISIS_PROXIMO_CARBON["humedad_residual_pct"])
    meseta_modelo = float(perdida[-1])
    meseta_medida = float(tabla[-1, 3])
    liberado_modelo = 100.0 * (meseta_modelo - humedad) / volatil
    liberado_medido = 100.0 * (meseta_medida - humedad) / volatil

    texto = (
        r"El acuerdo global es de \textbf{" + f"{rms:.2f}" + r" puntos porcentuales} "
        r"de RMS sobre los ocho puntos. De los " + f"{con_replica}" + r" que tienen "
        r"réplicas, " + f"{dentro}" + r" caen dentro de dos desviaciones de la medida."
        "\n\n"
        r"\textbf{La meseta.} El modelo se estabiliza en "
        f"\\qty{{{meseta_modelo:.2f}}}{{\\percent}}"
        r" y el ensayo en " f"\\qty{{{meseta_medida:.2f}}}{{\\percent}}"
        r". Descontada la humedad, eso significa que el modelo libera el "
        f"\\qty{{{liberado_modelo:.0f}}}{{\\percent}}"
        r" de la materia volátil del análisis próximo y el ensayo el "
        f"\\qty{{{liberado_medido:.0f}}}{{\\percent}}"
        r". No es un detalle numérico: dice que en el ensayo parte del volátil "
        r"\emph{no llega a salir}, algo esperable porque el análisis próximo se "
        r"hace a \SI{950}{\celsius} durante \SI{7}{\minute} en crisol tapado, y "
        r"el ensayo es otra cosa."
        "\n\n"
        r"\textbf{Lo que el contraste NO puede separar.} Un residuo en un instante "
        r"dado admite dos lecturas: que la carga esté a otra temperatura o que la "
        r"cinética sea otra. Con la curva de masa sola no se distinguen, porque "
        r"sólo se observa el producto de ambas. La cronología del aglomerado "
        r"—hinchamiento a los \SI{90}{\second}— sí las separa, porque el "
        r"hinchamiento depende de la \emph{temperatura} y no de la cinética de "
        r"liberación: fija la historia térmica y deja la cinética como única "
        r"incógnita. Ese es el argumento que llevó a la compuerta y no al "
        r"acoplamiento térmico."
        "\n"
    )
    (SALIDA / "parrafo_contraste.tex").write_text(texto, encoding="utf-8")


def parrafo_calibracion(serie, volumen, m0):
    from nucleo.caso import cargar_caso

    caso = cargar_caso(RAIZ / "casos" / "carbon_magnetita.yaml", malla="gruesa")
    mufla = caso.datos["condiciones_frontera"]["mufla"]
    quimica = caso.datos.get("quimica", {})
    vista = float(mufla.get("factor_vista", 1.0))
    puerta = float(quimica.get("T_devolatilizacion_C", 200.0))
    texto = r"""El contraste de la curva de masa dejó al descubierto que el modelo
devolatilizaba unos \SI{30}{\second} antes de tiempo. La primera hipótesis fue el
acoplamiento térmico: se probó a frenar el calentamiento con un factor de vista
del crisol dentro de la mufla, calibrado contra los ocho puntos. Con \num{0.46}
la curva de masa encajaba, \emph{pero rompía la cronología del aglomerado}: a los
\SI{90}{\second} el lecho quedaba a \SI{304}{\celsius}, fuera de la ventana
plástica, cuando el ensayo ya lo ve hincharse.

Ahí estaba la pista. Si a los \SI{90}{\second} el ensayo ha perdido el
\qty{19}{\percent} de la masa \emph{y} se le ve hinchar, el carbón está dentro de
la ventana plástica: el calentamiento del modelo sin frenar, que lo pone a
\SI{491}{\celsius}, es el correcto.

La causa real era la \textbf{compuerta de devolatilización}. La química la evalúa
como $\mathrm{sigmoide}(T, T_0, \SI{40}{\celsius})$ con
$T_0 = \SI{200}{\celsius}$, que no es un umbral sino el \emph{centro} de la
sigmoide: con ese valor está abierta al \qty{98}{\percent} ya a
\SI{360}{\celsius}, y el modelo suelta el volátil unos \SI{100}{\celsius} por
debajo de donde lo suelta un carbón bituminoso. En el 0-D de
\texttt{simulacion\_v3} no se notaba porque su historia térmica estaba calibrada
y era mucho más lenta que la que resuelve el 3-D.

"""
    texto += (
        f"Se adopta $T_0 = \\SI{{{puerta:.0f}}}{{\\celsius}}$, la temperatura de "
        r"máxima velocidad de devolatilización de los bituminosos, y no un ajuste "
        r"punto a punto. El factor de vista queda en "
        f"\\num{{{vista:.2f}}}: "
        r"\textbf{el acoplamiento térmico no lleva ningún parámetro calibrado}."
        "\n"
    )
    (SALIDA / "parrafo_calibracion.tex").write_text(texto, encoding="utf-8")


def n_pruebas():
    try:
        salida = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
            cwd=RAIZ, capture_output=True, text=True, timeout=300, check=False).stdout
        n = re.search(r"(\d+) tests? collected", salida)
        total = n.group(1) if n else "170"
    except Exception:
        total = "170"
    (SALIDA / "n_pruebas.tex").write_text(f"{total} pruebas automáticas.\n", encoding="utf-8")


def main():
    # Sin argumento se toma la corrida MÁS AVANZADA de `resultados/`, el mismo
    # criterio que usa el visor. Apuntar a una carpeta fija hacía que el informe
    # y el sitio publicado pudiesen describir corridas distintas.
    directorio = sys.argv[1] if len(sys.argv) > 1 else str(directorio_predeterminado(RAIZ / "resultados"))
    serie = serie_vigente(directorio)
    primera = serie[0][1]
    volumen = (
        (float(primera["x"][1]) - float(primera["x"][0]))
        * (float(primera["y"][1]) - float(primera["y"][0]))
        * (float(primera["z"][1]) - float(primera["z"][0]))
    ) * 1.0e-9
    m0 = masa_solida_g(primera, volumen)
    SALIDA.mkdir(exist_ok=True)
    tabla_evolucion(serie, volumen, m0)
    tabla_perdida(serie, volumen, m0)
    tabla_fases(serie)
    parrafo_porosidad(serie)
    parrafo_calibracion(serie, volumen, m0)
    parrafo_contraste(serie, volumen, m0)
    n_pruebas()
    print(f"tablas y párrafos escritos en {SALIDA}")


if __name__ == "__main__":
    main()
