"""Tablas y párrafos con cifras del documento de fenomenología.

Cada fragmento se escribe como un `.tex` que el documento incluye con \\input.
Ninguna cifra del documento está escrita a mano.

Uso:  python informe/tablas_fenomenologia.py [directorio_de_resultados]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ.parent / "simulacion_v3" / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ.parent / "simulacion_v3" / "src"))

from informe.datos_fenomenologia import (  # noqa: E402
    COHESION_AGLOMERADO,
    diagnosticos,
    fronteras_co,
)

SALIDA = RAIZ / "informe"


def num(valor, decimales=2):
    """Número con coma decimal, como se escribe en español.

    Sin separador de millares: ninguna magnitud del documento llega a mil, y
    el separador obligaría a un marcador intermedio para no pisar la coma.
    """
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "---"
    texto = f"{valor:.{decimales}f}"
    # Un residuo negativo diminuto se compone como «-0,00», que se lee como si
    # el signo significara algo. Al redondear a cero, cero es cero.
    if texto.lstrip("-").strip("0.,") == "":
        texto = texto.lstrip("-")
    return texto.replace(".", ",")


def cientifico(valor, cifras=2):
    """Notación científica en LaTeX: 3,16\\times10^{-6}."""
    if valor is None or not np.isfinite(valor) or valor == 0.0:
        return "0"
    exponente = int(np.floor(np.log10(abs(valor))))
    mantisa = valor / 10.0 ** exponente
    return rf"${num(mantisa, cifras)}\times10^{{{exponente}}}$"


def escribir(nombre, texto):
    (SALIDA / nombre).write_text(texto.rstrip() + "\n", encoding="utf-8")
    print(f"  {nombre}")


# Escalares del documento. Van a un único archivo de macros en vez de un
# `\input` por dato: `\input` es frágil y explota dentro de un argumento móvil
# como `\caption{}`, mientras que una macro se expande sin problemas.
MACROS: dict[str, str] = {}


def macro(nombre, valor):
    MACROS[nombre] = str(valor)


def escribir_macros():
    lineas = [
        r"% Generado por informe/tablas_fenomenologia.py. No editar a mano.",
    ]
    for nombre, valor in MACROS.items():
        lineas.append(rf"\newcommand{{\{nombre}}}{{{valor}}}")
    escribir("macros_fenomenologia.tex", "\n".join(lineas))


def _en(serie, t_objetivo, clave):
    t = np.asarray(serie["t"])
    return serie[clave][int(np.argmin(np.abs(t - t_objetivo)))]


# ------------------------------------------------------------------- tablas

def tabla_parametros(datos):
    dx, dy, dz = datos["paso_malla_mm"]
    s = datos["serie"]
    filas = [
        ("Carga de carbón", "0,75", "g"),
        ("Carga de concentrado", "0,25", "g"),
        ("Masa sólida inicial resuelta", num(datos["masa_inicial_g"], 4), "g"),
        ("Porosidad inicial del lecho", num(s["eps_media"][0], 3), "---"),
        ("Diámetro nominal de partícula", "175", r"$\mu$m"),
        ("Crisol: diámetro de base / boca", "25,0 / 29,5", "mm"),
        ("Crisol: altura, espesor de pared", "32,0 / 1,1", "mm"),
        ("Malla (gruesa)", rf"{num(dx,1)}$\times${num(dy,1)}$\times${num(dz,1)}", "mm"),
        ("Celdas del lecho", f"{int(_en(s, 720.0, 'celdas_aglomerado'))}", "---"),
        ("Temperatura inicial", num(s["T_lecho"][0] - 273.15, 1), r"$^\circ$C"),
        ("Temperatura final de mufla", num(s["T_mufla"][-1] - 273.15, 1), r"$^\circ$C"),
        ("Tiempo simulado", num(s["t"][-1], 0), "s"),
        ("Instantáneas guardadas", f"{len(s['t'])}", "---"),
        ("Emisividad de la pared", "0,80", "---"),
        ("Volátil liberable ($f_{vol}$)", "0,88", "---"),
        ("Compuerta de devolatilización", "450", r"$^\circ$C"),
        ("Índice de hinchamiento del carbón", "8", "ASTM D720"),
    ]
    cuerpo = "\n".join(rf"    {a} & {b} & {c} \\" for a, b, c in filas)
    escribir("tabla_fen_parametros.tex", (
        r"\begin{tabular}{lrl}" "\n"
        r"    \hline" "\n"
        r"    Parámetro & Valor & Unidad \\" "\n"
        r"    \hline" "\n"
        f"{cuerpo}\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))


def tabla_fases(datos):
    s = datos["serie"]
    nombres = {
        "volatil": "Volátil del carbón",
        "C": "Carbono fijo (char)",
        "Fe2O3": r"Hematita Fe$_2$O$_3$",
        "Fe3O4": r"Magnetita Fe$_3$O$_4$",
        "FeO": r"Wüstita FeO",
        "Fe": r"Hierro metálico Fe",
        "FeTiO3": r"Ilmenita FeTiO$_3$",
        "TiO2": r"Rutilo TiO$_2$",
        "SiO2": r"Cuarzo SiO$_2$",
        "Fe2SiO4": r"Fayalita Fe$_2$SiO$_4$",
        "ceniza": "Cenizas",
    }
    filas = []
    for fase, etiqueta in nombres.items():
        clave = f"m_{fase}"
        if clave not in s:
            continue
        inicial, final = 1000.0 * s[clave][0], 1000.0 * s[clave][-1]
        delta = final - inicial
        # «No cambia» con criterio RELATIVO. Un umbral absoluto diminuto marcaba
        # el carbono fijo como «se forma» por una variación de 3e-5 mg sobre
        # 405 mg: una parte en diez millones, que no es una transformación sino
        # el residuo del solucionador. El umbral de una parte en cien mil sigue
        # estando cuatro órdenes por debajo de lo que cualquier balanza vería.
        # Y un suelo ABSOLUTO además del relativo, porque el criterio relativo
        # no dice nada de una fase que empieza en cero: la fayalita aparecía
        # como «se forma» por 2e-9 mg, que es cero a todos los efectos.
        # 1e-4 mg son 0,1 microgramos, por debajo de cualquier balanza.
        relativo = abs(delta) / inicial if inicial > 0.0 else float("inf")
        if delta == 0.0 or relativo < 1.0e-5 or abs(delta) < 1.0e-4:
            veredicto = "no cambia"
        elif delta > 0:
            veredicto = "se forma"
        else:
            veredicto = "se consume"
        filas.append(rf"    {etiqueta} & {num(inicial,2)} & {num(final,2)} & "
                     rf"{num(delta,2)} & {veredicto} \\")
    escribir("tabla_fen_fases.tex", (
        r"\begin{tabular}{lrrrl}" "\n"
        r"    \hline" "\n"
        r"    Fase & $m(0)$ [mg] & $m(720)$ [mg] & $\Delta m$ [mg] & Resultado \\" "\n"
        r"    \hline" "\n"
        + "\n".join(filas) + "\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))


# Instantes de extracción que se tabulan. Los primeros son los que el
# laboratorio observó o midió; el resto reparte lo que queda del ensayo.
TIEMPOS_EXTRACCION = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 210.0, 360.0, 720.0)


def _indice_mas_cercano(s, objetivo):
    tiempos = np.asarray(s["t"], dtype=float)
    return int(np.argmin(np.abs(tiempos - float(objetivo))))


def tabla_fases_por_tiempo(datos):
    """Inventario de fases DENTRO de la mufla, en cada instante de extracción.

    Es el estado a \\SI{900}{\\celsius}, no lo que se recupera: para eso está
    `tabla_enfriado`, que aplica el eutectoide de la wüstita.
    """
    s = datos["serie"]
    columnas = (
        ("volatil", "volátil"),
        ("C", "char"),
        ("ceniza", "ceniza"),
        ("Fe2O3", r"Fe$_2$O$_3$"),
        ("Fe3O4", r"Fe$_3$O$_4$"),
        ("FeO", "FeO"),
        ("Fe", "Fe"),
        ("FeTiO3", r"FeTiO$_3$"),
    )
    presentes = [(clave, etiqueta) for clave, etiqueta in columnas if f"m_{clave}" in s]
    filas = []
    for objetivo in TIEMPOS_EXTRACCION:
        i = _indice_mas_cercano(s, objetivo)
        total = sum(s[f"m_{c}"][i] for c, _ in presentes)
        celdas = " & ".join(
            num(100.0 * s[f"m_{c}"][i] / total, 2) if total > 0 else "---"
            for c, _ in presentes
        )
        filas.append(
            rf"    {num(s['t'][i], 0)} & {num(1000.0 * total, 1)} & "
            rf"{num(s['eps_media'][i], 3)} & {celdas} \\"
        )
    cabecera = " & ".join(etiqueta for _, etiqueta in presentes)
    escribir("tabla_fen_fases_tiempo.tex", (
        r"\begin{tabular}{rrr" + "r" * len(presentes) + "}\n"
        r"    \hline" "\n"
        rf"     &  &  & \multicolumn{{{len(presentes)}}}{{c}}{{\% en masa del sólido}} \\" "\n"
        rf"    $t$ [s] & masa [mg] & $\varepsilon$ & {cabecera} \\" "\n"
        r"    \hline" "\n"
        + "\n".join(filas) + "\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))


def tabla_enfriado(datos):
    """Lo que se RECUPERA a temperatura ambiente, en cada instante de extracción.

    Por debajo de \\SI{570}{\\celsius} la wüstita deja de ser estable y se
    descompone, 4 FeO -> Fe3O4 + Fe. Se dan las dos cotas: temple, con la
    wüstita conservada entera, y enfriamiento lento, con ella descompuesta del
    todo. La lectura real está entre las dos.
    """
    from fisica import magnetismo as mag

    s = datos["serie"]
    enf = datos["enfriamiento"]
    fraccion = mag.fraccion_eutectoide(float(enf["tiempo_en_ventana_eutectoide_s"]))
    macro("datFraccionEutectoide", num(100.0 * fraccion, 0))

    fases = ("volatil", "C", "ceniza", "Fe2O3", "Fe3O4", "FeO", "Fe", "FeTiO3")
    volumen = float(np.prod(datos["paso_malla_mm"])) * 1.0e-9
    instantes = {round(float(t), 6): campos for t, campos in datos["instantaneas"]}
    tiempos_inst = sorted(instantes)

    filas = []
    for objetivo in TIEMPOS_EXTRACCION:
        i = _indice_mas_cercano(s, objetivo)
        t_fila = float(s["t"][i])
        clave = min(tiempos_inst, key=lambda x: abs(x - t_fila))
        solido = {f: np.asarray(v) for f, v in instantes[clave]["solido"].items()}

        # La masa NO cambia al enfriar: el eutectoide y la reoxidación por el
        # gas son reorganizaciones en estado sólido.
        total_mg = 1000.0 * sum(s[f"m_{f}"][i] for f in fases if f"m_{f}" in s)
        pct = lambda mg: 100.0 * mg / total_mg if total_mg > 0 else 0.0

        n_FeO = mag.fases_tras_enfriar(solido, volumen, 0.0)["FeO"]
        # «mufla» es el estado sin enfriar: ninguna transformación aplicada.
        estados = {"mufla": mag.composicion_tras_ruta(solido, volumen, f_eutectoide=0.0)}
        for etiqueta, ruta in (("tapa", mag.RUTA_CON_TAPA), ("sin", mag.RUTA_SIN_TAPA)):
            p = mag.parametros_de_ruta(ruta, n_FeO, float(s["T_lecho"][i]))
            estados[etiqueta] = mag.composicion_tras_ruta(
                solido, volumen,
                f_eutectoide=p["f_eutectoide"],
                f_reoxidacion_gas=p["f_reoxidacion_gas"],
                f_oxidacion_aire=p["f_oxidacion_aire"],
            )

        fijas = " & ".join(
            num(pct(1000.0 * s[f"m_{f}"][i]), 2)
            for f in ("volatil", "C", "ceniza", "FeTiO3")
        )

        def tercia(nombre: str, cifras: int = 2) -> str:
            """«mufla / con tapa / sin tapa» en la misma celda."""
            return " / ".join(
                num(pct(estados[e][f"{nombre}_mg"]), cifras)
                for e in ("mufla", "tapa", "sin")
            )

        ms = " / ".join(
            num(estados[e]["magnetizacion_Am2_kg"], 1)
            for e in ("mufla", "tapa", "sin")
        )

        filas.append(
            rf"    {num(t_fila, 0)} & {num(total_mg, 1)} & "
            rf"{num(s['eps_media'][i], 3)} & {fijas} & "
            rf"{tercia('Fe3O4')} & {tercia('FeO')} & {tercia('Fe')} & "
            rf"{tercia('Fe2O3')} & {ms} \\"
        )

    escribir("tabla_fen_enfriado.tex", (
        r"\begin{tabular}{rrrrrrrrrrrr}" "\n"
        r"    \hline" "\n"
        r"     &  &  & \multicolumn{4}{c}{iguales en los tres casos [\%]}"
        r" & \multicolumn{5}{c}{\textbf{en la mufla / con tapa / sin tapa}} \\" "\n"
        r"    $t$ [s] & masa [mg] & $\varepsilon$ & volátil & char & ceniza & FeTiO$_3$"
        r" & Fe$_3$O$_4$ [\%] & FeO [\%] & Fe [\%] & Fe$_2$O$_3$ [\%]"
        r" & $M_s$ [A m$^2$/kg] \\" "\n"
        r"    \hline" "\n"
        + "\n".join(filas) + "\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))


def tabla_rutas_de_enfriado(datos):
    """Compara las rutas de enfriado por lo magnético que dejan el aglomerado.

    Es la pregunta del laboratorio: si el material se va a usar como adsorbente
    y luego hay que recuperarlo con un imán, ¿con tapa o sin tapa?, ¿deprisa o
    despacio?
    """
    from fisica import magnetismo as mag

    ultima = datos["instantaneas"][-1][1]
    volumen = float(np.prod(datos["paso_malla_mm"])) * 1.0e-9
    solido = {f: np.asarray(v) for f, v in ultima["solido"].items()}
    rutas = mag.evaluacion_rutas_de_enfriado(solido, volumen)

    peor = min(r["magnetizacion_Am2_kg"] for r in rutas)
    filas = []
    for r in rutas:
        filas.append(
            rf"    {r['ruta']} & {num(r['tiempo_en_ventana_s'], 0)} & "
            rf"{num(100.0 * r['fraccion_eutectoide'], 0)} & "
            rf"{num(r['Fe3O4_mg'], 1)} & {num(r['FeO_mg'], 1)} & "
            rf"{num(r['Fe_mg'], 2)} & {num(r['Fe2O3_mg'], 1)} & "
            rf"\textbf{{{num(r['magnetizacion_Am2_kg'], 1)}}} & "
            rf"{num(100.0 * r['magnetizacion_Am2_kg'] / peor, 0)} \\"
        )

    mejor = max(rutas, key=lambda r: r["magnetizacion_Am2_kg"])
    macro("datMagnetMejorRuta", num(mejor["magnetizacion_Am2_kg"], 1))
    macro("datMagnetPeorRuta", num(peor, 1))
    macro("datMagnetGananciaRuta", num(100.0 * mejor["magnetizacion_Am2_kg"] / peor - 100.0, 0))
    macro("datFeMetalMejorRuta", num(mejor["Fe_mg"], 1))
    con_tapa = next(r for r in rutas if r["ruta"] == "Al aire, con tapa")
    macro("datReoxidaGas", num(100.0 * con_tapa["fraccion_reoxidada_por_gas"], 0))

    escribir("tabla_fen_rutas_enfriado.tex", (
        r"\begin{tabular}{lrrrrrrrr}" "\n"
        r"    \hline" "\n"
        r"    Ruta de enfriado & ventana & eutect. & Fe$_3$O$_4$ & FeO & Fe"
        r" & Fe$_2$O$_3$ & $M_s$ & rel. \\" "\n"
        r"     & [s] & [\%] & \multicolumn{4}{c}{[mg]} & [A m$^2$/kg] & [\%] \\" "\n"
        r"    \hline" "\n"
        + "\n".join(filas) + "\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))


def tabla_fronteras(datos):
    s = datos["serie"]
    fronteras = fronteras_co()
    x_gas = float(s["CO_sobre_COx"][-1])
    orden = (
        ("hematita", r"Fe$_2$O$_3 \rightarrow$ Fe$_3$O$_4$"),
        ("magnetita", r"Fe$_3$O$_4 \rightarrow$ FeO"),
        ("wustita", r"FeO $\rightarrow$ Fe"),
        ("ilmenita", r"FeTiO$_3 \rightarrow$ Fe + TiO$_2$"),
    )
    filas = []
    for clave, etiqueta in orden:
        umbral = fronteras[clave]
        ocurre = "sí" if x_gas > umbral else r"\textbf{no}"
        filas.append(rf"    {etiqueta} & {num(100 * umbral, 2)} & {ocurre} \\")
    escribir("tabla_fen_fronteras.tex", (
        r"\begin{tabular}{lrc}" "\n"
        r"    \hline" "\n"
        r"    Reducción & CO/(CO+CO$_2$) mínimo [\%] & ¿Ocurre? \\" "\n"
        r"    \hline" "\n"
        + "\n".join(filas) + "\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))
    macro("datGas", num(100 * x_gas, 1))
    macro("datUmbralIlmenita", num(100 * fronteras["ilmenita"], 1))
    macro("datUmbralWustita", num(100 * fronteras["wustita"], 1))
    macro("datUmbralMagnetita", num(100 * fronteras["magnetita"], 2))


def tabla_cronologia(datos):
    s = datos["serie"]
    filas = []
    for t in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 300.0, 720.0):
        filas.append(
            rf"    {num(t,0)} & {num(_en(s,t,'T_lecho') - 273.15, 0)} & "
            rf"{num(_en(s,t,'perdida_pct'), 2)} & {num(_en(s,t,'cohesion_max'), 3)} & "
            rf"{num(_en(s,t,'hinchamiento_medio'), 3)} & {num(_en(s,t,'eps_media'), 3)} & "
            rf"{num(_en(s,t,'D_huella_mm'), 1)} \\")
    escribir("tabla_fen_cronologia.tex", (
        r"\begin{tabular}{rrrrrrr}" "\n"
        r"    \hline" "\n"
        r"    $t$ [s] & $T_\text{lecho}$ [$^\circ$C] & pérdida [\%] & cohesión"
        r" & hinch. & $\varepsilon$ & $\varnothing$ [mm] \\" "\n"
        r"    \hline" "\n"
        + "\n".join(filas) + "\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))


def tabla_adimensionales(datos):
    s = datos["serie"]
    # Se declara el máximo además del valor final. Sólo con el final se diría
    # que la advección nunca cuenta, y no es cierto: durante el estallido de
    # devolatilización los Péclet pasan de 1 durante unos treinta segundos.
    def maximo(clave):
        return float(np.nanmax(np.abs(np.asarray(s[clave]))))

    filas = [
        ("Re de partícula", "Re$_p$", "Re_p",
         "inercia / viscosidad en el poro", "flujo de Darcy, sin inercia"),
        ("Péclet térmico", "Pe$_T$", "Pe_termico",
         "advección / conducción", "el calor se conduce"),
        ("Péclet másico", "Pe$_m$", "Pe_masico",
         "advección / difusión", "las especies difunden"),
        ("Damköhler", "Da", "Da",
         "reacción / transporte", "no hay control por transporte"),
        ("Rayleigh", "Ra", "Ra",
         "boyancia / difusión", "roza el umbral 1708"),
    ]

    def formato(valor):
        return cientifico(valor) if abs(valor) < 0.01 else num(valor, 2)

    cuerpo = "\n".join(
        rf"    {a} & {b} & {formato(maximo(c))} & {formato(s[c][-1])} & {d} & {e} \\"
        for a, b, c, d, e in filas)
    escribir("tabla_fen_adimensionales.tex", (
        r"\begin{tabular}{llrrll}" "\n"
        r"    \hline" "\n"
        r"    Número & Símb. & Máximo & A 720 s & Compara & Consecuencia \\" "\n"
        r"    \hline" "\n"
        f"{cuerpo}\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))
    macro("datPeMasicoMax", num(maximo("Pe_masico"), 1))
    macro("datPeTermicoMax", num(maximo("Pe_termico"), 1))
    macro("datRepMax", num(maximo("Re_p"), 2))
    t = np.asarray(s["t"])
    pico = t[int(np.nanargmax(np.abs(np.asarray(s["Pe_masico"]))))]
    macro("datTPico", num(float(pico), 0))


def tabla_aglomerado(datos):
    s = datos["serie"]
    t = np.asarray(s["t"])
    coh = np.asarray(s["cohesion_max"])
    formados = t[coh >= COHESION_AGLOMERADO]
    t_form = float(formados[0]) if formados.size else float("nan")
    filas = [
        ("Instante en que se forma (cohesión $\\geq$ 0,5)", num(t_form, 0), "s"),
        ("Diámetro de la huella", num(s["D_huella_mm"][-1], 1), "mm"),
        ("Altura resuelta", num(s["altura_aglomerado_mm"][-1], 1), "mm"),
        ("Altura libre predicha (hinchado)", num(s["altura_libre_mm"][-1], 1), "mm"),
        ("Volumen aparente", num(s["volumen_aglomerado_mm3"][-1], 0), "mm$^3$"),
        ("Volumen de materia sólida", num(s["volumen_solido_mm3"][-1], 0), "mm$^3$"),
        ("Diámetro de esfera de igual volumen", num(s["D_aglomerado_mm"][-1], 1), "mm"),
        ("Porosidad final", num(s["eps_media"][-1], 3), "---"),
        ("Factor de hinchamiento", num(s["hinchamiento_medio"][-1], 3), "---"),
        ("Diámetro de partícula de partida", "0,175", "mm"),
    ]
    cuerpo = "\n".join(rf"    {a} & {b} & {c} \\" for a, b, c in filas)
    escribir("tabla_fen_aglomerado.tex", (
        r"\begin{tabular}{lrl}" "\n"
        r"    \hline" "\n"
        r"    Magnitud & Valor & Unidad \\" "\n"
        r"    \hline" "\n"
        f"{cuerpo}\n"
        r"    \hline" "\n"
        r"\end{tabular}"
    ))
    macro("datTFormacion", num(t_form, 0))
    macro("datDiametro", num(s["D_huella_mm"][-1], 1))
    macro("datAltura", num(s["altura_aglomerado_mm"][-1], 1))
    macro("datAlturaLibre", num(s["altura_libre_mm"][-1], 1))
    macro("datDesfe", num(s["D_aglomerado_mm"][-1], 1))


def datos_sueltos(datos):
    s = datos["serie"]
    macro("datPerdidaFinal", num(s["perdida_pct"][-1], 2))
    macro("datNFotogramas", str(len(s["t"])))
    meseta = np.asarray(s["perdida_pct"])[np.asarray(s["t"]) >= 300.0]
    macro("datMeseta", f"{num(float(meseta.min()), 2)}--{num(float(meseta.max()), 2)}")
    # Instante en que la pérdida alcanza el 99 % de su valor final.
    t = np.asarray(s["t"])
    perdida = np.asarray(s["perdida_pct"])
    alcanzado = t[perdida >= 0.99 * perdida[-1]]
    macro("datTMeseta", num(float(alcanzado[0]), 0) if alcanzado.size else "---")
    macro("datRa", num(s["Ra"][-1], 0))
    macro("datRaCritico", num(datos["Ra_critico"], 0))
    macro("datUmax", cientifico(s["u_max"][-1]))
    macro("datFe", num(1000.0 * s["m_Fe"][-1], 2))
    macro("datIlmenita", num(1000.0 * s["m_FeTiO3"][-1], 2))
    macro("datFeo", num(1000.0 * s["m_FeO"][-1], 2))
    hierro_total = s["m_Fe"][-1] + s["m_FeO"][-1] + s["m_Fe3O4"][-1] + s["m_Fe2O3"][-1]
    macro("datPctFeMetal", num(100.0 * s["m_Fe"][-1] / hierro_total, 2) if hierro_total > 0 else "---")
    macro("datEpsFinal", num(s["eps_media"][-1], 3))
    macro("datEpsInicial", num(s["eps_media"][0], 3))
    macro("datHinchamiento", num(s["hinchamiento_medio"][-1], 3))
    macro("datCarbono", num(1000.0 * s["m_C"][-1], 2))
    macro("datVolatilInicial", num(1000.0 * s["m_volatil"][0], 2))
    macro("datVolatilFinal", num(1000.0 * s["m_volatil"][-1], 2))
    macro("datHematitaInicial", num(1000.0 * s["m_Fe2O3"][0], 2))
    macro("datMagnetitaInicial", num(1000.0 * s["m_Fe3O4"][0], 2))
    macro("datMagnetitaFinal", num(1000.0 * s["m_Fe3O4"][-1], 2))
    macro("datMagnetitaVariacion", num(
        100.0 * (s["m_Fe3O4"][-1] - s["m_Fe3O4"][0]) / s["m_Fe3O4"][0], 1))
    # Estado a los 30 y a los 90 s: los dos instantes que cita el texto al
    # contrastar con las observaciones cualitativas del laboratorio.
    macro("datTLechoTreinta", num(_en(s, 30.0, "T_lecho") - 273.15, 0))
    macro("datPerdidaTreinta", num(_en(s, 30.0, "perdida_pct"), 2))
    macro("datHinchamientoNoventa", num(_en(s, 90.0, "hinchamiento_medio"), 2))
    macro("datPerdidaNoventa", num(_en(s, 90.0, "perdida_pct"), 2))
    macro("datVolumenLecho", num(s["volumen_lecho_mm3"][0], 0))
    # Cuánto ha crecido el cuerpo respecto a la partícula de partida.
    d_particula_mm = 0.175
    razon = s["D_aglomerado_mm"][-1] / d_particula_mm
    macro("datCrecimientoLineal", num(razon, 0))
    macro("datCrecimientoVolumen", cientifico(razon ** 3, 1))

    # --- Prueba del iman y enfriado -------------------------------------
    m = np.asarray(s["magnetizacion_Am2_kg"])
    m_lenta = np.asarray(s["magnetizacion_lenta_Am2_kg"])
    macro("datMagnetInicial", num(float(m[0]), 2))
    macro("datMagnetMaxima", num(float(m.max()), 2))
    macro("datMagnetFinal", num(float(m[-1]), 2))
    macro("datMagnetPctFinal", num(100.0 * float(m[-1]) / float(m[0]), 0))
    macro("datMagnetLentaFinal", num(float(m_lenta[-1]), 2))
    macro("datMagnetTMaxima", num(float(t[int(np.argmax(m))]), 0))
    # Momento total, en mA m2: separa la quimica del hierro de la dilucion.
    mom = 1000.0 * np.asarray(s["momento_magnetico_Am2"])
    macro("datMomentoInicial", num(float(mom[0]), 2))
    macro("datMomentoMaximo", num(float(mom.max()), 2))
    macro("datMomentoFinal", num(float(mom[-1]), 2))
    macro("datMomentoPctFinal", num(100.0 * float(mom[-1]) / float(mom[0]), 0))
    enf = datos["enfriamiento"]
    macro("datEnfriaInicial", num(enf["velocidad_inicial_C_min"], 0))
    macro("datEnfriaEutectoide", num(enf["velocidad_en_eutectoide_C_min"], 0))
    macro("datEnfriaUmbral", num(enf["umbral_supresion_C_min"], 0))
    macro("datEnfriaVentana", num(enf["tiempo_en_ventana_eutectoide_s"], 0))
    macro("datEnfriaBiot", num(enf["biot_inicial"], 3))
    macro("datEnfriaCruce", num(enf["t_cruce_eutectoide_s"], 0))


def main() -> int:
    directorio = sys.argv[1] if len(sys.argv) > 1 else None
    datos = diagnosticos(directorio)
    print("tablas y datos de fenomenología:")
    tabla_parametros(datos)
    tabla_fases(datos)
    tabla_fases_por_tiempo(datos)
    tabla_enfriado(datos)
    tabla_rutas_de_enfriado(datos)
    tabla_fronteras(datos)
    tabla_cronologia(datos)
    tabla_adimensionales(datos)
    tabla_aglomerado(datos)
    datos_sueltos(datos)
    escribir_macros()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
