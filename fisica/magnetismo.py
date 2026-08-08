"""
Magnetización del aglomerado a temperatura ambiente, y qué le hace el enfriado.

DATO DE LABORATORIO QUE ORIGINA ESTE MÓDULO
-------------------------------------------
El usuario aportó dos observaciones del ensayo:

1. **El aglomerado se pega al imán.** Al final del ensayo, doce minutos en la
   mufla, **sigue pegándose, pero se siente más débil**.
2. Cuanto más tiempo pasa en la mufla, más débil es la respuesta.

Y una precisión sobre el procedimiento, que resultó ser decisiva: el crisol se
saca de la mufla al cumplirse el tiempo y **se deja enfriar al ambiente**. El
imán se pasa después, en frío.

Es una observación cualitativa —un imán contra una muestra, no un
magnetómetro—, y sirve para **falsar**, no para calibrar. Pero es el único dato
experimental que restringe la composición de fases del producto, porque no hay
caracterización post-ensayo de ninguna clase.

FÍSICA
------
A 900 °C **nada de esto es magnético**: la temperatura de Curie de la magnetita
son 585 °C y la del hierro 770 °C. Lo que mide el imán es el estado de fases
**después de enfriar**, no el que había dentro de la mufla. Por eso el
observable se evalúa a temperatura ambiente y el camino de enfriamiento importa.

A temperatura ambiente, y por kilogramo de fase:

===============  ===================  ==================
Fase             Orden magnético      M_s (A m2/kg)
===============  ===================  ==================
Fe3O4            ferrimagnético       92
Fe                ferromagnético      218
Fe2O3 hematita   antiferro inclinado  0,4
FeO wüstita      paramagnética        0     (T_Neel 198 K)
FeTiO3 ilmenita  paramagnética        0     (T_Neel 40 K)
char, ceniza     diamagnéticos        0
===============  ===================  ==================

La mezcla es **lineal en masa**: los momentos magnéticos son aditivos. No hay
ninguna regla de mezcla que calibrar.

De ahí sale la lectura de la observación. La magnetita vale 92 y la wüstita 0,
así que **reducir magnetita a wüstita apaga el imán**, y hacerlo poco a poco lo
va apagando poco a poco. Es exactamente el fenómeno que la industria conoce
como sobre-reducción en la tostación magnetizante.

EL ENFRIAMIENTO: NO ES UN DETALLE
---------------------------------
Por debajo de 570 °C la wüstita deja de ser estable y se descompone por vía
eutectoide, 4 FeO -> Fe3O4 + Fe. Los dos productos SÍ son magnéticos, y de
hecho por cada mol de hierro dan más momento que la magnetita de partida:

    magnetita               7,10 A m2 por mol de Fe
    eutectoide completo     8,37 A m2 por mol de Fe

O sea que **si el enfriamiento fuese lento, el aglomerado saldría MÁS magnético
que al empezar**, no menos. Como se observa lo contrario, la propia prueba del
imán exige que la wüstita se conserve en buena parte, es decir, que el
enfriamiento sea lo bastante rápido. Eso es información nueva sobre el ensayo,
obtenida sin ningún instrumento.

`enfriamiento_al_aire` calcula la curva y `veredicto_de_temple` la contrasta con
el umbral publicado. El resultado se reporta como **banda entre dos cotas**
—wüstita conservada y wüstita descompuesta del todo—, nunca como número único.

NADA DE ESTO ESTÁ VALIDADO
--------------------------
No hay VSM, ni Mössbauer, ni DRX del aglomerado después del ensayo. La única
restricción es la cualitativa de arriba, fijada como prueba en
`tests/test_magnetismo.py`: el modelo debe dejar el aglomerado todavía
magnético a los 720 s y debe hacerlo decrecer. Cualquier ajuste que apague el
imán antes queda refutado.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .adaptador_v3 import MASAS_MOLARES_SOLIDO_KG_MOL

# ---------------------------------------------------------------------------
# Datos magnéticos
# ---------------------------------------------------------------------------

REFERENCIA_MS = (
    "Hunt, Moskowitz y Banerjee (1995), 'Magnetic Properties of Rocks and "
    "Minerals', en Rock Physics and Phase Relations, AGU Reference Shelf 3, "
    "189-204; Tabla 3 (M_s y T_C a temperatura ambiente) y Fig. 9 (serie "
    "titanohematita). PDF en simulacion_v3/literatura/pdfs/."
)
REFERENCIA_WUSTITA = (
    "La wustita ordena antiferromagneticamente por debajo de T_Neel = 198 K, "
    "muy por debajo del ambiente, de modo que su magnetizacion de saturacion a "
    "300 K es nula a efectos de la prueba del iman."
)
REFERENCIA_TITANOHEMATITA = (
    "Ohara, Naka y Hashishin (2022), Science Advances 8, eabj2487: la solucion "
    "0,5 FeTiO3 - 0,5 Fe2O3 tiene M_s = 1,5 A m2/kg a 300 K."
)
REFERENCIA_EUTECTOIDE = (
    "Zorc, Nagode y Kosec (2024), 'Influence of the Cooling Rate on the Wustite "
    "Content in Oxide Layers...', High Temperature Corrosion of Materials: por "
    "encima de 700 grados C hace falta enfriar a mas de 1000 grados C/min para "
    "suprimir la transformacion eutectoide de la wustita."
)

VALIDACION = (
    "PREDICCION NO VALIDADA - la unica observacion es cualitativa (un iman "
    "contra la muestra); no hay VSM, Mossbauer ni DRX del aglomerado "
    "post-ensayo"
)

# Magnetización de saturación a temperatura ambiente, A m2/kg de fase pura.
# Todas las entradas son de la Tabla 3 de Hunt et al. (1995) salvo donde se
# indica. Las fases sin orden magnético a 300 K llevan cero explícito: no es un
# hueco, es un dato.
MAGNETIZACION_SATURACION_Am2_kg: dict[str, float] = {
    "Fe3O4": 92.0,      # ferrimagnética, T_C = 575-585 °C
    "Fe": 218.0,        # ferromagnético, T_C = 770 °C
    "Fe2O3": 0.4,       # antiferromagnética inclinada, T_N = 675 °C
    "FeO": 0.0,         # T_Neel 198 K: paramagnética a 300 K
    "FeTiO3": 0.0,      # ilmenita pura, T_Neel 40 K (Hunt: -233 °C)
    "TiO2": 0.0,
    "SiO2": 0.0,
    "Fe2SiO4": 0.0,     # fayalita, T_Neel 65 K
    "FeS": 0.0,         # troilita: antiferromagnética
    "C": 0.0,
    "ceniza": 0.0,
    "volatil": 0.0,
    "H2O_liq": 0.0,
}

# Temperaturas de Curie o de Néel, en kelvin. No entran en el cálculo: están
# para poder justificar por qué el observable se evalúa en frío y no a 900 °C.
TEMPERATURA_ORDEN_K: dict[str, float] = {
    "Fe3O4": 858.15,    # 585 °C
    "Fe": 1043.15,      # 770 °C
    "Fe2O3": 948.15,    # 675 °C
    "FeO": 198.0,
    "FeTiO3": 40.0,
}

# CALIBRABLE, rango [0,4 - 10,0] A m2/kg. Magnetización de la titanohematita de
# REF-M, la fase romboédrica R-3 con x(FeTiO3) = 0,49.
#
# Es el caso delicado del módulo y conviene decirlo: con ese x la fase cae justo
# sobre la transición de la serie hematita-ilmenita, que pasa de
# antiferromagnetismo inclinado a ferrimagnetismo en y ~ 0,45 (Hunt et al.,
# Fig. 9). A un lado M_s vale 0,4 y al otro llega a ~30; el valor depende del
# grado de orden Fe/Ti, que **la difracción de laboratorio no puede resolver**
# (Rwp 8,38 frente a 8,39 entre R-3 y R-3c).
#
# Se adopta 1,5 A m2/kg, medido sobre esa misma composición por Ohara et al.
# (2022). Da igual para el resultado: es el 0,5 % de la magnetización total del
# concentrado, así que ni el extremo bajo ni el alto del rango cambian nada de
# lo que se concluye. Se declara igualmente.
MS_TITANOHEMATITA_Am2_kg = 1.5
RANGO_MS_TITANOHEMATITA = (0.4, 10.0)

# En el modelo la fase R-3 va descompuesta en sus dos miembros extremos, y el
# componente FeTiO3 es el único que no reacciona: sirve de trazador de la fase
# entera. Para no perder ni duplicar el momento de la titanohematita se le
# atribuye todo, escalado por la razón de masas fase/componente, 23,15/11,047.
_RAZON_R3_SOBRE_ILMENITA = 23.15 / 11.047217655175212


# ---------------------------------------------------------------------------
# Enfriamiento al aire y descomposición eutectoide
# ---------------------------------------------------------------------------

SIGMA_STEFAN_BOLTZMANN = 5.670374419e-8
TEMPERATURA_EUTECTOIDE_K = 843.15   # 570 °C
# Por debajo de esta temperatura el eutectoide ya no avanza a velocidad
# apreciable en el tiempo de un enfriado al aire. Es un corte declarado, no una
# cinética medida: sirve para acotar la ventana, no para calcular una fracción.
TEMPERATURA_CONGELACION_K = 673.15  # 400 °C
VELOCIDAD_SUPRESION_C_MIN = 1000.0  # Zorc, Nagode y Kosec (2024)

# Geometría y propiedades del crisol Ni-Cr del ensayo, de `casos/carbon_magnetita.yaml`
# y `nucleo/perfil.py`: base 25,0 mm, boca 29,5 mm, altura 32,0 mm,
# 32,67 g de crisol y 15,87 g de tapa, Cp 500 J/kg/K, emisividad 0,80.
CRISOL_ENSAYO: dict[str, float] = {
    "masa_kg": (32.67 + 15.87) * 1.0e-3,
    "calor_especifico_J_kg_K": 500.0,
    # Cono truncado: lateral pi*(r1+r2)*s con s = sqrt(h^2 + (r2-r1)^2), mas
    # fondo y tapa. Se ignora el collar de la boca, que cambia el área en menos
    # del 2 % y no altera el veredicto.
    "area_m2": (
        math.pi * (0.0125 + 0.01475) * math.hypot(0.032, 0.01475 - 0.0125)
        + math.pi * 0.0125**2
        + math.pi * 0.01475**2
    ),
    # Altura del crisol: escala de la correlación de convección natural.
    "longitud_caracteristica_m": 0.032,
    # Volumen de METAL, masa/densidad; con el área da la longitud del Biot.
    "volumen_solido_m3": (32.67 + 15.87) * 1.0e-3 / 8400.0,
    "emisividad": 0.80,
    "conductividad_W_m_K": 16.0,
}

# El mismo crisol SIN la tapa. Pesa un tercio menos y por eso se enfría más
# rápido; el área cambia poco, porque la boca abierta radia por la cavidad casi
# lo mismo que radiaba el disco de la tapa. Sirve para separar el efecto térmico
# de la tapa del efecto químico, que es el que de verdad importa (véase
# `reoxidacion_con_la_tapa_puesta`).
CRISOL_SIN_TAPA: dict[str, float] = {
    "masa_kg": 32.67e-3,
    "calor_especifico_J_kg_K": 500.0,
    "area_m2": (
        math.pi * (0.0125 + 0.01475) * math.hypot(0.032, 0.01475 - 0.0125)
        + math.pi * 0.0125**2
        + math.pi * 0.01475**2
    ),
    "longitud_caracteristica_m": 0.032,
    "volumen_solido_m3": 32.67e-3 / 8400.0,
    "emisividad": 0.80,
    "conductividad_W_m_K": 16.0,
}

# El aglomerado solo, si se volcase fuera del crisol: disco de 22,8 mm de
# diámetro y ~5,4 mm de altura (el lecho hinchado), 0,72 g, Cp ~1000 J/kg/K.
AGLOMERADO_SOLO: dict[str, float] = {
    "masa_kg": 0.72e-3,
    "calor_especifico_J_kg_K": 1000.0,
    "area_m2": 2.0 * math.pi * 0.0114**2 + math.pi * 0.0228 * 0.0054,
    "longitud_caracteristica_m": 0.0054,
    "volumen_solido_m3": math.pi * 0.0114**2 * 0.0054,
    "emisividad": 0.85,
    "conductividad_W_m_K": 1.2,
}


def _coeficiente_conveccion(T_K: float, T_ambiente_K: float, longitud_m: float) -> float:
    """Convección natural en aire, correlación simplificada de placa/cilindro.

    ``h = 1,42 (dT/L)**0.25`` en W/m2/K (Holman, *Heat Transfer*, tabla de
    correlaciones simplificadas para aire a presión atmosférica). Es el término
    menor: a 900 °C la radiación aporta el 84 % del flujo.
    """

    delta = max(float(T_K) - float(T_ambiente_K), 0.0)
    if delta <= 0.0 or longitud_m <= 0.0:
        return 0.0
    return 1.42 * (delta / float(longitud_m)) ** 0.25


def flujo_de_enfriamiento_W_m2(
    T_K: float, T_ambiente_K: float = 298.15, *, cuerpo: Mapping[str, float] | None = None
) -> float:
    """Radiación más convección natural, en W/m2."""

    datos = dict(CRISOL_ENSAYO if cuerpo is None else cuerpo)
    T = float(T_K)
    Tamb = float(T_ambiente_K)
    if not math.isfinite(T) or T <= 0.0 or not math.isfinite(Tamb) or Tamb <= 0.0:
        raise ValueError("las temperaturas deben ser finitas y mayores que 0 K")
    radiacion = (
        float(datos["emisividad"]) * SIGMA_STEFAN_BOLTZMANN * (T**4 - Tamb**4)
    )
    conveccion = _coeficiente_conveccion(
        T, Tamb, float(datos["longitud_caracteristica_m"])
    ) * (T - Tamb)
    return float(radiacion + conveccion)


def numero_de_biot(
    T_K: float, T_ambiente_K: float = 298.15, *, cuerpo: Mapping[str, float] | None = None
) -> float:
    """Biot con el coeficiente total equivalente y la longitud V/A.

    Debe salir bastante menor que 1 para que la capacidad concentrada valga; si
    no, hay gradiente dentro del cuerpo y la curva es sólo indicativa. Se usa la
    longitud característica estándar, volumen de sólido entre área de
    intercambio, no el espesor: para el crisol el metal es una lámina de 1,1 mm
    en un cuerpo de 32 mm de alto, y el espesor sobrestimaría el Biot por un
    factor 20.
    """

    datos = dict(CRISOL_ENSAYO if cuerpo is None else cuerpo)
    delta = float(T_K) - float(T_ambiente_K)
    if delta <= 0.0:
        return 0.0
    h_total = flujo_de_enfriamiento_W_m2(T_K, T_ambiente_K, cuerpo=datos) / delta
    longitud = float(datos["volumen_solido_m3"]) / float(datos["area_m2"])
    return float(h_total * longitud / float(datos["conductividad_W_m_K"]))


def enfriamiento_al_aire(
    T_inicial_K: float = 1173.15,
    T_ambiente_K: float = 298.15,
    *,
    cuerpo: Mapping[str, float] | None = None,
    T_final_K: float = 473.15,
    dt_s: float = 0.05,
    t_max_s: float = 3600.0,
) -> dict[str, Any]:
    """Curva de enfriamiento por capacidad concentrada, con radiación y convección.

    Devuelve la trayectoria y, sobre todo, lo que decide la lectura del imán:
    la velocidad de enfriamiento al cruzar el eutectoide y el tiempo que el
    cuerpo pasa en la ventana en que la wüstita puede descomponerse.
    """

    datos = dict(CRISOL_ENSAYO if cuerpo is None else cuerpo)
    capacidad = float(datos["masa_kg"]) * float(datos["calor_especifico_J_kg_K"])
    if capacidad <= 0.0:
        raise ValueError("la capacidad térmica del cuerpo debe ser positiva")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s debe ser finito y positivo")

    T = float(T_inicial_K)
    t = 0.0
    tiempos = [t]
    temperaturas = [T]
    while T > float(T_final_K) and t < float(t_max_s):
        flujo = flujo_de_enfriamiento_W_m2(T, T_ambiente_K, cuerpo=datos)
        dT = -flujo * float(datos["area_m2"]) / capacidad * dt_s
        T = max(T + dT, float(T_ambiente_K) + 1.0e-9)
        t += dt_s
        tiempos.append(t)
        temperaturas.append(T)

    t_arr = np.asarray(tiempos, dtype=float)
    T_arr = np.asarray(temperaturas, dtype=float)

    def _cruce(objetivo: float) -> float | None:
        indices = np.flatnonzero(T_arr <= objetivo)
        return float(t_arr[indices[0]]) if indices.size else None

    def _velocidad_C_min(T_objetivo: float) -> float:
        flujo = flujo_de_enfriamiento_W_m2(T_objetivo, T_ambiente_K, cuerpo=datos)
        return float(60.0 * flujo * float(datos["area_m2"]) / capacidad)

    t_eutectoide = _cruce(TEMPERATURA_EUTECTOIDE_K)
    t_congelacion = _cruce(TEMPERATURA_CONGELACION_K)
    ventana = (
        None if t_eutectoide is None or t_congelacion is None
        else float(t_congelacion - t_eutectoide)
    )

    return {
        "tiempo_s": t_arr,
        "temperatura_K": T_arr,
        "capacidad_termica_J_K": capacidad,
        "biot_inicial": numero_de_biot(T_inicial_K, T_ambiente_K, cuerpo=datos),
        "velocidad_inicial_C_min": _velocidad_C_min(float(T_inicial_K)),
        "velocidad_en_eutectoide_C_min": _velocidad_C_min(TEMPERATURA_EUTECTOIDE_K),
        "velocidad_en_congelacion_C_min": _velocidad_C_min(TEMPERATURA_CONGELACION_K),
        "t_cruce_eutectoide_s": t_eutectoide,
        "t_cruce_congelacion_s": t_congelacion,
        "tiempo_en_ventana_eutectoide_s": ventana,
        "umbral_supresion_C_min": VELOCIDAD_SUPRESION_C_MIN,
        "referencia_eutectoide": REFERENCIA_EUTECTOIDE,
    }


# CALIBRABLES, los dos. Cinética del eutectoide de la wüstita al enfriar,
# escrita como una ley de Johnson--Mehl--Avrami sobre el tiempo que el cuerpo
# pasa en la ventana 570-400 °C:
#
#     f(t) = 1 - exp(-(t/tau)**n)
#
# De dónde salen tau y n, porque no son un ajuste a este ensayo sino a los dos
# únicos puntos medidos que se han encontrado. Zorc, Nagode y Kosec (2024)
# reportan, para muestras oxidadas a 700 °C, fracciones de wüstita en la capa de
# óxido de 0,17 enfriando a 100 °C/min y 0,41 enfriando a 1000 °C/min. Con la
# ventana de 170 K eso son 102 s y 10,2 s dentro de ella. Normalizando por la
# wüstita presente a temperatura (SUPUESTO: 0,50 de la capa; su cálculo CALPHAD
# da hasta 0,72 en la interfaz y menos hacia fuera) resultan extensiones de
# descomposición de 0,66 y 0,18, y de ahí tau = 92,3 s y n = 0,735.
#
# El supuesto del 0,50 es el eslabón débil de toda la cadena, y por eso el
# resultado principal NO es este número: son las DOS COTAS, f=0 y f=1. Este
# valor central se reporta como estimación ilustrativa y nada más. Los rangos
# cubren desde una cinética diez veces más rápida hasta diez veces más lenta.
TIEMPO_CARACTERISTICO_EUTECTOIDE_S = 92.3
RANGO_TIEMPO_CARACTERISTICO_S = (9.0, 900.0)
EXPONENTE_AVRAMI_EUTECTOIDE = 0.735
RANGO_EXPONENTE_AVRAMI = (0.5, 1.5)


def fraccion_eutectoide(
    tiempo_en_ventana_s: float,
    *,
    tau_s: float = TIEMPO_CARACTERISTICO_EUTECTOIDE_S,
    n: float = EXPONENTE_AVRAMI_EUTECTOIDE,
) -> float:
    """Fracción de la wüstita que alcanza a descomponerse al enfriar.

    0 es temple perfecto y 1 es descomposición completa. Es una ESTIMACIÓN
    ilustrativa: lo que se reporta como resultado son las dos cotas.
    """

    t = float(tiempo_en_ventana_s)
    if not math.isfinite(t) or t < 0.0:
        raise ValueError("tiempo_en_ventana_s debe ser finito y no negativo")
    if float(tau_s) <= 0.0 or float(n) <= 0.0:
        raise ValueError("tau_s y n deben ser positivos")
    if t == 0.0:
        return 0.0
    return float(1.0 - math.exp(-((t / float(tau_s)) ** float(n))))


# Lo que este modelo de enfriado NO representa, dicho antes de que alguien lo
# lea como una lista completa de fases del producto.
NO_MODELADO_AL_ENFRIAR = (
    "Reoxidacion en aire. El aglomerado sale a 900 grados C y se enfria al aire, "
    "de modo que su superficie puede reoxidarse: magnetita a hematita o "
    "maghemita, y hierro metalico a oxido. Iria en contra del magnetismo y "
    "restaria a las dos cotas, sobre todo a la de enfriamiento lento, que es la "
    "que pasa mas tiempo caliente.",
    "Combustion del char. Queda carbono fijo en cantidad y sale al aire "
    "incandescente; parte puede arder durante el enfriado. Eso quitaria masa no "
    "magnetica y SUBIRIA la magnetizacion especifica sin cambiar ninguna fase de "
    "hierro.",
    "Gradientes dentro del cuerpo. La capacidad concentrada da una sola "
    "temperatura; la superficie se enfria antes que el nucleo, asi que la "
    "fraccion de eutectoide no es uniforme.",
)


# Volumen libre dentro del crisol tapado, medido sobre la corrida: las celdas de
# lecho y de gas suman 15,47 cm3.
VOLUMEN_LIBRE_CRISOL_M3 = 15.47e-6

# Composición del gas residual al final de la corrida, en fracciones del total.
# Sólo importan las especies OXIDANTES frente a la wüstita, CO2 y H2O.
FRACCION_OXIDANTE_GAS_RESIDUAL = 0.207


def reoxidacion_con_la_tapa_puesta(
    moles_FeO: float,
    *,
    volumen_libre_m3: float = VOLUMEN_LIBRE_CRISOL_M3,
    T_K: float = 1173.15,
    fraccion_oxidante: float = FRACCION_OXIDANTE_GAS_RESIDUAL,
    presion_Pa: float = 101325.0,
) -> dict[str, Any]:
    """Cuánta wüstita puede devolver a magnetita el gas encerrado al enfriar.

    Con la tapa puesta pasan tres cosas distintas, y conviene no mezclarlas:

    1. **Desaparece la reoxidación por aire** y la combustión del char. La tapa
       es buena noticia para la limpieza de la predicción.
    2. **Aparece otra reoxidación, por el propio gas del crisol.** La frontera
       Fe3O4/FeO sube al bajar la temperatura --- 0,322 a 900 grados C, 0,456 a
       700, 0,589 en el eutectoide ---, mientras el gas se queda donde estaba.
       Es decir, un gas que a 900 grados C estaba justo en la frontera pasa a
       ser oxidante conforme el crisol se enfría, y devuelve wüstita a
       magnetita: 3 FeO + CO2 -> Fe3O4 + CO.
    3. El proceso **se autolimita**: al oxidar consume CO2 y produce CO, con lo
       que sube x_CO hasta reencontrar la frontera. El tope real es, por tanto,
       el inventario de oxidante que quepa en el crisol.

    Esa es la cuenta que hace esta función. Va en la MISMA dirección que el
    eutectoide: las dos devuelven magnetismo.

    OJO con usar el inventario de gas de la corrida para esto. El venteo del
    modelo es un sumidero de una sola dirección, así que al final del cálculo el
    crisol queda casi vacío ---unos 380 Pa--- y su inventario subestimaría la
    capacidad en tres órdenes de magnitud. Aquí se supone el crisol lleno a la
    presión que le corresponde, que es lo físico.
    """

    n_FeO = float(moles_FeO)
    if not math.isfinite(n_FeO) or n_FeO < 0.0:
        raise ValueError("moles_FeO debe ser finito y no negativo")
    if not 0.0 <= float(fraccion_oxidante) <= 1.0:
        raise ValueError("fraccion_oxidante debe pertenecer a [0, 1]")

    R = 8.314462618
    n_gas = float(presion_Pa) * float(volumen_libre_m3) / (R * float(T_K))
    n_oxidante = n_gas * float(fraccion_oxidante)
    # 3 FeO + CO2 -> Fe3O4 + CO: tres moles de wüstita por mol de oxidante.
    n_convertible = 3.0 * n_oxidante
    fraccion = 0.0 if n_FeO <= 0.0 else min(1.0, n_convertible / n_FeO)
    return {
        "moles_gas": n_gas,
        "moles_oxidante": n_oxidante,
        "moles_FeO_convertibles": n_convertible,
        "fraccion_de_la_wustita": fraccion,
        "reaccion": "3 FeO + CO2 -> Fe3O4 + CO",
        "nota": (
            "Cota de CAPACIDAD, no de cinetica: dice cuanta wustita alcanzaria a "
            "reoxidar el gas encerrado, no cuanta lo hace en el tiempo del "
            "enfriado. Va en la misma direccion que el eutectoide."
        ),
    }


def fases_tras_enfriar(
    solido_mol_m3: Mapping[str, Any],
    volumen_celda_m3: Any,
    fraccion_descompuesta: float = 0.0,
) -> dict[str, float]:
    """Inventario de fases A TEMPERATURA AMBIENTE, en moles.

    Es lo que encontraría un DRX del aglomerado recuperado, no lo que hay dentro
    de la mufla. La única transformación que se aplica es el eutectoide de la
    wüstita, 4 FeO -> Fe3O4 + Fe, con la extensión que se le pase. Todo lo demás
    se congela: la ilmenita, la titanohematita, el char y las cenizas no tienen
    ninguna transformación accesible entre 900 °C y el ambiente.

    Véase :data:`NO_MODELADO_AL_ENFRIAR` para lo que queda fuera.
    """

    f = float(fraccion_descompuesta)
    if not math.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ValueError("fraccion_descompuesta debe pertenecer a [0, 1]")

    volumen = np.asarray(volumen_celda_m3, dtype=float)
    # Se parte de TODAS las fases del inventario en cero, aunque no vengan en la
    # entrada: el resultado es una lista de fases completa, con sus ausencias
    # dichas explícitamente, y no una lista de las que casualmente había.
    moles: dict[str, float] = dict.fromkeys(MASAS_MOLARES_SOLIDO_KG_MOL, 0.0)
    for fase, concentracion in solido_mol_m3.items():
        if fase.startswith("_") or fase not in MASAS_MOLARES_SOLIDO_KG_MOL:
            continue
        moles[fase] = float(np.sum(np.asarray(concentracion, dtype=float) * volumen))

    n_FeO = moles.get("FeO", 0.0)
    if f > 0.0 and n_FeO > 0.0:
        transformada = f * n_FeO
        moles["FeO"] = n_FeO - transformada
        moles["Fe3O4"] = moles.get("Fe3O4", 0.0) + 0.25 * transformada
        moles["Fe"] = moles.get("Fe", 0.0) + 0.25 * transformada
    return moles


def masas_tras_enfriar_g(
    solido_mol_m3: Mapping[str, Any],
    volumen_celda_m3: Any,
    fraccion_descompuesta: float = 0.0,
) -> dict[str, float]:
    """Lo mismo que :func:`fases_tras_enfriar`, en gramos por fase."""

    moles = fases_tras_enfriar(solido_mol_m3, volumen_celda_m3, fraccion_descompuesta)
    return {
        fase: 1000.0 * n * MASAS_MOLARES_SOLIDO_KG_MOL[fase]
        for fase, n in moles.items()
    }


def veredicto_de_temple(
    *, cuerpo: Mapping[str, float] | None = None, **parametros: Any
) -> dict[str, Any]:
    """¿Es un temple, o no? La pregunta que hizo el laboratorio, con números.

    El criterio publicado es la velocidad de enfriamiento **al cruzar el
    eutectoide**, no la inicial: por encima de 1000 °C/min la transformación
    queda suprimida y la wüstita se conserva.
    """

    curva = enfriamiento_al_aire(cuerpo=cuerpo, **parametros)
    velocidad = float(curva["velocidad_en_eutectoide_C_min"])
    razon = velocidad / VELOCIDAD_SUPRESION_C_MIN
    if razon >= 1.0:
        veredicto = "TEMPLE: la wustita se conserva"
    elif razon >= 0.1:
        veredicto = "TEMPLE PARCIAL: la wustita se conserva en parte"
    else:
        veredicto = "ENFRIAMIENTO LENTO: la wustita se descompone"
    return {
        **curva,
        "razon_sobre_umbral": razon,
        "veredicto": veredicto,
    }


# ---------------------------------------------------------------------------
# Magnetización del inventario de fases
# ---------------------------------------------------------------------------


def _masas_por_fase_kg(
    solido_mol_m3: Mapping[str, Any], volumen_celda_m3: Any
) -> dict[str, np.ndarray]:
    volumen = np.asarray(volumen_celda_m3, dtype=float)
    masas: dict[str, np.ndarray] = {}
    for fase, concentracion in solido_mol_m3.items():
        if fase.startswith("_") or fase not in MASAS_MOLARES_SOLIDO_KG_MOL:
            continue
        c = np.asarray(concentracion, dtype=float)
        masas[fase] = c * volumen * MASAS_MOLARES_SOLIDO_KG_MOL[fase]
    return masas


def momento_magnetico_Am2(
    solido_mol_m3: Mapping[str, Any],
    volumen_celda_m3: Any,
    *,
    wustita_descompuesta: float = 0.0,
    ms_titanohematita: float = MS_TITANOHEMATITA_Am2_kg,
) -> float:
    """Momento magnético total del sólido, a temperatura ambiente, en A m2.

    `wustita_descompuesta` es la fracción de la wüstita que se transforma al
    enfriar por la vía eutectoide 4 FeO -> Fe3O4 + Fe. 0 es temple perfecto
    (la wüstita se conserva y no aporta nada) y 1 es enfriamiento lento
    (la wüstita se convierte entera en dos fases que sí son magnéticas). Las dos
    cotas se reportan siempre juntas: véase `resumen`.
    """

    fraccion = float(wustita_descompuesta)
    if not math.isfinite(fraccion) or not 0.0 <= fraccion <= 1.0:
        raise ValueError("wustita_descompuesta debe pertenecer a [0, 1]")
    ms_r3 = float(ms_titanohematita)
    if not math.isfinite(ms_r3) or ms_r3 < 0.0:
        raise ValueError("ms_titanohematita debe ser finita y no negativa")

    masas = _masas_por_fase_kg(solido_mol_m3, volumen_celda_m3)
    momento = 0.0
    for fase, masa in masas.items():
        ms = MAGNETIZACION_SATURACION_Am2_kg.get(fase, 0.0)
        if ms:
            momento += ms * float(np.sum(masa))

    # La titanohematita: se le atribuye a través de su componente FeTiO3, que es
    # el trazador de la fase entera porque no reacciona.
    if "FeTiO3" in masas:
        momento += ms_r3 * _RAZON_R3_SOBRE_ILMENITA * float(np.sum(masas["FeTiO3"]))

    # Descomposición eutectoide de la wüstita al enfriar: 4 FeO -> Fe3O4 + Fe.
    if fraccion > 0.0 and "FeO" in masas:
        n_FeO = float(np.sum(masas["FeO"])) / MASAS_MOLARES_SOLIDO_KG_MOL["FeO"]
        n_transformada = fraccion * n_FeO
        momento += 0.25 * n_transformada * MASAS_MOLARES_SOLIDO_KG_MOL["Fe3O4"] * (
            MAGNETIZACION_SATURACION_Am2_kg["Fe3O4"]
        )
        momento += 0.25 * n_transformada * MASAS_MOLARES_SOLIDO_KG_MOL["Fe"] * (
            MAGNETIZACION_SATURACION_Am2_kg["Fe"]
        )
    return float(momento)


def masa_solida_kg(solido_mol_m3: Mapping[str, Any], volumen_celda_m3: Any) -> float:
    """Masa total del sólido, para normalizar la magnetización."""

    masas = _masas_por_fase_kg(solido_mol_m3, volumen_celda_m3)
    return float(sum(float(np.sum(m)) for m in masas.values()))


def magnetizacion_Am2_kg(
    solido_mol_m3: Mapping[str, Any],
    volumen_celda_m3: Any,
    **parametros: Any,
) -> float:
    """Magnetización de saturación específica del sólido, en A m2/kg."""

    masa = masa_solida_kg(solido_mol_m3, volumen_celda_m3)
    if masa <= 0.0:
        return 0.0
    return momento_magnetico_Am2(solido_mol_m3, volumen_celda_m3, **parametros) / masa


def cotas_magnetizacion_Am2_kg(
    solido_mol_m3: Mapping[str, Any],
    volumen_celda_m3: Any,
    *,
    ms_titanohematita: float = MS_TITANOHEMATITA_Am2_kg,
) -> dict[str, float]:
    """Las dos cotas del observable, según qué le pase a la wüstita al enfriar.

    `temple` conserva la wüstita, que no es magnética a temperatura ambiente, y
    da la cota INFERIOR. `enfriamiento_lento` la descompone entera en
    magnetita más hierro, y da la SUPERIOR. La lectura real del imán está entre
    las dos, y más cerca de la primera cuanto más rápido se enfríe.
    """

    comun = {"ms_titanohematita": ms_titanohematita}
    return {
        "temple": magnetizacion_Am2_kg(
            solido_mol_m3, volumen_celda_m3, wustita_descompuesta=0.0, **comun
        ),
        "enfriamiento_lento": magnetizacion_Am2_kg(
            solido_mol_m3, volumen_celda_m3, wustita_descompuesta=1.0, **comun
        ),
    }


# Enfriamiento lento dentro de la mufla apagada. No se ha medido la curva de
# este horno: 8 °C/min es un valor representativo de una mufla de laboratorio de
# ~2 kW con la puerta cerrada, y se usa sólo como COTA de enfriamiento lento.
VELOCIDAD_MUFLA_APAGADA_C_MIN = 8.0

# Nombres de las rutas. `RUTA_CON_TAPA` es el procedimiento que se sigue hoy en
# el laboratorio: se saca el crisol tapado y se deja enfriar al ambiente.
RUTA_VOLCADO = "Volcado fuera del crisol"
RUTA_SIN_TAPA = "Al aire, sin tapa"
RUTA_CON_TAPA = "Al aire, con tapa"
RUTA_MUFLA = "En la mufla apagada, con tapa"
RUTA_DEL_ENSAYO = RUTA_CON_TAPA


def parametros_de_ruta(
    nombre: str, moles_FeO: float, T_extraccion_K: float = 1173.15
) -> dict[str, float]:
    """Extensiones de cada transformación para una ruta de enfriado dada.

    Devuelve lo que hay que pasarle a :func:`composicion_tras_ruta`. Se expone
    para que el informe pueda tabular el procedimiento REAL del laboratorio
    ---con la tapa puesta--- y no sólo las dos cotas abstractas.

    `T_extraccion_K` es la temperatura a la que estaba el cuerpo al sacarlo.
    Sólo interviene en la oxidación al aire, que se apaga por debajo de
    :data:`TEMPERATURA_MINIMA_OXIDACION_AIRE_K`.
    """

    cuerpos = {
        RUTA_VOLCADO: (AGLOMERADO_SOLO, False),
        RUTA_SIN_TAPA: (CRISOL_SIN_TAPA, False),
        RUTA_CON_TAPA: (CRISOL_ENSAYO, True),
        RUTA_MUFLA: (None, True),
    }
    if nombre not in cuerpos:
        raise ValueError(f"ruta desconocida: {nombre!r}; use {tuple(cuerpos)}")
    cuerpo, con_tapa = cuerpos[nombre]
    if cuerpo is None:
        ventana = 170.0 / VELOCIDAD_MUFLA_APAGADA_C_MIN * 60.0
    else:
        ventana = float(
            veredicto_de_temple(cuerpo=cuerpo)["tiempo_en_ventana_eutectoide_s"]
        )
    return {
        "tiempo_en_ventana_s": ventana,
        "f_eutectoide": fraccion_eutectoide(ventana),
        "f_reoxidacion_gas": (
            reoxidacion_con_la_tapa_puesta(moles_FeO)["fraccion_de_la_wustita"]
            if con_tapa else 0.0
        ),
        "f_oxidacion_aire": (
            0.0
            if con_tapa or float(T_extraccion_K) < TEMPERATURA_MINIMA_OXIDACION_AIRE_K
            else FRACCION_REOXIDADA_AL_AIRE
        ),
    }

# CALIBRABLE. Fracción de la magnetita que llega a reoxidarse a hematita en la
# superficie cuando el aglomerado se enfría destapado, expuesto al aire. No hay
# medida de esta muestra: es un orden de magnitud para poder comparar rutas, y
# el rango cubre desde despreciable hasta una costra apreciable. Su efecto es el
# contrario al de todo lo demás: RESTA magnetismo.
FRACCION_REOXIDADA_AL_AIRE = 0.05
RANGO_FRACCION_REOXIDADA_AL_AIRE = (0.0, 0.20)

# Por debajo de esta temperatura la oxidación de la magnetita al aire no avanza
# de forma apreciable en el tiempo de un enfriado. Sin esta compuerta el modelo
# decía que un aglomerado sacado a los \SI{30}{\second} ---que sigue a 205 °C y
# no ha reaccionado--- saldría con hematita en la superficie, lo cual es
# absurdo: si nunca estuvo caliente, no hay nada que oxidar al enfriarse.
TEMPERATURA_MINIMA_OXIDACION_AIRE_K = 673.15  # 400 °C


def composicion_tras_ruta(
    solido_mol_m3: Mapping[str, Any],
    volumen_celda_m3: Any,
    *,
    f_eutectoide: float,
    f_reoxidacion_gas: float = 0.0,
    f_oxidacion_aire: float = 0.0,
) -> dict[str, Any]:
    """Composición completa y magnetización tras una ruta de enfriado concreta.

    Se aplican en orden y son transformaciones DISTINTAS, no una misma cuenta:

    1. reoxidación de la wüstita por el gas encerrado, que sólo ocurre con la
       tapa puesta: 3 FeO + CO2 -> Fe3O4 + CO. Produce magnetita y **nada de
       hierro metálico**;
    2. eutectoide sobre la wüstita que quede: 4 FeO -> Fe3O4 + Fe. Éste **sí**
       produce hierro metálico;
    3. si el cuerpo está destapado, oxidación superficial de la magnetita al
       aire: 4 Fe3O4 + O2 -> 6 Fe2O3.

    Confundir (1) con (2) daría el mismo balance de wüstita pero un reparto
    Fe3O4/Fe distinto, y con él una magnetización distinta.
    """

    moles = fases_tras_enfriar(solido_mol_m3, volumen_celda_m3, 0.0)

    n_FeO = moles["FeO"]
    if f_reoxidacion_gas > 0.0 and n_FeO > 0.0:
        convertida = min(1.0, f_reoxidacion_gas) * n_FeO
        moles["FeO"] = n_FeO - convertida
        moles["Fe3O4"] += convertida / 3.0

    n_FeO = moles["FeO"]
    if f_eutectoide > 0.0 and n_FeO > 0.0:
        transformada = min(1.0, f_eutectoide) * n_FeO
        moles["FeO"] = n_FeO - transformada
        moles["Fe3O4"] += 0.25 * transformada
        moles["Fe"] += 0.25 * transformada

    if f_oxidacion_aire > 0.0 and moles["Fe3O4"] > 0.0:
        oxidada = min(1.0, f_oxidacion_aire) * moles["Fe3O4"]
        moles["Fe3O4"] -= oxidada
        moles["Fe2O3"] += 1.5 * oxidada

    masa = sum(n * MASAS_MOLARES_SOLIDO_KG_MOL[f] for f, n in moles.items())
    momento = sum(
        n * MASAS_MOLARES_SOLIDO_KG_MOL[f] * MAGNETIZACION_SATURACION_Am2_kg.get(f, 0.0)
        for f, n in moles.items()
    )
    momento += (
        MS_TITANOHEMATITA_Am2_kg
        * _RAZON_R3_SOBRE_ILMENITA
        * moles.get("FeTiO3", 0.0)
        * MASAS_MOLARES_SOLIDO_KG_MOL["FeTiO3"]
    )
    # Las masas molares están en kg/mol: para miligramos hace falta 1e6.
    masas_mg = {
        f: 1.0e6 * n * MASAS_MOLARES_SOLIDO_KG_MOL[f] for f, n in moles.items()
    }
    return {
        "moles": moles,
        "masas_mg": masas_mg,
        "Fe3O4_mg": masas_mg["Fe3O4"],
        "FeO_mg": masas_mg["FeO"],
        "Fe_mg": masas_mg["Fe"],
        "Fe2O3_mg": masas_mg["Fe2O3"],
        "masa_g": 1000.0 * masa,
        "magnetizacion_Am2_kg": momento / masa if masa > 0.0 else 0.0,
    }


# Nombre anterior, privado. Se conserva porque lo usa `evaluacion_rutas_de_enfriado`.
_magnetizacion_de_ruta = composicion_tras_ruta


def evaluacion_rutas_de_enfriado(
    solido_mol_m3: Mapping[str, Any], volumen_celda_m3: Any
) -> list[dict[str, Any]]:
    """Compara las rutas de enfriado por lo magnético que dejan el aglomerado.

    El criterio no es aquí conservar el estado de \\SI{900}{\\celsius}: es
    **maximizar la respuesta al imán**, porque de eso depende poder recuperar el
    material con un imán después de usarlo como adsorbente.

    Y el resultado es contraintuitivo. Todo lo que devuelve wüstita a magnetita
    ---el eutectoide y la reoxidación por el gas encerrado--- SUMA magnetismo,
    porque la wüstita no responde al imán y la magnetita sí. Lo único que resta
    es el aire, que oxida magnetita a hematita. De modo que, para este objetivo,
    conviene **enfriar despacio y con la tapa puesta**, justo lo contrario de lo
    que haría falta para fotografiar el estado de la mufla.
    """

    n_FeO = fases_tras_enfriar(solido_mol_m3, volumen_celda_m3, 0.0)["FeO"]
    tope_gas = reoxidacion_con_la_tapa_puesta(n_FeO)["fraccion_de_la_wustita"]

    def ventana(cuerpo):
        return float(
            veredicto_de_temple(cuerpo=cuerpo)["tiempo_en_ventana_eutectoide_s"]
        )

    # 570 -> 400 °C son 170 K; a velocidad constante, el tiempo en la ventana.
    ventana_mufla = 170.0 / VELOCIDAD_MUFLA_APAGADA_C_MIN * 60.0

    rutas = tuple(
        (nombre, t_ventana, f_gas, f_aire)
        for nombre, t_ventana, f_gas, f_aire in (
            (RUTA_VOLCADO, ventana(AGLOMERADO_SOLO), 0.0, FRACCION_REOXIDADA_AL_AIRE),
            (RUTA_SIN_TAPA, ventana(CRISOL_SIN_TAPA), 0.0, FRACCION_REOXIDADA_AL_AIRE),
            (RUTA_CON_TAPA, ventana(CRISOL_ENSAYO), tope_gas, 0.0),
            (RUTA_MUFLA, ventana_mufla, tope_gas, 0.0),
        )
    )

    salida: list[dict[str, Any]] = []
    for nombre, t_ventana, f_gas, f_aire in rutas:
        f_eut = fraccion_eutectoide(t_ventana)
        estado = _magnetizacion_de_ruta(
            solido_mol_m3, volumen_celda_m3,
            f_eutectoide=f_eut, f_reoxidacion_gas=f_gas, f_oxidacion_aire=f_aire,
        )
        salida.append({
            "ruta": nombre,
            "tiempo_en_ventana_s": t_ventana,
            "fraccion_eutectoide": f_eut,
            "fraccion_reoxidada_por_gas": f_gas,
            "fraccion_oxidada_al_aire": f_aire,
            **estado,
        })
    return salida


def resumen(
    solido_mol_m3: Mapping[str, Any] | None = None,
    volumen_celda_m3: Any = None,
    *,
    cuerpo: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Ficha del modelo, para que la interfaz muestre de dónde sale cada cosa."""

    ficha: dict[str, Any] = {
        "magnetizacion_saturacion_Am2_kg": dict(MAGNETIZACION_SATURACION_Am2_kg),
        "temperatura_orden_K": dict(TEMPERATURA_ORDEN_K),
        "ms_titanohematita_Am2_kg": MS_TITANOHEMATITA_Am2_kg,
        "rango_ms_titanohematita": RANGO_MS_TITANOHEMATITA,
        # Para que el cliente pueda repetir la cuenta sin escribir mineralogía.
        "razon_titanohematita_sobre_ilmenita": _RAZON_R3_SOBRE_ILMENITA,
        "eutectoide": {
            "reaccion": "4 FeO -> Fe3O4 + Fe",
            "moles_Fe3O4_por_mol_FeO": 0.25,
            "moles_Fe_por_mol_FeO": 0.25,
        },
        "regla_de_mezcla": "lineal en masa; los momentos magneticos son aditivos",
        "referencia_ms": REFERENCIA_MS,
        "referencia_wustita": REFERENCIA_WUSTITA,
        "referencia_titanohematita": REFERENCIA_TITANOHEMATITA,
        "validacion": VALIDACION,
    }
    enfriado = veredicto_de_temple(cuerpo=cuerpo)
    ficha["enfriamiento"] = {
        clave: enfriado[clave]
        for clave in (
            "capacidad_termica_J_K",
            "biot_inicial",
            "velocidad_inicial_C_min",
            "velocidad_en_eutectoide_C_min",
            "tiempo_en_ventana_eutectoide_s",
            "t_cruce_eutectoide_s",
            "umbral_supresion_C_min",
            "razon_sobre_umbral",
            "veredicto",
            "referencia_eutectoide",
        )
    }
    if solido_mol_m3 is not None and volumen_celda_m3 is not None:
        ficha["cotas_Am2_kg"] = cotas_magnetizacion_Am2_kg(
            solido_mol_m3, volumen_celda_m3
        )
    return ficha


__all__ = [
    "MAGNETIZACION_SATURACION_Am2_kg",
    "TEMPERATURA_ORDEN_K",
    "MS_TITANOHEMATITA_Am2_kg",
    "RANGO_MS_TITANOHEMATITA",
    "TEMPERATURA_EUTECTOIDE_K",
    "VELOCIDAD_SUPRESION_C_MIN",
    "CRISOL_ENSAYO",
    "AGLOMERADO_SOLO",
    "VALIDACION",
    "flujo_de_enfriamiento_W_m2",
    "numero_de_biot",
    "enfriamiento_al_aire",
    "veredicto_de_temple",
    "momento_magnetico_Am2",
    "masa_solida_kg",
    "magnetizacion_Am2_kg",
    "cotas_magnetizacion_Am2_kg",
    "resumen",
]
