"""Ejecuta un caso 3-D declarativo y guarda instantáneas para la interfaz."""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fisica import adaptador_v3
from nucleo.caso import (
    LECHO,
    cargar_caso,
    crear_fuente_masa,
    crear_integrador_quimico,
    integrar_caso,
    instantanea_desde_estado,
    momentum,
)
from nucleo.salida import cargar_serie, guardar_instantanea


def _analizador() -> argparse.ArgumentParser:
    analizador = argparse.ArgumentParser(
        description="Ejecuta la simulación 3-D carbón–magnetita y guarda NPZ reales",
    )
    analizador.add_argument(
        "--caso", type=Path, default=Path("casos/carbon_magnetita.yaml"),
        help="archivo YAML del ensayo",
    )
    analizador.add_argument(
        "--malla", choices=("gruesa", "media", "fina"), default="gruesa",
        help="preajuste espacial (gruesa por defecto)",
    )
    analizador.add_argument(
        "--t-final", type=float, default=None,
        help="tiempo físico final en s; por defecto usa el YAML (720 s)",
    )
    analizador.add_argument(
        "--intervalo-guardado", type=float, default=None,
        help="separación entre instantáneas en s; por defecto usa el YAML (5 s)",
    )
    analizador.add_argument(
        "--salida", type=Path, default=Path("resultados/simulacion"),
        help="directorio para las instantáneas NPZ",
    )
    return analizador


def _nombre_instantanea(t_s: float) -> str:
    microsegundos = int(round(float(t_s) * 1.0e6))
    return f"instantanea_{microsegundos:015d}us.npz"


# Estado persistente del campo de cohesion entre instantaneas. La coquizacion
# depende de la HISTORIA termica, no de la temperatura instantanea, asi que el
# campo debe acumularse a lo largo de toda la corrida.
_COHESION: dict[str, Any] = {"campo": None, "t_previo": 0.0, "carbon": None}


def _actualizar_cohesion(caso: Any, estado: Any) -> Any:
    """Avanza el campo de cohesion con la temperatura 3-D real del solucionador.

    Es lo que permite ver CRECER el aglomerado: como la pared se calienta antes
    que el nucleo del lecho, la cohesion supera el umbral primero junto a la
    pared y el aglomerado nace como varios nucleos que despues coalescen. Con
    una temperatura uniforme ese crecimiento no aparece: todas las celdas
    cruzarian el umbral a la vez.
    """
    from fisica import cohesion as _coh

    T = np.asarray(estado.T, dtype=float)
    campo = _COHESION["campo"]
    if campo is None:
        malla = caso.malla
        espaciado = (malla.dx_mm * 1e-3, malla.dx_mm * 1e-3, malla.dz_mm * 1e-3)
        campo = _coh.CampoCohesion(T.shape, espaciado_m=espaciado,
                                   densidad_bulk_kg_m3=736.0,
                                   fraccion_volumetrica=0.46)
    # Sólo hay carbón donde hay carga.
    #
    # Si se pasa 0,75 uniforme, el campo también coquiza la pared Ni-Cr, que es
    # lo primero en calentarse: en la corrida de 720 s eso dejaba 185 celdas de
    # pared marcadas como aglomerado, con c=1. No se ve en la vista 3-D (la
    # isosuperficie enmascara a lecho y gas), pero contamina todo escalar que
    # recorra el campo entero — el `cohesion_max` de la serie y la media que se
    # reporta frente al dato experimental de los 30 s. Con fracción de carbón
    # nula fuera del lecho la cohesión residual queda en ~6e-5.
    if _COHESION["carbon"] is None:
        _COHESION["carbon"] = np.where(
            np.asarray(caso.etiquetas) == LECHO, 0.75, 0.0,
        )
    dt = float(estado.t) - float(_COHESION["t_previo"])
    if dt > 0.0:
        campo = _coh.evolucionar(campo, T, dt,
                                 fraccion_carbon=_COHESION["carbon"],
                                 porosidad=0.54)
        _COHESION["t_previo"] = float(estado.t)
    _COHESION["campo"] = campo
    return campo


def _campo_hinchamiento(caso: Any, campo_coh: Any) -> np.ndarray:
    """Factor de expansión volumétrica del aglomerado, celda a celda.

    El carbón tiene índice de hinchamiento 8 (dato de laboratorio) y la
    magnetita lo atenúa: sólo hincha la fracción carbonosa y los granos
    minerales rompen la estructura de burbujas de la masa plástica. El factor
    avanza con el grado de plastificación, que es la variable irreversible que
    ya lleva `cohesion.HistoriaTermica`: el gas atrapado infla mientras el
    carbón está fluido y la estructura queda congelada al resolidificar.
    """
    from fisica import hinchamiento as _hin

    carga = caso.datos["carga"]
    m_carbon = float(carga["masa_carbon_g"])
    m_mineral = float(carga["masa_concentrado_g"])
    fraccion_inerte = m_mineral / (m_carbon + m_mineral)
    historia = campo_coh.historia_termica
    grado = np.asarray(historia.grado_plastificacion, dtype=float)
    factor = np.asarray(
        _hin.hinchamiento_local(grado, _hin.INDICE_HINCHAMIENTO_CARBON, fraccion_inerte),
        dtype=float,
    )
    # Sólo hincha donde hay carga; fuera del lecho el factor es exactamente 1.
    return np.where(np.asarray(caso.etiquetas) == LECHO, factor, 1.0)


def _magnetismo(caso: Any, estado: Any) -> dict[str, float] | None:
    """Magnetizacion de saturacion del lecho a temperatura ambiente.

    Es el observable que contrasta con la prueba del iman. Se evalua EN FRIO,
    no a 900 grados C: a la temperatura del ensayo nada de esto es magnetico
    (la temperatura de Curie de la magnetita son 585 grados C), y el iman se
    pasa despues de enfriar. Se guardan las dos cotas, segun la wustita se
    conserve o se descomponga al enfriar; vease `fisica/magnetismo.py`.
    """
    try:
        from fisica import magnetismo as _mag

        # Sobre TODO el solido, no sobre `etiquetas == LECHO`: el 12 % de la
        # carga esta en celdas cortadas por la frontera, que llevan etiqueta de
        # pared o de gas porque su centro cae fuera. Enmascarar por etiqueta
        # perderia ese 12 % y las cifras no cuadrarian con las masas por fase.
        solido = {
            fase: np.asarray(valores)
            for fase, valores in estado.solido.items()
            if not fase.startswith("_")
        }
        volumen = float(caso.malla.volumen_celda_mm3) * 1.0e-9
        cotas = _mag.cotas_magnetizacion_Am2_kg(solido, volumen)
        return {
            "magnetizacion_temple_Am2_kg": float(cotas["temple"]),
            "magnetizacion_enfriamiento_lento_Am2_kg": float(cotas["enfriamiento_lento"]),
            # El momento total, sin normalizar por masa. Hace falta para separar
            # dos efectos que la magnetizacion especifica mezcla: la quimica del
            # hierro y la simple perdida del 28 % de la masa por
            # devolatilizacion, que sube M sin que cambie ninguna fase.
            "momento_Am2": float(_mag.momento_magnetico_Am2(solido, volumen)),
            "masa_solida_lecho_kg": float(_mag.masa_solida_kg(solido, volumen)),
        }
    except Exception:
        return None


def _guardar(caso: Any, estado: Any, directorio: Path) -> Path:
    campos = instantanea_desde_estado(caso, estado)
    campo_coh = _actualizar_cohesion(caso, estado)
    campos["cohesion"] = np.asarray(campo_coh.c, dtype=float)
    campos["hinchamiento"] = _campo_hinchamiento(caso, campo_coh)
    # Numeros adimensionales DEL SOLUCIONADOR, con las propiedades reales de
    # cada celda. La interfaz los mostraba calculados con una funcion de
    # demostracion que llevaba densidad y viscosidad fijas y un Damkohler
    # inventado (una rampa lineal en el tiempo). Aqui se calculan una vez, con
    # lo que de verdad hay, y viajan en los metadatos de la instantanea.
    try:
        numeros = momentum.numeros_adimensionales(
            estado.u, estado.v, estado.w, estado.T,
            caso.propiedades, caso.malla,
            d_particula_m=float(caso.datos["carga"]["diametro_particula_m"]),
        )
        metadatos = dict(campos.get("metadatos") or {})
        metadatos["numeros_adimensionales"] = {
            nombre: float(valor) for nombre, valor in numeros.items()
        }
        campos["metadatos"] = metadatos
    except Exception:
        pass
    magnetismo = _magnetismo(caso, estado)
    if magnetismo is not None:
        metadatos = dict(campos.get("metadatos") or {})
        metadatos["magnetismo"] = magnetismo
        campos["metadatos"] = metadatos
    info = None
    try:
        from fisica import cohesion as _coh
        info = _coh.aglomerado(campo_coh)
    except Exception:
        info = None
    if info is not None:
        campos["aglomerado_volumen_cm3"] = float(info["volumen_m3"]) * 1e6
        campos["aglomerado_masa_g"] = float(info["masa_kg"]) * 1e3
        campos["aglomerado_componentes"] = int(info["numero_componentes"])
    return guardar_instantanea(
        campos,
        directorio / _nombre_instantanea(estado.t),
    )


def _balance(caso: Any, estado: Any, referencia: dict[str, Any]) -> dict[str, Any]:
    masa_solida = caso.masa_solida_kg(estado)
    masa_gas = caso.masa_gas_kg(estado)
    masa_total = masa_solida + masa_gas
    masa_venteada = float(getattr(estado, "masa_venteada_kg", 0.0))
    elementos = caso.inventario_elemental_mol(estado)
    elementos_venteados = {elemento: 0.0 for elemento in elementos}
    for especie, moles in getattr(estado, "moles_venteados", {}).items():
        composicion = adaptador_v3.COMPOSICION_ELEMENTAL.get(especie, {})
        for elemento in elementos_venteados:
            elementos_venteados[elemento] += composicion.get(elemento, 0.0) * float(moles)
    return {
        "masa_solida_kg": masa_solida,
        "masa_gas_kg": masa_gas,
        "masa_total_kg": masa_total,
        "masa_venteada_kg": masa_venteada,
        "masa_total_mas_venteo_kg": masa_total + masa_venteada,
        "error_masa_kg": masa_total + masa_venteada - referencia["masa_total_kg"],
        "error_masa_rel": (
            (masa_total + masa_venteada - referencia["masa_total_kg"])
            / referencia["masa_total_kg"]
        ),
        "elementos_mol": elementos,
        "elementos_venteados_mol": elementos_venteados,
        "errores_elementales_mol": {
            nombre: valor + elementos_venteados[nombre] - referencia["elementos_mol"][nombre]
            for nombre, valor in elementos.items()
        },
    }


def _referencia_balance(caso: Any, estado: Any) -> dict[str, Any]:
    return {
        "masa_solida_kg": caso.masa_solida_kg(estado),
        "masa_total_kg": caso.masa_solida_kg(estado) + caso.masa_gas_kg(estado),
        "elementos_mol": caso.inventario_elemental_mol(estado),
    }


def _limitador(diag: dict[str, Any], cfg: Any) -> str:
    dt = float(diag["dt"])
    limite = float(diag["limite_estabilidad"])
    if math.isfinite(limite) and math.isclose(dt, limite, rel_tol=2.0e-10, abs_tol=0.0):
        return str(diag["restringe"])
    if math.isclose(dt, cfg.dt_max, rel_tol=2.0e-10, abs_tol=0.0):
        return "dt_max_acople"
    return "crecimiento_o_fin_de_intervalo"


def _imprimir_progreso(
    caso: Any, estado: Any, diag: dict[str, Any], referencia: dict[str, Any],
) -> None:
    numeros = momentum.numeros_adimensionales(
        estado.u, estado.v, estado.w, estado.T,
        caso.propiedades, caso.malla,
        d_particula_m=float(caso.datos["carga"]["diametro_particula_m"]),
    )
    balance = _balance(caso, estado, referencia)
    print(
        f"t={estado.t:10.6f} s | paso={int(diag['n_paso']) + 1:6d} "
        f"| dt={float(diag['dt']):.3e} s | restringe={_limitador(diag, caso.config_acople)}"
    )
    print(
        "  adimensionales: "
        f"Re_p={numeros['Re_particula']:.3e}, Re_celda={numeros['Re_celda']:.3e}, "
        f"Ra={numeros['Ra']:.3e}, Da={numeros['Da']:.3e}, "
        f"Pe_masa={numeros['Pe_masico']:.3e}, Pe_termico={numeros['Pe_termico']:.3e}"
    )
    max_elemental = max(
        (abs(valor) for valor in balance["errores_elementales_mol"].values()),
        default=0.0,
    )
    print(
        "  conservación: "
        f"m_sólido={balance['masa_solida_kg'] * 1e3:.12f} g, "
        f"m_gas={balance['masa_gas_kg'] * 1e3:.12f} g, "
        f"m_venteo={balance['masa_venteada_kg'] * 1e3:.12f} g, "
        f"delta_m={balance['error_masa_kg']:.3e} kg "
        f"({balance['error_masa_rel']:.3e}), "
        f"máx|delta_n_elemento|={max_elemental:.3e} mol"
    )


def ejecutar(argumentos: argparse.Namespace) -> dict[str, Any]:
    if argumentos.t_final is not None and (
        not math.isfinite(argumentos.t_final) or argumentos.t_final <= 0.0
    ):
        raise ValueError("--t-final debe ser finito y positivo")
    if argumentos.intervalo_guardado is not None and (
        not math.isfinite(argumentos.intervalo_guardado)
        or argumentos.intervalo_guardado <= 0.0
    ):
        raise ValueError("--intervalo-guardado debe ser finito y positivo")

    caso = cargar_caso(argumentos.caso, malla=argumentos.malla)
    t_final = caso.t_final_s if argumentos.t_final is None else float(argumentos.t_final)
    intervalo = (
        caso.intervalo_guardado_s
        if argumentos.intervalo_guardado is None
        else float(argumentos.intervalo_guardado)
    )
    directorio = argumentos.salida.resolve()
    directorio.mkdir(parents=True, exist_ok=True)

    estado0 = caso.estado_inicial
    referencia = _referencia_balance(caso, estado0)
    guardadas: dict[int, Path] = {}
    inicial = _guardar(caso, estado0, directorio)
    guardadas[int(round(estado0.t * 1.0e6))] = inicial
    siguiente_guardado = intervalo

    print(
        f"Caso={caso.nombre} | malla={caso.preajuste_malla} {caso.malla.forma} "
        f"({caso.malla.n_celdas:,} celdas; dx={caso.malla.dx_mm:g} mm, "
        f"dz={caso.malla.dz_mm:g} mm)"
    )
    print(
        f"Curva real de mufla={caso.curva_mufla.fuente} "
        f"| T_mufla(0)={caso.curva_mufla(0.0):.2f} K "
        f"| carga sólida={caso.masa_solida_kg() * 1e3:.12f} g"
    )
    print(f"Instantánea inicial: {inicial}")

    def al_guardar(estado: Any, diag: dict[str, Any]) -> None:
        nonlocal siguiente_guardado
        if estado.t + 16.0 * np.finfo(float).eps < siguiente_guardado:
            return
        ruta = _guardar(caso, estado, directorio)
        guardadas[int(round(estado.t * 1.0e6))] = ruta
        siguiente_guardado += intervalo
        _imprimir_progreso(caso, estado, diag, referencia)

    inicio_reloj = time.perf_counter()
    final, historial = integrar_caso(
        caso,
        t_final,
        estado=estado0,
        cfg=caso.config_acople,
        quimica=crear_integrador_quimico(caso),
        fuente_masa=crear_fuente_masa(caso),
        al_guardar=al_guardar,
        intervalo_guardado=intervalo,
    )
    segundos_reloj = time.perf_counter() - inicio_reloj

    clave_final = int(round(final.t * 1.0e6))
    if clave_final not in guardadas:
        guardadas[clave_final] = _guardar(caso, final, directorio)
        if historial:
            _imprimir_progreso(caso, final, historial[-1], referencia)

    serie = cargar_serie(directorio)
    propias = {
        ruta.resolve() for ruta in guardadas.values()
    }
    serie_propia = [item for item in serie if Path(item["ruta"]).resolve() in propias]
    if len(serie_propia) != len(guardadas):
        raise RuntimeError("no se pudieron recargar todas las instantáneas recién generadas")

    balance_final = _balance(caso, final, referencia)
    limitadores = Counter(_limitador(diag, caso.config_acople) for diag in historial)
    if historial:
        ultimo = historial[-1]
        limite_estable = float(ultimo["limite_estabilidad"])
        dominante = (
            str(ultimo["restringe"])
            if math.isfinite(limite_estable)
            and limite_estable <= caso.config_acople.dt_max
            else "dt_max_acople"
        )
    else:
        dominante = "ninguno"
    divergencia_max = max(
        (float(diag.get("divergencia_residual", 0.0)) for diag in historial),
        default=0.0,
    )
    incompatibilidad_max = max(
        (abs(float(diag.get("incompatibilidad_divergencia", 0.0))) for diag in historial),
        default=0.0,
    )
    coste = segundos_reloj / t_final
    termico_final = {
        clave: float(historial[-1][clave])
        for clave in ("T_mufla_K", "T_pared_media_K", "T_lecho_media_K", "T_centro_lecho_K")
    } if historial else {}
    perdida_masa_pct = (
        (referencia["masa_solida_kg"] - caso.masa_solida_kg(final))
        / referencia["masa_solida_kg"] * 100.0
    )
    print("-" * 88)
    print(
        f"Final: t={final.t:.9f} s en {segundos_reloj:.3f} s de reloj "
        f"=> {coste:.3f} s_reloj/s_simulado"
    )
    print(
        f"Pasos={len(historial)} | limitador práctico en régimen={dominante} "
        f"| conteo transitorio={dict(limitadores)} "
        f"| divergencia residual máx={divergencia_max:.3e} s^-1 "
        f"| incompatibilidad Neumann máx={incompatibilidad_max:.3e} s^-1"
    )
    print(
        f"Balance final: delta_m={balance_final['error_masa_kg']:.6e} kg, "
        f"error relativo={balance_final['error_masa_rel']:.6e}, "
        "delta_n [mol]="
        + ", ".join(
            f"{nombre}:{valor:+.3e}"
            for nombre, valor in balance_final["errores_elementales_mol"].items()
        )
    )
    if termico_final:
        print(
            "Evolución térmica final: "
            f"mufla={termico_final['T_mufla_K']:.2f} K, "
            f"pared={termico_final['T_pared_media_K']:.2f} K, "
            f"lecho medio={termico_final['T_lecho_media_K']:.2f} K, "
            f"centro={termico_final['T_centro_lecho_K']:.2f} K"
        )
    print(
        f"Pérdida de masa sólida={perdida_masa_pct:.6f} % "
        f"| masa contabilizada por venteo={balance_final['masa_venteada_kg'] * 1e3:.9f} g"
    )
    print(
        f"Instantáneas compatibles recargadas={len(serie_propia)} "
        f"| directorio={directorio}"
    )
    return {
        "caso": caso,
        "estado_final": final,
        "historial": historial,
        "segundos_reloj": segundos_reloj,
        "segundos_reloj_por_simulado": coste,
        "balance_final": balance_final,
        "termico_final": termico_final,
        "perdida_masa_pct": perdida_masa_pct,
        "limitador_dominante": dominante,
        "divergencia_residual_max": divergencia_max,
        "incompatibilidad_divergencia_max": incompatibilidad_max,
        "instantaneas": serie_propia,
    }


def main(argv: Sequence[str] | None = None) -> int:
    argumentos = _analizador().parse_args(argv)
    ejecutar(argumentos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
