"""Serie de diagnósticos del documento de fenomenología.

Un solo sitio que abre la corrida y reduce cada instantánea a escalares. Las
figuras y las tablas del documento se construyen sobre esto, de modo que
ninguna cifra se calcula dos veces con dos criterios distintos.

Uso:  python informe/datos_fenomenologia.py [directorio_de_resultados]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ.parent / "simulacion_v3" / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ.parent / "simulacion_v3" / "src"))

from fisica.adaptador_v3 import MASAS_MOLARES_SOLIDO_KG_MOL  # noqa: E402
from fisica.fases_visuales import volumenes_molares_cm3_mol  # noqa: E402
from interfaz.app import (  # noqa: E402
    _recortar_a_la_corrida_vigente,
    directorio_predeterminado,
)
from nucleo.salida import cargar_instantanea  # noqa: E402

LECHO, GAS, PARED, TAPA = 3, 4, 1, 2

# Umbral de aglomerado operativo de `fisica/cohesion.py`: por debajo la mezcla
# todavía se comporta como polvo.
COHESION_AGLOMERADO = 0.5

# Fases que el modelo puede transformar y fases que no. Se declara aquí, y las
# pruebas comprueban contra la corrida que efectivamente sea así.
FASES_REACTIVAS = ("volatil", "C", "Fe2O3", "Fe3O4", "FeO", "Fe")
FASES_ESPECTADORAS = ("FeTiO3", "TiO2", "SiO2")


def serie_vigente(directorio: str | Path | None = None):
    """Instantáneas de la corrida vigente, ordenadas en el tiempo."""
    if directorio is None:
        directorio = directorio_predeterminado(RAIZ / "resultados")
    directorio = Path(directorio)
    rutas = sorted(directorio.glob("*.npz"))
    if not rutas:
        raise SystemExit(f"no hay instantáneas en {directorio}")
    cargadas = [(ruta, cargar_instantanea(ruta)) for ruta in rutas]
    vigentes = _recortar_a_la_corrida_vigente(
        cargadas, lambda par: par[1]["t"], lambda par: par[0].stat().st_mtime_ns,
    )
    return sorted(((float(c["t"]), c) for _, c in vigentes), key=lambda e: e[0])


def _paso_malla_m(campos) -> tuple[float, float, float]:
    return (
        (float(campos["x"][1]) - float(campos["x"][0])) * 1.0e-3,
        (float(campos["y"][1]) - float(campos["y"][0])) * 1.0e-3,
        (float(campos["z"][1]) - float(campos["z"][0])) * 1.0e-3,
    )


def _masas_por_fase_g(campos, volumen_celda_m3: float) -> dict[str, float]:
    """Gramos de cada fase sólida en todo el dominio."""
    return {
        fase: 1000.0 * float(np.sum(np.asarray(campos["solido"][fase])))
        * volumen_celda_m3 * masa_molar
        for fase, masa_molar in MASAS_MOLARES_SOLIDO_KG_MOL.items()
        if fase in campos["solido"]
    }


def _volumen_solido_m3(campos, volumen_celda_m3: float) -> float:
    """Volumen que ocupa la materia sólida, sin contar el poro.

    Se obtiene del inventario de fases por su volumen molar, no de la porosidad:
    así el volumen es consistente con la química que lo produjo.
    """
    volumenes = volumenes_molares_cm3_mol()
    from fisica.fases_visuales import MAPA_CAMPOS_SOLIDOS

    total_cm3 = 0.0
    for campo, valores in campos["solido"].items():
        clave = MAPA_CAMPOS_SOLIDOS.get(campo)
        if clave is None or clave not in volumenes:
            continue
        moles = float(np.sum(np.asarray(valores))) * volumen_celda_m3
        total_cm3 += moles * volumenes[clave]
    return total_cm3 * 1.0e-6


def fronteras_co(T_K: float = 1173.15) -> dict[str, float]:
    """Fracción mínima de CO que cada reducción exige, a la temperatura dada.

    Para `MO + CO -> M + CO2` la constante es K = [CO2]/[CO], de modo que el
    equilibrio se alcanza en x_CO = CO/(CO+CO2) = 1/(1+K). Por encima de ese
    valor la reacción avanza; por debajo, no. Sale de las tablas del propio
    proyecto —las mismas que usa el solucionador—, no de una cita.

    No se incluye Boudouard: su constante es K = p_CO^2/p_CO2 y la expresión
    1/(1+K) no le corresponde.
    """
    from fisica.tablas_termo import TablaTermoquimica

    tabla = TablaTermoquimica()
    reacciones = {
        "hematita": "hematita_CO",
        "magnetita": "magnetita_CO",
        "wustita": "wustita_CO",
        "ilmenita": "ilmenita_CO",
    }
    return {
        nombre: 1.0 / (1.0 + float(tabla.K_eq(reaccion, T_K)))
        for nombre, reaccion in reacciones.items()
    }


def _diametro_equivalente_mm(volumen_m3: float) -> float:
    """Diámetro de la esfera de igual volumen, en mm."""
    if volumen_m3 <= 0.0:
        return 0.0
    return 1.0e3 * (6.0 * volumen_m3 / np.pi) ** (1.0 / 3.0)


def diagnosticos(directorio: str | Path | None = None) -> dict:
    """Reduce la corrida entera a series temporales de escalares."""
    serie = serie_vigente(directorio)
    primera = serie[0][1]
    dx, dy, dz = _paso_malla_m(primera)
    volumen_celda = dx * dy * dz
    masas0 = _masas_por_fase_g(primera, volumen_celda)
    m0 = sum(masas0.values())

    fases = [f for f in MASAS_MOLARES_SOLIDO_KG_MOL if f in primera["solido"]]
    especies = list(primera["c_especies"])

    filas: dict[str, list] = {
        "t": [], "T_mufla": [], "T_pared": [], "T_lecho": [], "T_centro": [],
        "T_lecho_min": [], "T_lecho_max": [],
        "T_fondo_crisol": [], "T_lecho_base": [], "T_lecho_techo": [],
        "T_pared_lateral": [],
        "perdida_pct": [], "masa_solida_g": [],
        "eps_media": [], "cohesion_max": [], "cohesion_media": [],
        "hinchamiento_max": [], "hinchamiento_medio": [],
        "celdas_aglomerado": [], "volumen_aglomerado_mm3": [],
        "D_aglomerado_mm": [], "altura_aglomerado_mm": [],
        "D_huella_mm": [], "altura_libre_mm": [],
        "volumen_lecho_mm3": [],
        "volumen_solido_mm3": [], "D_solido_mm": [],
        "u_max": [], "CO_sobre_COx": [],
        "Re_p": [], "Re_celda": [], "Pe_termico": [], "Pe_masico": [],
        "Da": [], "Ra": [],
    }
    for fase in fases:
        filas[f"m_{fase}"] = []
    for especie in especies:
        filas[f"c_{especie}"] = []

    z_mm = np.asarray(primera["z"], dtype=float)
    for t, campos in serie:
        etiquetas = np.asarray(campos["etiquetas"])
        lecho = etiquetas == LECHO
        interior = np.isin(etiquetas, (LECHO, GAS))
        pared = np.isin(etiquetas, (PARED, TAPA))
        T = np.asarray(campos["T"])
        masas = _masas_por_fase_g(campos, volumen_celda)
        total = sum(masas.values())
        cohesion = np.asarray(campos["cohesion"])
        hinchamiento = np.asarray(campos.get("hinchamiento", np.ones_like(T)))
        aglomerado = lecho & (cohesion >= COHESION_AGLOMERADO)
        n_aglo = int(np.count_nonzero(aglomerado))
        # Volumen APARENTE del cuerpo cohesionado: sus celdas con el poro
        # dentro. No se multiplica por el factor de hinchamiento, que sería
        # contarlo dos veces: la malla es fija y el solucionador ya representa
        # el hinchamiento como el aumento de porosidad de esas mismas celdas
        # (eps pasa de 0,540 a 0,68), no ocupando celdas nuevas.
        volumen_aglo = n_aglo * volumen_celda
        altura = 0.0
        area_huella = 0.0
        if n_aglo:
            k = np.any(aglomerado, axis=(0, 1))
            altura = float(z_mm[k].max() - z_mm[k].min()) + float(z_mm[1] - z_mm[0])
            # Huella: la capa horizontal más ancha del cuerpo.
            por_capa = aglomerado.sum(axis=(0, 1))
            area_huella = float(por_capa.max()) * dx * dy
        volumen_solido = _volumen_solido_m3(campos, volumen_celda)

        numeros = (campos["metadatos"] or {}).get("numeros_adimensionales") or {}
        CO = float(np.mean(np.asarray(campos["c_especies"]["CO"])[interior]))
        CO2 = float(np.mean(np.asarray(campos["c_especies"]["CO2"])[interior]))

        filas["t"].append(float(t))
        filas["T_mufla"].append(max(float(np.max(T)), float(np.mean(T[pared]))))
        filas["T_pared"].append(float(np.mean(T[pared])) if np.any(pared) else np.nan)
        filas["T_lecho"].append(float(np.mean(T[lecho])) if np.any(lecho) else np.nan)
        filas["T_lecho_min"].append(float(np.min(T[lecho])) if np.any(lecho) else np.nan)
        filas["T_lecho_max"].append(float(np.max(T[lecho])) if np.any(lecho) else np.nan)
        # El camino real del calor es vertical: el fondo del crisol calienta la
        # base del lecho y el frente sube. Promediar TODA la pared mezcla el
        # fondo con el cilindro alto y con la tapa, y da un salto térmico de
        # signo engañoso.
        if np.any(lecho):
            capas = np.where(np.any(lecho, axis=(0, 1)))[0]
            k0, k1 = int(capas.min()), int(capas.max())
            base_crisol = pared.copy()
            base_crisol[:, :, k0:] = False
            lateral = pared.copy()
            lateral[:, :, :k0] = False
            lateral[:, :, k1 + 1:] = False
            filas["T_fondo_crisol"].append(
                float(np.mean(T[base_crisol])) if np.any(base_crisol) else np.nan)
            filas["T_pared_lateral"].append(
                float(np.mean(T[lateral])) if np.any(lateral) else np.nan)
            filas["T_lecho_base"].append(float(np.mean(T[:, :, k0][lecho[:, :, k0]])))
            filas["T_lecho_techo"].append(float(np.mean(T[:, :, k1][lecho[:, :, k1]])))
        else:
            for clave in ("T_fondo_crisol", "T_pared_lateral",
                          "T_lecho_base", "T_lecho_techo"):
                filas[clave].append(np.nan)
        centro = T[T.shape[0] // 2, T.shape[1] // 2, :]
        filas["T_centro"].append(float(centro[np.argmax(lecho[T.shape[0] // 2, T.shape[1] // 2, :])])
                                 if np.any(lecho) else np.nan)
        filas["masa_solida_g"].append(total)
        filas["perdida_pct"].append(100.0 * (m0 - total) / m0)
        filas["eps_media"].append(float(np.mean(np.asarray(campos["eps"])[lecho]))
                                  if np.any(lecho) else np.nan)
        filas["cohesion_max"].append(float(np.max(cohesion)))
        filas["cohesion_media"].append(float(np.mean(cohesion[lecho])) if np.any(lecho) else 0.0)
        filas["hinchamiento_max"].append(float(np.max(hinchamiento)))
        filas["hinchamiento_medio"].append(
            float(np.mean(hinchamiento[lecho])) if np.any(lecho) else 1.0)
        filas["celdas_aglomerado"].append(n_aglo)
        filas["volumen_aglomerado_mm3"].append(volumen_aglo * 1.0e9)
        filas["D_aglomerado_mm"].append(_diametro_equivalente_mm(volumen_aglo))
        filas["altura_aglomerado_mm"].append(altura)
        # Diámetro del disco de igual huella: el cuerpo es una pastilla ceñida a
        # la pared, no una esfera, y su anchura la fija el crisol.
        filas["D_huella_mm"].append(
            float(np.sqrt(4.0 * area_huella / np.pi)) * 1.0e3 if area_huella else 0.0)
        # Altura que tendría la superficie libre si el lecho pudiera subir. La
        # malla es fija y no lo representa; `fisica.hinchamiento` sí, y con
        # confinamiento lateral toda la expansión va a la altura.
        filas["altura_libre_mm"].append(
            altura * float(np.mean(hinchamiento[aglomerado])) if n_aglo else 0.0)
        # Envolvente del lecho entero, cohesionado o no: es el hueco que ocupa
        # la carga en el crisol desde el primer instante.
        filas["volumen_lecho_mm3"].append(
            float(np.count_nonzero(lecho)) * volumen_celda * 1.0e9)
        filas["volumen_solido_mm3"].append(volumen_solido * 1.0e9)
        filas["D_solido_mm"].append(_diametro_equivalente_mm(volumen_solido))
        filas["u_max"].append(float(numeros.get("u_max_m_s", np.nan)))
        filas["CO_sobre_COx"].append(CO / (CO + CO2) if (CO + CO2) > 0 else np.nan)
        filas["Re_p"].append(float(numeros.get("Re_particula", np.nan)))
        filas["Re_celda"].append(float(numeros.get("Re_celda", np.nan)))
        filas["Pe_termico"].append(float(numeros.get("Pe_termico", np.nan)))
        filas["Pe_masico"].append(float(numeros.get("Pe_masico", np.nan)))
        filas["Da"].append(float(numeros.get("Da", np.nan)))
        filas["Ra"].append(float(numeros.get("Ra", np.nan)))
        for fase in fases:
            filas[f"m_{fase}"].append(masas.get(fase, 0.0))
        for especie in especies:
            filas[f"c_{especie}"].append(
                float(np.mean(np.asarray(campos["c_especies"][especie])[interior])))

    return {
        "fases": fases,
        "especies": especies,
        "masa_inicial_g": m0,
        "masas_iniciales_g": masas0,
        "volumen_celda_m3": volumen_celda,
        "paso_malla_mm": [dx * 1e3, dy * 1e3, dz * 1e3],
        "Ra_critico": float(
            ((serie[-1][1]["metadatos"] or {}).get("numeros_adimensionales") or {}
             ).get("Ra_critico", np.nan)),
        "serie": {clave: np.asarray(valores).tolist() for clave, valores in filas.items()},
        "instantaneas": serie,
    }


def main() -> int:
    directorio = sys.argv[1] if len(sys.argv) > 1 else None
    datos = diagnosticos(directorio)
    resumen = {k: v for k, v in datos.items() if k != "instantaneas"}
    print(json.dumps({
        "n": len(datos["serie"]["t"]),
        "t_final": datos["serie"]["t"][-1],
        "masa_inicial_g": resumen["masa_inicial_g"],
        "perdida_final_pct": datos["serie"]["perdida_pct"][-1],
        "D_aglomerado_final_mm": datos["serie"]["D_aglomerado_mm"][-1],
        "altura_final_mm": datos["serie"]["altura_aglomerado_mm"][-1],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
