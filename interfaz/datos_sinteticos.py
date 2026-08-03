"""Datos demostrativos físicamente plausibles para desarrollar la interfaz.

La velocidad se construye analíticamente sobre una malla MAC. Las componentes
laterales proceden de una función de corriente y la componente vertical no
depende de z; por tanto, su divergencia discreta es nula salvo redondeo.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


ESPECIES = ("CO", "CO2", "H2", "H2O", "CH4", "N2", "O2")
FASES_SOLIDAS = ("Fe2O3", "Fe3O4", "FeO", "Fe", "C")
TIEMPO_FINAL_S = 720.0
FORMA_DEMO = (20, 20, 30)

# Condiciones y geometria reales del ensayo que la demostracion representa.
TEMPERATURA_MUFLA_K = 1173.15
EMISIVIDAD_CRISOL = 0.8
CONDUCTIVIDAD_PARED_W_MK = 16.0
ESPESOR_PARED_M = 1.1e-3
ALTURA_LECHO_MM = 3.26
RADIO_LECHO_MM = 11.4

# 50 052 = 388 bloques de 119 particulas de carbon + 10 de magnetita.
# La razon de la muestra es, por construccion, exactamente 11.9:1.
PARTICULAS_ENSAYO = {
    "malla_astm": 60,
    "apertura_malla_um": 250.0,
    "diametro_min_um": 100.0,
    "diametro_max_um": 250.0,
    "diametro_caracteristico_um": 175.0,
    "distribucion": "uniforme; no hay d10/d50/d90 medidos",
    "carbon_reales": 205_591,
    "magnetita_reales": 17_232,
    "total_real": 222_824,
    "carbon_muestra": 46_172,
    "magnetita_muestra": 3_880,
    "total_muestra": 50_052,
    "razon_numero_carbon_magnetita": 11.9,
    "factor_submuestreo": 222_824 / 50_052,
    "porosidad_inicial": 0.54,
    "volumen_lecho_cm3": 1.3593,
}


def _centros(limite_inferior: float, limite_superior: float, n: int) -> np.ndarray:
    caras = np.linspace(limite_inferior, limite_superior, n + 1, dtype=np.float64)
    return 0.5 * (caras[:-1] + caras[1:])


def _sigmoide(valor: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(valor, -40.0, 40.0)))


def _suavizar_gaussiano(campo: np.ndarray, radio: int = 1, sigma: float = 0.9) -> np.ndarray:
    """Filtro gaussiano separable con borde replicado, sin depender de SciPy."""
    radio = max(1, int(radio))
    distancias = np.arange(-radio, radio + 1, dtype=np.float64)
    pesos = np.exp(-(distancias ** 2) / (2.0 * sigma ** 2))
    pesos /= np.sum(pesos)
    resultado = np.asarray(campo, dtype=np.float64)
    for eje in range(resultado.ndim):
        relleno = [(0, 0)] * resultado.ndim
        relleno[eje] = (radio, radio)
        ampliado = np.pad(resultado, relleno, mode="edge")
        filtrado = np.zeros_like(resultado)
        for desplazamiento, peso in enumerate(pesos):
            rebanada = [slice(0, n) for n in resultado.shape]
            rebanada[eje] = slice(desplazamiento, desplazamiento + resultado.shape[eje])
            filtrado += peso * ampliado[tuple(rebanada)]
        resultado = filtrado
    return resultado


def estado_termico_sintetico(tiempo_s: float) -> dict[str, float]:
    """Magnitudes de la cadena mufla-pared-lecho para un tiempo dado."""
    T0 = 298.15
    t = max(float(tiempo_s), 0.0)
    tiempo_termico = t + 0.5
    fraccion_t = np.clip(t / TIEMPO_FINAL_S, 0.0, 1.0)
    respuesta_exterior = 1.0 - np.exp(-tiempo_termico / 18.0)
    respuesta_interior = 1.0 - np.exp(-tiempo_termico / 32.0)
    T_exterior = T0 + 0.992 * (TEMPERATURA_MUFLA_K - T0) * respuesta_exterior
    T_interior = T0 + 0.965 * (TEMPERATURA_MUFLA_K - T0) * respuesta_interior
    T_media = 0.5 * (T_exterior + T_interior)
    T_tapa = T0 + 0.982 * (TEMPERATURA_MUFLA_K - T0) * (
        1.0 - np.exp(-tiempo_termico / 24.0)
    )
    return {
        "T_mufla_K": TEMPERATURA_MUFLA_K,
        "T_pared_exterior_K": float(T_exterior),
        "T_pared_interior_K": float(T_interior),
        "T_pared_media_K": float(T_media),
        "T_tapa_K": float(T_tapa),
        "emisividad": EMISIVIDAD_CRISOL,
        "conductividad_pared_W_mK": CONDUCTIVIDAD_PARED_W_MK,
        "espesor_pared_m": ESPESOR_PARED_M,
        "conductividad_lecho_W_mK": float(1.2 - 0.6 * fraccion_t),
        "flujo_radiativo_W_m2": float(
            EMISIVIDAD_CRISOL * 5.670374419e-8
            * (TEMPERATURA_MUFLA_K ** 4 - T_exterior ** 4)
        ),
        "flujo_pared_W_m2": float(
            CONDUCTIVIDAD_PARED_W_MK / ESPESOR_PARED_M
            * max(T_exterior - T_interior, 0.0)
        ),
    }


def _etiquetar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    radio = np.hypot(X, Y)
    # Mismo perfil nominal de nucleo.perfil.PERFIL_ENSAYO: incluye el collar
    # fotografiado de 30,6 mm de diámetro a z≈21 mm.
    radio_ext = np.interp(
        Z, [0.0, 18.0, 21.0, 23.0, 32.0],
        [12.5, 14.0, 15.3, 14.75, 14.75],
    )
    radio_int = np.interp(
        Z, [2.0, 18.0, 21.0, 23.0, 32.0],
        [11.4, 12.9, 14.2, 13.65, 13.65],
    )
    dentro_ext = (Z <= 32.0) & (radio <= radio_ext)
    cavidad = (Z >= 2.0) & (Z <= 32.0) & (radio <= radio_int)
    etiquetas = np.zeros(X.shape, dtype=np.uint8)
    etiquetas[dentro_ext & ~cavidad] = 1
    etiquetas[cavidad] = 4
    etiquetas[cavidad & (Z <= 2.0 + ALTURA_LECHO_MM)] = 3
    tapa = (Z > 32.0) & (Z <= 34.0) & (radio <= 14.75)
    etiquetas[tapa] = 2
    return etiquetas


def generar_instantanea_sintetica(
    indice: int = 0,
    n_fotogramas: int = 25,
    forma: tuple[int, int, int] = FORMA_DEMO,
) -> dict[str, Any]:
    """Genera una instantánea completa con las formas del contrato §2."""
    if n_fotogramas < 2:
        raise ValueError("n_fotogramas debe ser al menos 2")
    if not 0 <= indice < n_fotogramas:
        raise IndexError("índice de fotograma fuera de rango")
    nx, ny, nz = (int(n) for n in forma)
    if min(nx, ny, nz) < 4:
        raise ValueError("cada dimensión de la malla debe ser al menos 4")

    x_caras = np.linspace(-14.75, 14.75, nx + 1, dtype=np.float64)
    y_caras = np.linspace(-14.75, 14.75, ny + 1, dtype=np.float64)
    z_caras = np.linspace(0.0, 34.0, nz + 1, dtype=np.float64)
    x = 0.5 * (x_caras[:-1] + x_caras[1:])
    y = 0.5 * (y_caras[:-1] + y_caras[1:])
    z = 0.5 * (z_caras[:-1] + z_caras[1:])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    etiquetas = _etiquetar(x, y, z)

    fraccion_t = indice / (n_fotogramas - 1)
    t = TIEMPO_FINAL_S * fraccion_t
    fase = 2.0 * np.pi * fraccion_t
    amplitud = 0.0022 + 0.0015 * fraccion_t
    escala_xy = 14.75

    # Campo MAC exactamente solenoidal en diferencias finitas: du/dx + dv/dy=0.
    perfil_z = (0.65 + 0.35 * np.cos(np.pi * (z - 17.0) / 34.0))[None, None, :]
    u = (
        amplitud
        * np.sin(np.pi * x_caras[:, None, None] / escala_xy)
        * np.cos(np.pi * y[None, :, None] / escala_xy)
        * perfil_z
    )
    v = (
        -amplitud
        * np.cos(np.pi * x[:, None, None] / escala_xy)
        * np.sin(np.pi * y_caras[None, :, None] / escala_xy)
        * perfil_z
    )
    pluma_xy = np.exp(-((X[:, :, 0] - 2.0 * np.sin(fase)) ** 2 + Y[:, :, 0] ** 2) / 42.0)
    retorno = 0.20
    w_plano = 0.0032 * (0.65 + 0.35 * fraccion_t) * (pluma_xy - retorno)
    w = np.repeat(w_plano[:, :, None], nz + 1, axis=2)

    # Cadena termica del ensayo: mufla -> radiacion -> pared exterior ->
    # conduccion Ni-Cr -> pared interior -> lecho.  La pared responde primero.
    # El pequeno desfase de 0.5 s representa el tiempo entre introducir el
    # crisol en la mufla y adquirir la primera instantanea demostrativa.
    T0 = 298.15
    tiempo_termico = t + 0.5
    termico = estado_termico_sintetico(t)
    T_pared_exterior = termico["T_pared_exterior_K"]
    T_pared_interior = termico["T_pared_interior_K"]
    T_pared_media = termico["T_pared_media_K"]
    T_tapa = termico["T_tapa_K"]

    radio = np.hypot(X, Y)
    # alpha efectiva decrece al consumirse/reorganizarse el carbon:
    # k_ef = 1.2 -> 0.6 W/(m K), representado por 0.72 -> 0.42 mm2/s.
    alpha_ef_mm2_s = 0.72 - 0.30 * fraccion_t
    penetracion_mm = 0.18 + 2.0 * np.sqrt(alpha_ef_mm2_s * tiempo_termico)
    distancia_fondo = np.maximum(Z - 2.0, 0.0)
    distancia_lateral = np.maximum(RADIO_LECHO_MM - radio, 0.0)
    frente_fondo = np.exp(-(distancia_fondo / penetracion_mm) ** 2)
    frente_lateral = np.exp(-(distancia_lateral / penetracion_mm) ** 2)
    penetracion_lecho = 1.0 - (1.0 - frente_fondo) * (1.0 - frente_lateral)
    respuesta_lecho = 1.0 - np.exp(-tiempo_termico / 55.0)
    T_lecho = T0 + (T_pared_interior - T0) * respuesta_lecho * penetracion_lecho

    # El gas de la cavidad se calienta despues de la pared y conserva un
    # gradiente hacia el eje. El exterior del dominio representa la mufla.
    distancia_pared = np.maximum(13.65 - radio, 0.0)
    penetracion_gas = np.exp(-(distancia_pared / (1.0 + penetracion_mm)) ** 2)
    T_gas = T0 + (T_pared_interior - T0) * (1.0 - np.exp(-tiempo_termico / 95.0)) * (0.38 + 0.62 * penetracion_gas)
    T = np.full(forma, TEMPERATURA_MUFLA_K, dtype=np.float64)
    T[etiquetas == 1] = T_pared_media
    T[etiquetas == 2] = T_tapa
    T[etiquetas == 4] = T_gas[etiquetas == 4]
    T[etiquetas == 3] = T_lecho[etiquetas == 3]

    frente_z = 2.0 + ALTURA_LECHO_MM * np.clip(
        penetracion_mm / ALTURA_LECHO_MM, 0.0, 1.0
    )

    P = 101_325.0 + 150.0 * (1.0 - Z / 34.0)
    P += 42.0 * np.exp(-(X * X + Y * Y) / 50.0) * np.cos(fase - Z / 12.0)

    desplazamiento = 2.6 * np.sin(0.7 * fase)
    pluma = np.exp(-((X - desplazamiento) ** 2 + (Y + 0.8 * np.cos(fase)) ** 2) / 38.0)
    pluma *= _sigmoide((Z - 2.5) / 1.4) * np.exp(-0.020 * np.maximum(Z - frente_z, 0.0) ** 2)
    zona_reaccion = np.exp(-((Z - frente_z) / 2.1) ** 2) * np.exp(-(X * X + Y * Y) / 95.0)
    calentamiento = np.clip((T - T0) / (TEMPERATURA_MUFLA_K - T0), 0.0, 1.0)
    c_especies = {
        "CO": 3.0 + 27.0 * pluma * (0.55 + 0.45 * fraccion_t),
        "CO2": 7.0 + 19.0 * zona_reaccion + 4.0 * pluma,
        "H2": 2.0 + 13.0 * pluma * (1.0 - 0.25 * fraccion_t),
        "H2O": 4.0 + 11.0 * zona_reaccion,
        "CH4": 0.8 + 7.0 * pluma * np.exp(-2.2 * fraccion_t),
        "N2": 25.0 + 0.7 * np.cos(Z / 8.0 + fase),
        "O2": 1.4 + 5.0 * (1.0 - calentamiento) * (1.0 - 0.75 * pluma),
    }

    mascara_lecho = etiquetas == 3
    progreso_termico = _sigmoide((T - 780.0) / 80.0)
    progreso = np.clip(fraccion_t * 1.45 * progreso_termico, 0.0, 0.985) * mascara_lecho
    mineral_inicial = 5_000.0
    solido = {
        "Fe2O3": mineral_inicial * (1.0 - progreso) * mascara_lecho,
        "Fe3O4": mineral_inicial * 0.36 * np.sin(np.pi * progreso) * mascara_lecho,
        "FeO": mineral_inicial * 0.46 * np.sin(0.78 * np.pi * progreso) ** 2 * mascara_lecho,
        "Fe": mineral_inicial * np.clip((progreso - 0.58) / 0.42, 0.0, 1.0) * mascara_lecho,
        "C": 8_500.0 * (1.0 - 0.68 * progreso) * mascara_lecho,
    }
    # Campo de fase continuo: no se umbraliza aqui. Marching cubes recibe una
    # transicion ancha y puede interpolar una superficie suave. La extension
    # fuera del lecho solo sirve de stencil numerico; etiquetas limita su render.
    madurez = np.clip((fraccion_t - 0.08) / 0.82, 0.0, 1.0)
    madurez = madurez * madurez * (3.0 - 2.0 * madurez)
    envolvente = 0.22 + 0.78 * np.exp(
        -((X - 0.7) ** 2 + (Y + 0.5) ** 2) / 155.0
        -((Z - 3.6) / 13.0) ** 2
    )
    activacion_termica = 0.32 + 0.68 * _sigmoide((T - 690.0) / 105.0)
    cohesion_cruda = 0.985 * madurez * envolvente * activacion_termica
    cohesion = np.clip(_suavizar_gaussiano(cohesion_cruda, radio=1), 0.0, 0.985)
    eps = np.ones(forma, dtype=np.float64)
    eps[etiquetas == 3] = 0.54 - 0.16 * cohesion[etiquetas == 3]
    eps[(etiquetas == 1) | (etiquetas == 2)] = 0.02
    eps[etiquetas == 0] = 1.0

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "etiquetas": etiquetas,
        "u": u,
        "v": v,
        "w": w,
        "P": P,
        "T": T,
        "c_especies": c_especies,
        "eps": eps,
        "solido": solido,
        "cohesion": cohesion,
        "termico": termico,
        "metadatos": {
            "fuente": "generador demostrativo analítico",
            "datos_sinteticos": True,
        },
    }


def velocidad_centrada(campos: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpola las componentes MAC a centros de celda."""
    return (
        0.5 * (campos["u"][:-1] + campos["u"][1:]),
        0.5 * (campos["v"][:, :-1] + campos["v"][:, 1:]),
        0.5 * (campos["w"][:, :, :-1] + campos["w"][:, :, 1:]),
    )


def _indice_mas_cercano(coordenadas: np.ndarray, valor: float) -> int:
    return int(np.clip(np.searchsorted(coordenadas, valor), 1, len(coordenadas) - 1) - (
        abs(coordenadas[np.clip(np.searchsorted(coordenadas, valor), 1, len(coordenadas) - 1)] - valor)
        >= abs(coordenadas[np.clip(np.searchsorted(coordenadas, valor), 1, len(coordenadas) - 1) - 1] - valor)
    ))


def _velocidad_en(
    punto_mm: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    velocidad: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    indices = (
        _indice_mas_cercano(x, float(punto_mm[0])),
        _indice_mas_cercano(y, float(punto_mm[1])),
        _indice_mas_cercano(z, float(punto_mm[2])),
    )
    return np.asarray([componente[indices] for componente in velocidad], dtype=np.float64)


def generar_linea_corriente(
    campos: dict[str, Any],
    semilla_mm: Iterable[float],
    paso_s: float = 0.075,
    max_pasos: int = 160,
) -> np.ndarray:
    """Integra una trayectoria RK2 y la recorta estrictamente al dominio."""
    x, y, z = (np.asarray(campos[eje]) for eje in ("x", "y", "z"))
    velocidad = velocidad_centrada(campos)
    punto = np.asarray(tuple(semilla_mm), dtype=np.float64)
    if punto.shape != (3,):
        raise ValueError("la semilla debe contener x, y, z")
    limites_min = np.array([x.min(), y.min(), z.min()])
    limites_max = np.array([x.max(), y.max(), z.max()])
    punto = np.clip(punto, limites_min, limites_max)
    trayectoria = [punto.copy()]
    for _ in range(max_pasos):
        v1 = _velocidad_en(punto, x, y, z, velocidad) * 1000.0
        medio = punto + 0.5 * paso_s * v1
        v2 = _velocidad_en(np.clip(medio, limites_min, limites_max), x, y, z, velocidad) * 1000.0
        siguiente = punto + paso_s * v2
        if np.any(siguiente < limites_min) or np.any(siguiente > limites_max):
            break
        radio = float(np.hypot(siguiente[0], siguiente[1]))
        # La envolvente conservadora histórica evita que la integración RK2
        # roce la pared incluso en el ensanchamiento local del collar.
        radio_int = 11.4 + (13.65 - 11.4) * np.clip(siguiente[2] / 32.0, 0.0, 1.0)
        if siguiente[2] < 2.0 or siguiente[2] > 31.8 or radio > radio_int:
            break
        trayectoria.append(siguiente.copy())
        punto = siguiente
    return np.asarray(trayectoria, dtype=np.float64)


def generar_lineas_corriente(campos: dict[str, Any], cantidad: int = 14) -> list[np.ndarray]:
    """Crea un haz reproducible de líneas dentro de la cavidad del crisol."""
    lineas: list[np.ndarray] = []
    for i in range(max(1, int(cantidad))):
        angulo = 2.0 * np.pi * i / max(1, cantidad)
        radio = 2.2 + 4.8 * ((i % 4) / 3.0)
        semilla = (radio * np.cos(angulo), radio * np.sin(angulo), 5.4 + 0.35 * (i % 3))
        linea = generar_linea_corriente(campos, semilla)
        if len(linea) > 1:
            lineas.append(linea)
    return lineas


def numeros_adimensionales_sinteticos(campos: dict[str, Any]) -> dict[str, float]:
    """Calcula indicadores demostrativos coherentes con el fotograma."""
    uc, vc, wc = velocidad_centrada(campos)
    velocidad_media = float(np.mean(np.sqrt(uc * uc + vc * vc + wc * wc)))
    T = np.asarray(campos["T"])
    longitud = 0.025
    rho, mu, alpha, difusividad = 1.05, 2.1e-5, 2.0e-5, 1.6e-5
    Re = rho * velocidad_media * longitud / mu
    Pe = velocidad_media * longitud / alpha
    Ra = 9.81 * (1.0 / 700.0) * max(float(T.max() - T.min()), 0.0) * longitud ** 3 / (alpha * 2.2e-5)
    Da = 0.08 + 8.5 * campos["t"] / TIEMPO_FINAL_S
    return {"Re": Re, "Ra": Ra, "Pe": Pe, "Da": Da, "Pe_masa": velocidad_media * longitud / difusividad}


__all__ = [
    "ESPECIES", "FASES_SOLIDAS", "FORMA_DEMO", "TIEMPO_FINAL_S",
    "PARTICULAS_ENSAYO", "TEMPERATURA_MUFLA_K", "estado_termico_sintetico",
    "generar_instantanea_sintetica", "generar_linea_corriente",
    "generar_lineas_corriente", "numeros_adimensionales_sinteticos",
    "velocidad_centrada",
]
