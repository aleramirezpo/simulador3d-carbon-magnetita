"""Verificación cuantitativa del transporte escalar por volúmenes finitos."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import optimize

from nucleo.transporte import (
    divergencia_flujo_advectivo,
    divergencia_flujo_difusivo,
    dt_estable_transporte,
    paso_energia,
    paso_especies,
)
from verificacion.analiticas import (
    adveccion_difusion_estacionaria_1d,
    conduccion_transitoria_placa,
    difusion_1d_semiinfinito,
    enfriamiento_newton,
)


@dataclass
class MallaPrueba:
    """Contrato mínimo de MallaVoxel sin construir un crisol completo."""

    forma: tuple[int, int, int]
    dx_mm: float
    dz_mm: float
    dy_mm: float | None = None

    def __post_init__(self) -> None:
        if self.dy_mm is None:
            self.dy_mm = self.dx_mm


def _velocidades_nulas(forma: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    nx, ny, nz = forma
    return (
        np.zeros((nx + 1, ny, nz)),
        np.zeros((nx, ny + 1, nz)),
        np.zeros((nx, ny, nz + 1)),
    )


def _reportar(request: pytest.FixtureRequest, mensaje: str) -> None:
    terminal = request.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(mensaje)


def test_divergencia_advectiva_es_conservativa_en_dominio_cerrado() -> None:
    forma = (9, 7, 5)
    malla = MallaPrueba(forma, dx_mm=0.5, dz_mm=0.25)
    rng = np.random.default_rng(1421)
    phi = rng.normal(size=forma)
    u, v, w = _velocidades_nulas(forma)
    u[1:-1] = rng.normal(scale=2.0e-7, size=u[1:-1].shape)
    v[:, 1:-1] = rng.normal(scale=2.0e-7, size=v[:, 1:-1].shape)
    w[:, :, 1:-1] = rng.normal(scale=2.0e-7, size=w[:, :, 1:-1].shape)

    for esquema in ("upwind", "central", "tvd_superbee"):
        divergencia = divergencia_flujo_advectivo(phi, u, v, w, malla, esquema)
        assert abs(float(np.sum(divergencia))) < 1.0e-12


def test_escalar_total_se_conserva_con_flujo_nulo() -> None:
    forma = (17, 4, 3)
    malla = MallaPrueba(forma, dx_mm=0.7, dz_mm=0.31)
    rng = np.random.default_rng(31)
    inicial = 0.2 + rng.random(forma)
    u, v, w = _velocidades_nulas(forma)
    campos = SimpleNamespace(u=u, v=v, w=w, c={"A": inicial})
    props = {"D": {"A": 2.5e-7}}

    final = paso_especies(campos, props, malla, dt=0.8, fuentes=None)["A"]
    assert np.sum(final) == pytest.approx(np.sum(inicial), rel=2.0e-12, abs=2.0e-12)


def _integrar_difusion_1d(
    n: int,
    L: float,
    D: float,
    tiempo: float,
    inicial: np.ndarray,
    izquierda: float,
    derecha: float,
) -> tuple[np.ndarray, np.ndarray]:
    h = L / n
    malla = MallaPrueba((n, 1, 1), dx_mm=1.0e3 * h, dz_mm=1.0e3 * h)
    u, v, w = _velocidades_nulas(malla.forma)
    estado = SimpleNamespace(u=u, v=v, w=w, c={"A": inicial.reshape(n, 1, 1).copy()})
    props = {
        "D": {"A": D},
        "esquema": "upwind",
        "condiciones_frontera": {
            "A": {
                "x_min": {"tipo": "dirichlet", "valor": izquierda},
                "x_max": {"tipo": "dirichlet", "valor": derecha},
            }
        },
    }
    # Backward Euler es de primer orden temporal: dt~h² hace que su error sea
    # del mismo orden que la discretización espacial centrada.
    dt_objetivo = 0.10 * h * h / D
    pasos = int(np.ceil(tiempo / dt_objetivo))
    dt = tiempo / pasos
    for _ in range(pasos):
        estado.c = paso_especies(estado, props, malla, dt, None)
    x = (np.arange(n) + 0.5) * h
    return x, estado.c["A"][:, 0, 0]


def _ordenes(errores: list[float]) -> list[float]:
    return [float(np.log(a / b) / np.log(2.0)) for a, b in zip(errores[:-1], errores[1:])]


def test_convergencia_difusion_semiinfinita_segundo_orden(request: pytest.FixtureRequest) -> None:
    D, tiempo, L = 2.0e-4, 0.20, 0.08
    errores = []
    for n in (20, 40, 80):
        x, numerica = _integrar_difusion_1d(
            n, L, D, tiempo, np.zeros(n), izquierda=1.0, derecha=0.0
        )
        exacta = difusion_1d_semiinfinito(x, tiempo, D, phi0=0.0, phis=1.0)
        errores.append(float(np.sqrt(np.mean((numerica - exacta) ** 2))))
    ordenes = _ordenes(errores)
    _reportar(request, f"orden difusión semiinfinita: {ordenes[-1]:.3f}; errores={errores}")
    assert 1.75 < ordenes[-1] < 2.25


def test_convergencia_conduccion_placa_segundo_orden(request: pytest.FixtureRequest) -> None:
    alpha, tiempo, L = 1.0e-4, 0.12, 0.04
    T0, Ts = 300.0, 700.0
    errores = []
    for n in (20, 40, 80):
        x, numerica = _integrar_difusion_1d(
            n, L, alpha, tiempo, np.full(n, T0), izquierda=Ts, derecha=Ts
        )
        exacta = conduccion_transitoria_placa(x, tiempo, alpha, L, T0, Ts, 250)
        errores.append(float(np.sqrt(np.mean((numerica - exacta) ** 2))))
    ordenes = _ordenes(errores)
    _reportar(request, f"orden conducción transitoria en placa: {ordenes[-1]:.3f}; errores={errores}")
    assert 1.75 < ordenes[-1] < 2.25


def _solucion_estacionaria_numerica(
    pe: float, esquema: str, n: int = 20
) -> tuple[np.ndarray, np.ndarray, float]:
    L, D, u0 = 1.0, 1.0, pe
    h = L / n
    malla = MallaPrueba((n, 1, 1), dx_mm=1.0e3 * h, dz_mm=1.0e3 * h)
    u, v, w = _velocidades_nulas(malla.forma)
    u[...] = u0
    x = (np.arange(n) + 0.5) * h

    def residual(vector: np.ndarray) -> np.ndarray:
        campo = vector.reshape(n, 1, 1)
        r = (-divergencia_flujo_advectivo(campo, u, v, w, malla, esquema)
             + divergencia_flujo_difusivo(campo, D, malla))[:, 0, 0]
        # Dirichlet en caras: distancia del centro a la frontera h/2.
        r[0] += 2.0 * D * (0.0 - vector[0]) / h**2
        r[-1] += 2.0 * D * (1.0 - vector[-1]) / h**2
        return r

    exacta = adveccion_difusion_estacionaria_1d(x, u0, D, L, 0.0, 1.0)
    if esquema in ("upwind", "central"):
        cero = residual(np.zeros(n))
        matriz = np.column_stack(
            [residual(np.eye(n)[j]) - cero for j in range(n)]
        )
        solucion = np.linalg.solve(matriz, -cero)
        norma_residuo = float(np.linalg.norm(residual(solucion), ord=np.inf))
    else:
        resultado = optimize.least_squares(
            residual,
            exacta,
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=4000,
        )
        assert resultado.success
        solucion = resultado.x
        norma_residuo = float(np.linalg.norm(residual(solucion), ord=np.inf))
    return x, solucion, norma_residuo


@pytest.mark.parametrize("pe", [1.0, 10.0, 50.0])
def test_esquemas_frente_adveccion_difusion_estacionaria(
    pe: float, request: pytest.FixtureRequest
) -> None:
    metricas: dict[str, tuple[float, float, float, float]] = {}
    for esquema in ("upwind", "central", "tvd_superbee"):
        x, numerica, residuo = _solucion_estacionaria_numerica(pe, esquema)
        exacta = adveccion_difusion_estacionaria_1d(x, pe, 1.0, 1.0, 0.0, 1.0)
        error = float(np.sqrt(np.mean((numerica - exacta) ** 2)))
        metricas[esquema] = (error, float(numerica.min()), float(numerica.max()), residuo)
    _reportar(
        request,
        f"Pe={pe:g}: " + "; ".join(
            f"{e}: L2={m[0]:.4e}, rango=[{m[1]:.4f},{m[2]:.4f}], Rinf={m[3]:.1e}"
            for e, m in metricas.items()
        ),
    )
    tol = 2.0e-9
    assert metricas["upwind"][1] >= -tol and metricas["upwind"][2] <= 1.0 + tol
    assert metricas["tvd_superbee"][1] >= -tol and metricas["tvd_superbee"][2] <= 1.0 + tol
    assert metricas["tvd_superbee"][0] < metricas["upwind"][0]
    cotas_error_tvd = {1.0: 5.0e-4, 10.0: 7.0e-3, 50.0: 4.0e-2}
    assert metricas["tvd_superbee"][0] < cotas_error_tvd[pe]
    if pe == 50.0:
        assert metricas["central"][1] < -1.0e-3 or metricas["central"][2] > 1.001


def test_monotonia_escalon_upwind_tvd_y_oscilacion_central() -> None:
    n, h, velocidad = 80, 1.0e-3, 0.1
    malla = MallaPrueba((n, 1, 1), dx_mm=1.0e3 * h, dz_mm=1.0e3 * h)
    phi = np.zeros((n, 1, 1))
    phi[n // 2:] = 1.0
    u, v, w = _velocidades_nulas(malla.forma)
    u[...] = velocidad
    dt = 0.4 * h / velocidad

    resultados = {
        e: phi - dt * divergencia_flujo_advectivo(phi, u, v, w, malla, e)
        for e in ("upwind", "central", "tvd_superbee")
    }
    for esquema in ("upwind", "tvd_superbee"):
        assert resultados[esquema].min() >= -2.0e-14
        assert resultados[esquema].max() <= 1.0 + 2.0e-14
    assert resultados["central"].min() < -1.0e-3 or resultados["central"].max() > 1.001


def test_media_armonica_en_interfaz_de_difusividad() -> None:
    n, L = 40, 1.0
    h = L / n
    malla = MallaPrueba((n, 1, 1), dx_mm=1.0e3 * h, dz_mm=1.0e3 * h)
    D1, D2 = 1.0, 100.0
    D = np.empty((n, 1, 1))
    D[: n // 2] = D1
    D[n // 2:] = D2
    x = (np.arange(n) + 0.5) * h
    resistencia = 0.5 / D1 + 0.5 / D2
    flujo_exacto = 1.0 / resistencia
    phi = np.where(
        x <= 0.5,
        flujo_exacto * x / D1,
        flujo_exacto * 0.5 / D1 + flujo_exacto * (x - 0.5) / D2,
    ).reshape(n, 1, 1)

    lap = divergencia_flujo_difusivo(phi, D, malla)[:, 0, 0]
    d_harmonica = 2.0 * D1 * D2 / (D1 + D2)
    flujo_interfaz = d_harmonica * (phi[n // 2, 0, 0] - phi[n // 2 - 1, 0, 0]) / h
    assert flujo_interfaz == pytest.approx(flujo_exacto, rel=2.0e-14)
    assert np.max(np.abs(lap[1:-1])) < 2.0e-11
    # La media aritmética produciría un error enorme en esta interfaz.
    flujo_aritmetico = 0.5 * (D1 + D2) * (phi[n // 2, 0, 0] - phi[n // 2 - 1, 0, 0]) / h
    assert abs(flujo_aritmetico / flujo_exacto - 1.0) > 20.0


def test_dt_estable_incluye_cfl_y_anisotropia_y_evitar_divergencia() -> None:
    forma = (18, 8, 6)
    malla = MallaPrueba(forma, dx_mm=0.5, dz_mm=0.25)
    u, v, w = _velocidades_nulas(forma)
    # Curl discreto de una función de corriente nodal: div(u,v)=0 a precisión
    # de máquina y las velocidades normales se anulan en las cuatro paredes.
    nx, ny, nz = forma
    xi = np.linspace(0.0, 1.0, nx + 1)[:, None, None]
    eta = np.linspace(0.0, 1.0, ny + 1)[None, :, None]
    psi = np.sin(np.pi * xi) * np.sin(np.pi * eta) * np.ones((1, 1, nz))
    u[...] = np.diff(psi, axis=1) / (0.5e-3)
    v[...] = -np.diff(psi, axis=0) / (0.5e-3)
    escala = 0.012 / max(np.max(np.abs(u)), np.max(np.abs(v)))
    u *= escala
    v *= escala
    D = 1.1e-7
    dt = dt_estable_transporte(u, v, w, D, malla)
    dx, dz = 0.5e-3, 0.25e-3
    esperado_adv = 0.5 / (np.max(np.abs(u)) / dx + np.max(np.abs(v)) / dx)
    esperado_dif = 0.5 / (D * (2.0 / dx**2 + 1.0 / dz**2))
    assert dt == pytest.approx(min(esperado_adv, esperado_dif), rel=2.0e-15)
    assert esperado_dif < 0.5 / (D * 3.0 / dx**2)  # dz es quien restringe

    rng = np.random.default_rng(2025)
    phi = rng.random(forma)
    minimo, maximo = float(phi.min()), float(phi.max())
    for _ in range(500):
        rhs = (-divergencia_flujo_advectivo(phi, u, v, w, malla, "upwind")
               + divergencia_flujo_difusivo(phi, D, malla))
        phi = phi + 0.9 * dt * rhs
    assert np.all(np.isfinite(phi))
    assert phi.min() >= minimo - 1.0e-12
    assert phi.max() <= maximo + 1.0e-12


def test_fracciones_en_frontera_inclinada_conservan_masa() -> None:
    forma = (24, 5, 16)
    malla = MallaPrueba(forma, dx_mm=0.5, dz_mm=0.25)
    i, _, k = np.indices(forma)
    # Banda oblicua con dos capas de celdas cortadas, no una máscara binaria.
    distancia = 9.4 + 0.42 * k - i
    fraccion = np.clip(distancia + 0.5, 0.0, 1.0)
    rng = np.random.default_rng(7)
    inicial = rng.random(forma)
    u, v, w = _velocidades_nulas(forma)
    campos = SimpleNamespace(u=u, v=v, w=w, c={"A": inicial})
    props = {"D": {"A": 3.0e-7}, "fraccion": fraccion}
    final = paso_especies(campos, props, malla, 0.25, None)["A"]
    masa_inicial = float(np.sum(fraccion * inicial))
    masa_final = float(np.sum(fraccion * final))
    assert masa_final == pytest.approx(masa_inicial, rel=3.0e-12, abs=3.0e-12)


def test_radiacion_linealizada_coincide_con_enfriamiento_newton() -> None:
    malla = MallaPrueba((1, 1, 1), dx_mm=10.0, dz_mm=10.0)
    u, v, w = _velocidades_nulas(malla.forma)
    T0, T_amb = 800.0, 790.0
    rho, cp, k = 2500.0, 800.0, 1.0e6
    emisividad = 0.8
    estado = SimpleNamespace(u=u, v=v, w=w, T=np.array([[[T0]]]))
    props = {"rho": rho, "cp": cp, "k": k}
    fuentes = {
        "condiciones_frontera": {
            "x_min": {
                "tipo": "radiacion", "T_ambiente": T_amb, "emisividad": emisividad
            }
        }
    }
    dt, pasos = 0.1, 100
    for _ in range(pasos):
        estado.T = paso_energia(estado, props, malla, dt, fuentes)

    sigma = 5.670374419e-8
    h_rad = emisividad * sigma * (T0 + T_amb) * (T0**2 + T_amb**2)
    # Área transversal arbitraria 1 m²: V=A*dx y m=rho*V.
    exacta = enfriamiento_newton(dt * pasos, T0, T_amb, h_rad, 1.0, rho * 0.01, cp)
    assert estado.T.item() == pytest.approx(float(exacta), rel=2.0e-5)


def test_la_difusion_de_calor_conserva_energia_a_traves_de_una_interfaz():
    """Gas contra metal: 3.100 veces de salto en rho*cp.

    La formulación con difusividad promediaba alpha y aplicaba el MISMO
    coeficiente a las dos celdas de la cara, de modo que la pared recibía en
    grados lo mismo que el gas cedía: 3.100 veces más energía de la que salía.
    El crisol se calentaba en menos de un segundo en vez del minuto que le
    corresponde por su masa, y con ello toda la cronología del ensayo.
    """
    forma = (4, 4, 20)
    malla = MallaPrueba(forma, dx_mm=2.0, dz_mm=1.0)
    metal = np.zeros(forma, dtype=bool)
    metal[:, :, forma[2] // 2:] = True

    rho_cp = np.where(metal, 8400.0 * 500.0, 1.2 * 1150.0)
    k = np.where(metal, 16.0, 0.075)
    T = np.where(metal, 300.0, 1100.0)

    props = {"k": k, "rho": np.where(metal, 8400.0, 1.2),
             "cp": np.where(metal, 500.0, 1150.0), "esquema": "upwind"}
    campos = {"T": T.copy(),
              "u": np.zeros((forma[0] + 1, forma[1], forma[2])),
              "v": np.zeros((forma[0], forma[1] + 1, forma[2])),
              "w": np.zeros((forma[0], forma[1], forma[2] + 1))}

    volumen = 2.0e-3 * 2.0e-3 * 1.0e-3
    energia_inicial = float(np.sum(rho_cp * T) * volumen)
    dt = 1.0e-4
    for _ in range(200):
        campos["T"] = paso_energia(campos, props, malla, dt, None)
    energia_final = float(np.sum(rho_cp * campos["T"]) * volumen)

    # Sin fuentes ni fronteras activas la energía debe conservarse.
    error = abs(energia_final - energia_inicial) / abs(energia_inicial)
    assert error < 1.0e-10, f"energía no conservada: error relativo {error:.3e}"

    # Y el metal no puede calentarse más de lo que da la energía del gas.
    calentamiento_metal = float(np.mean(campos["T"][metal]) - 300.0)
    capacidad_gas = float(np.sum(rho_cp[~metal]) * volumen)
    capacidad_metal = float(np.sum(rho_cp[metal]) * volumen)
    maximo_fisico = (1100.0 - 300.0) * capacidad_gas / (capacidad_gas + capacidad_metal)
    assert calentamiento_metal <= maximo_fisico * 1.05, (
        f"el metal subió {calentamiento_metal:.3f} K; el límite por balance de "
        f"energía es {maximo_fisico:.3f} K")


def test_la_adveccion_de_calor_no_enfria_donde_se_genera_el_gas():
    """div(uT) incluye T·div(u), que no transporta nada: es dilatación.

    Con fuente de masa —la devolatilización crea unos 19 cm3/s dentro del
    lecho— ese término restaba cientos de K/s a las celdas del lecho. El gas
    aparece a la temperatura del sólido que lo suelta, así que no puede enfriar
    la celda que lo genera.
    """
    forma = (5, 5, 5)
    malla = MallaPrueba(forma, dx_mm=1.0, dz_mm=1.0)
    T = np.full(forma, 700.0)
    # Campo con divergencia positiva pura: velocidad creciente en z.
    w = np.zeros((forma[0], forma[1], forma[2] + 1))
    for k in range(forma[2] + 1):
        w[:, :, k] = 0.01 * k
    campos = {"T": T.copy(),
              "u": np.zeros((forma[0] + 1, forma[1], forma[2])),
              "v": np.zeros((forma[0], forma[1] + 1, forma[2])),
              "w": w}
    props = {"k": np.full(forma, 1.0e-9), "rho": np.full(forma, 1.0),
             "cp": np.full(forma, 1.0), "esquema": "upwind"}

    dt = 1.0e-3
    resultado = paso_energia(campos, props, malla, dt, None)
    # Campo uniforme + divergencia pura: la temperatura no puede cambiar.
    # Sin la corrección el cambio sería T*div(u)*dt = 700*10*1e-3 = 7 K; lo que
    # queda (~1e-9 K) es el residuo del iterativo, rtol=1e-11 sobre 700 K.
    espurio = 700.0 * 10.0 * dt
    desviacion = float(np.max(np.abs(resultado - 700.0)))
    assert desviacion < 1.0e-6 * espurio, (
        f"la dilatación cambió la temperatura en {desviacion:.3e} K")
