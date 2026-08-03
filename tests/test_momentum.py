"""Verificación del solucionador de momentum contra soluciones exactas.

Un solucionador de Navier--Stokes que no se contrasta con una solución conocida
no es un solucionador: es una animación. Estas pruebas comprueban lo esencial:
incompresibilidad, el límite de Darcy, el perfil de Poiseuille y la ausencia de
modos espurios de presión.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))

import momentum as mm  # noqa: E402


class MallaFalsa:
    """Malla mínima con la interfaz que espera el solucionador."""

    def __init__(self, dx_mm: float, dz_mm: float | None = None):
        self.dx_mm = dx_mm
        self.dz_mm = dz_mm if dz_mm is not None else dx_mm


def _campos_cero(forma):
    nx, ny, nz = forma
    return (np.zeros((nx + 1, ny, nz)), np.zeros((nx, ny + 1, nz)),
            np.zeros((nx, ny, nz + 1)))


def _props(forma, eps_val=1.0, K_val=np.inf, mu=4.5e-5, rho=0.29, beta=0.0):
    return mm.PropiedadesMedio(
        rho=np.full(forma, rho), mu=mu,
        eps=np.full(forma, eps_val), K=np.full(forma, K_val), beta=beta)


# ---------------------------------------------------------------------------
def test_divergencia_es_exacta_para_campo_lineal():
    """div de un campo con divergencia conocida debe salir exacta."""
    nx = ny = nz = 8
    dx = dy = dz = 0.001
    x = (np.arange(nx + 1)) * dx
    u = np.tile(x[:, None, None], (1, ny, nz))          # u = x -> du/dx = 1
    v = np.zeros((nx, ny + 1, nz))
    w = np.zeros((nx, ny, nz + 1))
    div = mm.divergencia(u, v, w, dx, dy, dz)
    assert np.allclose(div, 1.0, atol=1e-9)


def test_proyeccion_produce_campo_solenoidal():
    """Tras un paso, la divergencia debe caer varios órdenes de magnitud.

    Es la prueba central del método de proyección: el campo corregido tiene que
    ser (casi) libre de divergencia.
    """
    forma = (10, 10, 10)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    rng = np.random.default_rng(0)
    u += rng.normal(0, 1e-3, u.shape)
    v += rng.normal(0, 1e-3, v.shape)
    w += rng.normal(0, 1e-3, w.shape)
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    props = _props(forma)

    res = mm.paso_momentum(u, v, w, P, T, props, malla, dt=1e-5,
                           cfg=mm.ConfigMomentum(con_adveccion=False))
    assert res["divergencia_residual"] < res["divergencia_inicial"] * 1e-6, (
        f"la proyección no redujo la divergencia: "
        f"{res['divergencia_inicial']:.3e} -> {res['divergencia_residual']:.3e}")


def _cfg_solo_proyeccion(paredes_en_el_borde: bool = True):
    return mm.ConfigMomentum(
        con_adveccion=False, con_viscoso=False, con_darcy=False,
        con_forchheimer=False, con_boyancia=False,
        paredes_en_el_borde=paredes_en_el_borde,
    )


def test_proyeccion_con_obstaculo_es_solenoidal_y_no_atraviesa_el_solido():
    """La matriz de presion debe tener exactamente el mismo grafo que el flujo."""
    forma = (12, 11, 10)
    malla = MallaFalsa(0.7, 0.4)
    solido = np.zeros(forma, dtype=bool)
    solido[4:8, 4:7, 3:7] = True
    rng = np.random.default_rng(917)
    u, v, w = (rng.normal(scale=2.0e-4, size=a.shape)
               for a in _campos_cero(forma))
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    dx, dz = malla.dx_mm * 1e-3, malla.dz_mm * 1e-3
    solver = mm.SolucionadorPresion(forma, dx, dx, dz)

    res = mm.paso_momentum(
        u, v, w, P, T, _props(forma), malla, 1.0e-5,
        solido=solido, cfg=_cfg_solo_proyeccion(), solucionador=solver,
    )
    div = mm.divergencia(res["u"], res["v"], res["w"], dx, dx, dz)
    assert np.max(np.abs(div[~solido])) < 1.0e-10

    # Cada componente MAC es normal a sus caras; toda cara adyacente a una
    # celda solida debe quedar cerrada exactamente, no solo dentro de tolerancia.
    assert np.all(res["u"][:-1][solido] == 0.0)
    assert np.all(res["u"][1:][solido] == 0.0)
    assert np.all(res["v"][:, :-1][solido] == 0.0)
    assert np.all(res["v"][:, 1:][solido] == 0.0)
    assert np.all(res["w"][:, :, :-1][solido] == 0.0)
    assert np.all(res["w"][:, :, 1:][solido] == 0.0)

    # La segunda llamada reutiliza matriz y factorizacion para la misma mascara.
    mm.paso_momentum(
        res["u"], res["v"], res["w"], res["P"], T, _props(forma),
        malla, 1.0e-5, solido=solido, cfg=_cfg_solo_proyeccion(),
        solucionador=solver,
    )
    assert solver.sistemas_enmascarados_construidos == 1


def test_compatibilidad_y_anclaje_se_calculan_solo_en_el_fluido():
    forma = (7, 6, 5)
    solido = np.zeros(forma, dtype=bool)
    solido[0] = True                    # la celda global 0 es solida
    solido[3, 2:4, 1:4] = True
    rhs = np.zeros(forma)
    rhs[solido] = 1.0e12                # no debe contaminar la media fluida
    solver = mm.SolucionadorPresion(forma, 1.0, 1.0, 1.0)
    phi = solver.resolver(rhs, solido=solido)
    assert np.all(phi[solido] == 0.0)
    assert np.all(phi[~solido] == 0.0)


def test_flujo_alrededor_de_obstaculo_conserva_el_caudal_entre_secciones():
    forma = (16, 12, 10)
    malla = MallaFalsa(1.0)
    solido = np.zeros(forma, dtype=bool)
    solido[6:10, 4:8, 3:7] = True
    u, v, w = _campos_cero(forma)
    u[:] = 3.0e-3
    P = np.zeros(forma)
    T = np.zeros(forma)

    res = mm.paso_momentum(
        u, v, w, P, T, _props(forma), malla, 1.0e-4,
        solido=solido, cfg=_cfg_solo_proyeccion(False),
    )
    caudal_antes = float(res["u"][3].sum())
    caudal_despues = float(res["u"][13].sum())
    escala = max(abs(caudal_antes), abs(caudal_despues), 1.0e-30)
    assert abs(caudal_antes - caudal_despues) / escala < 1.0e-12
    assert np.max(np.abs(res["v"])) > 0.0 or np.max(np.abs(res["w"])) > 0.0


def test_mascara_vacia_conserva_exactamente_la_ruta_simple():
    forma = (8, 7, 6)
    malla = MallaFalsa(0.5)
    rng = np.random.default_rng(51)
    campos = tuple(rng.normal(scale=1.0e-4, size=a.shape)
                   for a in _campos_cero(forma))
    P = np.zeros(forma)
    T = np.zeros(forma)
    cfg = _cfg_solo_proyeccion()
    sin_mascara = mm.paso_momentum(
        *campos, P, T, _props(forma), malla, 1.0e-5, cfg=cfg)
    mascara_vacia = mm.paso_momentum(
        *campos, P, T, _props(forma), malla, 1.0e-5,
        solido=np.zeros(forma, dtype=bool), cfg=cfg)
    for nombre in ("u", "v", "w", "P"):
        assert np.array_equal(sin_mascara[nombre], mascara_vacia[nombre])


def test_reposo_permanece_en_reposo():
    """Sin fuerzas, un fluido en reposo no puede ponerse en movimiento."""
    forma = (8, 8, 8)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    res = mm.paso_momentum(u, v, w, P, T, _props(forma), malla, dt=1e-4)
    assert np.abs(res["u"]).max() < 1e-14
    assert np.abs(res["v"]).max() < 1e-14
    assert np.abs(res["w"]).max() < 1e-14


def test_arrastre_de_darcy_tiene_la_magnitud_correcta():
    """El término de Darcy debe valer exactamente -(mu/K) eps u / rho.

    Se comprueba el operador aislado, sin proyección ni bucle temporal: así el
    fallo, si lo hay, sólo puede estar en el término.
    """
    forma = (6, 6, 6)
    malla = MallaFalsa(0.5)
    mu, K, rho, eps = 4.5e-5, 1.5e-10, 0.29, 0.54
    props = _props(forma, eps_val=eps, K_val=K, mu=mu, rho=rho)

    u0 = 0.01
    u, v, w = _campos_cero(forma)
    u += u0
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)

    dt = 1e-12  # tan pequeño que el paso es esencialmente la derivada
    # sin proyección: se verifica el término de Darcy aislado
    cfg = mm.ConfigMomentum(con_adveccion=False, con_viscoso=False,
                            con_forchheimer=False, con_boyancia=False,
                            con_proyeccion=False, paredes_en_el_borde=False)
    res = mm.paso_momentum(u, v, w, P, T, props, malla, dt, cfg=cfg)

    du_dt = float((res["u"][3, 3, 3] - u0) / dt)
    esperado = -(mu / K) * eps * u0 / rho
    assert math.isclose(du_dt, esperado, rel_tol=1e-6), (
        f"arrastre de Darcy: {du_dt:.6e}, esperado {esperado:.6e}")


def test_relajacion_de_darcy_sigue_la_exponencial_analitica():
    """Sin forzado, u(t) = u0 exp(-t/tau) con tau = rho K /(mu eps).

    Es la solución exacta de la ecuación de Darcy transitoria y verifica que el
    acoplamiento temporal es consistente, no sólo el término instantáneo.
    """
    forma = (6, 6, 6)
    malla = MallaFalsa(0.5)
    mu, K, rho, eps = 4.5e-5, 1.5e-10, 0.29, 0.54
    props = _props(forma, eps_val=eps, K_val=K, mu=mu, rho=rho)
    tau = rho * K / (mu * eps)

    u0 = 0.01
    u, v, w = _campos_cero(forma)
    u += u0
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    cfg = mm.ConfigMomentum(con_adveccion=False, con_viscoso=False,
                            con_forchheimer=False, con_boyancia=False,
                            con_proyeccion=False, paredes_en_el_borde=False)

    n_pasos = 400
    t_final = 2.0 * tau
    dt = t_final / n_pasos
    for _ in range(n_pasos):
        res = mm.paso_momentum(u, v, w, P, T, props, malla, dt, cfg=cfg)
        u, v, w, P = res["u"], res["v"], res["w"], res["P"]

    u_num = float(u[3, 3, 3])
    u_exacta = u0 * math.exp(-t_final / tau)
    # Euler explícito: error O(dt); con 400 pasos por 2 tau basta el 2 %
    assert abs(u_num - u_exacta) / u0 < 0.02, (
        f"relajación: {u_num:.6e}, exacta {u_exacta:.6e}, tau={tau:.3e} s")


def test_forchheimer_frena_mas_que_darcy_solo():
    """El término inercial sólo puede restar velocidad, nunca sumarla."""
    forma = (8, 8, 8)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    u += 1.0
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    props = _props(forma, eps_val=0.54, K_val=1.5e-10)

    sin_f = mm.paso_momentum(u, v, w, P, T, props, malla, 1e-7,
                             cfg=mm.ConfigMomentum(con_forchheimer=False))
    con_f = mm.paso_momentum(u, v, w, P, T, props, malla, 1e-7,
                             cfg=mm.ConfigMomentum(con_forchheimer=True))
    assert np.abs(con_f["u"]).max() <= np.abs(sin_f["u"]).max() + 1e-12


def test_boyancia_genera_recirculacion_no_flujo_neto():
    """En un recinto cerrado la boyancia produce recirculación, no ascenso neto.

    La versión ingenua de esta prueba (exigir w medio positivo en la zona
    caliente) es incorrecta: con paredes impermeables la incompresibilidad
    obliga a que el caudal vertical neto sea nulo. Lo que debe comprobarse es
    que aparece movimiento y que su promedio se anula.
    """
    forma = (8, 8, 12)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    T[:4, :, :] += 50.0                        # gradiente horizontal de temperatura
    props = _props(forma, beta=1.0 / 1173.15)

    res = mm.paso_momentum(u, v, w, P, T, props, malla, dt=1e-3,
                           cfg=mm.ConfigMomentum(con_adveccion=False))
    w_new = res["w"]
    assert np.abs(w_new).max() > 0.0, "la boyancia no puso el fluido en movimiento"
    caudal_neto = float(w_new[:, :, w_new.shape[2] // 2].sum())
    escala = float(np.abs(w_new).sum()) + 1e-30
    assert abs(caudal_neto) / escala < 1e-6, (
        "en un recinto cerrado el caudal vertical neto debe anularse")
    # el lado caliente sube y el frío baja: hay recirculación
    w_caliente = w_new[:4, :, 6].mean()
    w_frio = w_new[4:, :, 6].mean()
    assert w_caliente * w_frio < 0.0, "no se formó celda de recirculación"


def test_permeabilidad_kozeny_carman_valor_conocido():
    """K para el lecho del ensayo: d=175 um, eps=0,54 -> 1,52e-10 m2."""
    K = mm.PropiedadesMedio.permeabilidad_kozeny_carman(175e-6, np.array([0.54]))
    assert math.isclose(float(K[0]), 1.519e-10, rel_tol=0.02)


def test_gas_libre_no_tiene_resistencia_de_darcy():
    """Con eps -> 1 la permeabilidad es infinita: Navier-Stokes puro."""
    K = mm.PropiedadesMedio.permeabilidad_kozeny_carman(175e-6, np.array([0.9995]))
    assert not np.isfinite(K[0])


def test_solido_impone_velocidad_nula():
    forma = (8, 8, 8)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    u += 1.0
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    solido = np.zeros(forma, dtype=bool)
    solido[3:5] = True
    res = mm.paso_momentum(u, v, w, P, T, _props(forma), malla, 1e-6,
                           solido=solido)
    assert np.abs(res["u"][3:5]).max() < 1e-14


def test_numeros_adimensionales_del_ensayo():
    """Con las condiciones reales del ensayo, el régimen debe ser el previsto."""
    forma = (8, 8, 8)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    u += 0.0466                                  # 4,66 cm/s, velocidad superficial
    props = _props(forma, eps_val=0.54, K_val=1.519e-10, beta=1.0 / 1173.15)
    nums = mm.numeros_adimensionales(u, v, w, np.full(forma, 1173.15), props, malla)
    assert nums["Re_particula"] < 1.0, "debería ser flujo reptante"
    assert nums["Ra"] < nums["Ra_critico"], "la convección natural debería ser débil"
    interp = mm.interpretar_regimen(nums)
    assert "Darcy lineal" in interp["inercia_en_poro"]
    assert "débil" in interp["conveccion_natural"]


def test_dt_estable_es_positivo_y_finito():
    forma = (8, 8, 8)
    malla = MallaFalsa(0.5, 0.25)
    u, v, w = _campos_cero(forma)
    u += 0.05
    props = _props(forma, eps_val=0.54, K_val=1.519e-10)
    diagnostico = mm.dt_estable_momentum(
        u, v, w, props, malla, devolver_diagnostico=True)
    dt = diagnostico["global"]
    assert 0.0 < dt < np.inf
    assert diagnostico["restringe"] == "adveccion"
    assert diagnostico["viscoso_si_fuese_explicito"] < dt

    cfg_explicita = mm.ConfigMomentum(viscoso_implicito=False)
    dt_explicito = mm.dt_estable_momentum(
        u, v, w, props, malla, cfg=cfg_explicita)
    assert dt_explicito == diagnostico["viscoso_si_fuese_explicito"]


def test_fuente_neta_en_recinto_cerrado_es_incompatible():
    """Un recinto cerrado no admite generación neta de masa: exige salida.

    Con paredes impermeables por todas partes, la incompresibilidad impone
    ``integral(div u) = 0``. Si se inyecta masa sin salida, el problema de
    Poisson con Neumann puro es incompatible y el solucionador reparte el exceso
    de forma uniforme: es la proyección sobre el espacio de soluciones admisible,
    no un error numérico.

    Consecuencia física para el simulador: el venteo del crisol NO es un detalle
    opcional. La devolatilización genera 19 cm3/s de gas y ese caudal tiene que
    salir por algún sitio; sin frontera abierta el modelo no tiene solución.
    """
    forma = (10, 10, 10)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    props = _props(forma)
    fuente = np.zeros(forma)
    fuente[4:6, 4:6, 4:6] = 1e-3            # kg/m3/s, localizada
    res = mm.paso_momentum(u, v, w, P, T, props, malla, 1e-5, fuente=fuente,
                           cfg=mm.ConfigMomentum(con_adveccion=False))
    div = mm.divergencia(res["u"], res["v"], res["w"],
                         malla.dx_mm * 1e-3, malla.dx_mm * 1e-3, malla.dz_mm * 1e-3)
    objetivo = fuente / props.rho

    # el residuo es una constante: exactamente el promedio de la fuente
    residuo = div - objetivo
    assert np.allclose(residuo, residuo.flat[0], rtol=1e-6), (
        "el residuo debería ser una constante uniforme")
    assert math.isclose(float(residuo.flat[0]), -float(objetivo.mean()), rel_tol=1e-6)

    # y la divergencia total se anula, como exige el recinto cerrado
    assert abs(float(div.sum())) < 1e-9 * max(1.0, float(np.abs(objetivo).sum()))


def test_fuente_compensada_se_reproduce_exactamente():
    """Con fuente y sumidero equilibrados, la divergencia sí es la pedida.

    Es el caso compatible: lo que se genera en un punto se extrae en otro, que es
    justamente lo que hará el venteo del crisol.
    """
    forma = (10, 10, 10)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    props = _props(forma)
    fuente = np.zeros(forma)
    fuente[2:4, 4:6, 4:6] = 1e-3            # generación
    fuente[6:8, 4:6, 4:6] = -1e-3           # extracción equivalente
    assert abs(fuente.sum()) < 1e-18, "la prueba requiere fuente neta nula"

    res = mm.paso_momentum(u, v, w, P, T, props, malla, 1e-5, fuente=fuente,
                           cfg=mm.ConfigMomentum(con_adveccion=False))
    div = mm.divergencia(res["u"], res["v"], res["w"],
                         malla.dx_mm * 1e-3, malla.dx_mm * 1e-3, malla.dz_mm * 1e-3)
    objetivo = fuente / props.rho
    error = np.abs(div - objetivo).max() / np.abs(objetivo).max()
    assert error < 1e-6, f"error relativo {error:.3e}"


def test_fuerza_vectorial_acelera_sin_anadir_masa():
    """Una fuerza de cuerpo debe acelerar el fluido sin generar divergencia.

    Distingue los dos acoplamientos del solucionador: ``fuente`` añade masa (y
    por tanto divergencia), mientras que ``fuerza`` añade momentum manteniendo
    el campo solenoidal. Confundirlos es un error frecuente y aquí queda fijado.
    """
    forma = (8, 8, 8)
    malla = MallaFalsa(0.5)
    u, v, w = _campos_cero(forma)
    P = np.zeros(forma)
    T = np.full(forma, 1173.15)
    props = _props(forma)

    fx = np.zeros(forma)
    fy = np.zeros(forma)
    fz = np.full(forma, 1.0)          # N/m3 uniforme hacia arriba
    cfg = mm.ConfigMomentum(con_adveccion=False, con_proyeccion=False,
                            paredes_en_el_borde=False)
    dt = 1e-6
    res = mm.paso_momentum(u, v, w, P, T, props, malla, dt,
                           fuerza=(fx, fy, fz), cfg=cfg)
    esperado = dt * 1.0 / float(props.rho.mean())
    assert math.isclose(float(res["w"][4, 4, 4]), esperado, rel_tol=1e-9)

    # con proyección y paredes, la fuerza uniforme no puede producir flujo neto
    res2 = mm.paso_momentum(u, v, w, P, T, props, malla, dt,
                            fuerza=(fx, fy, fz),
                            cfg=mm.ConfigMomentum(con_adveccion=False))
    div = mm.divergencia(res2["u"], res2["v"], res2["w"],
                         malla.dx_mm * 1e-3, malla.dx_mm * 1e-3, malla.dz_mm * 1e-3)
    assert np.abs(div).max() < 1e-12


# ---------------------------------------------------------------------------
# Viscosidad implícita
# ---------------------------------------------------------------------------
def _cfg_difusion(viscoso_implicito: bool) -> mm.ConfigMomentum:
    return mm.ConfigMomentum(
        con_adveccion=False, con_darcy=False, con_forchheimer=False,
        con_boyancia=False, con_proyeccion=False,
        paredes_en_el_borde=False, viscoso_implicito=viscoso_implicito,
    )


def test_viscoso_implicito_reproduce_difusion_senoidal_analitica():
    """Un modo de Fourier debe decaer como exp(-nu*k^2*t), dentro del 1 %."""
    forma = (4, 64, 4)
    nx, ny, nz = forma
    malla = MallaFalsa(1.0)
    h = malla.dx_mm * 1e-3
    longitud = ny * h
    nu = 1.0e-4
    props = _props(forma, mu=nu, rho=1.0)
    props.mu_ef = nu

    # cos(pi*y/L) es un perfil senoidal compatible con Neumann homogéneo.
    perfil = np.cos(np.pi * (np.arange(ny) + 0.5) / ny)[None, :, None]
    u0 = np.broadcast_to(perfil, (nx + 1, ny, nz)).copy()
    _, v, w = _campos_cero(forma)
    P = np.zeros(forma)
    T = np.zeros(forma)
    u = u0.copy()

    dt, t_final = 0.05, 1.0
    solucionador_viscoso = mm.SolucionadorViscoso(h, h, h)
    for _ in range(round(t_final / dt)):
        res = mm.paso_momentum(
            u, v, w, P, T, props, malla, dt,
            cfg=_cfg_difusion(True),
            solucionador_viscoso=solucionador_viscoso,
        )
        u, v, w, P = res["u"], res["v"], res["w"], res["P"]

    k = np.pi / longitud
    exacta = u0 * np.exp(-nu * k * k * t_final)
    error_relativo = np.linalg.norm(u - exacta) / np.linalg.norm(exacta)
    assert error_relativo < 0.01, (
        f"difusión senoidal: error relativo {error_relativo:.6%}")
    # Tres componentes, una sola construcción por componente pese a 20 pasos.
    assert solucionador_viscoso.precondicionadores_construidos == 3


def test_viscoso_explicito_e_implicito_son_consistentes_para_dt_pequeno():
    forma = (10, 10, 10)
    malla = MallaFalsa(1.0)
    nu = 1.0e-4
    props = _props(forma, mu=nu, rho=1.0)
    props.mu_ef = nu
    rng = np.random.default_rng(2026)
    campos = tuple(rng.normal(size=a.shape) for a in _campos_cero(forma))
    P = np.zeros(forma)
    T = np.zeros(forma)

    dt_limite = 0.4 * 0.5 * (1.0e-3 ** 2) / nu
    dt = 1.0e-3 * dt_limite
    explicito = mm.paso_momentum(
        *campos, P, T, props, malla, dt, cfg=_cfg_difusion(False))
    implicito = mm.paso_momentum(
        *campos, P, T, props, malla, dt, cfg=_cfg_difusion(True))

    for nombre in ("u", "v", "w"):
        diferencia = np.linalg.norm(implicito[nombre] - explicito[nombre])
        escala = np.linalg.norm(explicito[nombre])
        assert diferencia / escala < 1.0e-5


def test_viscoso_implicito_es_estable_muy_por_encima_del_limite_explicito():
    forma = (8, 8, 8)
    malla = MallaFalsa(1.0)
    nu = 1.0e-4
    props = _props(forma, mu=nu, rho=1.0)
    props.mu_ef = nu
    indices = np.indices((forma[0] + 1, forma[1], forma[2])).sum(axis=0)
    u0 = (-1.0) ** indices
    _, v0, w0 = _campos_cero(forma)
    P0 = np.zeros(forma)
    T = np.zeros(forma)

    dt_limite = 0.4 * 0.5 * (1.0e-3 ** 2) / nu
    dt = 10.0 * dt_limite
    razones = {}
    for etiqueta, implicito in (("explicito", False), ("implicito", True)):
        u, v, w, P = u0.copy(), v0.copy(), w0.copy(), P0.copy()
        solucionador_viscoso = mm.SolucionadorViscoso(1.0e-3, 1.0e-3, 1.0e-3)
        for _ in range(6):
            res = mm.paso_momentum(
                u, v, w, P, T, props, malla, dt,
                cfg=_cfg_difusion(implicito),
                solucionador_viscoso=solucionador_viscoso,
            )
            u, v, w, P = res["u"], res["v"], res["w"], res["P"]
        razones[etiqueta] = np.linalg.norm(u) / np.linalg.norm(u0)

    assert razones["explicito"] > 1.0e6
    assert razones["implicito"] < 1.0


def test_viscoso_implicito_conserva_orden_temporal_uno():
    forma = (3, 32, 3)
    nx, ny, nz = forma
    malla = MallaFalsa(1.0)
    h = 1.0e-3
    nu = 1.0e-4
    props = _props(forma, mu=nu, rho=1.0)
    props.mu_ef = nu
    perfil = np.cos(np.pi * (np.arange(ny) + 0.5) / ny)[None, :, None]
    u0 = np.broadcast_to(perfil, (nx + 1, ny, nz)).copy()
    _, v0, w0 = _campos_cero(forma)
    P0 = np.zeros(forma)
    T = np.zeros(forma)
    t_final = 0.5
    lambda_discreto = -4.0 / h**2 * np.sin(np.pi / (2.0 * ny))**2
    exacta = u0 * np.exp(nu * lambda_discreto * t_final)

    errores = []
    for dt in (0.1, 0.05, 0.025, 0.0125):
        u, v, w, P = u0.copy(), v0.copy(), w0.copy(), P0.copy()
        solucionador_viscoso = mm.SolucionadorViscoso(h, h, h)
        for _ in range(round(t_final / dt)):
            res = mm.paso_momentum(
                u, v, w, P, T, props, malla, dt,
                cfg=_cfg_difusion(True),
                solucionador_viscoso=solucionador_viscoso,
            )
            u, v, w, P = res["u"], res["v"], res["w"], res["P"]
        errores.append(np.linalg.norm(u - exacta) / np.linalg.norm(exacta))

    ordenes = [math.log(errores[i] / errores[i + 1], 2.0)
               for i in range(len(errores) - 1)]
    assert min(ordenes[-2:]) == pytest.approx(1.0, abs=0.04)


def test_viscoso_resuelve_con_el_flujo_ya_en_reposo():
    """El caso que mató la corrida de 720 s en t=425 s.

    Cuando la devolatilización termina, el campo de velocidad se apaga y el
    término independiente cae a ~1e-13 m/s. Con atol=0 se estaría exigiendo un
    residuo de 1e-24 sobre una matriz cuya diagonal llega a 1e16 por el Darcy
    de las celdas sólidas: por debajo del suelo de redondeo, BiCGSTAB sufre
    breakdown (info=-10) y la corrida entera muere con la física en reposo.
    """
    forma = (17, 16, 35)
    rng = np.random.default_rng(3)
    nu = np.full(forma, 1.5e-4)
    darcy = np.full(forma, 1.0e2)
    # Celdas sólidas: K=1e-20 como en el caso real, 18 órdenes de contraste.
    solido = rng.random(forma) < 0.55
    darcy[solido] = nu[solido] / 1.0e-20
    forchheimer = np.zeros(forma)
    solucionador = mm.SolucionadorViscoso(2.0e-3, 2.0e-3, 1.0e-3)

    for escala in (1.0e-3, 1.0e-9, 1.0e-12, 1.0e-15, 1.0e-18, 0.0):
        b = escala * rng.standard_normal(forma)
        x = solucionador.resolver(b, "x", 0.5, nu, darcy, forchheimer)
        assert np.all(np.isfinite(x))
        A = solucionador._cache["x"]["A"]
        residuo = np.max(np.abs(b.reshape(-1) - A @ x.reshape(-1)))
        # El residuo debe ser despreciable frente al propio término fuente.
        assert residuo <= 1.0e-6 * max(float(np.max(np.abs(b))), 1.0e-30) + 1.0e-18
        # Y con fuente nula la solución debe ser exactamente nula.
        if escala == 0.0:
            assert np.all(x == 0.0)


def test_viscoso_en_regimen_vivo_no_pierde_precision_por_la_tolerancia():
    """La tolerancia absoluta no debe relajar el caso normal.

    Con ||b|| ~ 1e-1 manda rtol=1e-11 y atol=1e-20 no interviene: la solución
    debe seguir coincidiendo con la del sistema resuelto de forma directa.
    """
    from scipy.sparse.linalg import spsolve

    forma = (12, 12, 20)
    rng = np.random.default_rng(11)
    nu = np.full(forma, 1.5e-4)
    darcy = np.full(forma, 1.0e2)
    forchheimer = np.zeros(forma)
    solucionador = mm.SolucionadorViscoso(2.0e-3, 2.0e-3, 1.0e-3)
    b = 1.0e-3 * rng.standard_normal(forma)
    x = solucionador.resolver(b, "x", 0.5, nu, darcy, forchheimer)
    A = solucionador._cache["x"]["A"]
    exacta = spsolve(A.tocsc(), b.reshape(-1))
    assert np.max(np.abs(x.reshape(-1) - exacta)) <= 1.0e-9 * np.max(np.abs(exacta))
