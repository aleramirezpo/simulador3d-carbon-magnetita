"""Pruebas del splitting de Strang y del controlador adaptativo."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nucleo"))

import acople as ac  # noqa: E402
import geometria as geo  # noqa: E402
import momentum as mm  # noqa: E402


class MallaFalsa:
    def __init__(self, dx_mm: float = 1.0, dz_mm: float | None = None):
        self.dx_mm = dx_mm
        self.dy_mm = dx_mm
        self.dz_mm = dx_mm if dz_mm is None else dz_mm


def _estado(forma=(4, 4, 4)) -> ac.Estado:
    nx, ny, nz = forma
    return ac.Estado(
        t=0.0,
        u=np.zeros((nx + 1, ny, nz)),
        v=np.zeros((nx, ny + 1, nz)),
        w=np.zeros((nx, ny, nz + 1)),
        P=np.zeros(forma),
        T=np.full(forma, 1173.15),
        c={"A": np.full(forma, 2.0), "B": np.full(forma, 1.0)},
        eps=np.ones(forma),
    )


def _props(forma=(4, 4, 4), nu=1.0e-4) -> mm.PropiedadesMedio:
    return mm.PropiedadesMedio(
        rho=np.ones(forma), mu=nu, eps=np.ones(forma),
        K=np.full(forma, np.inf), C_F=0.0, beta=0.0, mu_ef=nu,
    )


def _masa_total(estado: ac.Estado) -> float:
    return float(sum(np.sum(campo) for campo in estado.c.values()))


def test_separacion_strang_conserva_masa_total():
    estado = _estado()
    masa_inicial = _masa_total(estado)
    subpasos = []

    def quimica_conservativa(est: ac.Estado, dt: float) -> ac.Estado:
        subpasos.append(dt)
        salida = est.copia()
        transferido = 0.2 * dt * salida.c["A"]
        salida.c["A"] -= transferido
        salida.c["B"] += transferido
        return salida

    cfg = ac.ConfigAcople(
        con_momentum=False, con_transporte=False, con_quimica=True)
    final, _ = ac.paso_global(
        estado, _props(), MallaFalsa(), 0.2, cfg,
        quimica=quimica_conservativa)

    assert subpasos == [0.1, 0.1]
    assert math.isclose(_masa_total(final), masa_inicial,
                        rel_tol=0.0, abs_tol=1.0e-12)


def test_bucle_adaptativo_alcanza_t_final_exactamente():
    estado = _estado()
    cfg = ac.ConfigAcople(
        dt_inicial=0.03, dt_min=1.0e-6, dt_max=0.04,
        factor_crecimiento=1.2, con_momentum=False,
        con_transporte=False, con_quimica=False,
    )
    t_final = 0.137
    final, historial = ac.integrar(
        estado, _props(), MallaFalsa(), t_final, cfg=cfg)

    assert final.t == t_final
    assert math.isclose(sum(paso["dt"] for paso in historial), t_final,
                        rel_tol=0.0, abs_tol=2.0e-16)
    assert all(paso["t"] <= t_final for paso in historial)


def test_paso_adaptativo_respeta_limite_de_estabilidad():
    estado = _estado()
    rng = np.random.default_rng(7)
    estado.u[:] = rng.normal(scale=1.0e-3, size=estado.u.shape)
    estado.v[:] = rng.normal(scale=1.0e-3, size=estado.v.shape)
    estado.w[:] = rng.normal(scale=1.0e-3, size=estado.w.shape)
    cfg_momentum = mm.ConfigMomentum(
        con_adveccion=False, con_darcy=False, con_forchheimer=False,
        con_boyancia=False, con_proyeccion=False,
        paredes_en_el_borde=False, viscoso_implicito=False,
    )
    cfg = ac.ConfigAcople(
        dt_inicial=1.0, dt_max=1.0, cfl=0.1,
        con_momentum=True, con_transporte=False, con_quimica=False,
        cfg_momentum=cfg_momentum,
    )
    malla = MallaFalsa(1.0)
    limite = ac.dt_estable(estado, _props(), malla, cfg)["global"]
    final, historial = ac.integrar(
        estado, _props(), malla, 2.5 * limite, cfg=cfg)

    assert final.t == 2.5 * limite
    assert historial
    for paso in historial:
        assert paso["dt"] <= paso["limite_estabilidad"] * (1.0 + 1.0e-14)
        assert paso["restringe"] == "viscoso_momentum_si_fuese_explicito"


def test_geometria_real_proyecta_sin_aviso_de_divergencia():
    """El acople infiere y reutiliza pared, tapa y exterior impermeables."""
    dominio = geo.dominio_del_ensayo(dx_mm=2.0, dz_mm=1.0)
    malla = dominio["malla"]
    etiquetas = dominio["etiquetas"]
    forma = malla.forma
    solido = ~np.isin(etiquetas, (geo.GAS, geo.LECHO))
    eps = np.where(solido, 1.0e-6, 1.0)
    estado = _estado(forma)
    estado.eps = eps.copy()
    rng = np.random.default_rng(20260801)
    estado.u[:] = rng.normal(scale=1.0e-6, size=estado.u.shape)
    estado.v[:] = rng.normal(scale=1.0e-6, size=estado.v.shape)
    estado.w[:] = rng.normal(scale=1.0e-6, size=estado.w.shape)
    props = mm.PropiedadesMedio(
        rho=np.ones(forma), mu=0.0, eps=eps,
        K=np.where(solido, 1.0e-20, np.inf), C_F=0.0, beta=0.0,
        mu_ef=0.0,
    )
    cfg = ac.ConfigAcople(
        con_transporte=False, con_quimica=False,
        tolerancia_divergencia=1.0e-8,
        cfg_momentum=mm.ConfigMomentum(
            con_adveccion=False, con_viscoso=False, con_darcy=False,
            con_forchheimer=False, con_boyancia=False,
        ),
    )

    with warnings.catch_warnings(record=True) as emitidos:
        warnings.simplefilter("always")
        final, diag = ac.paso_global(estado, props, malla, 1.0e-4, cfg)

    dx, dz = malla.dx_mm * 1.0e-3, malla.dz_mm * 1.0e-3
    div = mm.divergencia(final.u, final.v, final.w, dx, dx, dz)
    assert np.max(np.abs(div[~solido])) < 1.0e-8
    assert diag["divergencia_residual"] < 1.0e-8
    assert not [w for w in emitidos if "divergencia residual" in str(w.message)]


def test_el_transporte_de_energia_se_ejecuta_de_verdad():
    """El calor debe transportarse: la temperatura tiene que cambiar.

    Existió un fallo silencioso y grave: `_paso_transporte` comprobaba que las
    funciones de transporte existieran y devolvía el estado SIN llamarlas. El
    resultado era que la muestra permanecía a su temperatura inicial por mucho
    que la mufla estuviera a 900 C, y ninguna prueba se enteraba porque no
    saltaba ninguna excepción. Esta prueba fija ese contrato: tras un paso con
    transporte activo y un gradiente térmico impuesto, T debe haber cambiado.
    """
    import numpy as np
    import acople as ac
    import momentum as mom

    forma = (8, 8, 8)

    class _Malla:
        dx_mm, dz_mm = 1.0, 1.0
        shape = forma
        def __init__(self):
            self.forma = forma
    T = np.full(forma, 300.0)
    T[0, :, :] = 1173.15          # una cara caliente

    est = ac.Estado(
        t=0.0,
        u=np.zeros((9, 8, 8)), v=np.zeros((8, 9, 8)), w=np.zeros((8, 8, 9)),
        P=np.zeros(forma), T=T.copy(), c={"CO": np.zeros(forma)},
        eps=np.full(forma, 0.54),
    )
    props = mom.PropiedadesMedio(
        rho=np.full(forma, 0.29), mu=4.5e-5,
        eps=np.full(forma, 0.54), K=np.full(forma, 1.519e-10))

    cfg = ac.ConfigAcople(con_momentum=False, con_quimica=False,
                          con_transporte=True)
    est2, _ = ac.paso_global(
        est, props, _Malla(), dt=1.0e-3, cfg=cfg,
        propiedades_termicas={"alpha": 1.0e-5, "rho": props.rho,
                              "cp": 1000.0, "rho_cp": 1.0e6,
                              "D_CO": 1.5e-5},
    )

    assert not np.allclose(est2.T, T), (
        "la temperatura no cambió: el transporte de energía no se está ejecutando")
    # el calor fluye de caliente a frío, nunca al revés
    assert est2.T[1, 4, 4] > T[1, 4, 4], "la segunda capa debería haberse calentado"
    assert est2.T.max() <= T.max() + 1e-9, "no puede aparecer calor de la nada"
