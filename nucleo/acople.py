"""
Acoplamiento temporal: separación de operadores de Strang.

POR QUÉ HACE FALTA SEPARAR
--------------------------
El sistema reúne procesos con escalas de tiempo que difieren en varios órdenes
de magnitud:

  * relajación de Darcy en el lecho:  tau = rho K /(mu eps) ~ 1,8e-6 s
  * advección a través de una celda:  ~ 1e-2 s
  * difusión térmica en una celda:    ~ 1e-3 s
  * cinética química:                 rígida, con modos de 1e-9 s a 1e2 s

Integrar todo junto con un método explícito obligaría a un paso dictado por el
modo más rápido, que es la química. La separación permite tratar cada bloque con
el método que le conviene: la química con un integrador implícito y positivo, el
transporte de forma semi-implícita, y el momentum por proyección.

ESQUEMA DE STRANG
-----------------
    quimica(dt/2) -> momentum(dt) -> transporte(dt) -> quimica(dt/2)

Es simétrico y, por tanto, de segundo orden en el tiempo *para la separación*.
Conviene no engañarse con esto: el orden global lo limita el bloque menos
preciso. Con Euler explícito en el momentum, el conjunto es de primer orden. La
verificación por MMS ya lo confirma (orden temporal medido: 1,011). La simetría
de Strang sigue mereciendo la pena porque elimina el sesgo sistemático de la
separación de Lie, aunque no eleve el orden global.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import momentum as mom

try:
    import transporte as trans
    HAY_TRANSPORTE = True
except Exception:  # pragma: no cover
    trans = None
    HAY_TRANSPORTE = False


@dataclass
class ConfigAcople:
    """Ajustes del bucle temporal."""

    dt_inicial: float = 1.0e-4
    dt_min: float = 1.0e-9
    dt_max: float = 1.0e-1
    cfl: float = 0.4
    factor_crecimiento: float = 1.15     # cuánto puede crecer dt entre pasos
    factor_reduccion: float = 0.5
    con_momentum: bool = True
    con_transporte: bool = True
    con_quimica: bool = True
    tolerancia_divergencia: float = 1.0e-8
    verboso: bool = False
    cfg_momentum: mom.ConfigMomentum = field(default_factory=mom.ConfigMomentum)


@dataclass
class Estado:
    """Estado completo del sistema (contrato §2)."""

    t: float
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    P: np.ndarray
    T: np.ndarray
    c: dict[str, np.ndarray]
    eps: np.ndarray
    solido_fases: dict[str, np.ndarray] = field(default_factory=dict)
    cohesion: np.ndarray | None = None

    def copia(self) -> "Estado":
        return Estado(
            t=self.t, u=self.u.copy(), v=self.v.copy(), w=self.w.copy(),
            P=self.P.copy(), T=self.T.copy(),
            c={k: a.copy() for k, a in self.c.items()},
            eps=self.eps.copy(),
            solido_fases={k: a.copy() for k, a in self.solido_fases.items()},
            cohesion=None if self.cohesion is None else self.cohesion.copy(),
        )


def _mascara_solida_del_medio(
        props: mom.PropiedadesMedio, forma: tuple[int, int, int],
        ) -> np.ndarray | None:
    """Reconoce las celdas impermeables regularizadas del caso declarativo.

    El caso usa ``eps=1e-6`` en pared, tapa y exterior para evitar divisiones
    por cero; cualquier region fluida o porosa tiene una porosidad fisica muy
    superior. La mascara se calcula una vez al iniciar la integracion.
    """
    eps = np.broadcast_to(np.asarray(props.eps, dtype=float), forma)
    solido = eps <= 1.0e-5
    return solido if np.any(solido) else None


def dt_estable(estado: Estado, props: mom.PropiedadesMedio, malla: Any,
               cfg: ConfigAcople, difusividad_max: float = 2.3e-4,
               ) -> dict[str, float | str]:
    """Paso estable de cada bloque, y el mínimo que gobierna el conjunto.

    Devuelve todos los límites por separado, no sólo el mínimo: saber *cuál*
    restringe es lo que permite decidir dónde merece la pena invertir esfuerzo
    (por ejemplo, hacer implícito el bloque que manda).
    """
    dx = malla.dx_mm * 1e-3
    dz = malla.dz_mm * 1e-3
    h_min = min(dx, dz)

    u_c = 0.5 * (estado.u[:-1] + estado.u[1:])
    v_c = 0.5 * (estado.v[:, :-1] + estado.v[:, 1:])
    w_c = 0.5 * (estado.w[:, :, :-1] + estado.w[:, :, 1:])
    vel_max = float(np.sqrt(u_c**2 + v_c**2 + w_c**2).max())

    dt_adv = cfg.cfl * h_min / vel_max if vel_max > 0 else math.inf
    dt_dif_explicito = (cfg.cfl * 0.5 * h_min ** 2 / difusividad_max
                        if difusividad_max > 0 else math.inf)
    limites_mom = mom.dt_estable_momentum(
        estado.u, estado.v, estado.w, props, malla, seguridad=cfg.cfl,
        cfg=cfg.cfg_momentum, devolver_diagnostico=True,
    )

    # `transporte.paso_energia` y `paso_especies` resuelven la difusión de forma
    # implícita (matriz dispersa + BiCGSTAB), así que su criterio parabólico NO
    # limita el paso. Se conserva el valor como diagnóstico, porque saber cuánto
    # se está ganando con el tratamiento implícito es información útil, pero no
    # entra en el mínimo.
    limites = {
        "adveccion": dt_adv,
        "momentum": (float(limites_mom["global"])
                     if cfg.con_momentum else math.inf),
        "viscoso_momentum_si_fuese_explicito": float(
            limites_mom["viscoso_si_fuese_explicito"]),
        "darcy_momentum_si_fuese_explicito": float(
            limites_mom["darcy_si_fuese_explicito"]),
        "forchheimer_momentum_si_fuese_explicito": float(
            limites_mom["forchheimer_si_fuese_explicito"]),
        "difusion_transporte_si_fuese_explicita": dt_dif_explicito,
        # Alias histórico: se conserva para consumidores anteriores.
        "difusion_si_fuese_explicita": dt_dif_explicito,
    }
    activos: list[str] = []
    if cfg.con_transporte or (cfg.con_momentum and cfg.cfg_momentum.con_adveccion):
        activos.append("adveccion")
    if (cfg.con_momentum and cfg.cfg_momentum.con_viscoso
            and not cfg.cfg_momentum.viscoso_implicito):
        activos.append("viscoso_momentum_si_fuese_explicito")

    if activos:
        limites["global"] = min(limites[k] for k in activos)
        limites["restringe"] = min(activos, key=lambda k: limites[k])
    else:
        limites["global"] = math.inf
        limites["restringe"] = "ninguno"
    return limites


def paso_global(estado: Estado, props: mom.PropiedadesMedio, malla: Any,
                dt: float, cfg: ConfigAcople,
                quimica: Callable[[Estado, float], Estado] | None = None,
                fuente_masa: Callable[[Estado], np.ndarray] | None = None,
                solucionador: mom.SolucionadorPresion | None = None,
                solucionador_viscoso: mom.SolucionadorViscoso | None = None,
                propiedades_termicas: dict[str, Any] | None = None,
                fuentes_transporte: dict[str, Any] | None = None,
                solido: np.ndarray | None = None,
                ) -> tuple[Estado, dict[str, Any]]:
    """Un paso completo con separación de Strang.

    ``quimica`` es una función que avanza la química local un intervalo dado;
    se inyecta desde fuera para que el núcleo no dependa de ninguna química
    concreta (contrato §5).
    """
    diag: dict[str, Any] = {"dt": dt}
    est = estado.copia()

    # --- media etapa de química ---
    if cfg.con_quimica and quimica is not None:
        est = quimica(est, 0.5 * dt)

    # --- momentum ---
    if cfg.con_momentum:
        f_masa = fuente_masa(est) if fuente_masa is not None else None
        mascara_solida = (solido if solido is not None
                           else _mascara_solida_del_medio(props, est.P.shape))
        res = mom.paso_momentum(est.u, est.v, est.w, est.P, est.T, props, malla,
                                dt, solido=mascara_solida, fuente=f_masa,
                                cfg=cfg.cfg_momentum,
                                solucionador=solucionador,
                                solucionador_viscoso=solucionador_viscoso)
        est.u, est.v, est.w, est.P = res["u"], res["v"], res["w"], res["P"]
        diag["divergencia_residual"] = res["divergencia_residual"]
        diag["incompatibilidad_divergencia"] = res.get(
            "incompatibilidad_divergencia", 0.0)
        if (res["proyectado"]
                and res["divergencia_residual"] > cfg.tolerancia_divergencia):
            warnings.warn(
                f"divergencia residual {res['divergencia_residual']:.3e} por encima "
                f"de la tolerancia {cfg.tolerancia_divergencia:.1e}")

    # --- transporte de calor y especies ---
    if cfg.con_transporte and HAY_TRANSPORTE:
        est = _paso_transporte(est, props, malla, dt,
                               propiedades_termicas=propiedades_termicas,
                               fuentes=fuentes_transporte)

    # --- media etapa de química ---
    if cfg.con_quimica and quimica is not None:
        est = quimica(est, 0.5 * dt)

    est.t = estado.t + dt
    diag["t"] = est.t
    return est, diag


def _paso_transporte(est: Estado, props: mom.PropiedadesMedio, malla: Any,
                     dt: float, propiedades_termicas: dict[str, Any] | None = None,
                     fuentes: dict[str, Any] | None = None) -> Estado:
    """Avanza temperatura y especies llamando al módulo de transporte.

    .. note::
       Una versión anterior de esta función comprobaba que ``paso_energia`` y
       ``paso_especies`` existieran y devolvía el estado **sin ejecutarlas**. El
       efecto era silencioso y grave: la temperatura se quedaba congelada en su
       valor inicial (298,15 K) por mucho que la mufla estuviera a 900 °C, y sin
       embargo nada fallaba. De ahí que se añadiera la comprobación explícita de
       que el estado cambia realmente, más abajo.
    """
    if not HAY_TRANSPORTE:
        return est

    for nombre in ("paso_energia", "paso_especies"):
        if getattr(trans, nombre, None) is None:
            raise AttributeError(
                f"el módulo de transporte no expone {nombre}; revisar docs/CONTRATOS.md §4")

    campos = {"T": est.T, "c": est.c, "u": est.u, "v": est.v, "w": est.w,
              "eps": est.eps}
    props_t = dict(propiedades_termicas or {})
    props_t.setdefault("rho", props.rho)
    fuentes = fuentes or {}

    est.T = np.asarray(trans.paso_energia(campos, props_t, malla, dt,
                                          fuentes.get("energia")), dtype=float)
    campos["T"] = est.T
    est.c = {k: np.asarray(v, dtype=float) for k, v in
             trans.paso_especies(campos, props_t, malla, dt,
                                 fuentes.get("especies")).items()}
    return est


def integrar(estado: Estado, props: mom.PropiedadesMedio, malla: Any,
             t_final: float, cfg: ConfigAcople | None = None,
             quimica: Callable[[Estado, float], Estado] | None = None,
             fuente_masa: Callable[[Estado], np.ndarray] | None = None,
             al_guardar: Callable[[Estado, dict], None] | None = None,
             intervalo_guardado: float = 1.0,
             solido: np.ndarray | None = None,
             propiedades_termicas: dict[str, Any] | None = None,
             fuentes_transporte: dict[str, Any] | None = None,
             ) -> tuple[Estado, list[dict[str, Any]]]:
    """Bucle temporal con paso adaptativo.

    El paso se ajusta a los límites de estabilidad calculados en cada instante,
    con crecimiento acotado para evitar oscilaciones del propio controlador.
    """
    cfg = ConfigAcople() if cfg is None else cfg
    est = estado.copia()
    mascara_solida = (solido if solido is not None
                       else _mascara_solida_del_medio(props, est.P.shape))
    props_transporte = dict(propiedades_termicas or {})
    # Valores de referencia ya usados por el controlador de estabilidad. Los
    # llamadores con propiedades detalladas pueden sobrescribirlos.
    props_transporte.setdefault("alpha", 2.3e-4)
    props_transporte.setdefault("D", 1.5e-4)
    dt = cfg.dt_inicial
    historial: list[dict[str, Any]] = []
    solucionador = mom.SolucionadorPresion(
        est.P.shape, malla.dx_mm * 1e-3,
        getattr(malla, "dy_mm", malla.dx_mm) * 1e-3,
        malla.dz_mm * 1e-3)
    solucionador_viscoso = mom.SolucionadorViscoso(
        malla.dx_mm * 1e-3,
        getattr(malla, "dy_mm", malla.dx_mm) * 1e-3,
        malla.dz_mm * 1e-3,
    )

    proximo_guardado = est.t
    n_paso = 0
    while est.t < t_final - 1e-15:
        limites = dt_estable(est, props, malla, cfg)
        restante = t_final - est.t
        dt = max(min(dt * cfg.factor_crecimiento, cfg.dt_max), cfg.dt_min)
        # La última fracción puede ser menor que dt_min: truncarla es necesario
        # para alcanzar t_final exactamente y no sobrepasarlo. El límite de
        # estabilidad también prevalece sobre dt_min: nunca se gana robustez
        # imponiendo un piso que viole el CFL.
        dt = min(dt, float(limites["global"]), restante)
        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError(f"paso temporal inválido: dt={dt!r}")

        est, diag = paso_global(est, props, malla, dt, cfg, quimica,
                                fuente_masa, solucionador,
                                solucionador_viscoso,
                                propiedades_termicas=props_transporte,
                                fuentes_transporte=fuentes_transporte,
                                solido=mascara_solida)
        diag["restringe"] = limites["restringe"]
        diag["limite_estabilidad"] = limites["global"]
        diag["n_paso"] = n_paso
        historial.append(diag)
        n_paso += 1

        if al_guardar is not None and est.t >= proximo_guardado:
            al_guardar(est, diag)
            proximo_guardado += intervalo_guardado

        if cfg.verboso and n_paso % 100 == 0:
            print(f"  paso {n_paso}: t={est.t:.4f} s  dt={dt:.3e} s  "
                  f"restringe={limites['restringe']}")

    if abs(est.t - t_final) <= 8.0 * np.finfo(float).eps * max(1.0, abs(t_final)):
        est.t = float(t_final)
        if historial:
            historial[-1]["t"] = float(t_final)

    return est, historial
