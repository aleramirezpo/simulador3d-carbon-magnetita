"""Soluciones analíticas de referencia para verificar el transporte escalar."""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def difusion_1d_semiinfinito(
    x: np.ndarray | float,
    t: float,
    D: float,
    phi0: float,
    phis: float,
) -> np.ndarray:
    """Difusión en un semiespacio con concentración superficial constante.

    Referencia: J. Crank, *The Mathematics of Diffusion*, 2.ª ed., Oxford
    University Press (1975), §2.2, solución en función error. El dominio es
    ``x >= 0``, inicialmente ``phi0``, y ``phi(0,t) = phis``.
    """
    xx = np.asarray(x, dtype=float)
    if np.any(xx < 0.0):
        raise ValueError("x debe ser no negativo")
    if t < 0.0 or D <= 0.0:
        raise ValueError("t debe ser no negativo y D positivo")
    if t == 0.0:
        return np.where(xx == 0.0, phis, phi0).astype(float)
    eta = xx / (2.0 * np.sqrt(D * t))
    return phis + (phi0 - phis) * erf(eta)


def conduccion_transitoria_placa(
    x: np.ndarray | float,
    t: float,
    alpha: float,
    L: float,
    T0: float,
    Ts: float,
    n_terminos: int = 100,
) -> np.ndarray:
    """Temperatura en una placa ``0 <= x <= L`` cuyas dos caras están a ``Ts``.

    La placa parte de temperatura uniforme ``T0``. Se evalúa la serie de senos
    de términos impares. Referencia: H. S. Carslaw y J. C. Jaeger,
    *Conduction of Heat in Solids*, 2.ª ed., Oxford University Press (1959),
    capítulo II (placa con temperaturas superficiales prescritas).
    """
    xx = np.asarray(x, dtype=float)
    if L <= 0.0 or alpha <= 0.0 or t < 0.0 or n_terminos < 1:
        raise ValueError("L, alpha y n_terminos deben ser positivos; t no negativo")
    if np.any((xx < 0.0) | (xx > L)):
        raise ValueError("x debe pertenecer a [0, L]")
    if t == 0.0:
        # Las caras cambian instantáneamente a Ts; el interior conserva T0.
        return np.where((xx == 0.0) | (xx == L), Ts, T0).astype(float)
    n = 2 * np.arange(n_terminos, dtype=float) + 1.0
    # El eje inicial permite trabajar también con x escalar sin casos especiales.
    modos = n.reshape((-1,) + (1,) * xx.ndim)
    posicion = np.expand_dims(xx, 0)
    serie = np.sum(
        (4.0 / (np.pi * modos))
        * np.sin(modos * np.pi * posicion / L)
        * np.exp(-alpha * (modos * np.pi / L) ** 2 * t),
        axis=0,
    )
    return Ts + (T0 - Ts) * serie


def adveccion_difusion_estacionaria_1d(
    x: np.ndarray | float,
    u: float,
    D: float,
    L: float,
    phi0: float,
    phiL: float,
) -> np.ndarray:
    """Solución de ``u phi' = D phi''`` entre dos valores de Dirichlet.

    Referencia: S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*,
    Hemisphere (1980), capítulo 5, problema unidimensional de convección y
    difusión. La forma exponencial se evalúa de manera estable también para
    números de Péclet grandes.
    """
    xx = np.asarray(x, dtype=float)
    if D <= 0.0 or L <= 0.0:
        raise ValueError("D y L deben ser positivos")
    if np.any((xx < 0.0) | (xx > L)):
        raise ValueError("x debe pertenecer a [0, L]")
    pe = float(u) * L / D
    xi = xx / L
    if abs(pe) < 1.0e-8:
        razon = xi + 0.5 * pe * xi * (xi - 1.0)
    elif pe > 50.0:
        razon = (np.exp(pe * (xi - 1.0)) * (-np.expm1(-pe * xi))
                 / (-np.expm1(-pe)))
    elif pe < -50.0:
        # Simetría respecto a invertir el eje evita desbordar exp(-Pe).
        pe_pos = -pe
        razon_inversa = (np.exp(pe_pos * ((1.0 - xi) - 1.0))
                         * (-np.expm1(-pe_pos * (1.0 - xi)))
                         / (-np.expm1(-pe_pos)))
        razon = 1.0 - razon_inversa
    else:
        razon = np.expm1(pe * xi) / np.expm1(pe)
    return phi0 + (phiL - phi0) * razon


def pulso_gaussiano_advectado(
    x: np.ndarray | float,
    t: float,
    u: float,
    D: float,
    sigma0: float,
) -> np.ndarray:
    """Pulso gaussiano unitario que se traslada y ensancha en una recta infinita.

    Referencia: J. Crank, *The Mathematics of Diffusion*, 2.ª ed. (1975),
    solución fundamental gaussiana. El pulso inicial tiene máximo uno, centro
    cero y desviación ``sigma0``; el prefactor conserva su integral.
    """
    xx = np.asarray(x, dtype=float)
    if t < 0.0 or D < 0.0 or sigma0 <= 0.0:
        raise ValueError("t y D deben ser no negativos; sigma0 positivo")
    varianza = sigma0 * sigma0 + 2.0 * D * t
    return sigma0 / np.sqrt(varianza) * np.exp(-(xx - u * t) ** 2 / (2.0 * varianza))


def enfriamiento_newton(
    t: np.ndarray | float,
    T0: float,
    T_ambiente: float,
    h: float,
    area: float,
    masa: float | None = None,
    cp: float | None = None,
    volumen: float | None = None,
    *,
    rho: float | None = None,
) -> np.ndarray:
    """Temperatura de un cuerpo concentrado sometido a una frontera linealizada.

    ``T = T_ambiente + (T0-T_ambiente) exp[-h A t/(m cp)]``. La
    capacidad puede indicarse como ``masa, cp`` o como ``rho, volumen, cp``.
    También se admite la forma posicional ``rho, cp, volumen``.
    Para radiación, úsese
    ``h = eps*sigma*(Tref+T_ambiente)*(Tref**2+T_ambiente**2)``.

    Referencia: F. P. Incropera et al., *Fundamentals of Heat and Mass
    Transfer*, 7.ª ed., Wiley (2011), análisis de capacidad concentrada.
    """
    tt = np.asarray(t, dtype=float)
    if volumen is not None:
        densidad = rho if rho is not None else masa
        masa_efectiva = None if densidad is None else densidad * volumen
    else:
        masa_efectiva = masa
    if (np.any(tt < 0.0) or h < 0.0 or area < 0.0 or masa_efectiva is None
            or cp is None or masa_efectiva <= 0.0 or cp <= 0.0):
        raise ValueError("t, h y área no negativos; masa y cp positivos")
    return T_ambiente + (T0 - T_ambiente) * np.exp(-h * area * tt / (masa_efectiva * cp))


__all__ = [
    "difusion_1d_semiinfinito",
    "conduccion_transitoria_placa",
    "adveccion_difusion_estacionaria_1d",
    "pulso_gaussiano_advectado",
    "enfriamiento_newton",
]
