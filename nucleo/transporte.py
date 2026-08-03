"""Transporte conservativo de escalares en una malla cartesiana MAC.

Los escalares se almacenan en centros de celda y las velocidades en caras.  Las
separaciones de :class:`~nucleo.geometria.MallaVoxel` se convierten de milímetros
a metros al entrar en este módulo; el resto de las magnitudes usa SI.

La convención de signos es

``d(phi)/dt = -div(u phi) + div(D grad(phi)) + fuente``.

Las fronteras son de flujo difusivo nulo salvo que ``paso_energia`` o
``paso_especies`` reciban ``condiciones_frontera``. Las caras se nombran
``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min`` y ``z_max``. Cada valor es
un diccionario ``{"tipo": "dirichlet", "valor": ...}``,
``{"tipo": "neumann", "gradiente": ...}`` o, para energía,
``{"tipo": "radiacion", "T_ambiente": ..., "emisividad": 0.8}``.
En Neumann, ``gradiente`` es la derivada normal hacia fuera; alternativamente,
``flujo`` es el flujo físico positivo hacia fuera del dominio.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


_SIGMA_SB = 5.670374419e-8  # W m-2 K-4
_CARAS = {
    "x_min": (0, 0, ("x_min", "xmin", "oeste")),
    "x_max": (0, -1, ("x_max", "xmax", "este")),
    "y_min": (1, 0, ("y_min", "ymin", "sur")),
    "y_max": (1, -1, ("y_max", "ymax", "norte")),
    "z_min": (2, 0, ("z_min", "zmin", "fondo")),
    "z_max": (2, -1, ("z_max", "zmax", "tapa")),
}


def _obtener(objeto: Any, *nombres: str, defecto: Any = None) -> Any:
    """Obtiene el primer nombre presente de un mapeo u objeto."""
    if objeto is None:
        return defecto
    for nombre in nombres:
        if isinstance(objeto, Mapping) and nombre in objeto:
            return objeto[nombre]
        if hasattr(objeto, nombre):
            return getattr(objeto, nombre)
    return defecto


def _espaciados(malla: Any) -> tuple[float, float, float]:
    """Separaciones ``(dx, dy, dz)`` en metros."""
    dx = float(malla.dx_mm) * 1.0e-3
    dy = float(getattr(malla, "dy_mm", malla.dx_mm)) * 1.0e-3
    dz = float(malla.dz_mm) * 1.0e-3
    if min(dx, dy, dz) <= 0.0:
        raise ValueError("los espaciados de la malla deben ser positivos")
    return dx, dy, dz


def _como_campo(valor: Any, forma: tuple[int, ...], nombre: str) -> np.ndarray:
    campo = np.asarray(valor, dtype=float)
    if campo.ndim == 0:
        return np.full(forma, float(campo), dtype=float)
    try:
        return np.broadcast_to(campo, forma).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(f"{nombre} no se puede expandir a la forma {forma}") from exc


def _validar_escalar_y_velocidades(
    phi: Any, u: Any, v: Any, w: Any, malla: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forma = tuple(malla.forma)
    p = np.asarray(phi, dtype=float)
    uu, vv, ww = (np.asarray(a, dtype=float) for a in (u, v, w))
    formas_v = (
        (forma[0] + 1, forma[1], forma[2]),
        (forma[0], forma[1] + 1, forma[2]),
        (forma[0], forma[1], forma[2] + 1),
    )
    if p.shape != forma:
        raise ValueError(f"phi tiene forma {p.shape}; se esperaba {forma}")
    for nombre, campo, esperada in zip(("u", "v", "w"), (uu, vv, ww), formas_v):
        if campo.shape != esperada:
            raise ValueError(f"{nombre} tiene forma {campo.shape}; se esperaba {esperada}")
    return p, uu, vv, ww


def _limitador_superbee(r: np.ndarray) -> np.ndarray:
    """Limitador de Roe/Sweby Superbee, vectorizado."""
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0),
                                      np.minimum(r, 2.0)))


def _razon_segura(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    escala = np.maximum(1.0, np.maximum(np.abs(numerador), np.abs(denominador)))
    util = np.abs(denominador) > 32.0 * np.finfo(float).eps * escala
    return np.divide(numerador, denominador, out=np.zeros_like(denominador), where=util)


def _valor_escalar_en_caras(
    phi: np.ndarray, velocidad: np.ndarray, eje: int, esquema: str
) -> np.ndarray:
    """Reconstruye ``phi`` en todas las caras normales a ``eje``."""
    n = phi.shape[eje]
    caras = np.empty_like(velocidad, dtype=float)

    if n == 1:
        # No hay cara interior ni pendiente reconstruible en esta dirección.
        caras[...] = np.expand_dims(np.take(phi, 0, axis=eje), axis=eje)
        return caras

    sl_cara_int = [slice(None)] * 3
    sl_cara_int[eje] = slice(1, n)
    sl_izq = [slice(None)] * 3
    sl_izq[eje] = slice(0, n - 1)
    sl_der = [slice(None)] * 3
    sl_der[eje] = slice(1, n)
    izquierda = phi[tuple(sl_izq)]
    derecha = phi[tuple(sl_der)]
    vel_int = velocidad[tuple(sl_cara_int)]

    if esquema == "central":
        interior = 0.5 * (izquierda + derecha)
    elif esquema == "upwind":
        interior = np.where(vel_int >= 0.0, izquierda, derecha)
    elif esquema == "tvd_superbee":
        # La celda donante se reconstruye hacia la cara. Cerca de una frontera
        # no hay segunda celda aguas arriba y se degrada localmente a upwind.
        delta = derecha - izquierda
        aguas_pos = np.concatenate(
            (np.take(phi, [0], axis=eje), np.take(phi, range(0, n - 2), axis=eje)),
            axis=eje,
        )
        aguas_neg = np.concatenate(
            (np.take(phi, range(2, n), axis=eje), np.take(phi, [n - 1], axis=eje)),
            axis=eje,
        )
        r_pos = _razon_segura(izquierda - aguas_pos, delta)
        r_neg = _razon_segura(aguas_neg - derecha, delta)
        desde_izq = izquierda + 0.5 * _limitador_superbee(r_pos) * delta
        desde_der = derecha - 0.5 * _limitador_superbee(r_neg) * delta
        interior = np.where(vel_int >= 0.0, desde_izq, desde_der)
    else:
        raise ValueError(
            "esquema debe ser 'upwind', 'central' o 'tvd_superbee'"
        )

    caras[tuple(sl_cara_int)] = interior
    sl_min = [slice(None)] * 3
    sl_min[eje] = 0
    sl_max = [slice(None)] * 3
    sl_max[eje] = -1
    celda_min = [slice(None)] * 3
    celda_min[eje] = 0
    celda_max = [slice(None)] * 3
    celda_max[eje] = -1
    caras[tuple(sl_min)] = phi[tuple(celda_min)]
    caras[tuple(sl_max)] = phi[tuple(celda_max)]
    return caras


def divergencia_flujo_advectivo(
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    malla: Any,
    esquema: str = "tvd_superbee",
) -> np.ndarray:
    """Devuelve ``div(u*phi)`` mediante un único flujo conservativo por cara.

    ``upwind`` es monótono de primer orden, ``central`` es de segundo orden pero
    no acotado, y ``tvd_superbee`` usa reconstrucción MUSCL limitada. En una
    frontera sólo existe una celda interior, por lo que se extrapola con
    gradiente nulo; en un dominio cerrado las velocidades normales de esas
    caras deben ser cero.
    """
    esquema = str(esquema).lower()
    p, uu, vv, ww = _validar_escalar_y_velocidades(phi, u, v, w, malla)
    h = _espaciados(malla)
    resultado = np.zeros_like(p, dtype=float)
    for eje, (velocidad, paso) in enumerate(zip((uu, vv, ww), h)):
        flujo = velocidad * _valor_escalar_en_caras(p, velocidad, eje, esquema)
        resultado += np.diff(flujo, axis=eje) / paso
    return resultado


def divergencia_velocidad(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                          malla: Any, referencia: np.ndarray | None = None) -> np.ndarray:
    """``div(u)`` en centros de celda, con el mismo convenio MAC que el flujo.

    No es cero cuando hay fuente de masa: la devolatilización crea gas dentro
    del lecho. Se necesita para separar el transporte real de la dilatación.
    """
    forma = tuple(malla.forma)
    patron = np.zeros(forma, dtype=float) if referencia is None else referencia
    _, uu, vv, ww = _validar_escalar_y_velocidades(patron, u, v, w, malla)
    h = _espaciados(malla)
    divergencia = np.zeros(forma, dtype=float)
    for eje, (velocidad, paso) in enumerate(zip((uu, vv, ww), h)):
        divergencia += np.diff(velocidad, axis=eje) / paso
    return divergencia


def _media_armonica(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    suma = a + b
    return np.divide(2.0 * a * b, suma, out=np.zeros_like(suma), where=suma > 0.0)


def divergencia_flujo_difusivo(
    phi: np.ndarray,
    D: float | np.ndarray,
    malla: Any,
    fraccion: np.ndarray | None = None,
) -> np.ndarray:
    """Devuelve ``div(D grad(phi))`` con media armónica en cada cara.

    Las fronteras del bloque tienen flujo nulo. Si se proporciona ``fraccion``,
    el área abierta de una cara interior se aproxima por el mínimo de las dos
    fracciones vecinas y el balance se divide por el volumen activo. Por ello
    ``sum(fraccion * resultado)`` es cero (salvo redondeo) en un dominio cerrado.
    """
    forma = tuple(malla.forma)
    p = np.asarray(phi, dtype=float)
    if p.shape != forma:
        raise ValueError(f"phi tiene forma {p.shape}; se esperaba {forma}")
    dif = _como_campo(D, forma, "D")
    if np.any(dif < 0.0) or not np.all(np.isfinite(dif)):
        raise ValueError("D debe ser finita y no negativa")
    frac = (np.ones(forma, dtype=float) if fraccion is None
            else _como_campo(fraccion, forma, "fraccion"))
    if np.any((frac < 0.0) | (frac > 1.0)):
        raise ValueError("fraccion debe estar en [0, 1]")

    resultado = np.zeros_like(p)
    for eje, paso in enumerate(_espaciados(malla)):
        n = forma[eje]
        sl_i, sl_d = [slice(None)] * 3, [slice(None)] * 3
        sl_i[eje], sl_d[eje] = slice(0, n - 1), slice(1, n)
        si, sd = tuple(sl_i), tuple(sl_d)
        d_cara = _media_armonica(dif[si], dif[sd])
        area_relativa = np.minimum(frac[si], frac[sd])
        flujo_int = d_cara * area_relativa * (p[sd] - p[si]) / paso
        forma_caras = list(forma)
        forma_caras[eje] += 1
        flujo = np.zeros(forma_caras, dtype=float)
        sl_f = [slice(None)] * 3
        sl_f[eje] = slice(1, n)
        flujo[tuple(sl_f)] = flujo_int
        resultado += np.diff(flujo, axis=eje) / paso

    return np.divide(resultado, frac, out=np.zeros_like(resultado), where=frac > 0.0)


def _operador_difusion(
    D: float | np.ndarray, malla: Any, fraccion: np.ndarray | None,
    capacidad: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Matriz del mismo operador usado por ``divergencia_flujo_difusivo``.

    Con ``capacidad`` se construye la variante **conservativa en energía**: ``D``
    pasa a interpretarse como conductividad, la cara promedia conductividades y
    cada fila se divide por la capacidad volumétrica de SU celda.

    Sin ella, el operador promedia difusividades y aplica el mismo coeficiente a
    las dos celdas de la cara. Eso es correcto mientras ``rho*cp`` sea uniforme,
    pero **crea energía** en una interfaz entre materiales distintos: entre el
    gas (1,4e3 J/m3K) y la pared Ni-Cr (4,2e6 J/m3K) el salto es de 3.100 veces,
    y la pared recibía en grados lo mismo que el gas cedía, es decir 3.100 veces
    más energía de la que salía. El crisol se calentaba en menos de un segundo
    en vez de en el minuto que le corresponde por su masa.
    """
    forma = tuple(malla.forma)
    dif = _como_campo(D, forma, "D")
    if np.any(dif < 0.0) or not np.all(np.isfinite(dif)):
        raise ValueError("D debe ser finita y no negativa")
    frac = (np.ones(forma, dtype=float) if fraccion is None
            else _como_campo(fraccion, forma, "fraccion"))
    if np.any((frac < 0.0) | (frac > 1.0)):
        raise ValueError("fraccion debe estar en [0, 1]")

    indices = np.arange(np.prod(forma), dtype=np.int64).reshape(forma)
    filas: list[np.ndarray] = []
    columnas: list[np.ndarray] = []
    datos: list[np.ndarray] = []
    for eje, paso in enumerate(_espaciados(malla)):
        n = forma[eje]
        sl_i, sl_d = [slice(None)] * 3, [slice(None)] * 3
        sl_i[eje], sl_d[eje] = slice(0, n - 1), slice(1, n)
        si, sd = tuple(sl_i), tuple(sl_d)
        izq, der = indices[si].ravel(), indices[sd].ravel()
        d_cara = _media_armonica(dif[si], dif[sd])
        area_relativa = np.minimum(frac[si], frac[sd])
        base = d_cara * area_relativa / (paso * paso)
        divisor_izq = frac[si] if capacidad is None else frac[si] * capacidad[si]
        divisor_der = frac[sd] if capacidad is None else frac[sd] * capacidad[sd]
        a_izq = np.divide(base, divisor_izq, out=np.zeros_like(base), where=divisor_izq > 0.0).ravel()
        a_der = np.divide(base, divisor_der, out=np.zeros_like(base), where=divisor_der > 0.0).ravel()
        filas.extend((izq, izq, der, der))
        columnas.extend((der, izq, izq, der))
        datos.extend((a_izq, -a_izq, a_der, -a_der))
    n_total = int(np.prod(forma))
    operador = sparse.coo_matrix(
        (np.concatenate(datos), (np.concatenate(filas), np.concatenate(columnas))),
        shape=(n_total, n_total),
    ).tocsr()
    operador.sum_duplicates()
    return operador, dif, frac


def _seleccionar_cara(campo: Any, forma: tuple[int, int, int], eje: int, lado: int) -> np.ndarray:
    arr = np.asarray(campo, dtype=float)
    forma_cara = tuple(n for q, n in enumerate(forma) if q != eje)
    if arr.ndim == 0:
        return np.full(forma_cara, float(arr))
    if arr.shape == forma:
        return np.take(arr, lado, axis=eje)
    try:
        return np.broadcast_to(arr, forma_cara)
    except ValueError as exc:
        raise ValueError(f"valor de frontera incompatible con la cara de forma {forma_cara}") from exc


def _configuracion_cara(condiciones: Any, nombre: str, aliases: tuple[str, ...]) -> Any:
    if not isinstance(condiciones, Mapping):
        return None
    for alias in aliases:
        if alias in condiciones:
            return condiciones[alias]
    todas = _obtener(condiciones, "todas", "paredes", defecto=None)
    if todas is not None:
        return todas
    # En un modelo de crisol sin máscara explícita de caras, ``mufla`` denota
    # todas las fronteras externas y ``tapa`` puede sobrescribir z_max.
    if nombre != "z_max" and "mufla" in condiciones:
        return condiciones["mufla"]
    return None


def _condiciones_del_campo(condiciones: Any, campo: str | None) -> Any:
    if condiciones is None or not isinstance(condiciones, Mapping):
        return condiciones
    if campo is not None and campo in condiciones and isinstance(condiciones[campo], Mapping):
        return condiciones[campo]
    for clave in ("energia", "temperatura") if campo == "T" else ("especies",):
        if clave in condiciones and isinstance(condiciones[clave], Mapping):
            sub = condiciones[clave]
            if campo is not None and campo in sub and isinstance(sub[campo], Mapping):
                return sub[campo]
            if campo == "T":
                return sub
    return condiciones


def _terminos_frontera(
    condiciones: Any,
    campo_actual: np.ndarray,
    D: np.ndarray,
    malla: Any,
    fraccion: np.ndarray,
    *,
    capacidad: np.ndarray | None = None,
    conductividad: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal y término independiente de las fronteras para ``L phi + b``."""
    forma = tuple(malla.forma)
    diagonal = np.zeros(forma, dtype=float)
    termino = np.zeros(forma, dtype=float)
    pasos = _espaciados(malla)
    if condiciones is None:
        return diagonal.ravel(), termino.ravel()

    for nombre, (eje, lado, aliases) in _CARAS.items():
        cfg = _configuracion_cara(condiciones, nombre, aliases)
        if cfg is None:
            continue
        if isinstance(cfg, str):
            cfg = {"tipo": cfg}
        if not isinstance(cfg, Mapping):
            cfg = {"tipo": "dirichlet", "valor": cfg}
        tipo = str(_obtener(cfg, "tipo", defecto="neumann")).lower()
        sl = [slice(None)] * 3
        sl[eje] = lado
        s = tuple(sl)
        activa = fraccion[s] > 0.0
        paso = pasos[eje]

        if tipo == "dirichlet":
            valor = _seleccionar_cara(_obtener(cfg, "valor", "T", defecto=0.0), forma, eje, lado)
            coef = np.where(activa, 2.0 * D[s] / (paso * paso), 0.0)
            diagonal[s] -= coef
            termino[s] += coef * valor
        elif tipo == "neumann":
            gradiente = _obtener(cfg, "gradiente", defecto=None)
            if gradiente is not None:
                grad = _seleccionar_cara(gradiente, forma, eje, lado)
                termino[s] += np.where(activa, D[s] * grad / paso, 0.0)
            else:
                # ``flujo`` es positivo hacia fuera: entra con signo negativo
                # en el balance del volumen de control.
                flujo = _seleccionar_cara(_obtener(cfg, "flujo", "valor", defecto=0.0),
                                          forma, eje, lado)
                termino[s] -= np.where(activa, flujo / paso, 0.0)
        elif tipo in ("radiacion", "radiativa"):
            if capacidad is None:
                raise ValueError("la radiación sólo es válida en paso_energia")
            t_amb = _seleccionar_cara(
                _obtener(cfg, "T_ambiente", "T_mufla", "temperatura", "valor"),
                forma, eje, lado,
            )
            emisividad = _seleccionar_cara(
                _obtener(cfg, "emisividad", defecto=0.8), forma, eje, lado
            )
            t_ref = campo_actual[s]
            h_rad = emisividad * _SIGMA_SB * (t_ref + t_amb) * (t_ref * t_ref + t_amb * t_amb)
            k_cara = conductividad[s] if conductividad is not None else D[s] * capacidad[s]
            # Resistencia de media celda en serie con la radiación exterior.
            h_ef = np.divide(
                1.0,
                np.divide(1.0, h_rad, out=np.full_like(h_rad, np.inf), where=h_rad > 0.0)
                + np.divide(paso, 2.0 * k_cara, out=np.full_like(h_rad, np.inf), where=k_cara > 0.0),
                out=np.zeros_like(h_rad),
                where=(h_rad > 0.0) & (k_cara > 0.0),
            )
            beta = np.where(activa, h_ef / (capacidad[s] * paso), 0.0)
            diagonal[s] -= beta
            termino[s] += beta * t_amb
        else:
            raise ValueError(f"tipo de frontera desconocido: {tipo!r}")
    return diagonal.ravel(), termino.ravel()


def _resolver_implicito(
    actual: np.ndarray,
    rhs_explicito: np.ndarray,
    D: float | np.ndarray,
    malla: Any,
    dt: float,
    fraccion: np.ndarray | None,
    condiciones: Any,
    *,
    capacidad: np.ndarray | None = None,
    conductividad: np.ndarray | None = None,
    conservativo: bool = False,
) -> np.ndarray:
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt debe ser positivo y finito")
    L, dif, frac = _operador_difusion(
        D, malla, fraccion, capacidad if conservativo else None,
    )
    diag_bc, b_bc = _terminos_frontera(
        condiciones, actual, dif, malla, frac,
        capacidad=capacidad, conductividad=conductividad,
    )
    if np.any(diag_bc):
        L = L + sparse.diags(diag_bc, format="csr")
    n = actual.size
    A = sparse.eye(n, format="csr") - dt * L
    b = actual.ravel() + dt * (rhs_explicito.ravel() + b_bc)
    valores = actual.ravel().copy()
    activas = frac.ravel() > 0.0

    if not np.any(activas):
        return actual.copy()

    # RESTRICCIÓN A CELDAS ACTIVAS.
    #
    # Antes se resolvía sobre las n celdas del dominio, incluidas las sólidas.
    # Con la geometría real eso mete en la misma matriz propiedades que difieren
    # en 14 órdenes de magnitud (las celdas de pared llevan eps=1e-6 y K=1e-20
    # frente a los valores del fluido), el número de condición se dispara y
    # BiCGSTAB sufre breakdown: `info=-10`.
    #
    # La solución es la misma que ya se aplicó con éxito a la proyección de
    # presión: sacar del sistema las celdas que no participan, en lugar de
    # asignarles propiedades degeneradas. Allí la divergencia pasó de 1,2e-3 a
    # 1,7e-17 y además el paso resultó un 48 % más rápido, porque el sistema
    # reducido tiene menos incógnitas.
    if activas.all():
        A_r, b_r = A, b
    else:
        idx = np.flatnonzero(activas)
        A_r = A.tocsr()[idx][:, idx].tocsr()
        b_r = b[idx]

    m = A_r.shape[0]
    diag = A_r.diagonal()
    inv_diag = np.divide(1.0, diag, out=np.ones_like(diag), where=np.abs(diag) > 0.0)
    precondicionador = spla.LinearOperator((m, m), matvec=lambda x: inv_diag * x)
    kwargs = dict(M=precondicionador, maxiter=max(200, min(5000, m)), atol=0.0)
    try:
        solucion, info = spla.bicgstab(A_r, b_r, rtol=1.0e-11, **kwargs)
    except TypeError:  # SciPy < 1.12
        kwargs.pop("atol", None)
        solucion, info = spla.bicgstab(A_r, b_r, tol=1.0e-11, **kwargs)

    if info != 0:
        # Respaldo directo: si el sistema sigue siendo difícil para el método
        # iterativo, se resuelve exactamente. Cuesta más, pero es preferible a
        # abortar la simulación; y si tampoco así se puede, el problema es la
        # matriz y hay que verlo, no silenciarlo.
        try:
            solucion = spla.spsolve(A_r.tocsc(), b_r)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"el sistema implícito no se pudo resolver: BiCGSTAB info={info}, "
                f"y el respaldo directo falló ({exc}). Revisar el contraste de "
                f"propiedades entre celdas activas e inactivas."
            ) from exc
        if not np.all(np.isfinite(solucion)):
            raise RuntimeError("el respaldo directo devolvió valores no finitos")

    if activas.all():
        valores = solucion
    else:
        valores[activas] = solucion
    return valores.reshape(actual.shape)


def _velocidades(campos: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return campos.u, campos.v, campos.w
    except AttributeError:
        if isinstance(campos, Mapping):
            return campos["u"], campos["v"], campos["w"]
        raise ValueError("campos debe contener u, v y w") from None


def _fuente_energia(fuentes: Any, forma: tuple[int, ...], capacidad: np.ndarray) -> np.ndarray:
    if fuentes is None:
        return np.zeros(forma)
    if not isinstance(fuentes, Mapping):
        return _como_campo(fuentes, forma, "fuente de energía") / capacidad
    tasa_t = _obtener(fuentes, "fuente_T", "tasa_temperatura", defecto=None)
    if tasa_t is not None:
        return _como_campo(tasa_t, forma, "fuente_T")
    calor = _obtener(fuentes, "Q", "Q_reaccion", "energia", "calor", "fuente", defecto=0.0)
    return _como_campo(calor, forma, "fuente de energía") / capacidad


def paso_energia(campos: Any, props: Any, malla: Any, dt: float, fuentes: Any = None) -> np.ndarray:
    """Avanza la temperatura: advección explícita y difusión implícita.

    ``props`` puede ser un objeto o diccionario. Debe proporcionar ``alpha`` o
    bien ``k``, ``rho`` y ``cp``. Una fuente ``Q`` se interpreta en W/m³;
    ``fuente_T`` puede darse directamente en K/s.
    """
    T = np.asarray(_obtener(campos, "T"), dtype=float)
    forma = tuple(malla.forma)
    if T.shape != forma:
        raise ValueError(f"T tiene forma {T.shape}; se esperaba {forma}")
    rho = _como_campo(_obtener(props, "rho", "densidad", defecto=1.0), forma, "rho")
    cp = _como_campo(_obtener(props, "cp", "Cp", "calor_especifico", defecto=1.0), forma, "cp")
    capacidad = _como_campo(
        _obtener(props, "rho_cp", "capacidad_volumetrica", defecto=rho * cp),
        forma, "rho_cp",
    )
    if np.any(capacidad <= 0.0):
        raise ValueError("rho*cp debe ser positivo")
    k_val = _obtener(props, "k", "conductividad", "conductividad_termica", defecto=None)
    alpha_val = _obtener(props, "alpha", "difusividad_termica", defecto=None)
    if alpha_val is None and k_val is None:
        raise ValueError("props debe proporcionar alpha o conductividad k")
    if k_val is None:
        alpha = _como_campo(alpha_val, forma, "alpha")
        k = alpha * capacidad
    else:
        k = _como_campo(k_val, forma, "k")
        alpha = _como_campo(alpha_val, forma, "alpha") if alpha_val is not None else k / capacidad

    u, v, w = _velocidades(campos)
    esquema = _obtener(props, "esquema", "esquema_transporte", defecto="tvd_superbee")
    # FORMA ADVECTIVA, no en divergencia.
    #
    # div(uT) = u.grad(T) + T div(u). El segundo término no transporta nada:
    # es la dilatación del gas. Con fuente de masa div(u) NO es cero —la
    # devolatilización crea unos 19 cm3/s de gas dentro del lecho— y ese
    # término restaba cientos de K/s a las celdas del lecho (medido: -778 K/s
    # de media, hasta -2085). El efecto era que el lecho y el fondo del crisol
    # se quedaban clavados en ~470 K mientras la pared lateral estaba a 1045 K,
    # peleando la radiación contra un sumidero espurio.
    #
    # Físicamente el gas que aparece sale del sólido A LA TEMPERATURA LOCAL:
    # entra con la entalpía que se lleva, así que no puede enfriar la celda que
    # lo genera. Restituir T*div(u) deja exactamente u.grad(T), que es el
    # transporte real. Con campo solenoidal (todas las pruebas MMS) ambos
    # términos coinciden, porque div(u)=0.
    #
    # SEGUNDO FACTOR: sólo el gas advecta.
    #
    # El término va dividido por la capacidad de la celda, que en el lecho es la
    # del bulto (6,9e5 J/m3K). Pero quien se mueve es el gas (1,4e3 J/m3K): usar
    # la efectiva equivale a que el gas arrastre la entalpía del sólido, 500
    # veces mayor de la que puede llevar. Con `rho_cp_fluido` el coeficiente es
    # (rho.cp)_gas/(rho.cp)_ef, que vale 1 en el gas libre —donde nada cambia— y
    # ~2e-3 en el lecho. Es la formulación estándar de equilibrio térmico local
    # en medio poroso. Sin `rho_cp_fluido` se conserva el comportamiento previo.
    fluido = _obtener(props, "rho_cp_fluido", "capacidad_fluido", defecto=None)
    peso = 1.0 if fluido is None else np.divide(
        _como_campo(fluido, forma, "rho_cp_fluido"), capacidad,
        out=np.ones(forma, dtype=float), where=capacidad > 0.0,
    )
    explicito = peso * (
        -divergencia_flujo_advectivo(T, u, v, w, malla, esquema)
        + T * divergencia_velocidad(u, v, w, malla, T)
    )
    explicito += _fuente_energia(fuentes, forma, capacidad)
    frac = _obtener(props, "fraccion", "fraccion_volumetrica", defecto=None)
    condiciones = _obtener(fuentes, "condiciones_frontera", defecto=None)
    if condiciones is None:
        condiciones = _obtener(props, "condiciones_frontera", "fronteras", defecto=None)
    condiciones = _condiciones_del_campo(condiciones, "T")
    # Difusión conservativa en energía: coeficiente de cara con la CONDUCTIVIDAD
    # y división por la capacidad de cada celda. Con propiedades uniformes es
    # idéntica a la formulación con difusividad (k/rho.cp = alpha), así que las
    # pruebas analíticas y de MMS no cambian; sólo cambia en las interfaces
    # entre materiales, que es donde la otra creaba energía.
    return _resolver_implicito(
        T, explicito, k, malla, dt, frac, condiciones,
        capacidad=capacidad, conductividad=k, conservativo=True,
    )


def _difusividad_especie(props: Any, especie: str, forma: tuple[int, ...]) -> np.ndarray:
    todas = _obtener(props, "D_especies", "difusividades", "difusividad_especies", "D", defecto=None)
    if isinstance(todas, Mapping):
        if especie not in todas:
            raise KeyError(f"falta la difusividad de la especie {especie!r}")
        valor = todas[especie]
    elif todas is not None:
        valor = todas
    else:
        valor = _obtener(props, f"D_{especie}", defecto=None)
        if valor is None:
            raise KeyError(f"falta la difusividad de la especie {especie!r}")
    return _como_campo(valor, forma, f"D[{especie}]")


def _fuente_especie(fuentes: Any, especie: str, forma: tuple[int, ...]) -> np.ndarray:
    if fuentes is None or not isinstance(fuentes, Mapping):
        return np.zeros(forma)
    sub = _obtener(fuentes, "especies", "R_especie", "tasas", defecto=fuentes)
    valor = sub.get(especie, 0.0) if isinstance(sub, Mapping) else 0.0
    return _como_campo(valor, forma, f"fuente[{especie}]")


def paso_especies(campos: Any, props: Any, malla: Any, dt: float, fuentes: Any = None) -> dict[str, np.ndarray]:
    """Avanza todas las concentraciones con advección explícita y difusión implícita."""
    concentraciones = _obtener(campos, "c")
    if not isinstance(concentraciones, Mapping):
        raise ValueError("campos.c debe ser un diccionario de especies")
    forma = tuple(malla.forma)
    u, v, w = _velocidades(campos)
    esquema = _obtener(props, "esquema", "esquema_transporte", defecto="tvd_superbee")
    frac = _obtener(props, "fraccion", "fraccion_volumetrica", defecto=None)
    condiciones = _obtener(fuentes, "condiciones_frontera", defecto=None)
    if condiciones is None:
        condiciones = _obtener(props, "condiciones_frontera", "fronteras", defecto=None)

    nuevas: dict[str, np.ndarray] = {}
    for especie, concentracion in concentraciones.items():
        c = np.asarray(concentracion, dtype=float)
        if c.shape != forma:
            raise ValueError(f"c[{especie!r}] tiene forma {c.shape}; se esperaba {forma}")
        D = _difusividad_especie(props, especie, forma)
        explicito = -divergencia_flujo_advectivo(c, u, v, w, malla, esquema)
        explicito += _fuente_especie(fuentes, especie, forma)
        cc = _condiciones_del_campo(condiciones, especie)
        nuevas[especie] = _resolver_implicito(c, explicito, D, malla, dt, frac, cc)
    return nuevas


def _maximo_difusividad(D: Any) -> float:
    if D is None:
        return 0.0
    if isinstance(D, Mapping):
        return max((_maximo_difusividad(v) for v in D.values()), default=0.0)
    arr = np.asarray(D, dtype=float)
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError("D debe ser finita y no negativa")
    return float(np.max(arr, initial=0.0))


def dt_estable_transporte(
    u: Any,
    v: Any = None,
    w: Any = None,
    D: Any = None,
    malla: Any = None,
    cfl: float = 0.5,
    numero_fourier: float = 0.5,
) -> float:
    """Paso máximo por CFL advectivo y criterio de Fourier anisótropo.

    La llamada usual es ``(u, v, w, D, malla)``. También se admite
    ``(campos, props, malla)``; en ese caso se busca la mayor difusividad en
    ``props``. El límite difusivo se reporta aunque los pasos de este módulo
    traten la difusión implícitamente, pues es útil para exactitud y para otros
    integradores explícitos del acoplamiento.
    """
    # Compatibilidad con dt_estable_transporte(campos, props, malla).
    if malla is None and w is not None and hasattr(w, "forma") and _obtener(u, "u", defecto=None) is not None:
        campos, props, malla = u, v, w
        u, v, w = _velocidades(campos)
        D = _obtener(props, "D_especies", "difusividades", "D", "alpha", defecto=0.0)
    if malla is None:
        raise ValueError("se requiere malla")
    if not (0.0 < cfl <= 1.0) or not (0.0 < numero_fourier <= 0.5):
        raise ValueError("cfl debe estar en (0,1] y numero_fourier en (0,0.5]")
    dx, dy, dz = _espaciados(malla)
    uu, vv, ww = (np.asarray(a, dtype=float) for a in (u, v, w))
    tasa_adv = (float(np.max(np.abs(uu), initial=0.0)) / dx
                + float(np.max(np.abs(vv), initial=0.0)) / dy
                + float(np.max(np.abs(ww), initial=0.0)) / dz)
    dt_adv = np.inf if tasa_adv == 0.0 else cfl / tasa_adv
    d_max = _maximo_difusividad(D)
    tasa_dif = d_max * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2)
    dt_dif = np.inf if tasa_dif == 0.0 else numero_fourier / tasa_dif
    return float(min(dt_adv, dt_dif))


__all__ = [
    "divergencia_flujo_advectivo",
    "divergencia_flujo_difusivo",
    "paso_energia",
    "paso_especies",
    "dt_estable_transporte",
]
