"""Soluciones manufacturadas para verificar el sistema acoplado.

El convenio de las fuentes es

``phi_t + div(u phi) = div(D grad(phi)) + s_phi``

y, para momentum, el de :mod:`nucleo.momentum` documentado en
``docs/CONTRATOS.md``:

``rho/eps u_t + rho/eps**2 (u.grad)u = -grad(P) + mu_ef lap(u)
    - mu/K u - rho*C_F/sqrt(K)*|u|u + rho*g*beta*(T-T_ref) + f``.

Las expresiones se construyen y comprueban con SymPy cuando está disponible.
Se conservan fórmulas analíticas manuales equivalentes como alternativa: la
evaluación numérica no depende de SymPy una vez importado el módulo.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import inspect
import math
from typing import Any

import numpy as np


DOS_PI = 2.0 * np.pi
G = 9.80665


# ---------------------------------------------------------------------------
# Construcción simbólica y comprobaciones algebraicas
# ---------------------------------------------------------------------------
try:  # SymPy es opcional en producción.
    import sympy as sp
except ImportError:  # pragma: no cover - la instalación de CI incluye SymPy
    sp = None


VERIFICACION_SIMBOLICA: dict[str, bool | str] = {
    "sympy_disponible": sp is not None,
    "divergencia_velocidad": False,
    "fuente_escalar": False,
}

if sp is not None:
    _xs, _ys, _zs, _ts = sp.symbols("x y z t", real=True)
    _k, _a_phi, _lam_phi = sp.symbols("k a_phi lam_phi", positive=True)
    _a_u, _lam_u = sp.symbols("a_u lam_u", positive=True)

    _q = sp.cos(_k * _xs) * sp.cos(_k * _ys) * sp.cos(_k * _zs)
    _phi_s = 1 + _a_phi * sp.exp(-_lam_phi * _ts) * _q
    _derivadas_phi_s = {
        "valor": _phi_s,
        "dt": sp.diff(_phi_s, _ts),
        "dx": sp.diff(_phi_s, _xs),
        "dy": sp.diff(_phi_s, _ys),
        "dz": sp.diff(_phi_s, _zs),
        "dxx": sp.diff(_phi_s, _xs, 2),
        "dyy": sp.diff(_phi_s, _ys, 2),
        "dzz": sp.diff(_phi_s, _zs, 2),
    }
    _derivadas_phi_s["laplaciano"] = sum(
        _derivadas_phi_s[nombre] for nombre in ("dxx", "dyy", "dzz")
    )
    _args_phi = (_xs, _ys, _zs, _ts, _a_phi, _lam_phi, _k)
    _funciones_phi = {
        nombre: sp.lambdify(_args_phi, expresion, modules="numpy")
        for nombre, expresion in _derivadas_phi_s.items()
    }

    # U = rot(A), con
    # A=(0, sin(kx)cos(ky)/k,
    #       -sin(kx)cos(kz)/k + sin(ky)cos(kz)/k) a_u exp(-lam_u t).
    _factor_u = _a_u * sp.exp(-_lam_u * _ts)
    _potencial = sp.Matrix((
        0,
        _factor_u * sp.sin(_k * _xs) * sp.cos(_k * _ys) / _k,
        _factor_u * (
            -sp.sin(_k * _xs) * sp.cos(_k * _zs)
            + sp.sin(_k * _ys) * sp.cos(_k * _zs)
        ) / _k,
    ))
    _vel_s = sp.Matrix((
        sp.diff(_potencial[2], _ys) - sp.diff(_potencial[1], _zs),
        sp.diff(_potencial[0], _zs) - sp.diff(_potencial[2], _xs),
        sp.diff(_potencial[1], _xs) - sp.diff(_potencial[0], _ys),
    ))
    _coords = (_xs, _ys, _zs)
    _jac_s = _vel_s.jacobian(_coords)
    _dt_vel_s = _vel_s.diff(_ts)
    _lap_vel_s = sp.Matrix([
        sum(sp.diff(_vel_s[i], coord, 2) for coord in _coords)
        for i in range(3)
    ])
    _adv_vel_s = _jac_s * _vel_s
    _div_vel_s = sum(_jac_s[i, i] for i in range(3))
    VERIFICACION_SIMBOLICA["divergencia_velocidad"] = bool(
        sp.simplify(_div_vel_s) == 0
    )

    _D_s = sp.symbols("D", nonnegative=True)
    _grad_phi_s = sp.Matrix([
        _derivadas_phi_s["dx"], _derivadas_phi_s["dy"],
        _derivadas_phi_s["dz"],
    ])
    _fuente_phi_s = (
        _derivadas_phi_s["dt"]
        + _vel_s.dot(_grad_phi_s)
        - _D_s * _derivadas_phi_s["laplaciano"]
    )
    _residuo_phi_s = (
        _derivadas_phi_s["dt"]
        + sum(sp.diff(_phi_s * _vel_s[i], _coords[i]) for i in range(3))
        - _D_s * _derivadas_phi_s["laplaciano"]
        - _fuente_phi_s
    )
    VERIFICACION_SIMBOLICA["fuente_escalar"] = bool(
        sp.simplify(_residuo_phi_s) == 0
    )

    _args_vel = (_xs, _ys, _zs, _ts, _a_u, _lam_u, _k)
    _func_vel = sp.lambdify(_args_vel, _vel_s, modules="numpy")
    _func_dt_vel = sp.lambdify(_args_vel, _dt_vel_s, modules="numpy")
    _func_lap_vel = sp.lambdify(_args_vel, _lap_vel_s, modules="numpy")
    _func_adv_vel = sp.lambdify(_args_vel, _adv_vel_s, modules="numpy")
    _func_jac_vel = sp.lambdify(_args_vel, _jac_s, modules="numpy")
else:
    VERIFICACION_SIMBOLICA["divergencia_velocidad"] = (
        "comprobación manual: cada componente es independiente de su coordenada"
    )
    VERIFICACION_SIMBOLICA["fuente_escalar"] = (
        "comprobación manual por sustitución término a término"
    )


def _array(valor: Any) -> np.ndarray:
    """Convierte escalares o arreglos preservando la difusión de NumPy."""
    return np.asarray(valor, dtype=float)


def solucion_manufacturada_escalar(
    x: Any,
    y: Any,
    z: Any,
    t: float,
    *,
    amplitud: float = 0.2,
    tasa_temporal: float = 0.7,
    numero_onda: float = DOS_PI,
) -> dict[str, np.ndarray]:
    """Evalúa ``phi`` y sus derivadas analíticas.

    ``phi = 1 + A exp(-lambda t) cos(kx) cos(ky) cos(kz)``. La derivada
    normal es nula en las seis caras del cubo unidad para ``k=2*pi``, de modo
    que coincide con la frontera difusiva por defecto de transporte.

    El resultado contiene ``valor``, ``dt``, ``dx``, ``dy``, ``dz``,
    ``dxx``, ``dyy``, ``dzz`` y ``laplaciano``. ``phi`` es un alias de
    ``valor``. Si SymPy está instalado, estas funciones son las derivadas
    simbólicas lambdificadas; la rama manual implementa las mismas fórmulas.
    """
    xx, yy, zz = np.broadcast_arrays(_array(x), _array(y), _array(z))
    parametros = (xx, yy, zz, float(t), float(amplitud),
                  float(tasa_temporal), float(numero_onda))
    if sp is not None:
        resultado = {
            nombre: _array(funcion(*parametros))
            for nombre, funcion in _funciones_phi.items()
        }
    else:  # Fórmulas obtenidas al derivar la expresión escrita arriba.
        k = float(numero_onda)
        factor = float(amplitud) * np.exp(-float(tasa_temporal) * float(t))
        cx, cy, cz = np.cos(k * xx), np.cos(k * yy), np.cos(k * zz)
        sx, sy, sz = np.sin(k * xx), np.sin(k * yy), np.sin(k * zz)
        modo = factor * cx * cy * cz
        resultado = {
            "valor": 1.0 + modo,
            "dt": -float(tasa_temporal) * modo,
            "dx": -factor * k * sx * cy * cz,
            "dy": -factor * k * cx * sy * cz,
            "dz": -factor * k * cx * cy * sz,
            "dxx": -k * k * modo,
            "dyy": -k * k * modo,
            "dzz": -k * k * modo,
            "laplaciano": -3.0 * k * k * modo,
        }
    resultado["phi"] = resultado["valor"]
    return resultado


def solucion_manufacturada_velocidad(
    x: Any,
    y: Any,
    z: Any,
    t: float,
    *,
    amplitud: float = 0.1,
    tasa_temporal: float = 0.4,
    numero_onda: float = DOS_PI,
) -> dict[str, Any]:
    """Campo tridimensional solenoidal, construido como un rotacional.

    Su forma simplificada es ``a exp(-lambda*t)`` por
    ``(cos(ky)cos(kz), cos(kx)cos(kz), cos(kx)cos(ky))``. Cada componente es
    independiente de su propia coordenada, por lo que tanto la divergencia
    continua como la divergencia MAC son exactamente cero (salvo redondeo).
    Se devuelven también derivada temporal, jacobiano, laplaciano y término
    advectivo analíticos, necesarios para fabricar la fuente de momentum.
    """
    xx, yy, zz = np.broadcast_arrays(_array(x), _array(y), _array(z))
    k = float(numero_onda)
    lam = float(tasa_temporal)
    factor = float(amplitud) * np.exp(-lam * float(t))
    cx, cy, cz = np.cos(k * xx), np.cos(k * yy), np.cos(k * zz)
    sx, sy, sz = np.sin(k * xx), np.sin(k * yy), np.sin(k * zz)

    # Las expresiones explícitas evitan las dimensiones singleton que añade
    # lambdify al evaluar una Matrix sobre arreglos multidimensionales.
    u = factor * cy * cz
    v = factor * cx * cz
    w = factor * cx * cy
    ceros = np.zeros_like(u)
    jacobiano = (
        (ceros, -factor * k * sy * cz, -factor * k * cy * sz),
        (-factor * k * sx * cz, ceros, -factor * k * cx * sz),
        (-factor * k * sx * cy, -factor * k * cx * sy, ceros),
    )
    adveccion = (
        v * jacobiano[0][1] + w * jacobiano[0][2],
        u * jacobiano[1][0] + w * jacobiano[1][2],
        u * jacobiano[2][0] + v * jacobiano[2][1],
    )
    laplaciano = tuple(-2.0 * k * k * componente for componente in (u, v, w))
    derivada_t = tuple(-lam * componente for componente in (u, v, w))
    divergencia = jacobiano[0][0] + jacobiano[1][1] + jacobiano[2][2]
    return {
        "u": u,
        "v": v,
        "w": w,
        "velocidad": (u, v, w),
        "dt": derivada_t,
        "du_dt": derivada_t[0],
        "dv_dt": derivada_t[1],
        "dw_dt": derivada_t[2],
        "jacobiano": jacobiano,
        "laplaciano": laplaciano,
        "adveccion": adveccion,
        "divergencia": divergencia,
        "rapidez": np.sqrt(u * u + v * v + w * w),
    }


def fuente_mms_adveccion_difusion(
    x: Any,
    y: Any,
    z: Any,
    t: float,
    u: Any | None = None,
    v: Any | None = None,
    w: Any | None = None,
    D: float | np.ndarray = 0.01,
    *,
    difusividad: float | np.ndarray | None = None,
    divergencia_velocidad: float | np.ndarray = 0.0,
    amplitud: float = 0.2,
    tasa_temporal: float = 0.7,
    numero_onda: float = DOS_PI,
) -> np.ndarray:
    """Fuente que hace exacta la solución escalar manufacturada.

    Si no se da velocidad se usa :func:`solucion_manufacturada_velocidad`.
    ``D`` debe ser constante espacialmente; para una velocidad no solenoidal se
    puede proporcionar ``divergencia_velocidad`` y se incluye ``phi div(u)``.
    La identidad completa fue simplificada simbólicamente al importar el
    módulo y su resultado queda en ``VERIFICACION_SIMBOLICA['fuente_escalar']``.
    """
    if difusividad is not None:
        D = difusividad
    escalar = solucion_manufacturada_escalar(
        x, y, z, t, amplitud=amplitud,
        tasa_temporal=tasa_temporal, numero_onda=numero_onda,
    )
    if u is None and v is None and w is None:
        velocidad = solucion_manufacturada_velocidad(
            x, y, z, t, numero_onda=numero_onda
        )
        u, v, w = velocidad["velocidad"]
    else:
        u = 0.0 if u is None else u
        v = 0.0 if v is None else v
        w = 0.0 if w is None else w
    return (
        escalar["dt"]
        + _array(u) * escalar["dx"]
        + _array(v) * escalar["dy"]
        + _array(w) * escalar["dz"]
        + escalar["valor"] * _array(divergencia_velocidad)
        - _array(D) * escalar["laplaciano"]
    )


def _presion_manufacturada(
    x: Any,
    y: Any,
    z: Any,
    t: float,
    amplitud: float = 0.05,
    tasa_temporal: float = 0.3,
    numero_onda: float = DOS_PI,
) -> dict[str, Any]:
    xx, yy, zz = np.broadcast_arrays(_array(x), _array(y), _array(z))
    k = float(numero_onda)
    factor = float(amplitud) * np.exp(-float(tasa_temporal) * float(t))
    cx, cy, cz = np.cos(k * xx), np.cos(k * yy), np.cos(k * zz)
    sx, sy, sz = np.sin(k * xx), np.sin(k * yy), np.sin(k * zz)
    valor = factor * cx * cy * cz
    return {
        "valor": valor,
        "gradiente": (
            -factor * k * sx * cy * cz,
            -factor * k * cx * sy * cz,
            -factor * k * cx * cy * sz,
        ),
        "laplaciano": -3.0 * k * k * valor,
    }


def fuente_mms_momentum(
    x: Any,
    y: Any,
    z: Any,
    t: float,
    rho: float | np.ndarray = 1.2,
    mu: float = 0.02,
    eps: float | np.ndarray = 0.8,
    K: float | np.ndarray = 0.5,
    C_F: float = 0.3,
    *,
    mu_ef: float | None = None,
    beta: float = 0.01,
    T: Any | None = None,
    T_ref: float = 1.0,
    amplitud_velocidad: float = 0.1,
    tasa_temporal_velocidad: float = 0.4,
    amplitud_presion: float = 0.05,
    tasa_temporal_presion: float = 0.3,
    numero_onda: float = DOS_PI,
    devolver_terminos: bool = False,
) -> np.ndarray | dict[str, Any]:
    """Fuerza MMS de momentum, en N/m³, con todos los términos.

    La gravedad es ``(0, 0, -G)`` porque ``z`` apunta hacia arriba. Si ``T`` no
    se proporciona se usa un campo cosenoidal suave alrededor de ``T_ref`` para
    que el término de Boussinesq sea no nulo. ``K=np.inf`` desactiva de forma
    limpia Darcy y Forchheimer.

    Con ``devolver_terminos=True`` se devuelve un diccionario que permite
    auditar transitorio, advección, presión, viscosidad, Darcy, Forchheimer y
    boyancia por separado; en otro caso se devuelve un array cuya primera
    dimensión enumera ``(x, y, z)``.
    """
    vel = solucion_manufacturada_velocidad(
        x, y, z, t, amplitud=amplitud_velocidad,
        tasa_temporal=tasa_temporal_velocidad, numero_onda=numero_onda,
    )
    presion = _presion_manufacturada(
        x, y, z, t, amplitud_presion, tasa_temporal_presion, numero_onda
    )
    rho_a = _array(rho)
    eps_a = _array(eps)
    K_a = _array(K)
    if np.any(rho_a <= 0.0) or np.any(eps_a <= 0.0):
        raise ValueError("rho y eps deben ser positivos")
    if np.any(K_a <= 0.0):
        raise ValueError("K debe ser positiva o infinita")
    mu_efectiva = float(mu if mu_ef is None else mu_ef)
    inv_K = np.where(np.isfinite(K_a), 1.0 / K_a, 0.0)
    componentes = vel["velocidad"]
    rapidez = vel["rapidez"]

    transitorio = tuple(rho_a / eps_a * q for q in vel["dt"])
    adveccion = tuple(rho_a / (eps_a * eps_a) * q for q in vel["adveccion"])
    gradiente_presion = tuple(_array(q) for q in presion["gradiente"])
    viscoso = tuple(-mu_efectiva * q for q in vel["laplaciano"])
    darcy = tuple(float(mu) * inv_K * q for q in componentes)
    forchheimer = tuple(
        rho_a * float(C_F) * np.sqrt(inv_K) * rapidez * q
        for q in componentes
    )

    if T is None:
        esc = solucion_manufacturada_escalar(
            x, y, z, t, amplitud=1.0, tasa_temporal=0.2,
            numero_onda=numero_onda,
        )
        T = float(T_ref) + (esc["valor"] - 1.0)
    delta_T = _array(T) - float(T_ref)
    # En f se resta la boyancia del lado derecho. g_z=-G, luego el término
    # que se suma a f_z es +rho*G*beta*(T-T_ref).
    menos_boyancia = (
        np.zeros_like(delta_T), np.zeros_like(delta_T),
        rho_a * G * float(beta) * delta_T,
    )
    fuente = tuple(
        transitorio[i] + adveccion[i] + gradiente_presion[i]
        + viscoso[i] + darcy[i] + forchheimer[i] + menos_boyancia[i]
        for i in range(3)
    )
    fuente_array = np.stack(np.broadcast_arrays(*fuente), axis=0)
    if not devolver_terminos:
        return fuente_array
    return {
        "fuente": fuente_array,
        "transitorio": np.stack(np.broadcast_arrays(*transitorio), axis=0),
        "adveccion": np.stack(np.broadcast_arrays(*adveccion), axis=0),
        "gradiente_presion": np.stack(
            np.broadcast_arrays(*gradiente_presion), axis=0
        ),
        "menos_viscoso": np.stack(np.broadcast_arrays(*viscoso), axis=0),
        "darcy": np.stack(np.broadcast_arrays(*darcy), axis=0),
        "forchheimer": np.stack(np.broadcast_arrays(*forchheimer), axis=0),
        "menos_boyancia": np.stack(
            np.broadcast_arrays(*menos_boyancia), axis=0
        ),
        "presion": presion["valor"],
        "velocidad": componentes,
    }


# ---------------------------------------------------------------------------
# Utilidades genéricas de orden
# ---------------------------------------------------------------------------
def _tamano_malla(malla: Any) -> float:
    if np.isscalar(malla):
        valor = float(malla)
        if valor <= 0.0:
            raise ValueError("la malla debe ser positiva")
        return 1.0 / valor
    for nombre in ("h", "dx"):
        if hasattr(malla, nombre):
            return float(getattr(malla, nombre))
    if hasattr(malla, "dx_mm"):
        dx = float(malla.dx_mm) * 1.0e-3
        dy = float(getattr(malla, "dy_mm", malla.dx_mm)) * 1.0e-3
        dz = float(malla.dz_mm) * 1.0e-3
        return max(dx, dy, dz)
    if isinstance(malla, (tuple, list)) and malla:
        return 1.0 / float(min(malla))
    raise ValueError("no se pudo deducir h de la malla")


def _invocar_solucionador(
    solucionador: Callable[..., Any], malla: Any, fuente: Any
) -> Any:
    """Invoca protocolos comunes sin ocultar excepciones internas TypeError."""
    firma = inspect.signature(solucionador)
    intentos = ((malla, fuente), (malla,), ())
    for argumentos in intentos:
        try:
            firma.bind(*argumentos)
        except TypeError:
            continue
        return solucionador(*argumentos)
    try:
        firma.bind(malla=malla, fuente=fuente)
    except TypeError as exc:
        raise TypeError(
            "solucionador debe aceptar (malla, fuente), (malla) o palabras homónimas"
        ) from exc
    return solucionador(malla=malla, fuente=fuente)


def _extraer_resultado(
    resultado: Any, malla: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, Any]:
    mascara = None
    if isinstance(resultado, Mapping) and "numerica" in resultado and "exacta" in resultado:
        numerica = resultado["numerica"]
        exacta = resultado["exacta"]
        h = float(resultado.get("h", _tamano_malla(malla)))
        mascara = resultado.get("mascara")
    elif isinstance(resultado, tuple) and len(resultado) in (2, 3):
        numerica, exacta = resultado[:2]
        h = float(resultado[2]) if len(resultado) == 3 else _tamano_malla(malla)
    else:
        raise TypeError(
            "el solucionador debe devolver (numerica, exacta[, h]) o un mapeo "
            "con las claves numerica y exacta"
        )
    if not isinstance(numerica, Mapping):
        numerica = {"campo": numerica}
    if not isinstance(exacta, Mapping):
        exacta = {"campo": exacta}
    if set(numerica) != set(exacta):
        raise ValueError("numerica y exacta deben contener las mismas componentes")
    return (
        {str(k): _array(v) for k, v in numerica.items()},
        {str(k): _array(v) for k, v in exacta.items()},
        h,
        mascara,
    )


def _nombres_norma(norma: str | Iterable[str] | None) -> tuple[str, ...]:
    if norma is None or (isinstance(norma, str) and norma.lower() in ("todas", "all")):
        return ("L1", "L2", "Linf")
    nombres = (norma,) if isinstance(norma, str) else tuple(norma)
    normalizados = []
    for nombre in nombres:
        clave = str(nombre).lower().replace("∞", "inf")
        mapa = {"l1": "L1", "1": "L1", "l2": "L2", "2": "L2",
                "linf": "Linf", "inf": "Linf", "infinito": "Linf"}
        if clave not in mapa:
            raise ValueError(f"norma desconocida: {nombre!r}")
        normalizados.append(mapa[clave])
    return tuple(dict.fromkeys(normalizados))


def _norma_error(error: np.ndarray, nombre: str, mascara: Any = None) -> float:
    valores = np.abs(np.asarray(error, dtype=float))
    if mascara is not None:
        valores = valores[np.asarray(mascara, dtype=bool)]
    if valores.size == 0:
        raise ValueError("la máscara de error no contiene puntos")
    if nombre == "L1":
        return float(np.mean(valores))
    if nombre == "L2":
        return float(np.sqrt(np.mean(valores * valores)))
    return float(np.max(valores))


def verificar_orden(
    solucionador: Callable[..., Any],
    fuente: Any,
    mallas: Iterable[Any],
    norma: str | Iterable[str] | None = None,
) -> dict[str, Any]:
    """Ejecuta un solucionador y ajusta ``error = C h**p``.

    Cada ejecución debe devolver ``(numerica, exacta[, h])`` o un diccionario
    ``{'numerica': ..., 'exacta': ..., 'h': ..., 'mascara': ...}``. Los campos
    numérico y exacto pueden ser arrays o mapeos por componente. Se calculan
    siempre las normas solicitadas (las tres por defecto), órdenes globales por
    mínimos cuadrados y órdenes entre refinamientos consecutivos.
    """
    normas = _nombres_norma(norma)
    registros: list[tuple[float, dict[str, np.ndarray], dict[str, np.ndarray], Any]] = []
    for malla in mallas:
        resultado = _invocar_solucionador(solucionador, malla, fuente)
        numerica, exacta, h, mascara = _extraer_resultado(resultado, malla)
        if h <= 0.0 or not np.isfinite(h):
            raise ValueError("h debe ser positivo y finito")
        registros.append((h, numerica, exacta, mascara))
    if len(registros) < 2:
        raise ValueError("se requieren al menos dos mallas")
    registros.sort(key=lambda item: item[0], reverse=True)  # gruesa -> fina
    componentes = tuple(registros[0][1])
    errores: dict[str, dict[str, list[float]]] = {
        c: {n: [] for n in normas} for c in componentes
    }
    for _, numerica, exacta, mascara in registros:
        if tuple(numerica) != componentes:
            raise ValueError("todas las mallas deben devolver las mismas componentes")
        for componente in componentes:
            if numerica[componente].shape != exacta[componente].shape:
                raise ValueError(f"formas incompatibles en {componente!r}")
            diferencia = numerica[componente] - exacta[componente]
            for nombre_norma in normas:
                errores[componente][nombre_norma].append(
                    _norma_error(diferencia, nombre_norma, mascara)
                )
    h = np.asarray([r[0] for r in registros], dtype=float)
    ordenes: dict[str, dict[str, float]] = {c: {} for c in componentes}
    ordenes_pares: dict[str, dict[str, list[float]]] = {c: {} for c in componentes}
    monotono: dict[str, dict[str, bool]] = {c: {} for c in componentes}
    for componente in componentes:
        for nombre_norma in normas:
            e = np.asarray(errores[componente][nombre_norma], dtype=float)
            validos = np.isfinite(e) & (e > 0.0)
            ordenes[componente][nombre_norma] = (
                float(np.polyfit(np.log(h[validos]), np.log(e[validos]), 1)[0])
                if np.count_nonzero(validos) >= 2 else math.nan
            )
            pares = []
            for i in range(len(h) - 1):
                if e[i] > 0.0 and e[i + 1] > 0.0 and h[i] != h[i + 1]:
                    pares.append(float(np.log(e[i] / e[i + 1])
                                       / np.log(h[i] / h[i + 1])))
                else:
                    pares.append(math.nan)
            ordenes_pares[componente][nombre_norma] = pares
            monotono[componente][nombre_norma] = bool(np.all(np.diff(e) < 0.0))
    salida: dict[str, Any] = {
        "h": h,
        "errores": {
            c: {n: np.asarray(v, dtype=float) for n, v in por_norma.items()}
            for c, por_norma in errores.items()
        },
        "ordenes": ordenes,
        "ordenes_pares": ordenes_pares,
        "monotono": monotono,
        "normas": normas,
        "componentes": componentes,
    }
    if len(componentes) == 1 and len(normas) == 1:
        salida["orden"] = ordenes[componentes[0]][normas[0]]
    return salida


__all__ = [
    "DOS_PI",
    "VERIFICACION_SIMBOLICA",
    "solucion_manufacturada_escalar",
    "fuente_mms_adveccion_difusion",
    "solucion_manufacturada_velocidad",
    "fuente_mms_momentum",
    "verificar_orden",
]
