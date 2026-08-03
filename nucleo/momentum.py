"""
Ecuación de momentum: Navier--Stokes con términos de medio poroso.

ECUACIÓN RESUELTA
-----------------
Una sola ecuación cubre el lecho granular y el espacio libre de gas, en la
formulación de Darcy--Brinkman--Forchheimer:

    rho/eps * du/dt  +  rho/eps^2 (u.grad)u
        = -grad P
          + mu_ef * lap(u)                      (Brinkman, viscoso)
          - (mu/K) u                            (Darcy, sólo en el lecho)
          - (rho C_F / sqrt(K)) |u| u           (Forchheimer, inercial en poro)
          + rho g beta (T - T_ref)              (boyancia, Boussinesq)

En el espacio libre se toma ``eps = 1`` y ``K -> infinito``, con lo que los dos
términos porosos desaparecen y **la ecuación se reduce a Navier--Stokes puro**.
No hay dos solucionadores: hay uno con coeficientes que varían en el espacio.

POR QUÉ SE INCLUYEN TÉRMINOS QUE EN ESTE CASO SON PEQUEÑOS
----------------------------------------------------------
Para el ensayo carbón--titanomagnetita los números adimensionales dicen que
Forchheimer y boyancia son despreciables (Re_p = 0,053; Ra = 188 frente a 1708
crítico). Aun así se implementan, por dos razones: el programa debe servir para
otros casos donde sí dominen, y es preferible que los números revelen qué manda a
suponerlo de antemano. ``numeros_adimensionales`` los reporta en cada paso, de
modo que la afirmación es verificable en tiempo de ejecución y no un supuesto.

MÉTODO
------
Proyección de presión de Chorin sobre malla escalonada (MAC):

  1. predictor: se avanza el momentum sin el gradiente de presión -> u*
  2. Poisson: lap(phi) = rho/dt * div(u*), con Neumann homogéneo en paredes
  3. corrección: u = u* - dt/rho * grad(phi);  P += phi

La malla escalonada evita el modo de presión en tablero de ajedrez que aparece si
todo se coloca en el centro de la celda.

CONVENIO DE ÍNDICES (contrato §2)
---------------------------------
  u -> caras x, forma (nx+1, ny, nz)
  v -> caras y, forma (nx, ny+1, nz)
  w -> caras z, forma (nx, ny, nz+1)
  escalares -> centros, forma (nx, ny, nz)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, diags, identity, kron
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, bicgstab, factorized, spilu, spsolve

G = 9.80665  # m/s2


# ---------------------------------------------------------------------------
# Propiedades del medio
# ---------------------------------------------------------------------------
@dataclass
class PropiedadesMedio:
    """Propiedades por celda. Los arrays tienen la forma de los centros."""

    rho: np.ndarray          # kg/m3, densidad del gas
    mu: float                # Pa.s, viscosidad dinámica
    eps: np.ndarray          # porosidad, 1 en el gas libre
    K: np.ndarray            # m2, permeabilidad; np.inf en el gas libre
    C_F: float = 0.55        # coeficiente de Forchheimer (Ergun); adimensional
    beta: float = 0.0        # 1/K, expansión térmica; 0 desactiva la boyancia
    T_ref: float = 1173.15   # K
    mu_ef: float | None = None  # viscosidad efectiva de Brinkman

    def __post_init__(self) -> None:
        if self.mu_ef is None:
            # Brinkman clásico: mu_ef = mu/eps. En el gas libre eps=1 -> mu_ef=mu.
            self.mu_ef = self.mu

    @staticmethod
    def permeabilidad_kozeny_carman(d_particula_m: float, eps: np.ndarray) -> np.ndarray:
        """K = d^2 eps^3 / (150 (1-eps)^2).

        Correlación de Kozeny--Carman, válida para lechos de partículas casi
        esféricas y Re bajo. Para el concentrado del ensayo (d = 175 um,
        eps = 0,54) da K = 1,52e-10 m2.
        """
        eps = np.asarray(eps, dtype=float)
        uno_menos = np.maximum(1.0 - eps, 1e-12)
        K = d_particula_m ** 2 * eps ** 3 / (150.0 * uno_menos ** 2)
        # el gas libre (eps -> 1) no ofrece resistencia
        return np.where(eps >= 0.999, np.inf, K)


# ---------------------------------------------------------------------------
# Operadores de malla escalonada
# ---------------------------------------------------------------------------
def _a_caras_x(phi: np.ndarray) -> np.ndarray:
    """Interpola un escalar de centros a caras x, con extrapolación de borde."""
    n = phi.shape[0]
    cara = np.empty((n + 1,) + phi.shape[1:], dtype=float)
    cara[1:-1] = 0.5 * (phi[:-1] + phi[1:])
    cara[0] = phi[0]
    cara[-1] = phi[-1]
    return cara


def _a_caras_y(phi: np.ndarray) -> np.ndarray:
    n = phi.shape[1]
    cara = np.empty((phi.shape[0], n + 1, phi.shape[2]), dtype=float)
    cara[:, 1:-1] = 0.5 * (phi[:, :-1] + phi[:, 1:])
    cara[:, 0] = phi[:, 0]
    cara[:, -1] = phi[:, -1]
    return cara


def _a_caras_z(phi: np.ndarray) -> np.ndarray:
    n = phi.shape[2]
    cara = np.empty(phi.shape[:2] + (n + 1,), dtype=float)
    cara[:, :, 1:-1] = 0.5 * (phi[:, :, :-1] + phi[:, :, 1:])
    cara[:, :, 0] = phi[:, :, 0]
    cara[:, :, -1] = phi[:, :, -1]
    return cara


def divergencia(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                dx: float, dy: float, dz: float) -> np.ndarray:
    """div(u) en los centros de celda. Exacta y conservativa en malla MAC."""
    return ((u[1:, :, :] - u[:-1, :, :]) / dx
            + (v[:, 1:, :] - v[:, :-1, :]) / dy
            + (w[:, :, 1:] - w[:, :, :-1]) / dz)


def gradiente_a_caras(p: np.ndarray, dx: float, dy: float, dz: float,
                      solido: np.ndarray | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """grad(p) en caras, con gradiente normal nulo en toda pared solida."""
    nx, ny, nz = p.shape
    gx = np.zeros((nx + 1, ny, nz))
    gy = np.zeros((nx, ny + 1, nz))
    gz = np.zeros((nx, ny, nz + 1))
    if solido is None:
        gx[1:-1] = (p[1:] - p[:-1]) / dx
        gy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy
        gz[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / dz
    else:
        fluido = ~np.asarray(solido, dtype=bool)
        gx[1:-1] = np.where(
            fluido[:-1] & fluido[1:], (p[1:] - p[:-1]) / dx, 0.0)
        gy[:, 1:-1] = np.where(
            fluido[:, :-1] & fluido[:, 1:],
            (p[:, 1:] - p[:, :-1]) / dy, 0.0)
        gz[:, :, 1:-1] = np.where(
            fluido[:, :, :-1] & fluido[:, :, 1:],
            (p[:, :, 1:] - p[:, :, :-1]) / dz, 0.0)
    return gx, gy, gz


def _laplaciano_componente(f: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Laplaciano de una componente de velocidad, con bordes reflejados.

    El reflejo equivale a gradiente normal nulo en el borde del array; las
    paredes reales se imponen aparte con la máscara de sólido.
    """
    lap = np.zeros_like(f)
    for eje, h in ((0, dx), (1, dy), (2, dz)):
        adelante = np.roll(f, -1, axis=eje)
        atras = np.roll(f, 1, axis=eje)
        # bordes: copia del vecino interior -> derivada segunda nula
        sl_ini = [slice(None)] * 3
        sl_fin = [slice(None)] * 3
        sl_ini[eje] = 0
        sl_fin[eje] = -1
        atras[tuple(sl_ini)] = f[tuple(sl_ini)]
        adelante[tuple(sl_fin)] = f[tuple(sl_fin)]
        lap += (adelante - 2.0 * f + atras) / (h * h)
    return lap


def _adveccion_upwind(f: np.ndarray, u_c: np.ndarray, v_c: np.ndarray, w_c: np.ndarray,
                      dx: float, dy: float, dz: float) -> np.ndarray:
    """(u.grad)f con diferencias upwind de primer orden.

    Se usa upwind por robustez: con Re de celda pequeño (aquí ~1e-3) la difusión
    numérica que introduce es despreciable frente a la viscosidad física, y a
    cambio garantiza que no aparezcan oscilaciones.
    """
    out = np.zeros_like(f)
    for eje, (vel, h) in enumerate(((u_c, dx), (v_c, dy), (w_c, dz))):
        adelante = np.roll(f, -1, axis=eje)
        atras = np.roll(f, 1, axis=eje)
        sl_ini = [slice(None)] * 3
        sl_fin = [slice(None)] * 3
        sl_ini[eje] = 0
        sl_fin[eje] = -1
        atras[tuple(sl_ini)] = f[tuple(sl_ini)]
        adelante[tuple(sl_fin)] = f[tuple(sl_fin)]
        hacia_atras = (f - atras) / h
        hacia_adelante = (adelante - f) / h
        out += np.where(vel >= 0.0, vel * hacia_atras, vel * hacia_adelante)
    return out


# ---------------------------------------------------------------------------
# Ecuación de Poisson para la presión
# ---------------------------------------------------------------------------
def _laplaciano_1d(n: int, h: float) -> Any:
    """Laplaciano 1-D con Neumann homogéneo en ambos extremos."""
    if n == 1:
        return diags([np.zeros(1)], [0], format="csr")
    princ = np.full(n, -2.0)
    princ[0] = princ[-1] = -1.0          # Neumann: la celda de borde pierde un vecino
    L = diags([np.ones(n - 1), princ, np.ones(n - 1)], [-1, 0, 1], format="csr")
    return L / (h * h)


class SolucionadorPresion:
    """Poisson con Neumann puro, opcionalmente restringido al fluido.

    La ruta sin solidos conserva el producto de Kronecker original. Para una
    mascara no vacia se ensambla el grafo de celdas fluidas: cada cara que une
    dos celdas fluidas aporta los dos terminos fuera de diagonal y resta su
    peso a ambas diagonales. Las caras fluido--solido no aportan nada, que es
    exactamente Neumann homogeneo.

    Una mascara puede separar el fluido en varias componentes (por ejemplo, el
    interior y el exterior de un crisol cerrado). Cada componente necesita su
    propia condicion de compatibilidad y su propio anclaje de presion. Matriz y
    factorizacion se cachean por contenido de la mascara; llamadas sucesivas
    con el mismo objeto ni siquiera recalculan la clave.
    """

    def __init__(self, forma: tuple[int, int, int], dx: float, dy: float, dz: float):
        nx, ny, nz = forma
        self.forma = forma
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        Lx = _laplaciano_1d(nx, dx)
        Ly = _laplaciano_1d(ny, dy)
        Lz = _laplaciano_1d(nz, dz)
        Ix, Iy, Iz = identity(nx), identity(ny), identity(nz)
        self.A = (kron(kron(Lx, Iy), Iz)
                  + kron(kron(Ix, Ly), Iz)
                  + kron(kron(Ix, Iy), Lz)).tocsr()
        # Neumann puro deja la matriz singular (la presión está definida salvo
        # una constante). Se ancla una celda para que el sistema sea regular.
        self.A = self.A.tolil()
        self.A[0, :] = 0.0
        self.A[0, 0] = 1.0
        self.A = self.A.tocsc()
        self._resolver = None
        self._cache_mascaras: dict[bytes, dict[str, Any]] = {}
        self._ultima_mascara: np.ndarray | None = None
        self._ultimo_sistema: dict[str, Any] | None = None
        self.sistemas_enmascarados_construidos = 0

    def _construir_sistema_enmascarado(
            self, solido: np.ndarray) -> dict[str, Any]:
        """Construye el Laplaciano reducido usando solo caras fluido--fluido."""
        fluido = ~solido
        indices_fluidos = np.flatnonzero(fluido.reshape(-1))
        n_fluidas = indices_fluidos.size
        if n_fluidas == 0:
            return {
                "indices": indices_fluidos,
                "etiquetas": np.empty(0, dtype=np.int32),
                "conteos": np.empty(0, dtype=np.int64),
                "anclas": np.empty(0, dtype=np.int64),
                "A": None,
                "resolver": None,
            }

        global_a_local = np.full(solido.size, -1, dtype=np.int64)
        global_a_local[indices_fluidos] = np.arange(n_fluidas)
        diagonal = np.zeros(n_fluidas, dtype=float)
        filas: list[np.ndarray] = []
        columnas: list[np.ndarray] = []
        datos: list[np.ndarray] = []
        espaciados = (self.dx, self.dy, self.dz)

        for eje, h in enumerate(espaciados):
            izquierda = [slice(None)] * 3
            derecha = [slice(None)] * 3
            izquierda[eje] = slice(None, -1)
            derecha[eje] = slice(1, None)
            abiertas = fluido[tuple(izquierda)] & fluido[tuple(derecha)]
            coordenadas = list(np.nonzero(abiertas))
            if not coordenadas or coordenadas[0].size == 0:
                continue
            global_i = np.ravel_multi_index(tuple(coordenadas), self.forma)
            coordenadas[eje] = coordenadas[eje] + 1
            global_j = np.ravel_multi_index(tuple(coordenadas), self.forma)
            local_i = global_a_local[global_i]
            local_j = global_a_local[global_j]
            peso = 1.0 / (h * h)

            filas.extend((local_i, local_j))
            columnas.extend((local_j, local_i))
            datos.extend((np.full(local_i.size, peso),
                          np.full(local_j.size, peso)))
            np.add.at(diagonal, local_i, -peso)
            np.add.at(diagonal, local_j, -peso)

        base = np.arange(n_fluidas)
        filas.append(base)
        columnas.append(base)
        datos.append(diagonal)
        A = coo_matrix(
            (np.concatenate(datos),
             (np.concatenate(filas), np.concatenate(columnas))),
            shape=(n_fluidas, n_fluidas),
        ).tocsr()

        n_componentes, etiquetas = connected_components(A, directed=False)
        conteos = np.bincount(etiquetas, minlength=n_componentes)
        anclas = np.full(n_componentes, -1, dtype=np.int64)
        for indice, componente in enumerate(etiquetas):
            if anclas[componente] < 0:
                anclas[componente] = indice

        A = A.tolil()
        for ancla in anclas:
            A[ancla, :] = 0.0
            A[ancla, ancla] = 1.0
        return {
            "indices": indices_fluidos,
            "etiquetas": etiquetas,
            "conteos": conteos,
            "anclas": anclas,
            "A": A.tocsc(),
            "resolver": None,
        }

    def _sistema_enmascarado(self, solido: np.ndarray) -> dict[str, Any]:
        if solido.shape != self.forma:
            raise ValueError(
                f"solido tiene forma {solido.shape}; se esperaba {self.forma}")
        if solido is self._ultima_mascara and self._ultimo_sistema is not None:
            return self._ultimo_sistema

        clave = np.packbits(solido.reshape(-1)).tobytes()
        sistema = self._cache_mascaras.get(clave)
        if sistema is None:
            sistema = self._construir_sistema_enmascarado(solido)
            self._cache_mascaras[clave] = sistema
            self.sistemas_enmascarados_construidos += 1
        self._ultima_mascara = solido
        self._ultimo_sistema = sistema
        return sistema

    def quitar_media_por_componente(
            self, campo: np.ndarray, solido: np.ndarray,
            ) -> tuple[np.ndarray, float]:
        """Separa el residuo resoluble de la incompatibilidad Neumann."""
        mascara = np.asarray(solido, dtype=bool)
        sistema = self._sistema_enmascarado(mascara)
        indices = sistema["indices"]
        salida = np.zeros(self.forma, dtype=float)
        if indices.size == 0:
            return salida, 0.0
        valores = np.asarray(campo, dtype=float).reshape(-1)[indices].copy()
        etiquetas = sistema["etiquetas"]
        conteos = sistema["conteos"]
        medias = np.bincount(
            etiquetas, weights=valores, minlength=conteos.size) / conteos
        valores -= medias[etiquetas]
        salida.reshape(-1)[indices] = valores
        return salida, float(np.max(np.abs(medias), initial=0.0))

    def resolver(self, rhs: np.ndarray, directo: bool = True,
                 solido: np.ndarray | None = None) -> np.ndarray:
        if rhs.shape != self.forma:
            raise ValueError(
                f"rhs tiene forma {rhs.shape}; se esperaba {self.forma}")
        if solido is not None and np.any(solido):
            mascara = np.asarray(solido, dtype=bool)
            sistema = self._sistema_enmascarado(mascara)
            indices = sistema["indices"]
            if indices.size == 0:
                return np.zeros(self.forma, dtype=float)

            b = rhs.reshape(-1)[indices].copy()
            etiquetas = sistema["etiquetas"]
            conteos = sistema["conteos"]
            medias = np.bincount(
                etiquetas, weights=b, minlength=conteos.size) / conteos
            b -= medias[etiquetas]
            b[sistema["anclas"]] = 0.0
            A = sistema["A"]
            if directo:
                if sistema["resolver"] is None:
                    sistema["resolver"] = factorized(A)
                x = sistema["resolver"](b)
            else:
                x, info = bicgstab(A, b, rtol=1e-10, maxiter=2000)
                if info != 0:
                    raise RuntimeError(
                        f"BiCGSTAB enmascarado no convergio (info={info})")

            medias_x = np.bincount(
                etiquetas, weights=x, minlength=conteos.size) / conteos
            x -= medias_x[etiquetas]
            salida = np.zeros(self.forma, dtype=float)
            salida.reshape(-1)[indices] = x
            return salida

        b = rhs.reshape(-1).copy()
        # condición de compatibilidad: con Neumann puro, la integral del término
        # fuente debe anularse; si no, el sistema no tiene solución.
        b -= b.mean()
        b[0] = 0.0                        # coherente con la fila anclada
        if directo:
            if self._resolver is None:
                self._resolver = factorized(self.A)
            x = self._resolver(b)
        else:
            x, info = bicgstab(self.A, b, rtol=1e-10, maxiter=2000)
            if info != 0:
                raise RuntimeError(f"BiCGSTAB no convergió (info={info})")
        x -= x.mean()
        return x.reshape(self.forma)


class SolucionadorViscoso:
    """Resuelve el predictor viscoso sobre las tres mallas de caras MAC.

    Cada componente tiene una forma distinta y, por tanto, su propio
    laplaciano disperso.  El arrastre lineal de Darcy se incorpora en la
    diagonal del mismo sistema; Forchheimer, linealizado con la velocidad del
    paso anterior, se añade como una corrección diagonal en cada llamada.

    La matriz base y su ILU se conservan mientras no cambien ``dt``, la
    viscosidad cinemática ni Darcy. Esto evita reconstruir el precondicionador
    en los cientos de miles de pasos de una simulación real. La comparación de
    coeficientes se hace por valor porque las interpolaciones a caras crean
    arrays nuevos aunque las propiedades físicas no hayan cambiado.
    """

    def __init__(self, dx: float, dy: float, dz: float):
        self.espaciados = (float(dx), float(dy), float(dz))
        self._laplacianos: dict[tuple[int, int, int], Any] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        self.precondicionadores_construidos = 0
        self.respaldos_directos = 0

    def _laplaciano(self, forma: tuple[int, int, int]) -> Any:
        if forma not in self._laplacianos:
            nx, ny, nz = forma
            dx, dy, dz = self.espaciados
            Lx = _laplaciano_1d(nx, dx)
            Ly = _laplaciano_1d(ny, dy)
            Lz = _laplaciano_1d(nz, dz)
            Ix, Iy, Iz = identity(nx), identity(ny), identity(nz)
            self._laplacianos[forma] = (
                kron(kron(Lx, Iy), Iz)
                + kron(kron(Ix, Ly), Iz)
                + kron(kron(Ix, Iy), Lz)
            ).tocsr()
        return self._laplacianos[forma]

    @staticmethod
    def _misma_base(cache: dict[str, Any] | None, dt: float,
                    nu: np.ndarray, darcy: np.ndarray) -> bool:
        return bool(
            cache is not None
            and cache["dt"] == dt
            and np.array_equal(cache["nu"], nu)
            and np.array_equal(cache["darcy"], darcy)
        )

    def _sistema_base(self, componente: str, dt: float, nu: np.ndarray,
                      darcy: np.ndarray) -> dict[str, Any]:
        cache = self._cache.get(componente)
        if self._misma_base(cache, dt, nu, darcy):
            return cache  # type: ignore[return-value]

        lap = self._laplaciano(nu.shape)
        n = nu.size
        A = (
            identity(n, format="csr")
            + diags(dt * darcy.reshape(-1), format="csr")
            - diags(dt * nu.reshape(-1), format="csr") @ lap
        ).tocsr()

        # ILU es mucho más barato de aplicar que refactorizar el sistema en
        # cada paso. Si una malla patológica impide construirlo, el Jacobi
        # diagonal conserva una ruta iterativa robusta.
        try:
            ilu = spilu(A.tocsc(), drop_tol=1.0e-5, fill_factor=8.0)
            precondicionador = LinearOperator(A.shape, matvec=ilu.solve)
        except (RuntimeError, MemoryError):
            inversa_diagonal = 1.0 / A.diagonal()
            precondicionador = LinearOperator(
                A.shape, matvec=lambda x: inversa_diagonal * x)

        cache = {
            "dt": dt,
            "nu": nu.copy(),
            "darcy": darcy.copy(),
            "A": A,
            "M": precondicionador,
        }
        self._cache[componente] = cache
        self.precondicionadores_construidos += 1
        return cache

    def resolver(self, rhs: np.ndarray, componente: str, dt: float,
                 nu: np.ndarray, darcy: np.ndarray,
                 forchheimer: np.ndarray) -> np.ndarray:
        """Resuelve ``(I + dt*D - dt*nu*lap) u = rhs`` con BiCGSTAB."""
        nu = np.broadcast_to(np.asarray(nu, dtype=float), rhs.shape)
        darcy = np.broadcast_to(np.asarray(darcy, dtype=float), rhs.shape)
        forchheimer = np.broadcast_to(
            np.asarray(forchheimer, dtype=float), rhs.shape)
        cache = self._sistema_base(componente, dt, nu, darcy)
        A = cache["A"]
        if np.any(forchheimer != 0.0):
            A = A + diags(dt * forchheimer.reshape(-1), format="csr")

        b = np.asarray(rhs, dtype=float).reshape(-1)
        x0 = b / A.diagonal()
        # Tolerancia absoluta con sentido físico, no cero.
        #
        # Con atol=0 el objetivo es rtol*||b||, y cuando el flujo se apaga
        # ||b|| baja a ~1e-13 m/s: se estaría exigiendo un residuo de 1e-24 en
        # una matriz cuya diagonal llega a 1e16 por el arrastre de Darcy de las
        # celdas sólidas. Eso está por debajo del suelo de redondeo, rho se
        # anula y BiCGSTAB sufre breakdown (info=-10). Es lo que mató la
        # corrida de 720 s en t=425 s, con la física ya en reposo. Un residuo
        # de 1e-20 m/s es un nanómetro cada treinta años: cero a efectos
        # prácticos. En régimen vivo (||b|| ~ 1e-1) manda rtol y nada cambia.
        x, info = bicgstab(
            A, b, x0=x0, M=cache["M"], rtol=1.0e-11,
            atol=1.0e-20, maxiter=1000,
        )
        if info != 0:
            # Respaldo directo, como en transporte.py: una corrida de horas no
            # puede morir por un breakdown del iterativo. Si el directo también
            # falla, entonces sí es un problema real y se denuncia.
            try:
                x = spsolve(A.tocsc(), b)
                self.respaldos_directos += 1
            except Exception as exc:
                raise RuntimeError(
                    f"BiCGSTAB viscoso no convergió para {componente} "
                    f"(info={info}) y el respaldo directo falló: {exc}"
                ) from exc
            if not np.all(np.isfinite(x)):
                raise RuntimeError(
                    f"el respaldo directo del viscoso devolvió valores no "
                    f"finitos para {componente} (info del iterativo={info})")
        return x.reshape(rhs.shape)


# ---------------------------------------------------------------------------
# Paso de momentum
# ---------------------------------------------------------------------------
@dataclass
class ConfigMomentum:
    con_adveccion: bool = True
    con_viscoso: bool = True
    con_darcy: bool = True
    con_forchheimer: bool = True
    con_boyancia: bool = True
    viscoso_implicito: bool = True
    solucionador_directo: bool = True
    # Desactivar la proyección permite verificar los términos por separado. En
    # un recinto cerrado la proyección anula cualquier campo uniforme, porque no
    # es compatible con paredes impermeables; para comprobar, por ejemplo, la
    # relajación de Darcy hace falta el término sin la restricción de
    # incompresibilidad. NO usar en producción: sin proyección el campo deja de
    # ser solenoidal.
    con_proyeccion: bool = True
    # Impone velocidad nula en el borde del dominio. Se desactiva para pruebas
    # con condiciones periódicas o de flujo libre.
    paredes_en_el_borde: bool = True


def paso_momentum(u: np.ndarray, v: np.ndarray, w: np.ndarray, P: np.ndarray,
                  T: np.ndarray, props: PropiedadesMedio, malla: Any, dt: float,
                  solido: np.ndarray | None = None,
                  fuente: np.ndarray | None = None,
                  fuerza: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                  cfg: ConfigMomentum | None = None,
                  solucionador: SolucionadorPresion | None = None,
                  solucionador_viscoso: SolucionadorViscoso | None = None,
                  ) -> dict[str, Any]:
    """Avanza un paso de Navier--Stokes/Darcy--Brinkman--Forchheimer.

    Parameters
    ----------
    u, v, w : velocidades en caras (m/s)
    P : presión en centros (Pa)
    T : temperatura en centros (K), para la boyancia
    solido : máscara booleana de centros que son sólido impermeable
    fuente : término de generación de MASA en centros (kg/m3/s), por ejemplo la
        devolatilización; entra en la ecuación de Poisson como divergencia
        impuesta, que es como se acopla correctamente una fuente de masa.
    fuerza : terna de campos de FUERZA por unidad de volumen (N/m3), en centros.
        Es distinta de ``fuente``: aquella añade masa, ésta añade momentum sin
        añadir masa. Se necesita para la verificación por soluciones
        manufacturadas, donde el término que hace exacta la solución elegida es
        vectorial, y para cualquier fuerza de cuerpo externa.
    """
    cfg = ConfigMomentum() if cfg is None else cfg
    dx = malla.dx_mm * 1e-3
    dy = getattr(malla, "dy_mm", malla.dx_mm) * 1e-3
    dz = malla.dz_mm * 1e-3
    forma = P.shape

    mascara_solida: np.ndarray | None = None
    if solido is not None:
        candidata = np.asarray(solido, dtype=bool)
        if candidata.shape != forma:
            raise ValueError(
                f"solido tiene forma {candidata.shape}; se esperaba {forma}")
        if np.any(candidata):
            mascara_solida = candidata

    if solucionador is None:
        solucionador = SolucionadorPresion(forma, dx, dy, dz)

    # --- velocidades en centros, para los términos no lineales ---
    u_c = 0.5 * (u[:-1] + u[1:])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])
    w_c = 0.5 * (w[:, :, :-1] + w[:, :, 1:])

    rho, mu, eps = props.rho, props.mu, props.eps
    K = props.K
    inv_K = np.where(np.isfinite(K), 1.0 / np.maximum(K, 1e-30), 0.0)

    # --- aceleración en centros ---
    # El arrastre de Darcy es lineal en la velocidad y su tiempo de relajación
    # es tau = rho K /(mu eps). Para el lecho del ensayo vale 1,8e-6 s, de modo
    # que un tratamiento explícito impondría dt < tau y haría falta del orden de
    # 1e9 pasos para cubrir los 720 s del experimento. Al ser lineal se puede
    # integrar de forma exacta e incondicionalmente estable:
    #
    #     u^{n+1} = (u^n + dt * resto) / (1 + dt / tau)
    #
    # Con ello la restricción de paso desaparece sin perder exactitud: para
    # dt << tau el desarrollo coincide con el explícito, y para dt >> tau
    # reproduce el equilibrio de Darcy, que es el comportamiento correcto.
    coef_darcy = (mu * inv_K / rho) * eps if cfg.con_darcy else np.zeros(forma)

    if cfg.con_forchheimer:
        # Forchheimer es cuadrático; se linealiza con la magnitud del paso
        # anterior y se incorpora a la diagonal implícita.
        vel = np.sqrt(u_c**2 + v_c**2 + w_c**2)
        coef_forch = props.C_F * np.sqrt(inv_K) * vel
    else:
        coef_forch = np.zeros(forma)

    acel = {}
    for nombre, comp in (("x", u_c), ("y", v_c), ("z", w_c)):
        a = np.zeros(forma)
        if cfg.con_adveccion:
            a -= _adveccion_upwind(comp, u_c, v_c, w_c, dx, dy, dz) / np.maximum(eps, 1e-12)
        acel[nombre] = a

    if cfg.con_boyancia and props.beta != 0.0:
        acel["z"] += -G * props.beta * (T - props.T_ref)

    # Fuerza de cuerpo externa (N/m3 -> aceleración dividiendo por la densidad)
    if fuerza is not None:
        for nombre, f_comp in zip(("x", "y", "z"), fuerza):
            acel[nombre] += np.asarray(f_comp, dtype=float) / rho

    # --- predictor (sin presión), con arrastre y viscosidad implícitos ---
    # [I + dt*(Darcy + Forchheimer) - dt*nu*lap] u* = u + dt*a_explicita
    if cfg.con_viscoso and cfg.viscoso_implicito and solucionador_viscoso is None:
        solucionador_viscoso = SolucionadorViscoso(dx, dy, dz)

    nu_centros = np.broadcast_to(
        np.asarray(props.mu_ef, dtype=float), forma) / rho

    def _predictor(campo, nombre, a_caras):
        rhs = campo + dt * a_caras(acel[nombre])
        darcy_caras = a_caras(coef_darcy)
        forch_caras = a_caras(coef_forch)
        if cfg.con_viscoso and cfg.viscoso_implicito:
            assert solucionador_viscoso is not None
            return solucionador_viscoso.resolver(
                rhs, nombre, dt, a_caras(nu_centros),
                darcy_caras, forch_caras)
        if cfg.con_viscoso:
            rhs += dt * a_caras(nu_centros) * _laplaciano_componente(
                campo, dx, dy, dz)
        return rhs / (1.0 + dt * (darcy_caras + forch_caras))

    u_est = _predictor(u, "x", _a_caras_x)
    v_est = _predictor(v, "y", _a_caras_y)
    w_est = _predictor(w, "z", _a_caras_z)

    # --- fronteras: velocidad nula en el sólido y en el borde del dominio ---
    if mascara_solida is not None:
        u_est, v_est, w_est = _aplicar_solido(
            u_est, v_est, w_est, mascara_solida)
    if cfg.paredes_en_el_borde:
        u_est[0] = u_est[-1] = 0.0
        v_est[:, 0] = v_est[:, -1] = 0.0
        w_est[:, :, 0] = w_est[:, :, -1] = 0.0

    div_objetivo = np.zeros(forma) if fuente is None else fuente / rho
    div_est = divergencia(u_est, v_est, w_est, dx, dy, dz)

    fluido = None if mascara_solida is None else ~mascara_solida

    def _maximo_en_fluido(campo: np.ndarray) -> float:
        if fluido is None:
            return float(np.abs(campo).max())
        if not np.any(fluido):
            return 0.0
        return float(np.abs(campo[fluido]).max())

    if not cfg.con_proyeccion:
        residuo_inicial = _maximo_en_fluido(div_est - div_objetivo)
        return {
            "u": u_est, "v": v_est, "w": w_est, "P": P,
            "divergencia_residual": residuo_inicial,
            "divergencia_inicial": residuo_inicial,
            "incompatibilidad_divergencia": 0.0,
            "proyectado": False,
        }

    # --- Poisson ---
    rhs = (np.mean(rho) / dt) * (div_est - div_objetivo)
    phi = solucionador.resolver(
        rhs, directo=cfg.solucionador_directo, solido=mascara_solida)

    # --- corrección ---
    gx, gy, gz = gradiente_a_caras(
        phi, dx, dy, dz, solido=mascara_solida)
    factor = dt / np.mean(rho)
    u_new = u_est - factor * gx
    v_new = v_est - factor * gy
    w_new = w_est - factor * gz
    if mascara_solida is not None:
        # El gradiente sobre una cara solida ya es cero. Reaplicar la mascara
        # garantiza no-flujo exacto incluso si el predictor recibido no lo era.
        u_new, v_new, w_new = _aplicar_solido(
            u_new, v_new, w_new, mascara_solida)
    if cfg.paredes_en_el_borde:
        u_new[0] = u_new[-1] = 0.0
        v_new[:, 0] = v_new[:, -1] = 0.0
        w_new[:, :, 0] = w_new[:, :, -1] = 0.0

    div_final_cruda = (
        divergencia(u_new, v_new, w_new, dx, dy, dz) - div_objetivo)
    incompatibilidad = 0.0
    if mascara_solida is None:
        div_final = div_final_cruda
    else:
        # La media de cada componente Neumann es fisicamente incompatible y
        # se elimina del RHS. No es un error del solve: se informa aparte para
        # que el residual mida solo la precision de la proyeccion.
        div_final, incompatibilidad = solucionador.quitar_media_por_componente(
            div_final_cruda, mascara_solida)
    return {
        "u": u_new, "v": v_new, "w": w_new, "P": P + phi,
        "divergencia_residual": _maximo_en_fluido(div_final),
        "incompatibilidad_divergencia": incompatibilidad,
        "divergencia_inicial": _maximo_en_fluido(div_est - div_objetivo),
        "proyectado": True,
    }


def _aplicar_solido(u: np.ndarray, v: np.ndarray, w: np.ndarray, solido: np.ndarray):
    """Impone velocidad nula en las caras que tocan una celda sólida."""
    u = u.copy(); v = v.copy(); w = w.copy()
    u[:-1][solido] = 0.0
    u[1:][solido] = 0.0
    v[:, :-1][solido] = 0.0
    v[:, 1:][solido] = 0.0
    w[:, :, :-1][solido] = 0.0
    w[:, :, 1:][solido] = 0.0
    return u, v, w


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
def numeros_adimensionales(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                           T: np.ndarray, props: PropiedadesMedio, malla: Any,
                           d_particula_m: float = 175e-6,
                           L_caracteristica_m: float = 0.025,
                           dT_K: float = 50.0,
                           difusividad_masica: float = 1.5e-4,
                           ) -> dict[str, float]:
    """Números adimensionales del estado actual, con su interpretación.

    Se calculan en tiempo de ejecución en vez de suponerlos: así la afirmación
    de que un término es despreciable queda verificada, no asumida.
    """
    u_c = 0.5 * (u[:-1] + u[1:])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])
    w_c = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    vel = np.sqrt(u_c**2 + v_c**2 + w_c**2)
    u_max = float(vel.max()) if vel.size else 0.0
    rho_m = float(np.mean(props.rho))
    nu = props.mu / rho_m
    dx = malla.dx_mm * 1e-3

    alpha_t = 2.242e-4  # m2/s, difusividad térmica del gas a 900 C
    Ra = (G * props.beta * dT_K * L_caracteristica_m ** 3 / (nu * alpha_t)
          if props.beta > 0 else 0.0)
    K_med = float(np.median(props.K[np.isfinite(props.K)])) if np.isfinite(props.K).any() else np.inf

    return {
        "u_max_m_s": u_max,
        "Re_particula": rho_m * u_max * d_particula_m / props.mu,
        "Re_celda": rho_m * u_max * dx / props.mu,
        "Ra": Ra,
        "Ra_critico": 1708.0,
        "Da": (K_med / L_caracteristica_m ** 2) if np.isfinite(K_med) else np.inf,
        "Pe_masico": u_max * L_caracteristica_m / difusividad_masica,
        "Pe_termico": u_max * L_caracteristica_m / alpha_t,
    }


def interpretar_regimen(nums: dict[str, float]) -> dict[str, str]:
    """Traduce los números a afirmaciones sobre qué término domina."""
    fuera = {}
    fuera["inercia_en_poro"] = (
        "despreciable: Darcy lineal es válido" if nums["Re_particula"] < 1.0
        else "relevante: el término de Forchheimer importa")
    fuera["conveccion_natural"] = (
        "débil: la conducción domina en el gas" if nums["Ra"] < nums["Ra_critico"]
        else "significativa: hay celdas convectivas")
    fuera["transporte_masico"] = (
        "difusión dominante" if nums["Pe_masico"] < 1.0
        else ("advección y difusión comparables" if nums["Pe_masico"] < 10.0
              else "advección dominante"))
    fuera["estabilidad_adveccion"] = (
        "el esquema upwind introduce difusión numérica despreciable"
        if nums["Re_celda"] < 2.0 else
        "Re de celda alto: conviene refinar la malla o usar un esquema TVD")
    return fuera


def dt_estable_momentum(u, v, w, props: PropiedadesMedio, malla: Any,
                        seguridad: float = 0.4,
                        cfg: ConfigMomentum | None = None,
                        devolver_diagnostico: bool = False,
                        ) -> float | dict[str, float | str]:
    """Paso estable del bloque y límites inactivos como diagnóstico.

    Darcy, Forchheimer y, por defecto, Brinkman se tratan implícitamente. Sus
    escalas explícitas se reportan para cuantificar la ganancia, pero sólo los
    términos realmente explícitos entran en ``global``.
    """
    cfg = ConfigMomentum() if cfg is None else cfg
    dx = malla.dx_mm * 1e-3
    dy = getattr(malla, "dy_mm", malla.dx_mm) * 1e-3
    dz = malla.dz_mm * 1e-3
    h_min = min(dx, dy, dz)
    u_c = 0.5 * (u[:-1] + u[1:])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])
    w_c = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    vel = np.sqrt(u_c**2 + v_c**2 + w_c**2)
    u_max = float(vel.max())

    dt_adv = seguridad * h_min / u_max if u_max > 0 else np.inf
    nu_max = float(np.max(
        np.broadcast_to(np.asarray(props.mu_ef, dtype=float), props.rho.shape)
        / props.rho))
    dt_vis = (seguridad * 0.5 * h_min ** 2 / nu_max
              if nu_max > 0 else np.inf)

    inv_K = np.where(
        np.isfinite(props.K), 1.0 / np.maximum(props.K, 1.0e-30), 0.0)
    coef_darcy = props.mu * inv_K * props.eps / props.rho
    darcy_max = float(np.max(coef_darcy)) if cfg.con_darcy else 0.0
    dt_darcy = seguridad / darcy_max if darcy_max > 0 else np.inf
    coef_forch = props.C_F * np.sqrt(inv_K) * vel
    forch_max = float(np.max(coef_forch)) if cfg.con_forchheimer else 0.0
    dt_forch = seguridad / forch_max if forch_max > 0 else np.inf

    limites: dict[str, float | str] = {
        "adveccion": float(dt_adv),
        "viscoso_si_fuese_explicito": float(dt_vis),
        "darcy_si_fuese_explicito": float(dt_darcy),
        "forchheimer_si_fuese_explicito": float(dt_forch),
    }
    activos: list[str] = []
    if cfg.con_adveccion:
        activos.append("adveccion")
    if cfg.con_viscoso and not cfg.viscoso_implicito:
        activos.append("viscoso_si_fuese_explicito")

    if activos:
        restringe = min(activos, key=lambda nombre: float(limites[nombre]))
        global_ = float(limites[restringe])
    else:
        restringe = "ninguno"
        global_ = math.inf
    limites["global"] = global_
    limites["restringe"] = restringe
    return limites if devolver_diagnostico else global_
