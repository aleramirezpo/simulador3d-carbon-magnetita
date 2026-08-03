# Contratos entre módulos — `simulador3d`

Documento normativo. Define las interfaces que **todos** los módulos deben respetar
para poder desarrollarse en paralelo sin colisiones. Si un módulo necesita cambiar
un contrato, se cambia aquí primero.

---

## 0. Convenios globales

- **Unidades SI internamente** (m, kg, s, K, Pa, mol). Sólo la capa de presentación
  convierte a mm, °C, etc. La geometría es la única excepción documentada: sus
  cotas de entrada están en mm porque así vienen del laboratorio.
- **Ejes**: `z` es la vertical, contra la gravedad. El origen está en el fondo
  exterior del crisol, sobre el eje de revolución.
- **Orden de los índices**: `(i, j, k)` ↔ `(x, y, z)`, `indexing="ij"` de numpy.
- **Español** en docstrings, comentarios y mensajes al usuario. Los identificadores
  de código también en español, salvo símbolos físicos consagrados (`T`, `P`, `u`).
- Ningún módulo escribe en disco al importarse.

---

## 1. Malla — `nucleo/geometria.py` [IMPLEMENTADO]

```python
malla.forma            # (nx, ny, nz)
malla.dx_mm, malla.dz_mm
malla.x, malla.y, malla.z      # centros de celda, mm
malla.rejilla()                # -> (X, Y, Z) de forma `forma`
malla.volumen_celda_mm3
etiquetas: np.uint8            # VACIO=0, PARED_CRISOL=1, TAPA=2, LECHO=3, GAS=4
fraccion_volumetrica(malla, dentro, submuestreo) -> np.float64 en [0,1]
```

**Regla de oro**: las celdas cortadas por una frontera se tratan con su fracción
volumétrica, nunca como binarias. Es lo que hace conservativo al esquema.

---

## 2. Campos — malla escalonada (staggered / MAC)

Para que el acoplamiento velocidad–presión sea estable:

| Cantidad | Ubicación | Forma del array |
|---|---|---|
| `P`, `T`, `c_i`, `eps`, `alpha` | centro de celda | `(nx, ny, nz)` |
| `u` (componente x) | cara x | `(nx+1, ny, nz)` |
| `v` (componente y) | cara y | `(nx, ny+1, nz)` |
| `w` (componente z) | cara z | `(nx, ny, nz+1)` |

```python
@dataclass
class CamposEstado:
    t: float                      # s
    u, v, w: np.ndarray           # m/s, en caras
    P: np.ndarray                 # Pa, centros
    T: np.ndarray                 # K, centros
    c: dict[str, np.ndarray]      # mol/m3 por especie, centros
    eps: np.ndarray               # porosidad, centros
    solido: dict[str, np.ndarray] # mol/m3 de cada fase mineral, centros
    cohesion: np.ndarray          # [0,1], centros
    hinchamiento: np.ndarray      # >=1, centros; OPCIONAL
```

`hinchamiento` es el factor de expansión volumétrica del aglomerado
(`fisica/hinchamiento.py`): 1 significa sin hinchar. Es **opcional** en el NPZ;
al cargar una serie que no lo trae se rellena con unos, de modo que las
instantáneas anteriores siguen siendo legibles.

---

## 3. Momentum — `nucleo/momentum.py`

Resuelve Navier–Stokes con términos de medio poroso (Darcy–Brinkman–Forchheimer):

```
rho/eps * du/dt + rho/eps^2 (u.grad)u
    = -grad P + mu_ef lap(u) - (mu/K) u - (rho C_F/sqrt(K))|u|u + rho g beta (T-T0)
```

En el gas libre `eps=1` y `K=inf`, con lo que se reduce a Navier–Stokes puro.

```python
def paso_momentum(campos, props, malla, dt, cfg) -> CamposEstado
def numeros_adimensionales(campos, props, malla) -> dict[str, float]
    # {"Re_particula", "Re_celda", "Ra", "Da", "Pe_termico", "Ma"}
```

Acoplamiento por **proyección de presión** (Chorin): predictor sin presión,
ecuación de Poisson para la corrección, y proyección al campo solenoidal.

---

## 4. Transporte escalar — `nucleo/transporte.py`

```python
def divergencia_flujo_advectivo(phi, u, v, w, malla, esquema="upwind") -> np.ndarray
def divergencia_flujo_difusivo(phi, D, malla, fraccion=None) -> np.ndarray
def paso_energia(campos, props, malla, dt, fuentes) -> np.ndarray      # -> T
def paso_especies(campos, props, malla, dt, fuentes) -> dict           # -> c
```

Esquemas admitidos: `"upwind"` (robusto, 1.er orden), `"central"` (2.º orden),
`"tvd_superbee"` (2.º orden con limitador). El caso por defecto es TVD.

---

## 5. Química — `fisica/adaptador_v3.py`

Puente a los módulos ya validados de `simulacion_v3`. **No se duplica química.**

```python
def tasas_locales(T, c_gas, solido, eps, cfg) -> dict[str, np.ndarray]
    # devuelve {"R_especie": ..., "R_fase": ..., "Q_reaccion": ...}
def integrar_quimica_local(campos, dt, cfg) -> CamposEstado
```

La química se integra **por celda** con un solucionador rígido, dentro del
splitting de Strang. Reutiliza `gases.py`, `termodinamica_ext.py` y `grano.py`.

---

## 6. Acoplamiento — `nucleo/acople.py`

**Splitting de Strang**, de segundo orden:

```
quimica(dt/2) -> momentum(dt) -> transporte(dt) -> quimica(dt/2)
```

```python
def paso_global(campos, props, malla, dt, cfg) -> tuple[CamposEstado, dict]
def dt_estable(campos, props, malla, cfg) -> float   # CFL advectivo, difusivo y químico
```

---

## 7. Salida — `nucleo/salida.py`

```python
def guardar_instantanea(campos, ruta, formato="npz")   # npz | vtk | xdmf
def serie_temporal(...)  -> pd.DataFrame               # escalares integrados
```

La interfaz consume instantáneas; **nunca** importa el solucionador directamente.
Esto permite precalcular y reproducir sin recalcular.

---

## 8. Casos — `casos/*.yaml`

Un caso es declarativo. El motor no conoce ningún caso concreto.

```yaml
nombre: carbon_magnetita
geometria:
  tipo: crisol_perfil
  perfil_mm: [[0, 12.5], [18, 14.0], [21, 15.3], [23, 14.75], [32, 14.75]]
  espesor_pared_mm: 1.1
malla: {dx_mm: 0.5, dz_mm: 0.25}
fisica: [momentum, energia, especies, quimica, cohesion]
especies: [CO, CO2, H2, H2O, CH4, N2, O2]
condiciones_frontera:
  mufla: {tipo: radiacion, curva: curvas_t_vs_T.xlsx, emisividad: 0.8}
  tapa:  {tipo: venteo, conductancia: calculada}
tiempo: {t_final_s: 720, dt_inicial_s: 1e-3, adaptativo: true}
```

---

## 9. Verificación — `verificacion/`

Cada transporte necesita al menos una prueba con solución exacta:

| Módulo | Prueba de referencia |
|---|---|
| momentum | Poiseuille en canal; Darcy 1-D; orden de convergencia |
| energía | conducción transitoria (Carslaw & Jaeger) |
| especies | difusión pura (función error); advección–difusión estacionaria |
| acoplado | soluciones manufacturadas (MMS) con orden medido |
| global | conservación de masa, energía y especies; **consistencia con el 0-D de v3** |

La prueba de consistencia con `simulacion_v3` es la más importante: al elevar la
conductividad y la difusividad, el promedio volumétrico del 3D debe reproducir el
modelo 0-D ya calibrado. Si no lo hace, el acoplamiento está mal.

---

## 10. Regla sobre honestidad de los resultados

Los campos que la interfaz muestre y que **no** estén validados experimentalmente
deben ir rotulados como predicción. La caracterización disponible es del material
inicial: no existe medición del aglomerado después del ensayo, ni de la composición
del gas, ni del campo de velocidad. Ningún elemento de la interfaz puede sugerir lo
contrario.
