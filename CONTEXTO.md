# `simulador3d` — laboratorio virtual de medio poroso reactivo

Programa de simulación y visualización 3D de transporte de **momentum, calor y masa**
acoplado a reacción química heterogénea. El primer caso es el aglomerado
carbón–titanomagnetita del proyecto, pero el motor es genérico: un caso nuevo es un
archivo YAML, sin tocar el núcleo.

---

## 1. Cómo se ejecuta

```
interfaz\iniciar_simulador.bat          doble clic
python -m interfaz.app                  desde consola
python -m interfaz.app --headless       comprobación sin ventana
python -m pytest tests\ -v              suite completa (162 pruebas)
node tests\js\prueba_particulas.mjs     módulos del navegador, sin pytest
```

Funciona **sin conexión**: three.js está vendorizado en `interfaz\web\js\`.

---

## 2. Física implementada

### Momentum — Navier–Stokes con medio poroso

$$\frac{\rho}{\varepsilon}\frac{\partial\mathbf u}{\partial t}+\frac{\rho}{\varepsilon^2}(\mathbf u\cdot\nabla)\mathbf u=-\nabla P+\mu_{ef}\nabla^2\mathbf u-\frac{\mu}{K}\mathbf u-\frac{\rho C_F}{\sqrt K}|\mathbf u|\mathbf u+\rho\mathbf g\beta(T-T_0)$$

Una sola ecuación: en el gas libre ε=1 y K→∞, con lo que se reduce a Navier–Stokes puro.
Todos los términos están implementados, incluso los que en este caso son despreciables,
porque el programa debe servir para otros casos y porque es preferible que los números
adimensionales revelen qué domina a suponerlo. `numeros_adimensionales()` los reporta en
cada paso.

Acoplamiento presión–velocidad por **proyección de Chorin** sobre malla escalonada (MAC).

### Calor y masa

Advección–difusión–reacción por volúmenes finitos conservativos, con media **armónica** de
difusividades en las caras (la aritmética da error en interfaces discontinuas) y fracciones
volumétricas parciales en celdas cortadas por la frontera.

### Química

No se reimplementa: `fisica\adaptador_v3.py` es un **puente** a los módulos ya validados de
`simulacion_v3` (140 pruebas). La consistencia con el modelo 0-D se verifica a **1,95×10⁻¹⁶**.

---

## 3. Régimen del caso carbón–titanomagnetita

Calculado, no supuesto:

| Magnitud | Valor | Consecuencia |
|---|---|---|
| Tiempo de residencia del gas | **0,708 s** | el crisol se renueva 1,41 veces por segundo |
| Caudal por devolatilización | 19,03 cm³/s | frente a 13,5 cm³ de volumen libre |
| Rayleigh | **188** ≪ 1708 | convección natural débil |
| Reynolds de partícula | **0,053** ≪ 1 | Darcy lineal; Forchheimer despreciable |
| Péclet en el lecho | **1,01** | advección y difusión comparables |
| Péclet en el gas libre | 7,77 | advección domina |
| Caída de presión en el lecho | 45 Pa | 0,044 % de P_atm |

---

## 4. Geometría

Reconstruida de las cotas y **cerrada con la fotografía acotada** (`D:\pae\Imagen.pdf`), que
aportó una cota ausente de todos los archivos: un **collar de ⌀30,6 mm**, más ancho que la
boca, a ≈20,6 mm de altura.

| | Tronco simple | Perfil con collar |
|---|---|---|
| Masa calculada | 32,593 g | **32,670 g** |
| Error frente a 32,67 g declarados | −0,077 g | **−3,6×10⁻¹⁴ g** |

El collar explica exactamente la masa que faltaba. Parámetros ajustados: altura 20,633 mm,
espesor 3,097 mm — compatibles con lo que se ve en la foto.

**El lecho es un disco delgado**: 1,36 cm³ en 22,8 mm de diámetro son 3,26 mm de altura,
relación de aspecto 7,7:1. Por eso la malla es anisótropa.

---

## 5. Verificación

Ninguna afirmación de exactitud sin medirla.

| Prueba | Resultado | Teórico |
|---|---|---|
| Difusión frente a solución con función error | orden **2,012** | 2 |
| Conducción transitoria (Carslaw & Jaeger) | orden **2,010** | 2 |
| MMS, transporte escalar | orden **1,99** | 2 |
| MMS, momentum sin advección | orden **2,01** | 2 |
| MMS, Poisson de presión | orden **2,04** | 2 |
| MMS, momentum **con upwind** | orden **1,07** | 1 ⚠ |
| MMS, orden temporal | orden **1,01** | 1 |
| GCI de Roache (malla fina) | **0,75 %** | — |
| Relajación de Darcy vs. exponencial exacta | ✓ | — |
| Difusión de momentum vs. analítica | error **0,149 %** | <1 % |
| Consistencia con el 0-D de `simulacion_v3` | **1,95×10⁻¹⁶** | — |

**Hallazgo documentado, no ocultado**: la advección upwind degrada el momentum de orden 2 a
orden 1. Es aceptable aquí porque el Reynolds de celda es ~10⁻³ y la difusión numérica que
introduce queda muy por debajo de la viscosidad física, pero debe tenerse presente si el
programa se usa en regímenes advectivos.

### Esquemas de advección, comparados

| Pe | upwind | central | TVD superbee |
|---|---|---|---|
| 1 | L2 = 2,2×10⁻³ | 3,1×10⁻⁴ | 3,0×10⁻⁴ |
| 10 | 3,1×10⁻² | 5,1×10⁻³ | 5,1×10⁻³ |
| **50** | rango [0; 0,444] | **[−0,049; 0,444]** ⚠ | [0; 0,444] |

A Pe=50 el esquema central produce concentraciones **negativas**. TVD mantiene monotonía y
precisión: es el predeterminado.

---

## 6. Rendimiento — decisiones y ganancias medidas

### Paso de tiempo (malla 0,5 × 0,25 mm, 720 s de ensayo)

| Configuración | Paso estable | Pasos | Ganancia |
|---|---|---|---|
| Todo explícito | 3,92×10⁻⁷ s | 1.836.734.694 | — |
| + Darcy implícito | 5,43×10⁻⁵ s | 13.248.000 | ×139 |
| + viscoso implícito | 5,43×10⁻⁵ s | 13.248.000 | ×139 |
| **+ difusión implícita** | 2,15×10⁻³ s | **335.520** | **×5.474** |

El arrastre de Darcy es lineal en la velocidad, así que se integra de forma exacta e
incondicionalmente estable: `u = (u + dt·resto)/(1 + dt/tau)`. Sin eso, τ=1,8 µs obligaría a
mil ochocientos millones de pasos.

### Química (tabulación termoquímica)

La termodinámica depende sólo de T, luego se tabula una vez y se interpola.

| Celdas (campo heterogéneo) | Antes | Después | Ganancia |
|---|---|---|---|
| 512 | 2.338/s | 29.332/s | ×12,5 |
| 4.096 | 2.337/s | 164.287/s | ×70,3 |
| **32.768** | 2.381/s | **481.256/s** | **×202** |

---

## 7. Estructura

```
simulador3d/
├── nucleo/       geometria, perfil, momentum, transporte, acople, salida
├── fisica/       adaptador_v3 (puente a la química validada), tablas_termo
├── verificacion/ analiticas, mms, convergencia
├── interfaz/     app.py + web/ (three.js r180 real, 2 MB, vendorizado)
├── casos/        definiciones YAML
├── tests/        162 pruebas, + tests/js/ ejecutadas con Node
├── docs/         CONTRATOS.md — interfaces entre módulos (normativo)
└── resultados/
```

`docs\CONTRATOS.md` es normativo: define las interfaces para poder desarrollar en paralelo.
Cambiar una interfaz obliga a cambiarlo allí primero.

---

## 8. Honestidad de los resultados

**Ningún campo que muestre la interfaz está validado experimentalmente.** La caracterización
disponible es del material *inicial*; no existe medición del aglomerado tras el ensayo, ni de
la composición del gas, ni del campo de velocidad. Todo va rotulado como predicción, y así
debe seguir hasta que existan esas mediciones.

La interfaz ya muestra la salida **real** del solucionador y lo declara en la cabecera
(«RESULTADOS DEL SOLUCIONADOR — PREDICCIÓN»). Sólo cae a **datos sintéticos**, señalados
como tales de forma permanente, si no encuentra instantáneas. Las fracciones de fase de
la leyenda son igualmente predicción: no existe caracterización post-ensayo.

---

## 9. Estado y trabajo pendiente

**Hecho y verificado**: geometría, momentum, transporte, química acoplada, tabulación,
acoplamiento de Strang, verificación MMS, interfaz sobre three.js real.

**Pendiente**:
1. ~~Inspección visual de la interfaz~~ — **hecha** (§12.5): se ha visto renderizada en
   Chrome con datos reales, y la inspección encontró un defecto de geometría (§12.6).
2. ~~Sustituir los datos sintéticos por la salida del solucionador real~~ — **hecho** (§12).
3. ~~Modelo de cohesión y crecimiento del aglomerado~~ — **hecho** (§12.3).
4. Empaquetado a ejecutable con PyInstaller.
5. `LineSegments2` para líneas gruesas: es un addon de three.js que requiere dos ficheros más
   (`LineSegmentsGeometry`, `LineMaterial`) que no están vendorizados. Se usan `Line` y
   `LineSegments` estándar.
6. ~~Corrida completa de 720 s~~ — **hecha**: 145 instantáneas, 720 s (§12.9).

---

## 10. Gotchas del entorno

- **No escribir LaTeX ni regex desde heredocs de Python sin `r''`**: `\a`, `\n` y `\d` se
  convierten en bytes de control y corrompen el archivo en silencio.
- La consola de Windows usa cp1252: forzar `PYTHONIOENCODING=utf-8` al capturar la salida de
  un subproceso, o los acentos llegan mal decodificados.
- `pymupdf` sí puede rasterizar un PDF (no hay `pdftoppm`), pero exige rutas relativas: no
  acepta las rutas estilo `/d/pae/...` de Git Bash.
- MDPI bloquea la descarga automática de PDFs; J-Stage y unpkg funcionan.

---

## 11. PENDIENTE INMEDIATO — diagnóstico hecho, corrección por aplicar

### 11.1 BiCGSTAB no converge en el caso con geometría real

`test_caso.py::test_corrida_corta_estable_sin_nan_y_conservativa` falla con
`BiCGSTAB no convergió (info=-10)` (breakdown) en `transporte.py`.

**Causa localizada.** El caso marca las celdas sólidas con valores extremos en vez de
excluirlas:

```
eps:  min = 1e-6    max = 1        (fluido ~0,54 y 1)
K:    min = 1e-20   max = inf
```

Son 14 órdenes de magnitud de contraste dentro de la misma matriz. El sistema de
difusión implícita queda con un número de condición astronómico y el solucionador
iterativo sufre breakdown.

**Solución.** La misma que ya se aplicó con éxito a la presión: **resolver sólo en las
celdas activas**, excluyendo las sólidas del sistema en lugar de darles propiedades
degeneradas. En la proyección de presión ese cambio llevó la divergencia de 1,24e-3 a
1,69e-17 y además resultó un 48 % más rápido, porque el sistema reducido tiene menos
incógnitas (3.710 de 7.650). Conviene replicar el mismo patrón en `transporte.py`:
construir el operador sólo sobre las celdas de fluido y de lecho.

### 11.2 El venteo no está conectado

`test_caso.py::test_venteo_compensa_la_fuente_y_contabiliza_la_masa` falla. La
devolatilización genera unos 19 cm3/s de gas dentro de un recinto que el solucionador ve
como cerrado. Ya está demostrado en `test_momentum.py::test_fuente_neta_en_recinto_cerrado_es_incompatible`
que eso no tiene solución: la masa **debe** poder salir. Falta enlazar la máscara de
venteo (`caso._construir_mascara_venteo`, que ya existe) con la proyección de momentum.

### 11.3 Consecuencia visible

Mientras 11.1 y 11.2 no se resuelvan, la interfaz carga los resultados del solucionador
pero **la física no evoluciona**: el rango de temperatura se queda en 298,1–299,1 K
cuando debería llegar a 899 °C, y los números adimensionales salen 0. La interfaz en sí
está terminada y funciona a 30 fps; lo que le falta es una simulación que se caliente.

### 11.4 Y entonces se verá el crecimiento del aglomerado

La curva de cohesión con la historia térmica **uniforme** del 0-D hace que todas las
celdas crucen el umbral a la vez: el aglomerado aparece de golpe a 138,8 s
(0,8517 cm3, 0,6268 g, una sola pieza). El crecimiento **gradual**, de la pared hacia el
centro, sólo puede emerger del campo térmico 3-D con gradientes reales. Es precisamente
lo que aporta el 3-D frente al 0-D, y es lo que el usuario ha pedido ver.

---

## 12. Estado tras la corrección de los tres bugs bloqueantes

### 12.1 Resueltos

**BiCGSTAB (`info=-10`).** Corregido en `transporte.py::_resolver_implicito`
restringiendo el sistema a las celdas activas, igual que se hizo con la presión.
Las celdas sólidas llevan eps=1e-6 y K=1e-20: meterlas en la misma matriz que el
fluido daba 14 órdenes de contraste y el iterativo reventaba. Se añadió respaldo
directo (`spsolve`) por si el iterativo falla, que no se activa en la práctica.

**Venteo.** Conectado; `test_venteo_compensa_la_fuente` pasa y la masa venteada se
contabiliza (0,087 g a los 30 s).

**Transporte térmico.** Se ejecuta de verdad (antes había un stub que comprobaba
las funciones y devolvía el estado sin llamarlas).

### 12.2 La simulación real ya se calienta desde fuera

Corrida de 30 s en malla gruesa:

```
mufla  = 1123,15 K      (la curva real, con su caída inicial)
pared  = 1042,12 K      <- se calienta primero
lecho  =  491,30 K
centro =  467,44 K      <- lo último
```

Divergencia residual 1,19e-12 (era 1,4e-3). Conservación elemental ~1e-13 mol.
Pérdida de masa 8,99 % a 30 s. Coste: 9,4 s de reloj por segundo simulado en malla
gruesa, luego los 720 s son ~1,9 h: conviene precalcular una vez.

### 12.3 Crecimiento del aglomerado, con respaldo del modelo

`cohesion.py` incorpora ahora `perfil_termico_lecho()` y
`crecimiento_con_gradiente()`. Con el gradiente físico (la pared se calienta antes
que el núcleo) el aglomerado **crece** en vez de aparecer de golpe:

| Hito | t (s) |
|---|---|
| 10 % del volumen final | 61,3 |
| coalesce en UNA pieza | 62,5 |
| 50 % del volumen | 81,7 |
| 90 % del volumen | 119,0 |
| **máximo de fragmentos** | **28 núcleos** |

Final: 0,6983 cm3 y 0,5139 g. El aglomerado **nace como 28 núcleos junto a la pared
caliente** y coalesce. Eso es lo que el 0-D no puede predecir.

`correr_simulacion.py` calcula ya la cohesión con el campo T 3-D real del
solucionador y la guarda en cada instantánea, junto con volumen, masa y número de
componentes del aglomerado.

### 12.4 Paleta de fases

`fisica/fases_visuales.py` define 13 fases con su **color real de mineral** y
parámetros de material (metalicidad, rugosidad), cada uno con su referencia:
magnetita negra submetálica, hematita rojo pardo, ilmenita negro violáceo, rutilo
pardo rojizo, fayalita verde oliva, hierro gris plateado, char negro mate...
Separadas en fases iniciales y de producto, que es el orden en que aparecen.

### 12.5 Interfaz: hecho y comprobado en el navegador

**Por fin se ha visto renderizada** (Chrome, servidor local, datos reales de
`resultados/simulacion_larga`). Lo que antes era un pendiente de inspección visual
está resuelto: carga, orbita, reproduce y responde a los controles.

**1. Las partículas desaparecen al cohesionar.** Ya no crecen al aglomerarse (que
daba la impresión contraria a la física). El grano primero se acerca a su núcleo de
unión conservando tamaño y después se funde en la masa, que se dibuja con su propia
isosuperficie: así el mismo material no se representa dos veces. Verificado
numéricamente, no por inspección: a c=0,5 —el umbral operativo de
`cohesion.aglomerado`— el radio ya ha caído por debajo del 75 % y a c=0,7 queda
menos del 15 %, monótonamente.

**2. Coloreado por fase con el color real del mineral, ponderado por volumen.**
`fases_visuales.py` sirve ahora color, masa molar y densidad de cada fase, y el mapa
`campos_solidos` que traduce los campos NPZ (`C`→char, `ceniza`→cenizas,
`volatil`→carbón; `H2O_liq` no se colorea porque es humedad de poro, no una fase con
aspecto propio). El cliente **no duplica mineralogía**: hay una prueba que falla si
algún hexadecimal de la paleta aparece escrito a mano en el JavaScript.

- La mezcla pondera por **volumen** (mol/m³ × M/ρ), no por moles: un mol de Fe ocupa
  7,09 cm³ y uno de magnetita 44,79, así que ponderar por moles daría un lecho
  falsamente metálico. Densidades del catálogo de `simulacion_v3/src/superficie.py`.
- El promedio se hace en **luz lineal**; promediar sRGB oscurecería la mezcla.
- Leyenda en el panel izquierdo con la fracción volumétrica de cada fase sobre el
  sólido del lecho, las ausentes atenuadas, y las de producto resaltadas al aparecer.

Medido sobre la corrida real, la leyenda dice:

| | t = 0 s | t = 100 s |
|---|---:|---:|
| char | 51,89 % | 64,88 % |
| carbón (volátil) | 33,96 % | 17,56 % |
| magnetita | 5,92 % | 7,31 % |
| ilmenita | 1,59 % | 1,98 % |
| **wüstita** | — | **0,57 %** ← aparece |
| aglomerado | no cuaja | 0,072 cm³ · 5,8 % del lecho |

Coherente con el modelo: el volátil se va, el char se concentra, la ilmenita **no se
reduce** (sólo se concentra al perderse masa alrededor) y la primera fase de producto
que asoma es la wüstita.

**3. Transporte binario S3DF.** El cliente ya traía el decodificador
(`frame-cache.js`) pero **ningún servidor lo producía**: se pedía Float32 y llegaba
JSON. Implementado en `app.py` (`/api/fotograma?formato=float32`). El cliente ya no
convierte ~250.000 números desde texto decimal por fotograma, sino que copia memoria.
En bytes la ganancia es modesta (1,3×: los campos sólidos son casi todos ceros, y un
cero en JSON son 2 bytes); la ganancia real es de análisis. 51 fotogramas se precargan
en 13,7 s.

**4. Directorio de datos automático.** `--datos` sin argumento elige la corrida más
avanzada de `resultados/`, leyendo el tiempo del **nombre** del archivo para no abrir
decenas de megabytes sólo para decidir.

### 12.6 Defecto encontrado al inspeccionar, y corregido

La consola denunciaba `computeBoundingSphere(): Computed radius is NaN` en cada
reconstrucción. No era ruido: en `marching-cubes.js` las esquinas fuera de la máscara
valen `-Infinity`, y la interpolación calculaba
`(umbral − (−∞)) / (v − (−∞))` = `∞/∞` = **NaN**. Todo vértice de la isosuperficie
apoyado en el borde del lecho salía NaN, con la geometría y su esfera envolvente
corrompidas — justo en la frontera donde nace el aglomerado. Ahora, si un extremo no
es finito, la superficie se corta a mitad de arista, en la frontera de la máscara.

Debajo asomaba un segundo error, que el ruido del primero tapaba: `animar` leía
`estado.config.n_fotogramas` sin encadenamiento opcional, y el bucle de animación
arranca **antes** de que responda `/api/config`, así que lanzaba una excepción por
fotograma durante toda la carga. No rompía nada —el `requestAnimationFrame` siguiente
ya está pedido cuando salta— pero es exactamente lo que impide ver un error de verdad.
Con las dos correcciones la consola queda limpia de principio a fin.

### 12.7 Pruebas

162 en total. Las de los módulos del navegador ya no son comprobaciones de texto sobre
el fuente: `tests/js/*.mjs` construyen el objeto real bajo Node, le pasan instantáneas
y **miden** color de instancia, escala de matriz y vértices; `tests/test_interfaz.py`
los lanza y se salta si no hay Node. El contrato S3DF se verifica de extremo a extremo
codificando en Python y decodificando con el decodificador real del cliente.

### 12.8 Cuarto bug bloqueante: el viscoso reventaba con la física en reposo

El primer intento de corrida de 720 s **murió en t = 425 s**, tras tres horas, con
`BiCGSTAB viscoso no convergió para x (info=-10)`. No era un problema numérico del
lecho sino de la tolerancia:

- El registro lo enseña sin ambigüedad: `Re_p` cayó 9,2×10⁻⁴ → 6,0×10⁻⁷ → 6,4×10⁻⁹
  en tres guardados consecutivos, y la masa sólida se quedó fija en 0,7185816 g. La
  devolatilización había terminado y **el flujo se estaba apagando**.
- `momentum.SolucionadorViscoso.resolver` pedía `rtol=1e-11, atol=0.0`. Con ‖b‖ ~
  10⁻¹³ m/s eso exige un residuo de ~10⁻²⁴ sobre una matriz cuya diagonal llega a
  10¹⁶ por el arrastre de Darcy de las celdas sólidas: por debajo del suelo de
  redondeo. `rho` se anula y BiCGSTAB sufre breakdown.

O sea: el solucionador se caía justamente **porque la física había llegado al
reposo**, que es el resultado correcto.

Reproducido en una prueba antes de tocar nada (contraste de Darcy de 18 órdenes y
término fuente decreciente): falla a partir de ‖b‖ ~ 10⁻¹². Corregido con una
tolerancia absoluta con sentido físico —10⁻²⁰ m/s es un nanómetro cada treinta
años— más un **respaldo directo** (`spsolve`) como el que ya tenía `transporte.py`,
para que una corrida de horas no muera por un breakdown del iterativo. En régimen
vivo (‖b‖ ~ 10⁻¹) manda `rtol` y la solución sigue coincidiendo con la directa a
10⁻⁹ relativo: la corrección **no relaja el caso normal**.

### 12.9 La corrida completa de 720 s, terminada

> **SUPERADA por §14.** Los números de esta sección son del modelo **anterior** a
> corregir la ecuación de energía. La cronología que describen (lecho a 623 K
> pasados 300 s) es justamente la que no cuadraba con lo observado en el
> laboratorio. Se conservan porque documentan de qué se partía.

`resultados/simulacion_720s`: **145 instantáneas, 64 MB, t = 720,000 s exactos**, en
2.667,9 s de reloj = **44,5 min**, o 3,705 s de reloj por segundo simulado. (El ≈19
s/s que se estimó del intento fallido estaba contaminado: aquella corrida competía con
la suite de pruebas y dos servidores en la misma máquina. Sola, la malla gruesa va
cinco veces más rápido.) 5.235 pasos; en régimen el limitador es `dt_max_acople`, no
la estabilidad.

| | t = 0 s | t = 720 s |
|---|---:|---:|
| char | 51,89 % | **79,23 %** |
| carbón (volátil) | 33,96 % | **agotado** |
| cenizas | 5,57 % | 8,50 % |
| magnetita | 5,92 % | 5,86 % |
| ilmenita | 1,59 % | **2,42 %** (se concentra, no se reduce) |
| hematita | 0,90 % | **agotada** |
| wüstita | — | **3,61 %** |
| hierro metálico | — | **0,107 %** |
| aglomerado | no cuaja | **1,232 cm³ · 100 % del lecho** |

La secuencia es la que predice la termodinámica del proyecto: la hematita desaparece,
aparece wüstita con una traza de hierro metálico, y **la ilmenita no se reduce** —sólo
sube su fracción porque el volátil se ha ido—, exactamente lo que dice el CONTEXTO
raíz (haría falta 94,7 % de CO). Divergencia residual máxima 1,31×10⁻¹² s⁻¹. Pérdida
de masa sólida 28,14 %. Al final todo está a 1.172,2–1.173,2 K: equilibrio térmico.

La interfaz la toma sola por el directorio automático. Precarga de los 145 fotogramas:
~70 s y ~150 MB, 189 MB de heap; después reproduce a 30 fps sin peticiones HTTP.

### 12.10 El balance de masa cierra al 1,88 %, y se sabe por qué

El informe final dice `error relativo=1,876e-02` mientras los elementos cierran a
10⁻¹¹ mol. No es contradictorio: **el hidrógeno no está en el inventario elemental**
(`COMPOSICION_ELEMENTAL["H2"] = {}`, `H2O` sólo aporta O, `CH4` sólo aporta C).

La causa está aguas arriba, en una aproximación **documentada** de `simulacion_v3`: las
especies que el modelo no transporta se agregan a CH₄ conservando **carbono**, no masa
—alquitrán C₁₀H₈ → 10 CH₄ (8 H se vuelven 40), C₂H₄ → 2 CH₄, H₂S → H₂—. El resultado
es cuantificable: el reparto `_VOLATILES_MOL_POR_G` produce **1,0721 g de gas por cada
gramo de pseudoespecie volátil**, un 7,21 % de masa creada al devolatilizar.

Cuadra con lo observado: 0,2814 g de sólido perdido × 7,21 % = 20,3 mg, frente a los
19,1 mg de descuadre medido. **No es un defecto nuevo ni del simulador 3-D**: es el
precio conocido de agrupar el alquitrán en metano, y el balance elemental de C, O, Fe,
Ti y Si —lo que sí se rastrea— sigue cerrando a 10⁻¹¹ mol. Queda anotado porque el
informe imprime ese 1,88 % en cada corrida y merece explicación, no que se lo tome por
un error de integración. Corregirlo exigiría renormalizar el reparto de volátiles en
`simulacion_v3`, lo que movería la curva de pérdida de masa ya ajustada: es una
decisión de modelo, no una corrección de programación.

### 12.11 La pared del crisol figuraba como carbón coquizado

Al contrastar la corrida terminada con el dato experimental de §13 apareció otro
defecto, este de los datos guardados. `correr_simulacion._actualizar_cohesion` pasaba
`fraccion_carbon=0.75` **uniforme**, así que el campo de cohesión también coquizaba la
pared Ni-Cr, que es justamente lo primero en calentarse. A t = 720 s:

| etiqueta | celdas | c media | celdas c>0,5 |
|---|---:|---:|---:|
| **lecho** | 308 | 0,9996 | **308** ✔ |
| **pared** | 1.027 | 0,1803 | **185** ✘ |
| gas libre | 3.559 | 0,0002 | 0 |
| tapa | 342 | 0,0002 | 0 |

No se veía en la vista 3-D —la isosuperficie enmascara a lecho y gas— pero contaminaba
todo escalar recorrido sobre el campo entero: el `cohesion_max` de la serie y la media
que se compara con el ensayo. De hecho el **0,00506 de la tabla de §13 es la media
sobre TODO el dominio**; sobre el lecho, que es lo que significa, era 0,00061.

Corregido pasando la fracción de carbón enmascarada al lecho, que es la forma física de
decirlo: donde no hay carbón no hay coquización. Prueba añadida con pared más caliente
que el lecho: el lecho cohesiona por encima de 0,9 y fuera del lecho no queda ni una
celda por encima de 0,5.

La corrida se repitió con la máscara (2.691 s de reloj, 44,9 min). **La física es
idéntica** —mismos 5.235 pasos, misma divergencia 1,307×10⁻¹² s⁻¹, misma pérdida de
masa 28,141839 %—, porque la cohesión es diagnóstica y no realimenta al solucionador.
Lo que cambia es el campo guardado:

| etiqueta | c media (antes → después) | celdas c>0,5 |
|---|---|---:|
| lecho | 0,9996 → 0,9996 | 308 → **308** |
| pared | 0,1803 → **0,00018** | 185 → **0** |
| gas, tapa, exterior | 0,0002 → 0,0002 | 0 → 0 |

Crecimiento del aglomerado en la corrida definitiva, ya sólo sobre el lecho:

| t (s) | T lecho (K) | celdas cohesionadas | volumen (cm³) |
|---:|---:|---:|---:|
| 30 | 491,3 | **0** | 0 |
| 50 | 508,4 | 0 | 0 |
| 80 | 559,9 | 4 | 0,016 |
| 150 | 686,7 | 104 | 0,416 |
| 300 | 894,5 | 218 | 0,872 |
| 720 | 1172,2 | 308 | **1,232** |

**El criterio de falsación de §13 se sigue respetando**: a los 30 s el lecho está a
491,3 K y hay cero celdas cohesionadas.

### 12.12 Lo que queda

1. Empaquetado a ejecutable con PyInstaller.
2. `LineSegments2` para líneas gruesas (addon de three.js sin vendorizar).
3. A 900 °C el interior del crisol **emite**, y esa luz naranja tiñe las partículas:
   los colores de fase se leen bien en frío y en la leyenda, pero no en caliente. Es
   físicamente honesto, no un defecto; si se quisiera comparar fases en caliente habría
   que añadir un modo de iluminación neutra.

---

## 13. DATO EXPERIMENTAL DE VALIDACIÓN (aportado por el usuario)

**Observación de laboratorio:** «a los 30 segundos de calentarlo lo sacábamos de la
mufla y no había pasado nada».

Es el **primer y único dato experimental que restringe el modelo de cohesión**. Hasta
ahora toda predicción sobre el aglomerado iba rotulada como no validada, porque no
existe caracterización post-ensayo. Ésta es una restricción negativa, pero real.

### El modelo la respeta, sin haber sido ajustado para ello

Corrida real del solucionador 3-D (`resultados/simulacion_larga`):

Medido **sobre el lecho**, que es donde hay carbón. La versión anterior de esta tabla
promediaba sobre todo el dominio e incluía celdas de la pared del crisol, que no
contienen carga: véase §12.11.

| t (s) | T lecho (K) | cohesión media en el lecho | celdas de lecho con c>0,5 |
|---:|---:|---:|---:|
| 0 | 298,2 | 0,00000 | 0 |
| **30** | **491,3** | **0,00061** | **0** |
| 50 | 508,4 | 0,00270 | 0 |
| 80 | 559,9 | 0,02824 | 2 |
| 100 | 615,4 | 0,06071 | 18 |

Corregir el recuento **refuerza** el acuerdo con la observación: el aglomerado no
empieza a cuajar en el lecho hasta ~80 s, no a los 50 s. Las 2 celdas que la tabla
anterior daba a 50 s, y 11 de las 13 de 80 s, eran pared.

A los 30 s el lecho está a 491 K y la ventana termoplástica del carbón empieza a
623 K: el carbón ni siquiera ha comenzado a ablandarse. **Cero celdas cohesionadas.**
Sacarlo de la mufla en ese momento daría polvo suelto, que es lo observado.

Concuerda además con la Tabla 3: a 30 s la pérdida de masa medida fue **1,50 %**.

### Matiz físico importante — REVISADO, véase §14

La versión anterior de esta sección decía que el aglomerado **nace junto a la
pared**, porque el modelo mostraba el lecho 500 K por detrás del crisol. Eso era un
defecto de la ecuación de energía, no física: véase §14.2. Con las correcciones, el
crisol y su carga se calientan **juntos** y el gradiente dentro del lecho es de
decenas de kelvin, no de centenas. El aglomerado ya no nace en la pared.

### Uso correcto de este dato

Sirve para **falsar**, no para calibrar: cualquier ajuste de las constantes de
coquización que produjese aglomerado antes de los 30 s queda refutado por esta
observación. La prueba automática existe:
`tests/test_hinchamiento.py::test_a_los_30_segundos_no_hay_aglomerado`.

---

## 14. DATOS NUEVOS DEL LABORATORIO Y LO QUE DESTAPARON

El usuario aportó dos observaciones más del ensayo. Ninguna es una medición
instrumentada, pero las tres juntas —con la de los 30 s de §13— forman la
**única cronología experimental** que restringe el modelo:

| t (s) | Observado |
|---:|---|
| 30 | no ha pasado nada: al sacarlo, polvo suelto |
| **90** | **ya es más grande, crece y se hincha** |
| **120** | **ya está bien formado** |

Y un dato de caracterización: el carbón tiene **índice de hinchamiento IH 8**
(ensayo del botón en crisol, ASTM D720), pero **mezclado con la magnetita se
hincha menos**.

### 14.1 El modelo no reproducía esa cronología, y eso llevó al fondo del problema

Con la corrida anterior el lecho llegaba a 623 K —el inicio de la ventana
termoplástica— pasados **300 s**, así que a los 120 s no podía haber nada
formado. La tentación era recalibrar las constantes de coquización. No hacía
falta: el desajuste venía de la ecuación de energía.

### 14.2 Tres defectos en la ecuación de energía

**(a) El término advectivo estaba en forma de divergencia.** `div(uT)` es
`u·grad(T) + T·div(u)`. El segundo sumando no transporta nada: es dilatación.
Con fuente de masa —la devolatilización crea unos 19 cm³/s de gas **dentro** del
lecho— `div(u)` no es cero, y ese término restaba al lecho **−778 K/s de media,
hasta −2085 K/s** (medido sobre la instantánea de 30 s). El gas que aparece sale
del sólido a la temperatura local: no puede enfriar la celda que lo genera.

**(b) La advección transportaba con la capacidad térmica del bulto.** El término
va dividido por `rho·cp` de la celda, que en el lecho es la del conjunto
(6,9×10⁵ J/m³K), pero quien se mueve es el gas (1,4×10³). Equivalía a que el gas
arrastrase la entalpía del sólido, 500 veces mayor de la que puede llevar. Ahora
el coeficiente es `(rho·cp)_gas/(rho·cp)_ef`, que vale 1 en el gas libre y ~2×10⁻³
en el lecho: la formulación estándar de equilibrio térmico local en medio poroso.

**(c) La difusión promediaba difusividades, no conductividades.** El operador
usaba media armónica de `alpha` y aplicaba **el mismo coeficiente a las dos
celdas de cada cara**. Eso es correcto con `rho·cp` uniforme, pero en la interfaz
gas/metal el salto es de **3.100 veces**: la pared recibía en grados lo mismo que
el gas cedía, o sea 3.100 veces más energía de la que salía. El crisol se
calentaba en menos de un segundo en vez del minuto que le corresponde por sus
32,67 g de Ni-Cr. Ahora la cara promedia **conductividades** y cada fila se divide
por la capacidad de su celda; con propiedades uniformes es idéntica a la anterior,
así que las pruebas analíticas y de MMS no cambian.

Los tres se comprobaron por separado antes de tocar nada, y quedan fijados con
pruebas: `test_la_difusion_de_calor_conserva_energia_a_traves_de_una_interfaz`
(gas contra metal, error relativo < 10⁻¹⁰) y
`test_la_adveccion_de_calor_no_enfria_donde_se_genera_el_gas`.

**Los defectos (a) y (c) se compensaban parcialmente**: uno enfriaba el lecho y el
otro sobrecalentaba el crisol. Por eso la corrida anterior respetaba el dato de
los 30 s —por las razones equivocadas— y fallaba en los de 90 y 120 s.

### 14.3 Consecuencia: el aglomerado ya NO nace en la pared

Con la energía bien planteada, crisol, gas y lecho suben **juntos** (se
diferencian en decenas de kelvin, no en 500). El lecho ya no está frío mientras la
pared arde, así que la afirmación de §12.3 y §13 —«nace como 28 núcleos junto a la
pared caliente»— **queda retirada**: era consecuencia del retraso espurio. El
gradiente radial que queda es de signo contrario y de pocos kelvin: el periférico
está en contacto con la pared, que es masiva y va ligeramente por detrás del gas.

Se reescribió también la prueba que codificaba aquel retraso
(`test_tras_10_s_el_conjunto_se_calienta_sin_retrasos_espurios`): ahora exige que
nada supere a la mufla y que **no reaparezca** un desacople pared–lecho mayor de
150 K, que es la firma del defecto corregido.

### 14.4 Modelo de hinchamiento — `fisica/hinchamiento.py`

El hinchamiento no es ganancia de masa: es el gas de pirólisis atrapado en la masa
plástica, que la infla mientras el carbón está fluido y queda congelado al
resolidificar. Por eso sigue al **grado de plastificación** —la variable interna
irreversible que ya llevaba `cohesion.HistoriaTermica`— y no a la temperatura
instantánea.

| | Valor |
|---|---|
| Carbón solo, IH 8 | **×2,00** en volumen |
| Con 25 % de concentrado | **×1,65** |
| Atenuación por los inertes | a 65 % del hinchamiento del carbón solo |
| Altura del lecho | 3,26 mm → **5,38 mm** |
| Volumen | 1,36 cm³ → **2,24 cm³** |

Los inertes reducen por dos vías, agrupadas en un factor `(1-f_inerte)^n`: sólo
hincha la fracción carbonosa, y los granos minerales rompen la estructura de
burbujas. Los dos parámetros (expansión por unidad de IH y exponente del inerte)
están declarados **CALIBRABLES** con su rango, como los de `cohesion.py`: no hay
dilatometría de esta muestra.

La pared del crisol impide la expansión radial, así que el hinchamiento se
resuelve **hacia arriba**: la altura escala con todo el factor volumétrico. En la
interfaz la isosuperficie del aglomerado y las partículas se elevan con el campo
`hinchamiento`, que es nuevo en el contrato NPZ (opcional; las series anteriores
se cargan con unos).

### 14.5 Interfaz

- **Cada grano es de UNA fase, sorteada según la composición de su celda.** Antes
  cada partícula llevaba el color de la *mezcla* local, con lo que todos los granos
  salían del mismo gris promedio y las fases no se distinguían. Un lecho real es un
  mosaico: ahora se ve char negro, ceniza clara, hematita rojiza y magnetita gris,
  y cuando aparece un 0,1 % de hierro metálico aparecen **unos pocos granos
  brillantes**, en vez de aclararse todo imperceptiblemente. El sorteo es
  determinista por partícula, así que un grano no cambia de mineral entre
  fotogramas sin que cambie la composición.
- **Modo de luz neutra.** A 900 °C el interior del crisol emite y tiñe de naranja
  cada grano, justo cuando aparecen las fases nuevas. El conmutador apaga la
  emisión y blanquea las luces. No cambia ningún campo ni ningún número: va
  rotulado en el visor mientras está activo.
- **Se fueron los cuadros blancos.** `PointsMaterial` sin mapa dibuja cada punto
  como un **cuadrado** opaco; con `size: 7` llenaban la vista de cuadrados que no
  representaban nada. Ahora los trazadores de las líneas de corriente y los pulsos
  de radiación usan una textura radial con alfa: motas redondas.
- **Panel de hinchamiento**: factor actual, IH del carbón y atenuación por la
  magnetita, con su referencia.

### 14.6 El modelo reproduce la cronología SIN calibrar nada

Es el resultado que da valor a lo anterior. Las constantes no se tocaron —la
ventana plástica de van Krevelen, `TIEMPO_COQUIZACION_S = 15 s`, el IH 8— y aun
así, corregida la energía, la corrida cae sobre las tres observaciones:

| t (s) | T lecho (K) | hinchamiento | cohesión media | celdas c>0,5 | Observado |
|---:|---:|---:|---:|---:|---|
| 0 | 298,2 | ×1,000 | 0,000 | 0 / 308 | |
| **30** | **477,2** | **×1,000** | **0,000** | **0** | **nada, polvo suelto** ✔ |
| 60 | 629,0 | ×1,062 | 0,000 | 0 | entra en la ventana plástica |
| **90** | **757,8** | **×1,570** | 0,000 | 0 | **crece y se hincha** ✔ |
| **120** | **862,8** | **×1,589** | **0,576** | **306** | **bien formado** ✔ |
| 180 | 1017,7 | ×1,589 | 0,931 | 308 | |
| 300 | 1138,4 | ×1,589 | 0,998 | 308 | |
| 720 | 1172,0 | ×1,589 | 1,000 | 308 | equilibrio térmico |

La secuencia sale sola: el carbón entra en la ventana plástica hacia los 60 s y
**se hincha primero** —el gas atrapado infla la masa fluida—, y sólo cuando pasa
de 500 °C resolidifica y coquiza, que es lo que da la pieza cohesionada. Hinchar
antes que cuajar no se impuso: es lo que sale de la física, y coincide con lo que
se vio en el laboratorio.

Las tres observaciones están fijadas como pruebas en
`tests/test_hinchamiento.py`, y siguen sirviendo para **falsar**: cualquier
recalibración futura que forme el aglomerado antes de los 30 s, que no lo hinche
a los 90 o que no lo tenga formado a los 120 queda refutada.

### 14.7 La corrida definitiva

`resultados/simulacion_720s`, con la energía corregida y el campo de hinchamiento:
**145 instantáneas, t = 720,000 s exactos, en 2.238,6 s de reloj = 37,3 min**
(3,109 s de reloj por segundo simulado; 4.848 pasos). Es más rápida que la
anterior pese a resolver una física más viva, porque el paso ya no lo estrangula
un transitorio espurio.

| Diagnóstico | Antes de §14 | Ahora |
|---|---|---|
| Divergencia residual máxima | 1,31×10⁻¹² s⁻¹ | **8,37×10⁻¹³ s⁻¹** |
| Error elemental máximo | 1,97×10⁻¹¹ mol | **1,30×10⁻¹³ mol** |
| Pérdida de masa sólida | 28,142 % | 28,350 % |
| Estado térmico final | mufla 1172,15 / pared 1172,15 / lecho 1172,15 | mufla 1172,15 / pared **1171,48** / lecho **1172,01** |
| Lecho a 623 K (ventana plástica) | t ≈ 300 s | **t ≈ 58 s** |

El descuadre de masa del 1,88 % **no cambia**, como debe ser: es el hidrógeno de
§12.10, ajeno a la ecuación de energía.

Un detalle que ahora sí cuadra: al final la pared queda **0,5 K por debajo** del
lecho y de la mufla. Antes las tres coincidían en 1172,15 K exactos, que era la
firma del operador que creaba energía en la interfaz: el sólido se pegaba al
ambiente en vez de quedar ligeramente por detrás por su propia inercia y por la
pérdida hacia el exterior.

### 14.8 Lo que queda pendiente, y una aproximación que conviene tener presente

1. El exterior del crisol intercambia con la mufla por **conducción** a través de
   las celdas de gas que lo rodean, no por radiación. En ese hueco la radiación
   domina por un factor ~10 (h_rad ≈ 241 W/m²K frente a h_cond ≈ 25). Que la
   cronología resultante caiga sobre las tres observaciones sugiere que el camino
   térmico neto es aproximadamente correcto, pero **no es una validación de ese
   detalle**: si se modelara la radiación explícitamente habría que volver a
   comprobar los tres instantes.
2. Empaquetado a ejecutable con PyInstaller.
3. `LineSegments2` para líneas gruesas (addon de three.js sin vendorizar).
4. En `resultados/simulacion_720s` conviven **143 instantáneas de corridas
   anteriores (65,7 MB)** con las 145 vigentes, porque los instantes de guardado
   caen en microsegundos distintos y no se sobrescriben. Ni la interfaz ni las
   pruebas las mezclan —ambas toman sólo lo escrito a partir del t=0 más
   reciente—, pero convendría borrarlas.

---

## 15. LIMPIEZA Y CORRECCIONES DE LA ÚLTIMA RONDA

### 15.1 Se borraron 377 MB de detritos

| Qué | Tamaño |
|---|---:|
| `.pytest_tmp_pressure_full/` | 155 MB |
| `.pytest_cache/pytest-of-Alejandro/` | 78 MB |
| `interfazpytest-final/` (un `--basetemp` mal escrito) | 78 MB |
| 143 instantáneas de corridas superadas en `resultados/simulacion_720s/` | 66 MB |
| `interfaz/pytest-of-Alejandro/`, `tests/_tmp_pytest_*`, `__pycache__` | ~0,6 MB |

Ninguno estaba en git. Se añadió `.gitignore` para que no vuelvan a acumularse,
y también para las salidas del solucionador, que se regeneran con
`python correr_simulacion.py` y pesan decenas de MB. Los CSV y PNG de
verificación sí se conservan: son el respaldo de las tablas de este documento.

Se eliminó `fases_visuales.color_dominante`, que quedó sin uso al pasar al
sorteo de fase por grano (§14.5).

### 15.2 La conversión marcaba 97,6 % en t=0

`interfaz/app.py` calculaba la conversión de hematita como `1 − Fe2O3/5000`, con
5.000 mol/m³ escritos a mano. La hematita inicial de esta carga es **126 mol/m³**,
así que el campo «Conversión de Fe₂O₃» arrancaba en **97,6 %**: la interfaz decía
que la reducción estaba casi terminada antes de empezar.

Ahora la referencia es el campo de hematita de la primera instantánea de la serie,
**celda a celda**. En t=0 la conversión es exactamente 0 en todo el lecho.

### 15.3 Los números adimensionales eran de demostración

El panel mostraba Re, Ra, Pe y Da calculados por
`datos_sinteticos.numeros_adimensionales_sinteticos`, con densidad y viscosidad
fijas y un Damköhler que es literalmente `0,08 + 8,5·t/720`: una rampa. Iban
rotulados «Sintético», pero se mostraban también con datos reales.

`correr_simulacion` los calcula ahora con
`momentum.numeros_adimensionales` —las propiedades reales de cada celda— y los
guarda en los metadatos de la instantánea; `guardar_instantanea` conserva los
diagnósticos escalares que aporte quien la genera. La interfaz los usa si están y
rotula el origen: «Solucionador» o «Sintético».

### 15.4 La porosidad estaba congelada

`eps` se fijaba en 0,54 al cargar el caso y **no se actualizaba nunca**, mientras
el sólido perdía el 28 % de su masa por devolatilización. De la porosidad dependen
la permeabilidad de Kozeny–Carman, la conductividad efectiva del lecho y la
corrección de tortuosidad de las difusividades: las tres quedaban congeladas
también.

Ahora `caso.actualizar_porosidad` la recalcula en cada paso a partir del volumen
de sólido que queda —suma de `c_i·M_i/ρ_i` sobre las fases— y refresca la
permeabilidad. Se aplica el cambio **relativo** sobre la porosidad inicial
calibrada, no el valor absoluto: la fracción sólida que dan los volúmenes molares
(0,413) no coincide con la que declara el caso (0,46, de las densidades aparentes
del YAML), y con la razón esa discrepancia se cancela y en t=0 sale exactamente
0,54.

El hinchamiento **no** entra en la porosidad. Sobre malla fija el lecho no puede
crecer de volumen; lo que el inventario describe es el hueco que dejan los
volátiles al marcharse. La expansión se representa aparte, como campo de
diagnóstico y en la visualización.

**Umbral de actualización.** La primera versión reescribía la permeabilidad en
cada paso, y eso invalidaba **siempre** la caché de la factorización ILU del
solucionador viscoso, indexada por `(dt, nu, darcy)`: medido, la corrida se volvió
**diez veces más lenta**. La porosidad evoluciona en la escala de la
devolatilización —decenas de segundos—, así que sólo se aplica cuando ha cambiado
más de 5×10⁻³. Conserva la física y devuelve la caché.

---

## 16. CORRIDA DEFINITIVA Y CONTRASTE CUANTITATIVO

`resultados/simulacion_720s`: **145 instantáneas, t = 720,000 s, en 33,5 min**
(2,79 s de reloj por segundo simulado, 4.840 pasos). Divergencia residual
8,39×10⁻¹³ s⁻¹, error elemental 1,26×10⁻¹³ mol.

### 16.1 La reducción se detiene en wüstita, y el modelo lo obtiene solo

| t (s) | Fe₂O₃ | Fe₃O₄ | FeO | Fe |
|---:|---:|---:|---:|---:|
| 0 | 37.588 | 168.150 | 0 | 0 |
| 80 | **0** | 102.868 | 271.003 | 19,3 |
| 720 | 0 | 53.609 | 418.778 | **20,3** |

(mol/m³ sumados en el lecho). El hierro metálico se queda en el **0,004 % del
inventario de hierro**: no se forma. Es lo que predice el tamponamiento
CO/(CO+CO₂) = 0,00758 sobre la frontera Fe₃O₄/FeO (0,00757) del CONTEXTO raíz, y
el 3-D lo reproduce sin que se le impusiera.

**A partir de los 150 s no ocurre nada más**: los inventarios de 300 y 720 s son
idénticos a los de 150. Agotado el volátil no queda reductor y la gasificación del
char a 900 °C es despreciable (Da = 1,6×10⁻¹⁷).

### 16.2 La curva de pérdida de masa medida (8 puntos) — y lo que dice

La Tabla 3 del artículo tiene ocho puntos medidos entre 30 y 720 s. **No se
estaban usando para contrastar el 3-D.**

| t (s) | medida (%) | modelo (%) | diferencia |
|---:|---:|---:|---:|
| 30 | 1,50 | **1,18** | **−0,32** |
| 60 | 2,00 | 24,75 | **+22,75** |
| 90 | 19,25 ± 1,77 | 28,07 | +8,82 |
| 120 | 23,43 ± 0,28 | 28,26 | +4,83 |
| 150 | 24,83 ± 0,58 | 28,35 | +3,52 |
| 360 | 25,20 | 28,35 | +3,15 |
| 720 | 25,20 | 28,35 | +3,15 |

**El modelo devolatiliza unos 30 s antes de tiempo.** El ensayo pierde el 2 % a
los 60 s y salta al 19,25 % a los 90; el modelo ya va por el 24,75 % a los 60. La
devolatilización real ocurre entre 60 y 90 s y la del modelo entre 40 y 60.

Esto **matiza a la baja** lo dicho en §14.6: el modelo reproduce los hitos del
**aglomerado** (30 / 90 / 120 s) pero llega temprano al de la **masa**. Y apunta
en la misma dirección que el análisis del acoplamiento térmico: modelar la
radiación mufla → crisol explícitamente aceleraría el calentamiento y
**empeoraría** este contraste, no lo mejoraría.

La meseta del modelo queda 3,15 puntos alta: libera toda la materia volátil del
análisis próximo (35,30 % del carbón) y el ensayo se queda en 25,20 %.

Lo que sí reproduce es la **forma**: subida brusca y meseta. Ambos se congelan,
por la razón de §16.1.

Tres pruebas fijan el contraste en `tests/test_perdida_masa.py`.

### 16.3 Informe

El informe es **`informe/informe.pdf`** (LaTeX). Se reconstruye entero con
`python informe/construir.py`, que genera las figuras y las tablas de la corrida y
compila el documento. **Ninguna cifra del informe está escrita a mano**: todas
entran por `\input` desde `figuras_informe.py` y `tablas_informe.py`.

---

## 17. LA DEVOLATILIZACIÓN LLEGABA 30 s ANTES: LA COMPUERTA ESTABA MAL CENTRADA

El contraste de §16.2 dejó una discrepancia grande: el ensayo pierde el 2 % a los
60 s y el modelo el 24,75 %. Buscarle la causa costó dos hipótesis y tres
corridas.

### 17.1 Primera hipótesis, descartada: el acoplamiento térmico

Si el crisol se calentaba demasiado rápido, bastaría con frenarlo. Se introdujo un
**factor de vista** del crisol dentro de la mufla —la emisividad efectiva es
`emisividad × factor_vista`— y se calibró contra los ocho puntos.

Con factor 0,46 la curva de masa encajaba muy bien:

| t (s) | medido | vista 1,0 | vista 0,46 |
|---:|---:|---:|---:|
| 60 | 2,00 | 24,75 | **2,99** |
| 90 | 19,25 ± 1,77 | 28,07 | **20,45** |

**Pero rompía la cronología del aglomerado.** Con ese frenado, a los 120 s el
lecho está a 670 K y todavía no ha cuajado; a los 90 s ni siquiera ha empezado a
hincharse. Las dos observaciones de laboratorio quedaban incumplidas.

Ahí estaba la pista: si a los 90 s el ensayo ha perdido el 19 % de la masa **y**
se le ve hincharse, el carbón tiene que estar dentro de la ventana plástica
(350-500 °C) a los 90 s. El modelo frenado lo pone a 304 °C. El modelo sin frenar
lo pone a 491 °C, que es lo correcto. **El calentamiento no era el problema.**

### 17.2 La causa real: la compuerta de devolatilización

`modelo_multifase` evalúa la liberación de volátiles como
`sigmoide(T−273,15, T_inicio_C, 40)`, con `T_inicio_C = 200` °C. Pero ese valor
no es un umbral: es el **centro** de una sigmoide de 40 °C de ancho.

| T (°C) | compuerta con 200 | compuerta con 450 |
|---:|---:|---:|
| 200 | 0,500 | 0,002 |
| 300 | 0,924 | 0,023 |
| 358 | 0,981 | 0,091 |
| 450 | 0,998 | 0,500 |
| 491 | 0,999 | 0,736 |

Con 200 °C la compuerta está abierta al 98 % ya a 360 °C: el modelo suelta el
volátil **unos 100 °C por debajo** de donde lo suelta un carbón bituminoso. El
máximo de velocidad de devolatilización de un carbón coquizable está hacia
430-470 °C.

**Por qué no se notaba en la v3.** Su historia térmica 0-D estaba *calibrada* y
era mucho más lenta que la que resuelve el 3-D: entraba a 350 °C hacia los 90 s.
Con esa historia, una compuerta centrada en 200 °C daba la curva correcta. Otra
compensación entre dos errores, como las de §14.2.

### 17.3 Corrección

`quimica.T_devolatilizacion_C` en el caso, aplicado por
`caso.ajustar_devolatilizacion` al diccionario compartido con el 0-D, de modo que
ambos siguen usando el mismo valor y su consistencia se mantiene. Valor adoptado:
**450 °C**, de la temperatura de máxima velocidad de devolatilización de los
bituminosos, no ajustado punto a punto.

Se revirtió el factor de vista a 1,0: el acoplamiento térmico queda **sin
calibrar**, resuelto por la física.

Además se añadió el **calor de pirólisis** (endotérmico, 500 kJ/kg de volátil,
CALIBRABLE en 200-1400), que faltaba: el modelo trataba la devolatilización como
si liberar los volátiles fuese gratis.


---

## 18. PARA RETOMAR EL TRABAJO

> **Esta sección está SUPERADA por §19.** Se conserva porque describe el estado
> del que se partía. Para el estado actual, lea §19 y luego la lista de
> pendientes de §19.8.

### 18.1 Estado en esta parada

- Corrida definitiva de 720 s con la compuerta de devolatilización corregida
  (`quimica.T_devolatilizacion_C: 450`), `f_vol_liberable: 0,88`, calor de
  pirólisis, porosidad viva y los números adimensionales del solucionador.
- Informe LaTeX en `informe/informe.pdf`, 13 páginas, 10 figuras y 3 tablas, todo
  generado de la corrida con `python informe/construir.py`.
- 171 pruebas.

### 18.2 Lo primero que hay que hacer al retomar

1. **Comprobar que la corrida definitiva terminó** y regenerar el informe:
   ```
   python informe/construir.py
   python -m pytest tests/ -q
   ```
   Si la corrida quedó a medias, relanzarla:
   `python correr_simulacion.py --malla gruesa --salida resultados/simulacion_720s`

2. **Revisar el contraste de la curva de masa** (§16.2 y la sección homónima del
   informe). Es el número que mide si el modelo va bien.

### 18.3 Pendientes, por orden de valor

| # | Qué | Por qué |
|---|---|---|
| 1 | Recalibrar `simulacion_v3` con la compuerta a 450 °C | Su ajuste actual y la compuerta baja se sostienen mutuamente (§17). Es lo que desbloquea todo lo demás. |
| 2 | Malla media hasta 130 s (~5 h) | Extendería la independencia de malla a la ventana plástica, que es donde pasa todo. |
| 3 | Radiación mufla→crisol con factores de vista geométricos | Ahora es conducción por el gas circundante. |
| 4 | Empaquetado con PyInstaller | Un `.exe` que no exija Python instalado. |
| 5 | Cierre de masa del 1,88 % (hidrógeno del alquitrán agrupado en CH4) | Decisión de modelo: renormalizar movería la curva ya ajustada. |
| 6 | `LineSegments2` para líneas gruesas en la interfaz | Cosmético. |

Nada de esto es urgente frente a lo que falta del **laboratorio**: sin
caracterización post-ensayo, las 47 fases predichas siguen siendo predicción.

---

## 19. LA PRUEBA DEL IMÁN, Y LOS DOS DEFECTOS TERMOQUÍMICOS QUE DESTAPÓ

### 19.1 El dato nuevo

El usuario aportó una observación más, y de procedimiento:

| Observado | |
|---|---|
| El aglomerado **se pega al imán** | |
| Cuanto **más tiempo en la mufla, más débil** el magnetismo | |
| **Al final del ensayo todavía se pega**, sólo que más flojo | |
| Se saca el crisol y **se deja enfriar al ambiente** | procedimiento |

Es cualitativa —un imán contra una muestra, no un magnetómetro—, pero es el
**único dato experimental que restringe la composición de fases del producto**:
toda la caracterización disponible es del material inicial.

### 19.2 Qué decía el modelo, medido antes de tocar nada

Calculando la magnetización de saturación a temperatura ambiente del inventario
de fases de la corrida vigente (magnetita 92 A m²/kg, hierro 218, hematita 0,4,
wüstita e ilmenita 0; Hunt, Moskowitz y Banerjee 1995, Tabla 3):

| t (s) | 0 | 30 | 60 | 90 | 120 | 150 | 720 |
|---|---|---|---|---|---|---|---|
| M/M₀ | 1,000 | 1,007 | **1,109** | 0,181 | **0,003** | 0,028 | 0,028 |
| Fe₃O₄ | 100 % | 100 % | 107 % | 13 % | 0,0008 % | **0** | **0** |

**El modelo quedaba falsado.** Consumía toda la magnetita hacia los 150 s y
dejaba un aglomerado prácticamente no magnético, cuando en el laboratorio sigue
respondiendo al imán a los doce minutos. Y producía 1,06 % de hierro metálico,
no el 0,004 % que declaraban §16.1 y el CONTEXTO raíz — esas tablas eran de una
corrida anterior a la corrección de la compuerta de §17 y estaban obsoletas.

### 19.3 La causa: la frontera Fe₃O₄/FeO estaba mal por un factor 42

El gas del lecho está en CO/(CO+CO₂) = 0,49–0,72 y `termodinamica_ext` ponía la
frontera de reducción de la magnetita en **0,00757**. Con esa frontera cualquier
gas la reduce entera. La tabla NIST-JANAF —que es la fuente que el propio módulo
declara— la pone en **0,3222** a 900 °C, de acuerdo con el diagrama clásico de
Baur–Glaessner.

Eran **dos** defectos de datos, y los dos empujaban en el mismo sentido:

**(a) El Cp de Fe₃O₄ extrapolado fuera de su rango.** Era un Maier–Kelley de
rango bajo convertido a Shomate, `Cp = 111,8 + 0,106 T`, extendido hasta 1400 K.
A 1173 K daba 236 J/mol/K. JANAF (tabla Fe-032) tiene una **transición lambda
magnética en 900 K** por encima de la cual Cp = 200,832 J/mol/K constante hasta
1870 K. Corregido con un ajuste Maier–Kelley a los ocho puntos de JANAF por
debajo de 900 K (error máximo 0,23 J/mol/K) y el valor exacto por encima: las
integrales de H y S cierran contra JANAF a 0,01 kJ y 0,008 J/K.

**(b) FeO estequiométrico en lugar de wüstita.** La fase que de verdad coexiste
con la magnetita es Fe(0,947)O (JANAF Fe-001), bastante menos estable. Con FeO
estequiométrico la frontera sale en 0,052 en vez de 0,32: seis veces menos CO
del necesario. Se añadió la especie `FeO_wustita`, que es el Shomate de FeO
desplazado en ΔH = +8,2493 kJ/mol y ΔS = +1,0222 J/mol/K, y **sólo** las seis
reacciones del par magnetita/wüstita/hierro la usan. Las reacciones en que el
FeO entra como componente de otro compuesto (fayalita, ulvöspinela, hercinita)
siguen con el FeO estequiométrico, porque los datos de esas fases están
referidos a él: la ΔG de formación de la fayalita se queda en los −8,2722 kJ de
siempre.

**El desplazamiento no se ajustó contra el ensayo**: el objetivo era tabla
primaria. Y se valida contra dos cosas que **no** entraron en el ajuste:

| | modelo | JANAF |
|---|---|---|
| Frontera FeO/Fe a 900 °C | 0,7091 | 0,6843 |
| Eutectoide de la wüstita | **613,7 °C** | **615,6 °C** |

### 19.4 Y el «invariante robusto» no era independiente

`calibracion.PARAMETROS_RESTRINGIDOS_FISICA` fija `k_magnetita = 0,20` con este
comentario: *«seleccionado en la rama que conserva Fe3O4 a 720 s y reproduce
CO/(CO+CO2)=0,00757»*. O sea que la constante se eligió para que el gas cayera
sobre una frontera equivocada, y el resultado se citaba después como invariante
termodinámico. Es el **cuarto** caso del patrón de dos errores que se compensan.

Corregida la termodinámica, con los parámetros **por omisión** el 0-D da
CO/(CO+CO₂) = 0,3227 frente a la frontera 0,3222, la magnetita **gana** un 8,6 %
al heredar el hierro de la hematita, y la conversión de ilmenita sigue siendo
exactamente cero. El enunciado se sostiene; el número era falso.

### 19.5 El enfriamiento: ¿es un temple?

Lo preguntó el laboratorio y hay que responderlo con números, porque de eso
depende cómo se lee el imán. Por debajo de 570 °C la wüstita se descompone,
4 FeO → Fe₃O₄ + Fe, y **los dos productos sí son magnéticos**:

| | A m² por mol de Fe |
|---|---|
| magnetita de partida | 7,10 |
| eutectoide completo | **8,37** |

Es decir: **si el enfriamiento fuese lento, el aglomerado saldría más magnético
que al empezar**, no menos. Como se observa lo contrario, la propia prueba del
imán exige que la wüstita se conserve en buena parte. Eso es información nueva
sobre el ensayo, obtenida sin ningún instrumento.

`fisica/magnetismo.py` resuelve la curva por capacidad concentrada con radiación
(ε = 0,80) y convección natural:

| | crisol + tapa (48,5 g) | aglomerado solo (0,72 g) |
|---|---|---|
| Biot | 0,011 | 0,20 |
| velocidad a 900 °C | 984 °C/min | 11.617 °C/min |
| **velocidad al cruzar 570 °C** | **304 °C/min** | **3.788 °C/min** |
| tiempo en la ventana 570–400 °C | 49,8 s | 3,9 s |
| veredicto | **temple parcial** | temple |

El umbral publicado para suprimir la transformación eutectoide por encima de
700 °C es **1000 °C/min** (Zorc, Nagode y Kosec, *High Temperature Corrosion of
Materials*, 2024). Sacar el crisol entero se queda tres veces por debajo: es
rápido para un horno y lento para un temple. Por eso el observable se reporta
como **banda entre dos cotas** y no como número único.

**Consecuencia práctica para el laboratorio**: si interesa que el imán lea el
estado que había a 900 °C, basta con volcar el aglomerado fuera del crisol al
sacarlo. Sin el crisol se enfría dos órdenes de magnitud más rápido y sí es un
temple.

### 19.6 Lo que se añadió

- `fisica/magnetismo.py`: M_s por fase con su referencia, temperaturas de orden,
  mezcla lineal en masa, las dos cotas del enfriado, curva de enfriamiento y
  veredicto de temple. La titanohematita se declara **CALIBRABLE** en
  [0,4 – 10] A m²/kg porque su x = 0,49 cae justo sobre la transición de la
  serie (y ≈ 0,45, Hunt Fig. 9) y el DRX no resuelve el orden Fe/Ti; da igual,
  es el 0,5 % del total, y eso también se dice.
- `tests/test_magnetismo.py`: dos falsadores contra la corrida real —al final
  responde al imán, y decrece— más las propiedades del modelo puro.
- Panel «Prueba del imán» en el visor, con las dos cotas y el veredicto del
  enfriado. El cliente no codifica ningún dato magnético: los recibe en
  `/api/config`.
- Figura y macros del informe.

### 19.7 De paso: la corrida del informe y la del sitio eran distintas

`informe/construir.py` apuntaba por omisión a `resultados/simulacion_720s`, una
carpeta con **328 NPZ de dos corridas mezcladas**, mientras el sitio publicaba
`simulacion_720s_completa`. Además los tres scripts del informe reimplementaban
a mano el recorte a la corrida vigente. El recorte pasó a `nucleo/salida.py`
(`recortar_a_la_corrida_vigente` y `serie_vigente`) y ahora lo usan los cuatro
consumidores: visor, selector, informe y pruebas. El informe elige por omisión
la corrida más avanzada, igual que el visor.

### 19.8 Pendientes, por orden de valor

| # | Qué | Por qué |
|---|---|---|
| 1 | **Recalibrar `simulacion_v3`** con la compuerta a 450 °C y la termodinámica corregida | `k_magnetita = 0,20` se eligió contra una frontera equivocada (§19.4) y ya no puede citarse como restricción física. Es lo que desbloquea todo lo demás. |
| 2 | **VSM del aglomerado** a varios tiempos | Convertiría la prueba del imán de cualitativa en medida y fijaría el reparto Fe3O4/FeO directamente. Es lo más barato del laboratorio y lo que más rinde. |
| 3 | Registrar la curva de enfriamiento real | El modelo dice temple parcial (304 °C/min contra un umbral de 1000). Con un termopar se cerraría la banda de §19.5. |
| 4 | Malla media hasta 130 s (~5 h) | Extendería la independencia de malla a la ventana plástica. |
| 5 | Radiación mufla→crisol con factores de vista geométricos | Ahora es conducción por el gas circundante. |
| 6 | Actividad del Fe2O3 dentro de la titanohematita | Medida y declarada despreciable para este par (mueve la frontera de 2,3e-5 a 1,8e-4, tres órdenes por debajo del gas), pero sin implementar. |
| 7 | Empaquetado con PyInstaller | Un `.exe` que no exija Python instalado. |
| 8 | Cierre de masa del 1,88 % (hidrógeno del alquitrán agrupado en CH4) | Decisión de modelo: renormalizar movería la curva ya ajustada. |
| 9 | `LineSegments2` para líneas gruesas en la interfaz | Cosmético. |

Nada de esto es urgente frente a lo que falta del **laboratorio**. La prueba del
imán acaba de demostrar hasta qué punto: una observación cualitativa, sin
instrumento, destapó dos defectos termoquímicos que llevaban escondidos desde el
principio y que ningún ajuste de la curva de masa habría revelado.

### 19.9 LA DISCREPANCIA QUE QUEDA ABIERTA

Es lo más útil de todo el capítulo, y no se ha tocado.

Con la termodinámica corregida el modelo **ya no queda falsado**: al final del
ensayo sigue habiendo magnetita (97,2 % de la inicial) y el aglomerado responde
al imán. Pero el modelo **se congela hacia los 150 s**: agotado el volátil no
queda reductor, la gasificación del char está prácticamente apagada
(Da ≈ 1,6×10⁻¹⁷) y ninguna fase se mueve más. Su predicción es que un aglomerado
sacado a los 300 s y otro a los 720 s responden al imán **igual**.

El laboratorio dice que **sigue** perdiendo capacidad magnética con el tiempo.
Eso no se reproduce.

Trazas del modelo (cota de temple, sobre el lecho):

| t (s) | Fe₃O₄ (µmol) | FeO (µmol) | Fe (% del Fe) | M (A m²/kg) | momento (mA m²) | masa (g) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 731,1 | 0 | 0 | 17,77 | 15,66 | 0,881 |
| 60 | 838,2 | 0 | 0 | 21,03 | 17,93 | 0,853 |
| 100 | 785,4 | 158,9 | 0,48 | **25,39** | 16,95 | 0,668 |
| 140 | 710,9 | 353,4 | 1,62 | 23,60 | 15,72 | 0,666 |
| ≥150 | **710,8** | **353,7** | **1,62** | **23,59** | **15,72** | 0,666 |

Dos lecturas hay que hacer a la vez, y por eso se reportan las dos:

- La **magnetización específica** sube un 33 % respecto del inicio. Gran parte de
  esa subida **no es química de hierro**: el lecho pierde el 24 % de su masa al
  devolatilizarse, y eso sube el cociente sin que cambie ninguna fase.
- El **momento total** empieza en 15,66 mA m², sube a 17,93 (hematita → magnetita)
  y vuelve a 15,72. En neto, la química del hierro es casi un empate.

**A dónde apunta la discrepancia.** Para que la magnetita siga reduciéndose
después de los 150 s tiene que seguir habiendo CO, y el único carbono disponible
es el char, que sigue ahí (24.000 mol/m³ de media) junto con CO₂ a 900 °C, donde
el equilibrio de Boudouard favorece fuertemente al CO. Lo que falta es cinética:
`k_boudouard` es **uno de los parámetros que `calibracion.py` declara NO
identificables** con la curva de pérdida de masa.

> **La prueba del imán sí podría identificarlo.** Sería el primer observable del
> proyecto que restringe `k_boudouard`. Una curva de magnetización de saturación
> contra tiempo —un VSM de una tarde— lo fijaría.

No se ha ajustado. Ajustar `k_boudouard` hasta que la curva baje sería
exactamente el error que este proyecto ya ha pagado cuatro veces. Queda como
`tests/test_magnetismo.py::test_el_magnetismo_sigue_bajando_despues_de_los_150_segundos`,
marcada `xfail(strict=True)`: avisará el día en que el modelo la reproduzca.

### 19.10 QUÉ SE RECUPERA: LAS FASES A TEMPERATURA AMBIENTE

Faltaba, y era una laguna real: todo lo que reportaba el informe era el estado
**dentro** de la mufla. Lo que el laboratorio tiene en la mano es otra cosa.

Entre los 900 °C y la mesa hay **una sola** transformación accesible: por debajo
de 570 °C la wüstita deja de ser estable y se descompone,
4 FeO → Fe₃O₄ + Fe. La ilmenita, la titanohematita, el char y las cenizas no
tienen ninguna ruta, así que bajan congelados.

`fisica/magnetismo.py` añade:

- `fraccion_eutectoide(tiempo_en_ventana_s)`, ley de Avrami **CALIBRABLE**
  (τ = 92,3 s, n = 0,735) anclada en las dos únicas fracciones medidas que se
  encontraron: Zorc, Nagode y Kosec (2024) dan 0,17 de wüstita retenida
  enfriando a 100 °C/min y 0,41 a 1000 °C/min. El eslabón débil es el supuesto
  de cuánta wüstita había en sus probetas (0,50 de la capa), y por eso **el
  resultado que se defiende son las dos cotas**, no este número.
- `fases_tras_enfriar()` y `masas_tras_enfriar_g()`, que devuelven el inventario
  completo a temperatura ambiente, con las fases ausentes en cero explícito.
- `NO_MODELADO_AL_ENFRIAR`, con las tres cosas que quedan fuera y en qué
  dirección tira cada una.

Los dos informes llevan ahora **dos tablas por tiempo de extracción**
(0, 30, 60, 90, 120, 150, 210, 360 y 720 s):

| | contenido |
|---|---|
| `tabla_fen_fases_tiempo` | inventario de fases **en la mufla**, en mg |
| `tabla_fen_enfriado` | lo que se **recupera en frío**, Fe₃O₄/FeO/Fe y M_s, con las dos cotas |

Y la tabla de estado de `informe.tex` pasa a llevar también la cohesión y la
temperatura en °C junto a la porosidad, el hinchamiento y la pérdida.

Lo que se recupera, en las dos cotas (mg):

| t (s) | Fe₃O₄ temple → lento | FeO temple → lento | Fe temple → lento | M_s temple → lento |
|---:|---|---|---|---|
| 90 | 210,4 → 218,4 | 10,0 → 0 | 0,19 → 2,13 | 25,7 → 27,2 |
| 120 | 192,9 → 211,1 | 22,6 → 0 | 3,07 → 7,46 | 24,5 → 27,9 |
| ≥150 | 178,7 → 205,0 | 32,6 → 0 | 5,50 → 11,84 | 23,5 → 28,5 |

Con la ventana de 49,8 s del crisol, la estimación central es que se
descompondría el **47 %** de la wüstita: a mitad de camino entre las dos
columnas. Volcando el aglomerado solo, un 9 %.

**Lo que el enfriado hace y esto no recoge**, declarado en el informe: la
reoxidación en aire de la superficie (restaría magnetismo), la combustión del
char al salir incandescente (subiría la magnetización específica sin cambiar
ninguna fase de hierro) y los gradientes internos (la superficie cruza el
eutectoide antes que el núcleo). Las tres se cierran con lo mismo: un DRX del
aglomerado recuperado a dos o tres tiempos.

### 19.11 CON TAPA O SIN TAPA: QUÉ ENFRIADO DEJA EL AGLOMERADO MÁS MAGNÉTICO

El usuario aportó el **objetivo del material**: se va a usar como adsorbente y
después hay que recuperarlo con un imán. Eso cambia la pregunta. No se trata de
conservar el estado de la mufla, sino de **maximizar la respuesta al imán**, y
para ese objetivo la respuesta se invierte.

> Todo lo que devuelve wüstita a magnetita **suma** magnetismo, porque la
> wüstita no responde al imán y la magnetita sí. Lo único que resta es el aire,
> que oxida magnetita a hematita. Conviene enfriar **despacio y con la tapa
> puesta**, justo lo contrario de lo que haría falta para fotografiar el estado
> de la mufla.

Con la tapa puesta pasan tres cosas distintas, y no hay que mezclarlas:

1. **Desaparece la reoxidación por aire** y la combustión del char: dos de los
   tres límites de §19.10 se van.
2. **Aparece otra reoxidación, la del propio gas encerrado, y juega a favor.**
   La frontera Fe₃O₄/FeO *sube* al bajar la temperatura —0,322 a 900 °C, 0,456 a
   700, 0,589 en el eutectoide— mientras el gas se queda donde estaba. Un gas
   que a 900 °C estaba sobre la frontera pasa a ser oxidante al enfriarse:
   3 FeO + CO₂ → Fe₃O₄ + CO. El tope es el inventario de CO₂ del crisol, que
   alcanza para el **22 %** de la wüstita, y el proceso se autolimita porque al
   oxidar sube x_CO hasta reencontrar la frontera.
3. **Térmicamente la tapa frena poco**: son 15,87 g de 48,54.

| Ruta | ventana (s) | eutect. | Fe₃O₄ | FeO | Fe | Fe₂O₃ | M_s | rel. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Volcado fuera del crisol | 4 | 9 % | 172,1 | 29,6 | 6,09 | 9,4 | **22,9** | 100 % |
| Al aire, sin tapa | 34 | 38 % | 179,2 | 20,3 | 7,90 | 9,8 | **24,2** | 106 % |
| Al aire, con tapa | 50 | 47 % | 196,1 | 13,5 | 7,83 | 0 | **26,3** | 115 % |
| En la mufla apagada, con tapa | 1275 | 100 % | 206,9 | 0 | 10,44 | 0 | **28,3** | 124 % |

(masas en mg, M_s en A m²/kg)

**El contrapeso, declarado**: la ruta más magnética es también la que deja más
**hierro metálico**, 10,4 mg. Es la fase menos estable en medio acuoso —se
corroe, pierde magnetismo con el tiempo y suelta Fe²⁺ al sistema—, así que si la
adsorción es en agua parte de la ganancia se perderá y además introduce una
variable nueva. «Al aire, con tapa» conserva casi todo el beneficio con la mitad
de hierro metálico.

**Qué tan firme es.** El *orden* de las cuatro rutas es robusto: sale de que la
wüstita vale 0 y la magnetita 92 A m²/kg, que es dato de tabla. Los valores no:
dependen de la cinética del eutectoide, anclada en dos puntos de una referencia
sobre capas de óxido en acero, y de una fracción de oxidación al aire que es un
orden de magnitud declarado (CALIBRABLE, 0-20 %). El cuadro sostiene la
dirección y el tamaño del efecto, no la tercera cifra.

**Y medirlo cuesta poco**: dos aglomerados del mismo tiempo de mufla, uno
enfriado tapado y otro destapado, y un VSM.
