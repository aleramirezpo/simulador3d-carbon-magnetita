# Simulador 3-D de transporte y reacción en aglomerados carbón–magnetita

Simulación 3-D de momentum, calor, masa y química de un aglomerado de carbón
(0,75 g) con concentrado de magnetita (0,25 g) calentado en mufla a 900 °C, y
visualizador interactivo de la corrida.

**Visor en línea:** https://aleramirezpo.github.io/simulador3d-carbon-magnetita/

Se abre en cualquier PC con un navegador moderno: no hace falta Python, ni
instalar nada, ni tener los resultados en disco. La primera carga baja la serie
completa de campos en Float32 y a partir de ahí la reproducción es local, a
30 fps interpolados y sin una sola petición más.

Lo que viaja son unos 190 KB por fotograma: el Float32 crudo ocupa 1,1 MB pero
GitHub Pages lo sirve con gzip y comprime 5,7 veces. `sitio/manifiesto.json`
declara el peso en disco y el peso real de la descarga.

Trabajo experimental base: A. Ramírez Polo, S. Puerta Araque, G. Neira Arenas
(Universidad Nacional de Colombia, sede Medellín — Laboratorio de Carbones).

---

## Aviso sobre el alcance de lo que se ve

Todo lo que muestra el visor es **predicción del modelo, sin validación
experimental de fases**. Conviene decirlo con precisión, porque el proyecto ha
sido cuidadoso en separar lo medido de lo calculado:

- El concentrado **no es magnetita pura**: por DRX-Rietveld de la fracción
  magnética (REF-M, Rwp 8,38 %) es 76,85 % Fe₃O₄ + 23,15 % de una
  **titanohematita** R-3, miembro intermedio de la serie hematita–ilmenita con
  x(FeTiO₃) = 0,49. El modelo la descompone en sus dos extremos: 12,10 % Fe₂O₃
  + 11,05 % FeTiO₃. **No hay hematita libre ni cuarzo** — están en la fracción
  no magnética, que no entra al ensayo.
- La pérdida de masa medida **no mide la reducción**: el 95 % es
  devolatilización del carbón y sólo el 2,2 % es oxígeno de los óxidos.
- El grado de reducción **no es identificable** con los 8 puntos
  experimentales disponibles: α varía entre 0,037 y 0,171 sin que empeore el
  ajuste. Un R² alto sobre la pérdida de masa no valida ninguna afirmación
  sobre fases.
- Lo robusto es termodinámico, no ajustado: el gas se tampona **sobre la
  frontera Fe₃O₄/FeO** y la ilmenita **no se reduce** a 900 °C (requeriría
  94,7 % de CO). La frontera está en CO/(CO+CO₂) = 0,3222 a 900 °C según
  NIST-JANAF; el valor de 0,00757 que se citaba antes venía de dos defectos de
  datos termoquímicos ya corregidos.
- El aglomerado **sigue respondiendo al imán al final del ensayo, más débil**.
  Es la única restricción experimental sobre las fases del producto, y el visor
  la muestra en el panel «Prueba del imán» con las dos cotas que impone el
  enfriado.

Toda la caracterización disponible es del material **inicial**, así que ninguna
fase predicha tiene contraste experimental. Falta, sobre todo, DRX/Mössbauer/SEM
del aglomerado **después** del ensayo.

## Qué muestra el visor

- Campo volumétrico por ray marching: temperatura, presión, velocidad,
  concentración de especies gaseosas, dominancia advectiva/difusiva.
- El lecho como partículas discretas coloreadas por fase mineral, con el color
  real de cada mineral y conmutador de luz neutra para poder compararlas.
- Geometría real del crisol (perfil de revolución con collar y tapa).
- Números adimensionales calculados por el solucionador celda a celda, junto a
  los valores de referencia del ensayo.
- Cortes, perfiles trazados sobre la vista 3-D y lectura local por raycaster.

## Estructura

| Directorio | Contenido |
|---|---|
| `nucleo/` | Malla, geometría del crisol, solucionadores, formato de salida |
| `fisica/` | Química, termodinámica, cohesión, hinchamiento, paleta de fases |
| `casos/` | Definición del caso `carbon_magnetita.yaml` |
| `interfaz/` | Servidor local, cliente 3-D (`web/`) y exportador estático |
| `verificacion/` | MMS, convergencia, consistencia contra el modelo 0-D |
| `tests/` | Suite de pruebas (Python + módulos del navegador en Node) |
| `informe/` | Informe LaTeX; ninguna cifra está escrita a mano |
| `sitio/` | Sitio estático publicado (generado, no editar a mano) |
| `CONTEXTO.md` | Bitácora técnica del proyecto — empezar por aquí |

## Ejecución local

```bash
python correr_simulacion.py        # corrida completa de 720 s (~35-60 min)
python -m interfaz.app             # interfaz 3-D contra los resultados en disco
python -m pytest tests/ -v         # suite completa
python informe/construir.py        # regenera figuras, tablas y el PDF
```

Los NPZ del solucionador (`resultados/simulacion*/`) no se versionan: pesan
cientos de MB y se regeneran con `correr_simulacion.py`.

## Cómo se publica el visor

`sitio/` se genera congelando las respuestas del servidor local en archivos:

```bash
python -m interfaz.exportar_estatico --salida sitio --limpiar
```

El exportador llama a las **mismas** funciones que sirven la API local, así que
no hay una segunda implementación que se pueda desviar; sólo cambia el
transporte:

| Servidor local | Sitio estático |
|---|---|
| `/api/config` | `datos/config.json` |
| `/api/fotograma?indice=N` | `datos/fotograma_NNNN.bin` (S3DF, bytes idénticos) |
| `/api/lineas?indice=N` | `datos/lineas_NNNN.json` |
| `/api/serie` | `datos/serie.json` |

El cliente resuelve unas u otras en `interfaz/web/js/rutas.js`, según el
`index.html` declare o no `window.SIMULADOR3D_ESTATICO`. Hay una prueba
(`test_el_sitio_estatico_sirve_los_mismos_bytes_que_la_api`) que compara byte a
byte el fotograma exportado con el que entrega el servidor.

La interfaz no pide nada a ningún CDN: three.js está vendorizado y hay una
prueba que lo comprueba.

## Corrida publicada

La que abre por omisión la interfaz local: la más avanzada y coherente de
`resultados/`. El manifiesto del sitio (`sitio/manifiesto.json`) declara cuál
es, cuántos fotogramas trae y hasta qué tiempo físico llega.

## Referencias

Hammam 2022 (energía de activación de la reducción) · Kiamehr 2017 (k_ef) ·
Chase 1998 (NIST-JANAF) · van Krevelen 1993 (ventana termoplástica y
carbonización). La bibliografía de ilmenita y titanomagnetita está fichada en
`BIBLIOGRAFIA.md` del modelo 0-D.
