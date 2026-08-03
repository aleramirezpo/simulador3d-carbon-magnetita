/**
 * Decodifica un fotograma S3DF con el decodificador REAL del cliente y
 * escribe un resumen JSON por stdout, para que `tests/test_interfaz.py` lo
 * compare con los arreglos numpy de origen. Así el contrato binario se
 * verifica de extremo a extremo y no sólo por inspección del formato.
 *
 * Uso: node tests/js/decodificar_s3df.mjs <archivo.bin>
 */
import { readFileSync } from 'node:fs';
import { decodificarFotogramaBinario, compactarFotogramaFloat32 } from '../../interfaz/web/js/frame-cache.js';

const ruta = process.argv[2];
if (!ruta) {
  console.error('falta la ruta del fotograma binario');
  process.exit(2);
}
const bytes = readFileSync(ruta);
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const fotograma = compactarFotogramaFloat32(decodificarFotogramaBinario(buffer));

const resumen = (valores) => ({
  n: valores.length,
  primero: valores[0],
  ultimo: valores[valores.length - 1],
  suma: valores.reduce((a, b) => a + b, 0),
});

const campos = {};
for (const nombre of ['x', 'y', 'z', 'u', 'v', 'w', 'P', 'T', 'eps', 'cohesion', 'conversion', 'reduccion', 'etiquetas']) {
  campos[nombre] = resumen(fotograma[nombre]);
}
for (const grupo of ['c_especies', 'solido']) {
  for (const [nombre, valores] of Object.entries(fotograma[grupo] || {})) {
    campos[`${grupo}.${nombre}`] = resumen(valores);
  }
}
process.stdout.write(JSON.stringify({
  t: fotograma.t,
  forma: fotograma.forma,
  tieneTermico: Boolean(fotograma.termico),
  tieneNumeros: Boolean(fotograma.numeros),
  campos,
}));
