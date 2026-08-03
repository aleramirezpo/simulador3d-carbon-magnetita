/**
 * Pruebas del extractor de isosuperficies, ejecutadas en Node.
 *
 * Cubre el caso que rompía la geometría en el borde del lecho: las esquinas
 * fuera de la máscara valen -Infinity y la interpolación daba Inf/Inf = NaN.
 *
 * Uso: node tests/js/prueba_marching_cubes.mjs
 */
import assert from 'node:assert/strict';
import { extraerIsosuperficieMarchingCubes, suavizarCampoGaussiano } from '../../interfaz/web/js/marching-cubes.js';

const FORMA = [6, 6, 6];
const CELDAS = 6 * 6 * 6;
const ejes = { x: [], y: [], z: [] };
for (let i = 0; i < 6; i += 1) { ejes.x.push(i); ejes.y.push(i); ejes.z.push(i); }
const indice3 = (i, j, k) => (i * FORMA[1] + j) * FORMA[2] + k;

const pruebas = [];
const prueba = (nombre, fn) => pruebas.push([nombre, fn]);

function campoYMascara({ interiorDesde = 1, interiorHasta = 4, valorDentro = 1 } = {}) {
  const valores = new Float32Array(CELDAS);
  const etiquetas = new Uint8Array(CELDAS);   // 0 = pared, 3 = lecho
  for (let i = 0; i < 6; i += 1) for (let j = 0; j < 6; j += 1) for (let k = 0; k < 6; k += 1) {
    const dentro = i >= interiorDesde && i <= interiorHasta
      && j >= interiorDesde && j <= interiorHasta
      && k >= interiorDesde && k <= interiorHasta;
    const q = indice3(i, j, k);
    etiquetas[q] = dentro ? 3 : 0;
    valores[q] = dentro ? valorDentro : 0;
  }
  return { valores, etiquetas };
}

prueba('el borde de la máscara no produce vértices NaN', () => {
  // Todo el interior por encima del umbral: la superficie cae justo sobre la
  // frontera de la máscara, que es donde aparecía el NaN.
  const { valores, etiquetas } = campoYMascara({ valorDentro: 1 });
  const geometria = extraerIsosuperficieMarchingCubes(valores, FORMA, ejes, 0.5, etiquetas);
  const posiciones = geometria.getAttribute('position').array;
  assert.ok(posiciones.length > 0, 'debería generarse superficie');
  const nan = [...posiciones].filter((v) => !Number.isFinite(v));
  assert.equal(nan.length, 0, `${nan.length} coordenadas no finitas de ${posiciones.length}`);
  assert.ok(Number.isFinite(geometria.boundingSphere.radius),
    `radio no finito: ${geometria.boundingSphere.radius}`);
  assert.ok(geometria.boundingSphere.radius > 0);
});

prueba('la superficie vacía tiene esfera envolvente válida', () => {
  const { valores, etiquetas } = campoYMascara({ valorDentro: 0 });
  const geometria = extraerIsosuperficieMarchingCubes(valores, FORMA, ejes, 0.5, etiquetas);
  assert.equal(geometria.getAttribute('position').count, 0);
  assert.ok(geometria.boundingSphere && Number.isFinite(geometria.boundingSphere.radius));
});

prueba('la superficie se queda dentro del dominio', () => {
  const { valores, etiquetas } = campoYMascara();
  const geometria = extraerIsosuperficieMarchingCubes(valores, FORMA, ejes, 0.5, etiquetas);
  const p = geometria.getAttribute('position').array;
  for (let n = 0; n < p.length; n += 1) {
    assert.ok(p[n] >= 0 && p[n] <= 5, `coordenada fuera del dominio: ${p[n]}`);
  }
});

prueba('el suavizado gaussiano no sale del rango ni introduce NaN', () => {
  const { valores } = campoYMascara();
  const suave = suavizarCampoGaussiano(valores, FORMA, 2, 1.05);
  assert.ok([...suave].every(Number.isFinite));
  // Un filtro de pesos positivos que suman 1 no puede rebasar los extremos.
  assert.ok(Math.min(...suave) >= Math.min(...valores) - 1e-6);
  assert.ok(Math.max(...suave) <= Math.max(...valores) + 1e-6);
  // Y debe reducir el salto entre celdas vecinas, que es para lo que está.
  const variacion = (a) => {
    let total = 0;
    for (let i = 0; i < 5; i += 1) for (let j = 0; j < 6; j += 1) for (let k = 0; k < 6; k += 1) {
      total += Math.abs(a[indice3(i + 1, j, k)] - a[indice3(i, j, k)]);
    }
    return total;
  };
  assert.ok(variacion(suave) < variacion(valores),
    `el suavizado debe reducir la variación: ${variacion(suave)} vs ${variacion(valores)}`);
});

let fallos = 0;
for (const [nombre, fn] of pruebas) {
  try {
    fn();
    console.log(`ok   ${nombre}`);
  } catch (error) {
    fallos += 1;
    console.error(`FALLO ${nombre}\n      ${error.message}`);
  }
}
console.log(`${pruebas.length - fallos}/${pruebas.length} pruebas de marching cubes correctas`);
process.exit(fallos ? 1 : 0);
