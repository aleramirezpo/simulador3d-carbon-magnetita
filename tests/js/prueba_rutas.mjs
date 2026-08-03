/**
 * Pruebas del resolutor de rutas, ejecutadas en Node.
 *
 * La interfaz tiene ahora dos orígenes de datos —el servidor local y el sitio
 * estático publicado— y una sola implementación del cliente. Lo que se
 * comprueba aquí es que el mismo módulo produce las rutas de cada modo y que
 * el modo estático sale RELATIVO: en GitHub Pages el sitio no cuelga de la
 * raíz del dominio sino de `/<repositorio>/`, y una ruta absoluta daría 404 en
 * todos los fotogramas.
 *
 * Uso: node tests/js/prueba_rutas.mjs
 */
import assert from 'node:assert/strict';

const pruebas = [];
const prueba = (nombre, fn) => pruebas.push([nombre, fn]);

// El módulo consulta `window`; en Node no existe y hay que fabricarlo antes de
// importarlo, igual que hace el navegador.
globalThis.window = {};
const { baseEstatica, modoCache, rutaConfig, rutaFotograma, rutaLineas } =
  await import('../../interfaz/web/js/rutas.js');

const conBase = (base, fn) => {
  const anterior = globalThis.window.SIMULADOR3D_ESTATICO;
  globalThis.window.SIMULADOR3D_ESTATICO = base;
  try { fn(); } finally { globalThis.window.SIMULADOR3D_ESTATICO = anterior; }
};

prueba('sin declaración se usan las rutas del servidor local', () => {
  conBase(undefined, () => {
    assert.equal(baseEstatica(), null);
    assert.equal(rutaConfig(), '/api/config');
    assert.equal(rutaFotograma(7), '/api/fotograma?indice=7&formato=float32');
    assert.equal(rutaLineas(7), '/api/lineas?indice=7');
    assert.equal(modoCache(), 'no-store');
  });
});

prueba('el modo estático apunta a archivos y nunca a la raíz del dominio', () => {
  conBase('datos', () => {
    assert.equal(rutaConfig(), 'datos/config.json');
    assert.equal(rutaFotograma(7), 'datos/fotograma_0007.bin');
    assert.equal(rutaLineas(7), 'datos/lineas_0007.json');
    for (const ruta of [rutaConfig(), rutaFotograma(0), rutaLineas(0)]) {
      assert.ok(!ruta.startsWith('/'), `ruta absoluta en modo estático: ${ruta}`);
    }
  });
});

prueba('el índice se rellena a cuatro cifras, como los archivos exportados', () => {
  conBase('datos', () => {
    assert.equal(rutaFotograma(0), 'datos/fotograma_0000.bin');
    assert.equal(rutaFotograma(53), 'datos/fotograma_0053.bin');
    assert.equal(rutaFotograma(1234), 'datos/fotograma_1234.bin');
  });
});

prueba('el sitio estático deja que el navegador cachee los fotogramas', () => {
  // Una serie completa son decenas de MB de archivos inmutables: con
  // `no-store` se volverían a bajar enteros en cada recarga de la página.
  conBase('datos', () => assert.equal(modoCache(), 'default'));
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
console.log(`${pruebas.length - fallos}/${pruebas.length} pruebas de rutas correctas`);
process.exit(fallos ? 1 : 0);
