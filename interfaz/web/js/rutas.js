/**
 * Resolución de las rutas de datos, para los dos modos de la interfaz.
 *
 * La misma interfaz corre contra dos orígenes distintos:
 *
 *   - El servidor local de `interfaz.app`, que calcula cada respuesta al
 *     vuelo y la publica en `/api/config`, `/api/fotograma` y `/api/lineas`.
 *   - Un sitio estático (GitHub Pages), donde no hay servidor: cada una de
 *     esas respuestas se exportó antes a un archivo con
 *     `interfaz/exportar_estatico.py`.
 *
 * El `index.html` del sitio estático declara `window.SIMULADOR3D_ESTATICO` con
 * el prefijo —relativo— del directorio de datos. Si no está declarado se usan
 * las rutas del servidor de siempre, de modo que el modo local no cambia.
 *
 * El prefijo es relativo a propósito: en Pages el sitio no cuelga de la raíz
 * del dominio sino de `/<repositorio>/`, y una ruta absoluta daría 404.
 */

const indice4 = (indice) => String(indice).padStart(4, '0');

/** Prefijo de datos estáticos, o null si se está sirviendo desde el servidor. */
export function baseEstatica() {
  return (typeof window !== 'undefined' && window.SIMULADOR3D_ESTATICO) || null;
}

export function rutaConfig() {
  const base = baseEstatica();
  return base ? `${base}/config.json` : '/api/config';
}

export function rutaFotograma(indice) {
  const base = baseEstatica();
  return base
    ? `${base}/fotograma_${indice4(indice)}.bin`
    : `/api/fotograma?indice=${indice}&formato=float32`;
}

export function rutaLineas(indice) {
  const base = baseEstatica();
  return base ? `${base}/lineas_${indice4(indice)}.json` : `/api/lineas?indice=${indice}`;
}

/**
 * `no-store` contra el servidor local, donde los datos cambian mientras una
 * corrida se escribe. En el sitio estático los archivos son inmutables y
 * conviene lo contrario: que la caché del navegador se los quede, porque una
 * serie completa son decenas de MB y se vuelven a pedir en cada recarga.
 */
export function modoCache() {
  return baseEstatica() ? 'default' : 'no-store';
}
