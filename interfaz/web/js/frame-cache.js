import { modoCache, rutaFotograma, rutaLineas } from './rutas.js';

const CAMPOS_ESCALARES = Object.freeze([
  'u', 'v', 'w', 'P', 'T', 'eps', 'cohesion', 'hinchamiento', 'conversion', 'reduccion',
]);

const MAGIC_FLOAT32 = 'S3DF';
const CABECERA_BINARIA_BYTES = 8;

function asignarRuta(objeto, ruta, valor) {
  const partes = ruta.split('.');
  let destino = objeto;
  for (let i = 0; i < partes.length - 1; i += 1) {
    destino[partes[i]] ??= {};
    destino = destino[partes[i]];
  }
  destino[partes.at(-1)] = valor;
}

/**
 * Decodifica el contrato binario S3DF: magic "S3DF", longitud uint32 LE,
 * cabecera JSON, relleno a 4 bytes y bloques Float32/Uint8 descritos por ruta.
 * Así el servidor puede enviar los campos densos sin serializar números JSON.
 */
export function decodificarFotogramaBinario(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 4));
  if (magic !== MAGIC_FLOAT32) throw new Error('Formato binario S3DF no reconocido');
  const vista = new DataView(arrayBuffer);
  const longitudCabecera = vista.getUint32(4, true);
  const finCabecera = CABECERA_BINARIA_BYTES + longitudCabecera;
  const inicioDatos = (finCabecera + 3) & ~3;
  const cabecera = JSON.parse(new TextDecoder().decode(bytes.subarray(CABECERA_BINARIA_BYTES, finCabecera)));
  const fotograma = cabecera.fotograma || {};
  for (const bloque of cabecera.bloques || []) {
    const inicio = inicioDatos + bloque.offset;
    const longitud = bloque.longitud;
    const valores = bloque.tipo === 'u8'
      ? new Uint8Array(arrayBuffer.slice(inicio, inicio + longitud))
      : new Float32Array(arrayBuffer.slice(inicio, inicio + longitud * Float32Array.BYTES_PER_ELEMENT));
    asignarRuta(fotograma, bloque.ruta, valores);
  }
  return fotograma;
}

function float32(valores) {
  return valores instanceof Float32Array ? valores : Float32Array.from(valores || []);
}

/** Libera la memoria de los Array de Number de JSON tan pronto como llegan. */
export function compactarFotogramaFloat32(fotograma) {
  for (const campo of CAMPOS_ESCALARES) fotograma[campo] = float32(fotograma[campo]);
  fotograma.x = float32(fotograma.x);
  fotograma.y = float32(fotograma.y);
  fotograma.z = float32(fotograma.z);
  fotograma.etiquetas = fotograma.etiquetas instanceof Uint8Array
    ? fotograma.etiquetas : Uint8Array.from(fotograma.etiquetas || []);
  for (const grupo of ['c_especies', 'solido']) {
    for (const [nombre, valores] of Object.entries(fotograma[grupo] || {})) {
      fotograma[grupo][nombre] = float32(valores);
    }
  }
  return fotograma;
}

async function descargarFotogramaFloat32(indice) {
  const respuesta = await fetch(rutaFotograma(indice), {
    headers: { Accept: 'application/vnd.simulador3d.float32, application/octet-stream;q=0.9, application/json;q=0.5' },
    cache: modoCache(),
  });
  if (!respuesta.ok) throw new Error(`No se pudo precargar el fotograma ${indice}`);
  // arrayBuffer evita response.json() y permite usar directamente Float32 cuando
  // el servidor ofrece S3DF. Se conserva compatibilidad con servidores 1.0 JSON.
  const buffer = await respuesta.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const esS3df = bytes.length >= 4 && new TextDecoder().decode(bytes.subarray(0, 4)) === MAGIC_FLOAT32;
  const fotograma = esS3df
    ? decodificarFotogramaBinario(buffer)
    : JSON.parse(new TextDecoder().decode(bytes));
  fotograma.metadatos ??= {};
  fotograma.metadatos.transporte_cliente = esS3df ? 'Float32 binario S3DF' : 'JSON compatible compactado a Float32';
  return compactarFotogramaFloat32(fotograma);
}

async function descargarLineas(indice) {
  const respuesta = await fetch(rutaLineas(indice), { cache: modoCache() });
  if (!respuesta.ok) throw new Error(`No se pudieron precargar las líneas ${indice}`);
  return (await respuesta.json()).lineas;
}

/**
 * Precarga acotada: nunca hay peticiones HTTP durante la reproducción. La
 * concurrencia corta mantiene ocupado el servidor sin disparar el uso de RAM.
 */
export class CacheFotogramas {
  constructor(cantidad, { concurrencia = 3, alProgresar = () => {} } = {}) {
    this.cantidad = cantidad;
    this.concurrencia = Math.max(1, Math.min(concurrencia, cantidad));
    this.alProgresar = alProgresar;
    this.fotogramas = new Array(cantidad);
    this.lineas = new Array(cantidad);
    this.completa = false;
    // Numero de fotogramas ya descargados. Permite reproducir con lo que hay
    // en vez de esperar a que la serie entera este disponible.
    this.listos = 0;
  }

  async precargar() {
    let siguiente = 0;
    let completados = 0;
    const trabajador = async () => {
      while (siguiente < this.cantidad) {
        const indice = siguiente;
        siguiente += 1;
        const fotograma = await descargarFotogramaFloat32(indice);
        const lineas = await descargarLineas(indice);
        this.fotogramas[indice] = fotograma;
        this.lineas[indice] = lineas;
        completados += 1;
        // `listos` debe ser el PREFIJO CONTIGUO disponible, no el numero total
        // de descargas terminadas: con varias descargas en paralelo los
        // fotogramas llegan desordenados, y reproducir hasta un indice que aun
        // tiene huecos por delante daria saltos.
        let contiguos = 0;
        while (contiguos < this.cantidad && this.fotogramas[contiguos] !== undefined) {
          contiguos += 1;
        }
        this.listos = contiguos;
        this.alProgresar(completados, this.cantidad, fotograma.metadatos.transporte_cliente);
      }
    };
    await Promise.all(Array.from({ length: this.concurrencia }, trabajador));
    this.completa = true;
    return this;
  }

  obtener(indice) { return this.fotogramas[indice]; }
  obtenerLineas(indice) { return this.lineas[indice] || []; }
}

function interpolarNumero(a, b, mezcla) {
  return Number(a) + (Number(b) - Number(a)) * mezcla;
}

function interpolarArreglo(a, b, mezcla, salida) {
  const destino = salida?.length === a.length ? salida : new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 1) destino[i] = a[i] + (b[i] - a[i]) * mezcla;
  return destino;
}

function interpolarRegistro(a = {}, b = {}, mezcla, salida = {}) {
  for (const clave of new Set([...Object.keys(a), ...Object.keys(b)])) {
    salida[clave] = Number.isFinite(Number(a[clave])) && Number.isFinite(Number(b[clave]))
      ? interpolarNumero(a[clave], b[clave], mezcla)
      : (mezcla < 0.5 ? a[clave] : b[clave]);
  }
  return salida;
}

/**
 * Interpolación lineal reutilizable de todos los campos escalares. El objeto de
 * salida y sus Float32Array se reciclan para sostener 30 actualizaciones/s sin GC.
 */
export function interpolarFotogramas(a, b, mezcla, salida = {}) {
  const t = Math.max(0, Math.min(1, mezcla));
  salida.t = interpolarNumero(a.t, b.t, t);
  salida.forma = a.forma;
  salida.x = a.x;
  salida.y = a.y;
  salida.z = a.z;
  salida.etiquetas = a.etiquetas;
  salida.metadatos = a.metadatos;
  for (const campo of CAMPOS_ESCALARES) {
    salida[campo] = interpolarArreglo(a[campo], b[campo], t, salida[campo]);
  }
  salida.c_especies ??= {};
  for (const nombre of Object.keys(a.c_especies || {})) {
    salida.c_especies[nombre] = interpolarArreglo(
      a.c_especies[nombre], b.c_especies[nombre], t, salida.c_especies[nombre],
    );
  }
  salida.solido ??= {};
  for (const nombre of Object.keys(a.solido || {})) {
    salida.solido[nombre] = interpolarArreglo(a.solido[nombre], b.solido[nombre], t, salida.solido[nombre]);
  }
  salida.termico = interpolarRegistro(a.termico, b.termico, t, salida.termico);
  salida.numeros = interpolarRegistro(a.numeros, b.numeros, t, salida.numeros);
  return salida;
}

