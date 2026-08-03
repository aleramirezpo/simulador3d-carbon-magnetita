import * as THREE from './three.module.js';

// Tabla clásica de Lorensen y Cline: 256 configuraciones × 16 aristas.
// Se conserva compactada en base64 para no inflar el módulo con 4096 enteros.
const CASOS_BASE64 = '/////////////////////wAIA/////////////////8AAQn/////////////////AQgDCQgB/////////////wECCv////////////////8ACAMBAgr/////////////CQIKAAIJ/////////////wIIAwIKCAoJCP////////8DCwL/////////////////AAsCCAsA/////////////wEJAAIDC/////////////8BCwIBCQsJCAv/////////AwoBCwoD/////////////wAKAQAICggLCv////////8DCQADCwkLCgn/////////CQgKCggL/////////////wQHCP////////////////8EAwAHAwT/////////////AAEJCAQH/////////////wQBCQQHAQcDAf////////8BAgoIBAf/////////////AwQHAwAEAQIK/////////wkCCgkAAggEB/////////8CCgkCCQcCBwMHCQT/////CAQHAwsC/////////////wsEBwsCBAIABP////////8JAAEIBAcCAwv/////////BAcLCQQLCQsCCQIB/////wMKAQMLCgcIBP////////8BCwoBBAsBAAQHCwT/////BAcICQALCQsKCwAD/////wQHCwQLCQkLCv////////8JBQT/////////////////CQUEAAgD/////////////wAFBAEFAP////////////8IBQQIAwUDAQX/////////AQIKCQUE/////////////wMACAECCgQJBf////////8FAgoFBAIEAAL/////////AgoFAwIFAwUEAwQI/////wkFBAIDC/////////////8ACwIACAsECQX/////////AAUEAAEFAgML/////////wIBBQIFCAIICwQIBf////8KAwsKAQMJBQT/////////BAkFAAgBCAoBCAsK/////wUEAAUACwULCgsAA/////8FBAgFCAoKCAv/////////CQcIBQcJ/////////////wkDAAkFAwUHA/////////8ABwgAAQcBBQf/////////AQUDAwUH/////////////wkHCAkFBwoBAv////////8KAQIJBQAFAwAFBwP/////CAACCAIFCAUHCgUC/////wIKBQIFAwMFB/////////8HCQUHCAkDCwL/////////CQUHCQcCCQIAAgcL/////wIDCwABCAEHCAEFB/////8LAgELAQcHAQX/////////CQUICAUHCgEDCgML/////wUHAAUACQcLAAEACgsKAP8LCgALAAMKBQAIAAcFBwD/CwoFBwsF/////////////woGBf////////////////8ACAMFCgb/////////////CQABBQoG/////////////wEIAwEJCAUKBv////////8BBgUCBgH/////////////AQYFAQIGAwAI/////////wkGBQkABgACBv////////8FCQgFCAIFAgYDAgj/////AgMLCgYF/////////////wsACAsCAAoGBf////////8AAQkCAwsFCgb/////////BQoGAQkCCQsCCQgL/////wYDCwYFAwUBA/////////8ACAsACwUABQEFCwb/////AwsGAAMGAAYFAAUJ/////wYFCQYJCwsJCP////////8FCgYEBwj/////////////BAMABAcDBgUK/////////wEJAAUKBggEB/////////8KBgUBCQcBBwMHCQT/////BgECBgUBBAcI/////////wECBQUCBgMABAMEB/////8IBAcJAAUABgUAAgb/////BwMJBwkEAwIJBQkGAgYJ/wMLAgcIBAoGBf////////8FCgYEBwIEAgACBwv/////AAEJBAcIAgMLBQoG/////wkCAQkLAgkECwcLBAUKBv8IBAcDCwUDBQEFCwb/////BQELBQsGAQALBwsEAAQL/wAFCQAGBQADBgsGAwgEB/8GBQkGCQsEBwkHCwn/////CgQJBgQK/////////////wQKBgQJCgAIA/////////8KAAEKBgAGBAD/////////CAMBCAEGCAYEBgEK/////wEECQECBAIGBP////////8DAAgBAgkCBAkCBgT/////AAIEBAIG/////////////wgDAggCBAQCBv////////8KBAkKBgQLAgP/////////AAgCAggLBAkKBAoG/////wMLAgABBgAGBAYBCv////8GBAEGAQoECAECAQsICwH/CQYECQMGCQEDCwYD/////wgLAQgBAAsGAQkBBAYEAf8DCwYDBgAABgT/////////BgQICwYI/////////////wcKBgcICggJCv////////8ABwMACgcACQoGBwr/////CgYHAQoHAQcIAQgA/////woGBwoHAQEHA/////////8BAgYBBggBCAkIBgf/////AgYJAgkBBgcJAAkDBwMJ/wcIAAcABgYAAv////////8HAwIGBwL/////////////AgMLCgYICggJCAYH/////wIABwIHCwAJBwYHCgkKB/8BCAABBwgBCgcGBwoCAwv/CwIBCwEHCgYBBgcB/////wgJBggGBwkBBgsGAwEDBv8ACQELBgf/////////////BwgABwAGAwsACwYA/////wcLBv////////////////8HBgv/////////////////AwAICwcG/////////////wABCQsHBv////////////8IAQkIAwELBwb/////////CgECBgsH/////////////wECCgMACAYLB/////////8CCQACCgkGCwf/////////BgsHAgoDCggDCgkI/////wcCAwYCB/////////////8HAAgHBgAGAgD/////////AgcGAgMHAAEJ/////////wEGAgEIBgEJCAgHBv////8KBwYKAQcBAwf/////////CgcGAQcKAQgHAQAI/////wADBwAHCgAKCQYKB/////8HBgoHCggICgn/////////BggECwgG/////////////wMGCwMABgAEBv////////8IBgsIBAYJAAH/////////CQQGCQYDCQMBCwMG/////wYIBAYLCAIKAf////////8BAgoDAAsABgsABAb/////BAsIBAYLAAIJAgoJ/////woJAwoDAgkEAwsDBgQGA/8IAgMIBAIEBgL/////////AAQCBAYC/////////////wEJAAIDBAIEBgQDCP////8BCQQBBAICBAb/////////CAEDCAYBCAQGBgoB/////woBAAoABgYABP////////8EBgMEAwgGCgMAAwkKCQP/CgkEBgoE/////////////wQJBQcGC/////////////8ACAMECQULBwb/////////BQABBQQABwYL/////////wsHBggDBAMFBAMBBf////8JBQQKAQIHBgv/////////BgsHAQIKAAgDBAkF/////wcGCwUECgQCCgQAAv////8DBAgDBQQDAgUKBQILBwb/BwIDBwYCBQQJ/////////wkFBAAIBgAGAgYIB/////8DBgIDBwYBBQAFBAD/////BgIIBggHAgEIBAgFAQUI/wkFBAoBBgEHBgEDB/////8BBgoBBwYBAAcIBwAJBQT/BAAKBAoFAAMKBgoHAwcK/wcGCgcKCAUECgQICv////8GCQUGCwkLCAn/////////AwYLAAYDAAUGAAkF/////wALCAAFCwABBQUGC/////8GCwMGAwUFAwH/////////AQIKCQULCQsICwUG/////wALAwAGCwAJBgUGCQECCv8LCAULBQYIAAUKBQIAAgX/BgsDBgMFAgoDCgUD/////wUICQUCCAUGAgMIAv////8JBQYJBgAABgL/////////AQUIAQgABQYIAwgCBgII/wEFBgIBBv////////////8BAwYBBgoDCAYFBgkICQb/CgEACgAGCQUABQYA/////wADCAUGCv////////////8KBQb/////////////////CwUKBwUL/////////////wsFCgsHBQgDAP////////8FCwcFCgsBCQD/////////CgcFCgsHCQgBCAMB/////wsBAgsHAQcFAf////////8ACAMBAgcBBwUHAgv/////CQcFCQIHCQACAgsH/////wcFAgcCCwUJAgMCCAkIAv8CBQoCAwUDBwX/////////CAIACAUCCAcFCgIF/////wkAAQUKAwUDBwMKAv////8JCAIJAgEIBwIKAgUHBQL/AQMFAwcF/////////////wAIBwAHAQEHBf////////8JAAMJAwUFAwf/////////CQgHBQkH/////////////wUIBAUKCAoLCP////////8FAAQFCwAFCgsLAwD/////AAEJCAQKCAoLCgQF/////woLBAoEBQsDBAkEAQMBBP8CBQECCAUCCwgEBQj/////AAQLAAsDBAULAgsBBQEL/wACBQAFCQILBQQFCAsIBf8JBAUCCwP/////////////AgUKAwUCAwQFAwgE/////wUKAgUCBAQCAP////////8DCgIDBQoDCAUEBQgAAQn/BQoCBQIEAQkCCQQC/////wgEBQgFAwMFAf////////8ABAUBAAX/////////////CAQFCAUDCQAFAAMF/////wkEBf////////////////8ECwcECQsJCgv/////////AAgDBAkHCQsHCQoL/////wEKCwELBAEEAAcEC/////8DAQQDBAgBCgQHBAsKCwT/BAsHCQsECQILCQEC/////wkHBAkLBwkBCwILAQAIA/8LBwQLBAICBAD/////////CwcECwQCCAMEAwIE/////wIJCgIHCQIDBwcECf////8JCgcJBwQKAgcIBwACAAf/AwcKAwoCBwQKAQoABAAK/wEKAggHBP////////////8ECQEEAQcHAQP/////////BAkBBAEHAAgBCAcB/////wQAAwcEA/////////////8ECAf/////////////////CQoICgsI/////////////wMACQMJCwsJCv////////8AAQoACggICgv/////////AwEKCwMK/////////////wECCwELCQkLCP////////8DAAkDCQsBAgkCCwn/////AAILCAAL/////////////wMCC/////////////////8CAwgCCAoKCAn/////////CQoCAAkC/////////////wIDCAIICgABCAEKCP////8BCgL/////////////////AQMICQEI/////////////wAJAf////////////////8AAwj//////////////////////////////////////w==';

const ESQUINAS = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
const ARISTAS = [
  [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
  [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
];

let tablaCasos;

function obtenerTablaCasos() {
  if (tablaCasos) return tablaCasos;
  const binario = atob(CASOS_BASE64);
  tablaCasos = new Int8Array(binario.length);
  for (let i = 0; i < binario.length; i += 1) {
    const valor = binario.charCodeAt(i);
    tablaCasos[i] = valor > 127 ? valor - 256 : valor;
  }
  return tablaCasos;
}

function indice3(i, j, k, forma) {
  return (i * forma[1] + j) * forma[2] + k;
}

function interpolarVertice(a, b, va, vb, umbral) {
  const diferencia = vb - va;
  // Las esquinas fuera de la máscara valen -Infinity. Sin este caso,
  // (umbral - (-Inf)) / (vb - (-Inf)) es Inf/Inf = NaN y el vértice sale NaN:
  // la geometría queda corrupta y su esfera envolvente también. Cuando un
  // extremo no es finito la superficie se corta en la frontera de la máscara,
  // es decir a mitad de arista.
  const t = !Number.isFinite(va) || !Number.isFinite(vb) || Math.abs(diferencia) < 1e-12
    ? 0.5
    : THREE.MathUtils.clamp((umbral - va) / diferencia, 0, 1);
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

/** Filtro gaussiano separable para eliminar escalones antes de polygonizar. */
export function suavizarCampoGaussiano(valores, forma, radio = 1, sigma = 0.9) {
  const r = Math.max(1, Math.round(radio));
  const pesos = new Float32Array(2 * r + 1);
  let suma = 0;
  for (let d = -r; d <= r; d += 1) {
    const peso = Math.exp(-(d * d) / (2 * sigma * sigma));
    pesos[d + r] = peso;
    suma += peso;
  }
  for (let n = 0; n < pesos.length; n += 1) pesos[n] /= suma;
  const [nx, ny, nz] = forma;
  const filtrarEje = (entrada, eje) => {
    const salida = new Float32Array(entrada.length);
    for (let i = 0; i < nx; i += 1) for (let j = 0; j < ny; j += 1) for (let k = 0; k < nz; k += 1) {
      let valor = 0;
      for (let d = -r; d <= r; d += 1) {
        const ii = eje === 0 ? THREE.MathUtils.clamp(i + d, 0, nx - 1) : i;
        const jj = eje === 1 ? THREE.MathUtils.clamp(j + d, 0, ny - 1) : j;
        const kk = eje === 2 ? THREE.MathUtils.clamp(k + d, 0, nz - 1) : k;
        valor += entrada[indice3(ii, jj, kk, forma)] * pesos[d + r];
      }
      salida[indice3(i, j, k, forma)] = valor;
    }
    return salida;
  };
  return filtrarEje(filtrarEje(filtrarEje(valores, 0), 1), 2);
}

function indexarVertices(vertices, tolerancia = 1e-5) {
  const posiciones = [];
  const indices = [];
  const mapa = new Map();
  for (let n = 0; n < vertices.length; n += 3) {
    const x = vertices[n]; const y = vertices[n + 1]; const z = vertices[n + 2];
    const clave = `${Math.round(x / tolerancia)},${Math.round(y / tolerancia)},${Math.round(z / tolerancia)}`;
    let indice = mapa.get(clave);
    if (indice === undefined) {
      indice = posiciones.length / 3;
      mapa.set(clave, indice);
      posiciones.push(x, y, z);
    }
    indices.push(indice);
  }
  return { posiciones, indices };
}

export function extraerIsosuperficieMarchingCubes(valores, forma, ejes, umbral, etiquetas) {
  const tabla = obtenerTablaCasos();
  const [nx, ny, nz] = forma;
  const vertices = [];
  for (let i = 0; i < nx - 1; i += 1) {
    for (let j = 0; j < ny - 1; j += 1) {
      for (let k = 0; k < nz - 1; k += 1) {
        const escalares = new Array(8);
        const posiciones = new Array(8);
        let caso = 0;
        for (let esquina = 0; esquina < 8; esquina += 1) {
          const [di, dj, dk] = ESQUINAS[esquina];
          const q = indice3(i + di, j + dj, k + dk, forma);
          const interior = etiquetas[q] === 3 || etiquetas[q] === 4;
          escalares[esquina] = interior && Number.isFinite(valores[q]) ? valores[q] : -Infinity;
          posiciones[esquina] = [ejes.x[i + di], ejes.y[j + dj], ejes.z[k + dk]];
          if (escalares[esquina] >= umbral) caso |= (1 << esquina);
        }
        if (caso === 0 || caso === 255) continue;
        const desplazamiento = caso * 16;
        const cacheAristas = new Array(12);
        for (let n = 0; n < 16 && tabla[desplazamiento + n] >= 0; n += 1) {
          const arista = tabla[desplazamiento + n];
          if (!cacheAristas[arista]) {
            const [ea, eb] = ARISTAS[arista];
            cacheAristas[arista] = interpolarVertice(
              posiciones[ea], posiciones[eb], escalares[ea], escalares[eb], umbral,
            );
          }
          vertices.push(...cacheAristas[arista]);
        }
      }
    }
  }
  const geometria = new THREE.BufferGeometry();
  const indexada = indexarVertices(vertices);
  geometria.setAttribute('position', new THREE.Float32BufferAttribute(indexada.posiciones, 3));
  // Al compartir índices, computeVertexNormals promedia las caras adyacentes en
  // cada vértice. En una geometría no indexada sólo produciría normales planas.
  geometria.setIndex(indexada.indices);
  if (indexada.posiciones.length) {
    geometria.computeVertexNormals();
    geometria.normalizeNormals();
    geometria.computeBoundingSphere();
  } else {
    // Isosuperficie vacía: es lo normal antes de que exista el aglomerado o
    // mientras nada supera la isoterma. computeBoundingSphere sobre cero
    // vértices da radio NaN y three lo denuncia por consola en cada fotograma.
    geometria.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 0);
  }
  return geometria;
}
