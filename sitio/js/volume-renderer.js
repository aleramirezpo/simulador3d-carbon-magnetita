import * as THREE from './three.module.js';

// El volumen usa texturas 3D reales. La alternativa de planos mantiene una
// ruta compatible para GPU integradas donde el ray marching sea demasiado caro.
export const VOLUME_VERTEX_SHADER = /* glsl */`
  out vec3 vPosicionLocal;
  out vec3 vOrigenLocal;
  #include <clipping_planes_pars_vertex>

  void main() {
    vPosicionLocal = position;
    vOrigenLocal = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
  }
`;

export const VOLUME_FRAGMENT_SHADER = /* glsl */`
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uVolumen;
  uniform sampler2D uPaleta;
  uniform float uOpacidad;
  uniform float uPasos;
  uniform vec3 uCajaMax;
  in vec3 vPosicionLocal;
  in vec3 vOrigenLocal;
  out vec4 colorFragmento;
  #include <clipping_planes_pars_fragment>

  vec2 interseccionCaja(vec3 origen, vec3 direccion) {
    vec3 inversa = 1.0 / direccion;
    vec3 t0 = (-vec3(0.5) - origen) * inversa;
    vec3 t1 = (uCajaMax - origen) * inversa;
    vec3 menor = min(t0, t1);
    vec3 mayor = max(t0, t1);
    float entrada = max(max(menor.x, menor.y), menor.z);
    float salida = min(min(mayor.x, mayor.y), mayor.z);
    return vec2(entrada, salida);
  }

  void main() {
    // El volumen recorta su intervalo de integración con uCajaMax; aplicar aquí
    // el descarte superficial estándar ocultaría rayos cuya cara posterior cae
    // fuera del corte, aunque atraviesen el subvolumen conservado.
    vec3 direccion = normalize(vPosicionLocal - vOrigenLocal);
    vec2 limites = interseccionCaja(vOrigenLocal, direccion);
    if (limites.x > limites.y) discard;
    limites.x = max(limites.x, 0.0);
    float longitud = limites.y - limites.x;
    float paso = longitud / max(uPasos, 1.0);
    vec3 posicion = vOrigenLocal + direccion * (limites.x + 0.5 * paso);
    vec4 acumulado = vec4(0.0);

    for (int muestra = 0; muestra < 192; muestra++) {
      if (float(muestra) >= uPasos || acumulado.a > 0.985) break;
      float valor = texture(uVolumen, posicion + vec3(0.5)).r;
      vec3 color = texture(uPaleta, vec2(valor, 0.5)).rgb;
      float densidad = smoothstep(0.015, 1.0, valor);
      float alfa = 1.0 - exp(-densidad * uOpacidad * paso * 5.5);
      acumulado.rgb += (1.0 - acumulado.a) * alfa * color;
      acumulado.a += (1.0 - acumulado.a) * alfa;
      posicion += direccion * paso;
    }
    if (acumulado.a < 0.01) discard;
    colorFragmento = acumulado;
  }
`;

function indiceCampo(i, j, k, forma) {
  return (i * forma[1] + j) * forma[2] + k;
}

export function crearTexturaPaleta(interpolarColor) {
  const datos = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i += 1) {
    const color = interpolarColor(i / 255);
    datos[i * 4] = Math.round(color[0] * 255);
    datos[i * 4 + 1] = Math.round(color[1] * 255);
    datos[i * 4 + 2] = Math.round(color[2] * 255);
    datos[i * 4 + 3] = 255;
  }
  const textura = new THREE.DataTexture(datos, 256, 1, THREE.RGBAFormat);
  textura.colorSpace = THREE.SRGBColorSpace;
  textura.minFilter = THREE.LinearFilter;
  textura.magFilter = THREE.LinearFilter;
  textura.needsUpdate = true;
  return textura;
}

export function crearTexturaVolumetrica(valores, etiquetas, forma, normalizar, soloLecho = false) {
  const [nx, ny, nz] = forma;
  const datos = new Uint8Array(nx * ny * nz);
  escribirTexturaVolumetrica(datos, valores, etiquetas, forma, normalizar, soloLecho);
  const textura = new THREE.Data3DTexture(datos, nx, ny, nz);
  textura.format = THREE.RedFormat;
  textura.type = THREE.UnsignedByteType;
  textura.minFilter = THREE.LinearFilter;
  textura.magFilter = THREE.LinearFilter;
  textura.unpackAlignment = 1;
  textura.needsUpdate = true;
  return textura;
}

function escribirTexturaVolumetrica(datos, valores, etiquetas, forma, normalizar, soloLecho) {
  const [nx, ny, nz] = forma;
  // Data3DTexture ordena x como el índice rápido; la API serializa en orden C
  // (z rápido), por lo que la transposición explícita es imprescindible.
  for (let k = 0; k < nz; k += 1) {
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const origen = indiceCampo(i, j, k, forma);
        const interior = soloLecho ? etiquetas[origen] === 3 : (etiquetas[origen] === 3 || etiquetas[origen] === 4);
        const destino = i + nx * (j + ny * k);
        datos[destino] = interior ? Math.round(THREE.MathUtils.clamp(normalizar(valores[origen]), 0, 1) * 255) : 0;
      }
    }
  }
}

/** Actualiza el mismo buffer GPU durante la interpolación; no crea texturas. */
export function actualizarTexturaVolumetrica(textura, valores, etiquetas, forma, normalizar, soloLecho = false) {
  const datos = textura?.image?.data;
  if (!datos || datos.length !== forma[0] * forma[1] * forma[2]) return false;
  escribirTexturaVolumetrica(datos, valores, etiquetas, forma, normalizar, soloLecho);
  textura.needsUpdate = true;
  return true;
}

export function crearVolumenRayMarching({ textura, paleta, dimensiones, centro, clippingPlanes, opacidad, pasos, cajaMax }) {
  const geometria = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: VOLUME_VERTEX_SHADER,
    fragmentShader: VOLUME_FRAGMENT_SHADER,
    uniforms: {
      uVolumen: { value: textura },
      uPaleta: { value: paleta },
      uOpacidad: { value: opacidad },
      uPasos: { value: pasos },
      uCajaMax: { value: cajaMax },
    },
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
    clipping: true,
    clippingPlanes,
  });
  const volumen = new THREE.Mesh(geometria, material);
  volumen.name = 'Volumen ray marching Data3DTexture';
  volumen.scale.copy(dimensiones);
  volumen.position.copy(centro);
  volumen.renderOrder = 2;
  return volumen;
}

export function crearPlanosApilados({ valores, etiquetas, forma, ejes, limites, normalizar, interpolarColor, clippingPlanes, opacidad, soloLecho = false }) {
  const grupo = new THREE.Group();
  grupo.name = 'Alternativa de planos DataTexture';
  const [nx, ny, nz] = forma;
  const pasoK = Math.max(1, Math.ceil(nz / 18));
  const ancho = (ejes.x.at(-1) - ejes.x[0]) + (ejes.x[1] - ejes.x[0]);
  const alto = (ejes.y.at(-1) - ejes.y[0]) + (ejes.y[1] - ejes.y[0]);
  void limites;
  for (let k = 0; k < nz; k += pasoK) {
    const datos = new Uint8Array(nx * ny * 4);
    for (let j = 0; j < ny; j += 1) {
      for (let i = 0; i < nx; i += 1) {
        const q = indiceCampo(i, j, k, forma);
        const interior = soloLecho ? etiquetas[q] === 3 : (etiquetas[q] === 3 || etiquetas[q] === 4);
        const destino = (i + nx * j) * 4;
        const t = THREE.MathUtils.clamp(normalizar(valores[q]), 0, 1);
        const color = interpolarColor(t);
        datos[destino] = Math.round(color[0] * 255);
        datos[destino + 1] = Math.round(color[1] * 255);
        datos[destino + 2] = Math.round(color[2] * 255);
        datos[destino + 3] = interior ? Math.round(255 * opacidad * 0.24) : 0;
      }
    }
    const textura = new THREE.DataTexture(datos, nx, ny, THREE.RGBAFormat);
    textura.colorSpace = THREE.SRGBColorSpace;
    textura.minFilter = THREE.LinearFilter;
    textura.magFilter = THREE.LinearFilter;
    textura.needsUpdate = true;
    const material = new THREE.MeshBasicMaterial({
      map: textura,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      clippingPlanes,
    });
    const plano = new THREE.Mesh(new THREE.PlaneGeometry(ancho, alto), material);
    plano.position.set((ejes.x[0] + ejes.x.at(-1)) / 2, (ejes.y[0] + ejes.y.at(-1)) / 2, ejes.z[k]);
    plano.renderOrder = 2 + k / Math.max(nz, 1);
    grupo.add(plano);
  }
  return grupo;
}

export function liberarObjetoVolumetrico(objeto) {
  objeto.traverse((hijo) => {
    hijo.geometry?.dispose();
    if (hijo.material) {
      hijo.material.map?.dispose();
      hijo.material.uniforms?.uVolumen?.value?.dispose();
      hijo.material.uniforms?.uPaleta?.value?.dispose();
      hijo.material.dispose();
    }
  });
}
