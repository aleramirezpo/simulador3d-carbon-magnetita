import * as THREE from './three.module.js';
import { OrbitControls } from './OrbitControls.js';
import { extraerIsosuperficieMarchingCubes, suavizarCampoGaussiano } from './marching-cubes.js';
import { SistemaParticulasLecho } from './particle-system.js';
import { CacheFotogramas, interpolarFotogramas } from './frame-cache.js';
import { modoCache, rutaConfig } from './rutas.js';
import {
  actualizarTexturaVolumetrica,
  crearPlanosApilados,
  crearTexturaPaleta,
  crearTexturaVolumetrica,
  crearVolumenRayMarching,
  liberarObjetoVolumetrico,
} from './volume-renderer.js';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const lienzo = $('#lienzo-3d');
const visor = $('#visor');

const renderer = new THREE.WebGLRenderer({
  canvas: lienzo,
  antialias: true,
  alpha: false,
  preserveDrawingBuffer: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.localClippingEnabled = true;

const escena = new THREE.Scene();
escena.background = new THREE.Color(0x091014);
escena.fog = new THREE.FogExp2(0x091014, 0.006);
const camara = new THREE.PerspectiveCamera(38, 1, 0.1, 500);
camara.up.set(0, 0, 1);
camara.position.set(50, -52, 48);

const controles = new OrbitControls(camara, lienzo);
controles.target.set(0, 0, 16);
controles.enableDamping = true;
controles.dampingFactor = 0.065;
controles.enableZoom = true;
controles.zoomSpeed = 1.15;
controles.zoomToCursor = true;
controles.minDistance = 4.5;
controles.maxDistance = 155;
controles.screenSpacePanning = true;
controles.update();

const raycaster = new THREE.Raycaster();
const puntero = new THREE.Vector2();
const planosCorte = {
  x: new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0),
  y: new THREE.Plane(new THREE.Vector3(0, -1, 0), 15.3),
  z: new THREE.Plane(new THREE.Vector3(0, 0, -1), 34),
};
const ayudantesCorte = {
  x: new THREE.PlaneHelper(planosCorte.x, 43, 0x56b4e9),
  y: new THREE.PlaneHelper(planosCorte.y, 43, 0x7ac36a),
  z: new THREE.PlaneHelper(planosCorte.z, 43, 0xe69f00),
};

const estado = {
  config: null,
  fotograma: null,
  indice: 0,
  serie: [],
  limites: [0, 1],
  campoActivo: [],
  nombreCampo: 'Temperatura',
  unidadCampo: 'K',
  dinamicos: [],
  lineasDatos: [],
  particulas: null,
  faseParticulas: 0,
  fasesVistas: new Map(),
  magnetizacionInicial: undefined,
  reproduciendo: false,
  velocidad: 1,
  cacheFotogramas: null,
  fotogramaInterpolado: {},
  cursorReproduccion: 0,
  relojReproduccionMs: 0,
  acumuladorReproduccionMs: 0,
  ultimoTiempoReproduccion: null,
  indiceReconstruido: -1,
  volumenDinamico: null,
  cargando: false,
  reconstruccionPendiente: false,
  puntosPerfil: [],
  lineaPerfil: null,
  modoPerfil: false,
  marcador: null,
  sistemaParticulas: null,
  superficiesTermicas: [],
  mufla: null,
  luces: null,
  flechasRadiacion: null,
  pulsosRadiacion: null,
  faseRadiacion: 0,
};

const FPS_REPRODUCCION = 30;
const PASO_REPRODUCCION_MS = 1000 / FPS_REPRODUCCION;
const DURACION_TRAMO_MS = 650;

const PALETAS = {
  cividis: [[0, [0.00, 0.13, 0.30]], [0.25, [0.23, 0.30, 0.42]], [0.5, [0.49, 0.49, 0.47]], [0.75, [0.76, 0.68, 0.39]], [1, [0.99, 0.91, 0.22]]],
  viridis: [[0, [0.27, 0.00, 0.33]], [0.25, [0.23, 0.32, 0.55]], [0.5, [0.13, 0.57, 0.55]], [0.75, [0.37, 0.79, 0.38]], [1, [0.99, 0.91, 0.14]]],
  divergente: [[0, [0.09, 0.27, 0.53]], [0.5, [0.84, 0.86, 0.84]], [1, [0.82, 0.32, 0.11]]],
};

function interpolarColor(valor, nombre = $('#paleta').value) {
  const t = THREE.MathUtils.clamp(Number.isFinite(valor) ? valor : 0, 0, 1);
  const paradas = PALETAS[nombre];
  for (let i = 1; i < paradas.length; i += 1) {
    if (t <= paradas[i][0]) {
      const [a, ca] = paradas[i - 1];
      const [b, cb] = paradas[i];
      const f = (t - a) / (b - a || 1);
      return ca.map((v, k) => v + (cb[k] - v) * f);
    }
  }
  return paradas.at(-1)[1];
}

function gradienteCss(nombre) {
  return `linear-gradient(90deg,${PALETAS[nombre].map(([p, c]) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${p * 100}%`).join(',')})`;
}

function indice3(i, j, k, forma) {
  return (i * forma[1] + j) * forma[2] + k;
}

function formatear(valor) {
  if (!Number.isFinite(valor)) return '—';
  const absoluto = Math.abs(valor);
  if ((absoluto > 0 && absoluto < 0.01) || absoluto >= 1e4) return valor.toExponential(2).replace('.', ',');
  return valor.toLocaleString('es-CO', { maximumFractionDigits: absoluto < 10 ? 3 : absoluto < 100 ? 2 : 1 });
}

function percentil(valores, p) {
  if (!valores.length) return 0;
  const ordenados = [...valores].sort((a, b) => a - b);
  return ordenados[Math.floor((ordenados.length - 1) * p)];
}

function planosActivos() {
  return ['x', 'y', 'z'].filter((eje) => $(`#activar-corte-${eje}`).checked).map((eje) => planosCorte[eje]);
}

function materialRecortable(Clase, parametros) {
  const material = new Clase({ ...parametros, clippingPlanes: planosActivos() });
  material.userData.recortable = true;
  return material;
}

function instalarIluminacion() {
  const hemisferio = new THREE.HemisphereLight(0xb8d8e8, 0x111820, 1.45);
  escena.add(hemisferio);
  const principal = new THREE.DirectionalLight(0xfff0d7, 3.1);
  principal.position.set(-32, -22, 64);
  escena.add(principal);
  const relleno = new THREE.PointLight(0x58a6a6, 48, 105, 2);
  relleno.position.set(34, 20, 28);
  escena.add(relleno);
  estado.luces = { hemisferio, principal, relleno };
}

/**
 * Modo de luz neutra para comparar fases.
 *
 * A 900 °C el interior del crisol EMITE, y esa luz naranja tiñe cada grano: los
 * colores minerales dejan de distinguirse justo cuando aparecen las fases
 * nuevas. El modo neutro apaga la emisión de las superficies y blanquea las
 * luces, de modo que lo que se ve es el color propio de cada fase. No cambia
 * ningún campo ni ningún número: es sólo iluminación, y va rotulado como tal.
 */
function luzNeutraActiva() {
  return Boolean($('#luz-neutra')?.checked);
}

function aplicarIluminacion() {
  const neutra = luzNeutraActiva();
  const luces = estado.luces;
  if (luces) {
    luces.hemisferio.color.setHex(neutra ? 0xffffff : 0xb8d8e8);
    luces.hemisferio.groundColor.setHex(neutra ? 0x9aa2a8 : 0x111820);
    luces.hemisferio.intensity = neutra ? 2.6 : 1.45;
    luces.principal.color.setHex(neutra ? 0xffffff : 0xfff0d7);
    luces.principal.intensity = neutra ? 2.2 : 3.1;
    luces.relleno.color.setHex(neutra ? 0xffffff : 0x58a6a6);
    luces.relleno.intensity = neutra ? 26 : 48;
  }
  if (estado.mufla) estado.mufla.visible = !neutra;
  renderer.toneMappingExposure = neutra ? 0.92 : 1.05;
  escena.background.setHex(neutra ? 0x141a1e : 0x091014);
  $('#aviso-luz')?.classList.toggle('oculto', !neutra);
  if (estado.fotograma) actualizarSuperficiesTermicas(estado.fotograma.termico);
}

function construirCrisol(geometriaConfig) {
  const grupo = new THREE.Group();
  grupo.name = 'Crisol perfilado con collar fotográfico';
  const material = materialRecortable(THREE.MeshPhysicalMaterial, {
    color: 0xa8bbc3,
    metalness: 0.22,
    roughness: 0.34,
    transmission: 0.08,
    thickness: 1.1,
    ior: 1.46,
    transparent: true,
    opacity: 0.40,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const perfilExterior = geometriaConfig.perfil_exterior_mm.map(([z, r]) => new THREE.Vector2(r, z));
  const perfilInterior = geometriaConfig.perfil_interior_mm.map(([z, r]) => new THREE.Vector2(r, z));
  for (const [nombre, perfil] of [['exterior', perfilExterior], ['cavidad', perfilInterior]]) {
    const malla = new THREE.Mesh(new THREE.LatheGeometry(perfil, 128), material.clone());
    malla.name = `Superficie ${nombre}`;
    malla.userData.superficieTermica = 'pared';
    malla.rotation.x = Math.PI / 2;
    malla.renderOrder = 7;
    grupo.add(malla);
  }
  const [zBase, rBase] = geometriaConfig.perfil_exterior_mm[0];
  const fondo = new THREE.Mesh(new THREE.CylinderGeometry(rBase, rBase, 2, 128), material.clone());
  fondo.rotation.x = Math.PI / 2;
  fondo.position.z = zBase + 1;
  fondo.name = 'Fondo conductor del crisol';
  fondo.userData.superficieTermica = 'pared';
  fondo.renderOrder = 7;
  grupo.add(fondo);

  const [zBoca, rBoca] = geometriaConfig.perfil_exterior_mm.at(-1);
  const rInterior = geometriaConfig.perfil_interior_mm.at(-1)[1];
  const labio = new THREE.Mesh(new THREE.RingGeometry(rInterior, rBoca, 128), material.clone());
  labio.position.z = zBoca;
  labio.name = 'Labio conductor del crisol';
  labio.userData.superficieTermica = 'pared';
  labio.renderOrder = 7;
  grupo.add(labio);

  const tapa = new THREE.Mesh(
    new THREE.CylinderGeometry(rBoca, rBoca, geometriaConfig.espesor_tapa_mm, 128),
    materialRecortable(THREE.MeshPhysicalMaterial, {
      color: 0x8da2ac, metalness: 0.28, roughness: 0.42, transparent: true,
      opacity: 0.52, depthWrite: false, side: THREE.DoubleSide,
    }),
  );
  tapa.rotation.x = Math.PI / 2;
  tapa.position.z = zBoca + geometriaConfig.espesor_tapa_mm / 2;
  tapa.name = 'Tapa cerrada del crisol del ensayo';
  tapa.userData.superficieTermica = 'tapa';
  tapa.renderOrder = 8;
  grupo.add(tapa);

  const bordes = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.LatheGeometry(perfilExterior, 48), 18),
    materialRecortable(THREE.LineBasicMaterial, { color: 0x9eb4bd, transparent: true, opacity: 0.24 }),
  );
  bordes.rotation.x = Math.PI / 2;
  bordes.renderOrder = 9;
  grupo.add(bordes);
  escena.add(grupo);
  estado.superficiesTermicas = [
    ...grupo.children.filter((objeto) => objeto.userData.superficieTermica),
  ];

  const lineas = [];
  for (let n = -20; n <= 20; n += 5) lineas.push(-20, n, 0, 20, n, 0, n, -20, 0, n, 20, 0);
  const base = new THREE.BufferGeometry();
  base.setAttribute('position', new THREE.Float32BufferAttribute(lineas, 3));
  const rejilla = new THREE.LineSegments(base, new THREE.LineBasicMaterial({ color: 0x2b424c, transparent: true, opacity: 0.46 }));
  rejilla.material.userData.recortable = false;
  escena.add(rejilla);
}

function colorMetalTermico(temperaturaK) {
  const nivel = THREE.MathUtils.clamp((temperaturaK - 298.15) / (1173.15 - 298.15), 0, 1);
  const frio = new THREE.Color(0x788b94);
  const rojo = new THREE.Color(0xc94e28);
  const caliente = new THREE.Color(0xffb24b);
  return nivel < 0.68
    ? frio.clone().lerp(rojo, nivel / 0.68)
    : rojo.clone().lerp(caliente, (nivel - 0.68) / 0.32);
}

function actualizarSuperficiesTermicas(termico) {
  if (!termico) return;
  estado.superficiesTermicas.forEach((superficie) => {
    const esTapa = superficie.userData.superficieTermica === 'tapa';
    const temperatura = esTapa ? termico.T_tapa_K : termico.T_pared_media_K;
    const nivel = THREE.MathUtils.clamp((temperatura - 298.15) / 875, 0, 1);
    const neutra = luzNeutraActiva();
    superficie.material.color.copy(neutra ? new THREE.Color(0x9fb0b8) : colorMetalTermico(temperatura));
    superficie.material.emissive.copy(colorMetalTermico(temperatura))
      .multiplyScalar(neutra ? 0.0 : 0.52 * nivel);
    superficie.material.emissiveIntensity = neutra ? 0.0 : 0.25 + 0.95 * nivel;
    superficie.material.opacity = neutra
      ? (esTapa ? 0.30 : 0.22)
      : (esTapa ? 0.52 + 0.13 * nivel : 0.38 + 0.10 * nivel);
    superficie.material.needsUpdate = true;
  });
  $('#temperatura-mufla').textContent = `${formatear(termico.T_mufla_K - 273.15)} °C`;
  $('#temperatura-pared').textContent = `${formatear(termico.T_pared_media_K - 273.15)} °C`;
  $('#flujo-radiativo').textContent = `${formatear(termico.flujo_radiativo_W_m2 / 1000)} kW/m²`;
  actualizarFlechasRadiacion(termico);
}

function construirMufla() {
  const grupo = new THREE.Group();
  grupo.name = 'Mufla radiante a 900 C';
  const camisa = new THREE.Mesh(
    new THREE.CylinderGeometry(24.5, 24.5, 43, 64, 1, true),
    new THREE.MeshBasicMaterial({
      color: 0xd64a25, transparent: true, opacity: 0.055,
      side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending,
    }),
  );
  camisa.rotation.x = Math.PI / 2;
  camisa.position.z = 18.5;
  camisa.name = 'Entorno radiante de la mufla';
  grupo.add(camisa);
  for (const z of [-1.5, 9, 19.5, 30, 40.5]) {
    const aro = new THREE.Mesh(
      new THREE.TorusGeometry(24.5, 0.035, 5, 96),
      new THREE.MeshBasicMaterial({ color: 0xe15b32, transparent: true, opacity: 0.28 }),
    );
    aro.position.z = z;
    grupo.add(aro);
  }

  const rutas = [];
  for (const z of [5, 16, 27]) for (let n = 0; n < 12; n += 1) {
    const angulo = 2 * Math.PI * n / 12 + (z / 11) * 0.12;
    const exterior = new THREE.Vector3(22.6 * Math.cos(angulo), 22.6 * Math.sin(angulo), z);
    const destino = new THREE.Vector3(14.9 * Math.cos(angulo), 14.9 * Math.sin(angulo), z);
    rutas.push({ exterior, destino });
  }
  for (let n = 0; n < 12; n += 1) {
    const angulo = 2 * Math.PI * n / 12;
    const radio = n % 2 ? 9.5 : 4.8;
    rutas.push({
      exterior: new THREE.Vector3(radio * Math.cos(angulo), radio * Math.sin(angulo), 40.5),
      destino: new THREE.Vector3(radio * Math.cos(angulo), radio * Math.sin(angulo), 34.2),
    });
  }

  const materialFlecha = new THREE.MeshStandardMaterial({
    color: 0xff9d3b, emissive: 0xff5b16, emissiveIntensity: 1.1,
    roughness: 0.42, transparent: true, opacity: 0.78, depthWrite: false,
  });
  const astas = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.045, 0.045, 1, 6), materialFlecha, rutas.length);
  const puntas = new THREE.InstancedMesh(new THREE.ConeGeometry(0.16, 0.46, 8), materialFlecha.clone(), rutas.length);
  astas.name = 'Flechas de radiacion mufla hacia crisol';
  puntas.name = 'Puntas de flujo radiativo incidente';
  astas.frustumCulled = false;
  puntas.frustumCulled = false;
  grupo.add(astas, puntas);
  estado.flechasRadiacion = { rutas, astas, puntas };

  const posiciones = new Float32Array(rutas.length * 3);
  const geometriaPulsos = new THREE.BufferGeometry();
  geometriaPulsos.setAttribute('position', new THREE.Float32BufferAttribute(posiciones, 3));
  const puntos = new THREE.Points(
    geometriaPulsos,
    new THREE.PointsMaterial({
      color: 0xffca6a, size: 0.78, transparent: true, opacity: 0.92,
      map: texturaPuntoRedondo(), alphaTest: 0.04,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
    }),
  );
  puntos.name = 'Pulsos de radiacion incidente';
  grupo.add(puntos);
  estado.pulsosRadiacion = { rutas, puntos, intensidad: 1 };
  estado.mufla = grupo;
  escena.add(grupo);
}

function actualizarFlechasRadiacion(termico) {
  if (!estado.flechasRadiacion) return;
  const intensidad = THREE.MathUtils.clamp(termico.flujo_radiativo_W_m2 / 85000, 0.08, 1);
  const matriz = new THREE.Matrix4();
  const ejeY = new THREE.Vector3(0, 1, 0);
  estado.flechasRadiacion.rutas.forEach(({ exterior, destino }, indice) => {
    const direccion = destino.clone().sub(exterior).normalize();
    const largo = exterior.distanceTo(destino) * (0.25 + 0.75 * intensidad);
    const cuaternion = new THREE.Quaternion().setFromUnitVectors(ejeY, direccion);
    matriz.compose(exterior.clone().addScaledVector(direccion, largo * 0.40), cuaternion, new THREE.Vector3(1, largo * 0.72, 1));
    estado.flechasRadiacion.astas.setMatrixAt(indice, matriz);
    matriz.compose(exterior.clone().addScaledVector(direccion, largo * 0.82), cuaternion, new THREE.Vector3(1, 1, 1));
    estado.flechasRadiacion.puntas.setMatrixAt(indice, matriz);
  });
  estado.flechasRadiacion.astas.instanceMatrix.needsUpdate = true;
  estado.flechasRadiacion.puntas.instanceMatrix.needsUpdate = true;
  estado.flechasRadiacion.astas.visible = $('#capa-radiacion').checked;
  estado.flechasRadiacion.puntas.visible = $('#capa-radiacion').checked;
  estado.pulsosRadiacion.intensidad = intensidad;
  estado.pulsosRadiacion.puntos.visible = $('#capa-radiacion').checked;
}

function animarPulsosRadiacion(deltaSegundos) {
  if (!estado.pulsosRadiacion || !estado.pulsosRadiacion.puntos.visible) return;
  estado.faseRadiacion = (estado.faseRadiacion + deltaSegundos * (0.32 + 0.9 * estado.pulsosRadiacion.intensidad)) % 1;
  const atributo = estado.pulsosRadiacion.puntos.geometry.getAttribute('position');
  estado.pulsosRadiacion.rutas.forEach(({ exterior, destino }, indice) => {
    const avance = (estado.faseRadiacion + indice * 0.173) % 1;
    atributo.setXYZ(
      indice,
      THREE.MathUtils.lerp(exterior.x, destino.x, avance),
      THREE.MathUtils.lerp(exterior.y, destino.y, avance),
      THREE.MathUtils.lerp(exterior.z, destino.z, avance),
    );
  });
  atributo.needsUpdate = true;
}

function construirSistemaParticulas() {
  estado.sistemaParticulas = new SistemaParticulasLecho({ clippingPlanes: planosActivos() });
  escena.add(estado.sistemaParticulas.grupo);
}

/**
 * Leyenda de fases con el color real de cada mineral.
 *
 * La paleta, el mapa de campos y el volumen molar llegan de
 * `fisica/fases_visuales.paleta_web()`: aquí no se codifica mineralogía, sólo
 * se presenta. Las fases iniciales van primero y las de producto después, que
 * es el orden en que aparecen al reproducir.
 */
function construirLeyendaFases() {
  const paleta = estado.config?.fases;
  const lista = $('#leyenda-fases');
  lista.innerHTML = '';
  if (!paleta?.fases || !Object.keys(paleta.fases).length) {
    $('#nota-fases').textContent = 'Paleta de fases no disponible en este servidor.';
    return;
  }
  const orden = paleta.orden_leyenda?.length ? paleta.orden_leyenda : Object.keys(paleta.fases);
  for (const clave of orden) {
    const fase = paleta.fases[clave];
    if (!fase) continue;
    const item = document.createElement('li');
    item.className = 'fase-item ausente';
    item.dataset.fase = clave;
    item.title = `${fase.descripcion || ''}${fase.densidad_g_cm3 ? ` · ${fase.densidad_g_cm3} g/cm³` : ''}`;
    const muestra = document.createElement('i');
    muestra.className = 'muestra-fase';
    muestra.style.background = fase.color;
    const nombre = document.createElement('span');
    nombre.className = 'fase-nombre';
    nombre.textContent = fase.nombre;
    const valor = document.createElement('span');
    valor.className = 'fase-valor';
    valor.textContent = '—';
    item.append(muestra, nombre, valor);
    lista.append(item);
  }
  estado.fasesVistas = new Map();
  $('#nota-fases').textContent = `${paleta.nota} Ponderación: ${paleta.ponderacion}.`;
}

/**
 * Fracción de VOLUMEN de cada fase sobre el sólido del lecho.
 *
 * Los campos vienen en mol/m³; el volumen molar M/rho los convierte a volumen,
 * que es lo que determina el aspecto de la mezcla. Ponderar por moles daría
 * demasiado peso a las fases de molécula pequeña.
 */
function fraccionesDeFase(f) {
  const paleta = estado.config?.fases;
  if (!paleta?.campos_solidos) return { fracciones: {}, total: 0 };
  const acumulado = {};
  let total = 0;
  for (const [campo, clave] of Object.entries(paleta.campos_solidos)) {
    const fase = clave ? paleta.fases[clave] : null;
    const valores = f.solido?.[campo];
    if (!fase || !valores || !(fase.volumen_molar_cm3_mol > 0)) continue;
    let suma = 0;
    for (let q = 0; q < valores.length; q += 1) if (f.etiquetas[q] === 3) suma += valores[q];
    const volumen = suma * fase.volumen_molar_cm3_mol;
    if (!(volumen > 0)) continue;
    acumulado[clave] = (acumulado[clave] || 0) + volumen;
    total += volumen;
  }
  if (!(total > 0)) return { fracciones: {}, total: 0 };
  const fracciones = {};
  for (const [clave, volumen] of Object.entries(acumulado)) fracciones[clave] = volumen / total;
  return { fracciones, total };
}

/**
 * Magnetización de saturación del lecho a temperatura ambiente, en A m²/kg.
 *
 * Igual que con la mineralogía, aquí NO se codifica ningún dato magnético: las
 * M_s por fase y las masas molares llegan del servidor en `config.magnetismo` y
 * `config.fases`. Esto sólo suma.
 *
 * Devuelve las dos cotas. `temple` conserva la wüstita, que a temperatura
 * ambiente no responde al imán; `lenta` la descompone entera por la vía
 * eutectoide en magnetita más hierro, que sí responden.
 */
function magnetizacionDelLecho(f) {
  const mag = estado.config?.magnetismo;
  const paleta = estado.config?.fases;
  if (!mag?.magnetizacion_saturacion_Am2_kg || !paleta?.campos_solidos) return null;
  const dx = (f.x[1] - f.x[0]) / 1000; const dy = (f.y[1] - f.y[0]) / 1000; const dz = (f.z[1] - f.z[0]) / 1000;
  const volumenCelda = dx * dy * dz;
  let momento = 0; let momentoLento = 0; let masa = 0; let molesFeO = 0;
  for (const [campo, clave] of Object.entries(paleta.campos_solidos)) {
    const valores = f.solido?.[campo];
    const fase = clave ? paleta.fases[clave] : null;
    if (!valores || !fase || !(fase.masa_molar_g_mol > 0)) continue;
    // Todo el sólido, sin filtrar por etiqueta: el 12 % de la carga está en
    // celdas cortadas por la frontera del lecho, que llevan etiqueta de pared o
    // de gas porque su centro cae fuera.
    let suma = 0;
    for (let q = 0; q < valores.length; q += 1) suma += valores[q];
    const moles = suma * volumenCelda;                       // mol
    const masaFase = moles * fase.masa_molar_g_mol / 1000;   // kg
    masa += masaFase;
    momento += masaFase * (mag.magnetizacion_saturacion_Am2_kg[campo] || 0);
    if (campo === 'FeTiO3') momento += masaFase * mag.ms_titanohematita_Am2_kg * mag.razon_titanohematita_sobre_ilmenita;
    if (campo === 'FeO') molesFeO += moles;
  }
  if (!(masa > 0)) return null;
  const eut = mag.eutectoide || {};
  const mFe3O4 = paleta.fases.Fe3O4?.masa_molar_g_mol || 0;
  const mFe = paleta.fases.Fe?.masa_molar_g_mol || 0;
  momentoLento = momento
    + molesFeO * (eut.moles_Fe3O4_por_mol_FeO || 0) * mFe3O4 / 1000 * (mag.magnetizacion_saturacion_Am2_kg.Fe3O4 || 0)
    + molesFeO * (eut.moles_Fe_por_mol_FeO || 0) * mFe / 1000 * (mag.magnetizacion_saturacion_Am2_kg.Fe || 0);
  return { temple: momento / masa, lenta: momentoLento / masa };
}

/** Volumen del aglomerado: celdas por encima del umbral operativo 0,5. */
function volumenAglomeradoCm3(f) {
  const dx = (f.x[1] - f.x[0]) / 10; const dy = (f.y[1] - f.y[0]) / 10; const dz = (f.z[1] - f.z[0]) / 10;
  const volumenCelda = dx * dy * dz;
  let celdas = 0; let celdasLecho = 0;
  for (let q = 0; q < f.cohesion.length; q += 1) {
    if (f.etiquetas[q] !== 3) continue;
    celdasLecho += 1;
    if (f.cohesion[q] >= 0.5) celdas += 1;
  }
  return { volumen: celdas * volumenCelda, fraccion: celdas / Math.max(celdasLecho, 1) };
}

function actualizarLeyendaFases(f) {
  const lista = $('#leyenda-fases');
  if (!lista.children.length || !f) return;
  const { fracciones } = fraccionesDeFase(f);
  const aglomerado = volumenAglomeradoCm3(f);
  // El hinchamiento se informa siempre, exista o no aglomerado cohesionado: la
  // masa plástica se infla antes de resolidificar.
  const hinchado = factorHinchamiento(f);
  const filaHinchamiento = $('#valor-hinchamiento');
  if (filaHinchamiento) {
    filaHinchamiento.textContent = hinchado > 1.0005
      ? `×${formatear(hinchado)} en volumen · +${formatear(100 * (hinchado - 1))} %`
      : 'todavía no hincha';
  }
  let presentes = 0;
  for (const item of lista.children) {
    const clave = item.dataset.fase;
    const valor = item.querySelector('.fase-valor');
    if (clave === 'aglomerado') {
      const hay = aglomerado.fraccion > 0;
      item.classList.toggle('ausente', !hay);
      valor.textContent = hay
        ? `${formatear(aglomerado.volumen * hinchado)} cm³ · ${formatear(100 * aglomerado.fraccion)} %`
        : 'aún no cuaja';
      continue;
    }
    const fraccion = fracciones[clave] || 0;
    const hay = fraccion > 1e-4;
    item.classList.toggle('ausente', !hay);
    if (hay) presentes += 1;
    valor.textContent = hay ? `${formatear(100 * fraccion)} %` : '—';
    // «Nueva» marca la aparición reciente de una fase de producto: es lo que
    // permite ver surgir cada fase con su color. Se guarda el instante de la
    // primera aparición y el resalte dura 45 s de tiempo simulado.
    if (hay && !estado.fasesVistas.has(clave)) estado.fasesVistas.set(clave, f.t);
    const aparicion = estado.fasesVistas.get(clave);
    const reciente = hay && aparicion !== undefined && f.t - aparicion < 45
      && estado.config.fases.fases[clave]?.estado === 'producto';
    item.classList.toggle('nueva', Boolean(reciente));
    if (reciente) item.title = `${item.title.split(' · Apareció')[0]} · Apareció a t = ${formatear(aparicion)} s`;
  }
  $('#fases-activas').textContent = `${presentes} presentes · predicción`;
  actualizarPanelMagnetismo(f);
}

/** Panel de la prueba del imán: las dos cotas y cuánto queda del inicial. */
function actualizarPanelMagnetismo(f) {
  const destino = $('#valor-magnetizacion');
  if (!destino) return;
  const m = magnetizacionDelLecho(f);
  if (!m) { destino.textContent = 'no disponible'; return; }
  if (estado.magnetizacionInicial === undefined && f.t <= 0.001) {
    estado.magnetizacionInicial = m.temple;
  }
  destino.textContent = `${formatear(m.temple)} A m²/kg`;
  $('#valor-magnetizacion-lenta').textContent = `${formatear(m.lenta)} A m²/kg`;
  const inicial = estado.magnetizacionInicial;
  $('#magnetismo-relativo').textContent = inicial > 0
    ? `${formatear(100 * m.temple / inicial)} % del inicial`
    : '—';
}

function configurarAyudantesCorte() {
  Object.values(ayudantesCorte).forEach((ayudante) => {
    ayudante.visible = false;
    ayudante.material.userData.recortable = false;
    escena.add(ayudante);
  });
}

function actualizarRecorte(reconstruirVolumen = false) {
  for (const eje of ['x', 'y', 'z']) {
    const valor = Number($(`#corte-${eje}`).value);
    planosCorte[eje].constant = valor;
    $(`#valor-corte-${eje}`).textContent = `${valor.toLocaleString('es-CO', { maximumFractionDigits: 1 })} mm`;
    ayudantesCorte[eje].visible = $('#mostrar-planos').checked && $(`#activar-corte-${eje}`).checked;
  }
  const activos = planosActivos();
  escena.traverse((objeto) => {
    const materiales = Array.isArray(objeto.material) ? objeto.material : [objeto.material];
    materiales.filter(Boolean).forEach((material) => {
      if (material.userData.recortable) {
        const cambioPrograma = (material.clippingPlanes?.length || 0) !== activos.length;
        material.clippingPlanes = activos;
        if (cambioPrograma) material.needsUpdate = true;
      }
    });
  });
  if (reconstruirVolumen) agendarReconstruccion();
}

function gradiente(escalar, f) {
  const [nx, ny, nz] = f.forma;
  const dx = (f.x[1] - f.x[0]) / 1000;
  const dy = (f.y[1] - f.y[0]) / 1000;
  const dz = (f.z[1] - f.z[0]) / 1000;
  const gx = new Float32Array(escalar.length);
  const gy = new Float32Array(escalar.length);
  const gz = new Float32Array(escalar.length);
  for (let i = 0; i < nx; i += 1) for (let j = 0; j < ny; j += 1) for (let k = 0; k < nz; k += 1) {
    const q = indice3(i, j, k, f.forma);
    const im = Math.max(i - 1, 0); const ip = Math.min(i + 1, nx - 1);
    const jm = Math.max(j - 1, 0); const jp = Math.min(j + 1, ny - 1);
    const km = Math.max(k - 1, 0); const kp = Math.min(k + 1, nz - 1);
    gx[q] = (escalar[indice3(ip, j, k, f.forma)] - escalar[indice3(im, j, k, f.forma)]) / ((ip - im) * dx || dx);
    gy[q] = (escalar[indice3(i, jp, k, f.forma)] - escalar[indice3(i, jm, k, f.forma)]) / ((jp - jm) * dy || dy);
    gz[q] = (escalar[indice3(i, j, kp, f.forma)] - escalar[indice3(i, j, km, f.forma)]) / ((kp - km) * dz || dz);
  }
  return [gx, gy, gz];
}

function campoSeleccionado() {
  const f = estado.fotograma;
  const seleccion = $('#campo').value;
  const especie = $('#especie').value;
  if (seleccion === 'velocidad') return { valores: f.u.map((u, q) => Math.hypot(u, f.v[q], f.w[q])), nombre: 'Magnitud de velocidad', unidad: 'm/s', soloLecho: false };
  if (seleccion === 'P') return { valores: f.P, nombre: 'Presión', unidad: 'Pa', soloLecho: false };
  if (seleccion === 'T') return { valores: f.T, nombre: 'Temperatura', unidad: 'K', soloLecho: false };
  if (seleccion === 'concentracion') return { valores: f.c_especies[especie], nombre: `Concentración de ${especie}`, unidad: 'mol/m³', soloLecho: false };
  if (seleccion === 'conversion') return { valores: f.conversion, nombre: 'Conversión de Fe₂O₃', unidad: '1', soloLecho: true };
  if (seleccion === 'reduccion') return { valores: f.reduccion, nombre: 'Grado de reducción local', unidad: '1', soloLecho: true };
  if (seleccion === 'cohesion') return { valores: f.cohesion, nombre: 'Cohesión del aglomerado', unidad: '1', soloLecho: true };
  if (seleccion === 'eps') return { valores: f.eps, nombre: 'Porosidad', unidad: '1', soloLecho: true };
  const concentracion = f.c_especies[especie];
  const [gx, gy, gz] = gradiente(concentracion, f);
  const D = 1.6e-5;
  return {
    valores: concentracion.map((c, q) => Math.log10((Math.hypot(f.u[q], f.v[q], f.w[q]) * Math.max(c, 1e-12) + 1e-12) / (D * Math.hypot(gx[q], gy[q], gz[q]) + 1e-12))),
    nombre: `Dominio advectivo / difusivo (${especie})`, unidad: 'log₁₀(A/D)', soloLecho: false, divergente: true,
  };
}

function calcularLimites(campo) {
  const f = estado.fotograma;
  let validos = campo.valores.filter((v, q) => Number.isFinite(v) && (campo.soloLecho ? f.etiquetas[q] === 3 : (f.etiquetas[q] === 3 || f.etiquetas[q] === 4)));
  if ($('#tipo-escala').value === 'log') validos = validos.filter((v) => v > 0);
  let minimo = percentil(validos, 0.02);
  let maximo = percentil(validos, 0.98);
  if (!$('#limites-auto').checked) {
    minimo = Number($('#limite-min').value);
    maximo = Number($('#limite-max').value);
  } else {
    $('#limite-min').value = Number.isFinite(minimo) ? minimo.toPrecision(5) : 0;
    $('#limite-max').value = Number.isFinite(maximo) ? maximo.toPrecision(5) : 1;
  }
  if (!(maximo > minimo)) maximo = minimo + 1;
  return [minimo, maximo];
}

function normalizar(valor, minimo = estado.limites[0], maximo = estado.limites[1]) {
  if ($('#tipo-escala').value === 'log') {
    const piso = Math.max(minimo, 1e-12);
    return (Math.log10(Math.max(valor, piso)) - Math.log10(piso)) / (Math.log10(Math.max(maximo, piso * 1.0001)) - Math.log10(piso));
  }
  return (valor - minimo) / (maximo - minimo || 1);
}

function dimensionesVolumen(f) {
  const dx = f.x[1] - f.x[0]; const dy = f.y[1] - f.y[0]; const dz = f.z[1] - f.z[0];
  return {
    dimensiones: new THREE.Vector3(f.x.at(-1) - f.x[0] + dx, f.y.at(-1) - f.y[0] + dy, f.z.at(-1) - f.z[0] + dz),
    centro: new THREE.Vector3((f.x[0] + f.x.at(-1)) / 2, (f.y[0] + f.y.at(-1)) / 2, (f.z[0] + f.z.at(-1)) / 2),
  };
}

function construirCampoVolumetrico(campo) {
  const f = estado.fotograma;
  const nombrePaleta = campo.divergente ? 'divergente' : $('#paleta').value;
  const normalizador = (v) => normalizar(v);
  const opacidad = Number($('#opacidad').value) / 100;
  const { dimensiones, centro } = dimensionesVolumen(f);
  let objeto;
  if ($('#modo-volumen').value === 'raymarch') {
    const textura = crearTexturaVolumetrica(campo.valores, f.etiquetas, f.forma, normalizador, campo.soloLecho);
    const paleta = crearTexturaPaleta((t) => interpolarColor(t, nombrePaleta));
    const cajaMax = new THREE.Vector3(0.5, 0.5, 0.5);
    ['x', 'y', 'z'].forEach((eje) => {
      if ($(`#activar-corte-${eje}`).checked) cajaMax[eje] = THREE.MathUtils.clamp((Number($(`#corte-${eje}`).value) - centro[eje]) / dimensiones[eje], -0.5, 0.5);
    });
    objeto = crearVolumenRayMarching({
      textura, paleta, dimensiones, centro, clippingPlanes: planosActivos(), cajaMax,
      opacidad, pasos: Number($('#calidad-volumen').value),
    });
    estado.volumenDinamico = { textura, soloLecho: campo.soloLecho };
  } else {
    objeto = crearPlanosApilados({
      valores: campo.valores, etiquetas: f.etiquetas, forma: f.forma,
      ejes: { x: f.x, y: f.y, z: f.z }, limites: estado.limites,
      normalizar: normalizador, interpolarColor: (t) => interpolarColor(t, nombrePaleta),
      clippingPlanes: planosActivos(), opacidad, soloLecho: campo.soloLecho,
    });
  }
  objeto.userData.volumetrico = true;
  escena.add(objeto);
  estado.dinamicos.push(objeto);
}

function construirFlechasInstanciadas(componentes, color, separacion = 3, escala = 4.0, opacidad = 0.9) {
  const f = estado.fotograma;
  const [ax, ay, az] = componentes;
  const [nx, ny, nz] = f.forma;
  let maximo = 0;
  for (let q = 0; q < ax.length; q += 1) if (f.etiquetas[q] === 3 || f.etiquetas[q] === 4) maximo = Math.max(maximo, Math.hypot(ax[q], ay[q], az[q]));
  const datos = [];
  for (let i = 1; i < nx - 1; i += separacion) for (let j = 1; j < ny - 1; j += separacion) for (let k = 1; k < nz - 1; k += separacion) {
    const q = indice3(i, j, k, f.forma);
    if (f.etiquetas[q] !== 3 && f.etiquetas[q] !== 4) continue;
    const magnitud = Math.hypot(ax[q], ay[q], az[q]);
    if (magnitud < maximo * 0.035) continue;
    datos.push({ posicion: new THREE.Vector3(f.x[i], f.y[j], f.z[k]), direccion: new THREE.Vector3(ax[q], ay[q], az[q]).normalize(), largo: escala * (0.25 + 0.75 * magnitud / (maximo || 1)) });
  }
  if (!datos.length) return;
  const material = materialRecortable(THREE.MeshStandardMaterial, { color, emissive: color, emissiveIntensity: 0.24, roughness: 0.38, transparent: opacidad < 1, opacity: opacidad, depthWrite: false });
  const astas = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.035, 0.035, 1, 7), material, datos.length);
  const puntas = new THREE.InstancedMesh(new THREE.ConeGeometry(0.11, 0.30, 9), material.clone(), datos.length);
  const ejeY = new THREE.Vector3(0, 1, 0);
  const matriz = new THREE.Matrix4();
  datos.forEach((dato, indice) => {
    const cuaternion = new THREE.Quaternion().setFromUnitVectors(ejeY, dato.direccion);
    matriz.compose(dato.posicion.clone().addScaledVector(dato.direccion, dato.largo * 0.325), cuaternion, new THREE.Vector3(1, dato.largo * 0.65, 1));
    astas.setMatrixAt(indice, matriz);
    matriz.compose(dato.posicion.clone().addScaledVector(dato.direccion, dato.largo * 0.80), cuaternion, new THREE.Vector3(1, dato.largo, 1));
    puntas.setMatrixAt(indice, matriz);
  });
  astas.instanceMatrix.needsUpdate = true; puntas.instanceMatrix.needsUpdate = true;
  astas.renderOrder = 5; puntas.renderOrder = 5;
  escena.add(astas, puntas);
  estado.dinamicos.push(astas, puntas);
}

function construirCapasVectoriales() {
  const f = estado.fotograma;
  if ($('#capa-flechas').checked) construirFlechasInstanciadas([f.u, f.v, f.w], 0x56b4e9, 3, 4.0, 0.88);
  if ($('#capa-calor').checked) construirFlechasInstanciadas(gradiente(f.T, f).map((a) => a.map((v) => -0.6 * v)), 0xe69f00, 3, 3.7, 0.86);
  const especie = f.c_especies[$('#especie').value];
  if ($('#capa-advectivo').checked) construirFlechasInstanciadas([f.u.map((v, q) => v * especie[q]), f.v.map((v, q) => v * especie[q]), f.w.map((v, q) => v * especie[q])], 0x7ac36a, 3, 3.5, 0.86);
  if ($('#capa-difusivo').checked) construirFlechasInstanciadas(gradiente(especie, f).map((a) => a.map((v) => -1.6e-5 * v)), 0xcc79a7, 3, 3.5, 0.86);
}

/**
 * Factor de hinchamiento representativo del aglomerado en la instantánea.
 *
 * El carbón tiene índice de hinchamiento 8 y la magnetita lo atenúa; el campo
 * `hinchamiento` viene calculado por `fisica/hinchamiento.py`. Aquí sólo se
 * promedia sobre las celdas que ya cohesionaron, que son las que forman cuerpo.
 */
function factorHinchamiento(f) {
  if (!f?.hinchamiento) return 1;
  // Media sobre el LECHO, sin ponderar por cohesión.
  //
  // Ponderar por cohesión daba 1 —«todavía no hincha»— justo en la fase que hay
  // que ver: hacia los 90 s el carbón está plástico y el gas atrapado ya lo ha
  // inflado un 57 %, pero aún no ha resolidificado, así que la cohesión es cero
  // y el promedio ponderado no tenía con qué pesar. Hinchar precede a cuajar.
  let suma = 0; let n = 0;
  for (let q = 0; q < f.hinchamiento.length; q += 1) {
    if (f.etiquetas[q] !== 3) continue;
    suma += f.hinchamiento[q]; n += 1;
  }
  return n > 0 ? suma / n : 1;
}

/**
 * Empuja la isosuperficie hacia arriba según el hinchamiento.
 *
 * La pared del crisol impide la expansión radial, así que la masa plástica sólo
 * puede crecer en vertical: la altura escala con todo el factor volumétrico
 * (`hinchamiento.altura_hinchada_mm` con confinamiento lateral). El suelo del
 * lecho no se mueve porque descansa sobre el fondo del crisol.
 */
function hincharGeometria(geometria, f, factor) {
  if (!(factor > 1.0001)) return;
  const dz = f.z[1] - f.z[0];
  let base = Infinity;
  for (let q = 0; q < f.etiquetas.length; q += 1) {
    if (f.etiquetas[q] !== 3) continue;
    const k = q % f.forma[2];
    base = Math.min(base, f.z[k] - dz / 2);
  }
  if (!Number.isFinite(base)) return;
  const posicion = geometria.getAttribute('position');
  for (let i = 0; i < posicion.count; i += 1) {
    const z = posicion.getZ(i);
    posicion.setZ(i, base + (z - base) * factor);
  }
  posicion.needsUpdate = true;
  geometria.computeVertexNormals();
  geometria.computeBoundingSphere();
}

function construirIsosuperficie(valores, umbral, color, opacidad = 0.34, tipo = 'T') {
  const f = estado.fotograma;
  const esAglomerado = tipo === 'cohesion';
  // Dos celdas de filtro gaussiano eliminan el escalonado que la malla gruesa
  // imprimía sobre la cohesión antes de ejecutar marching cubes.
  const campoSuave = esAglomerado ? suavizarCampoGaussiano(valores, f.forma, 2, 1.05) : valores;
  const geometria = extraerIsosuperficieMarchingCubes(campoSuave, f.forma, { x: f.x, y: f.y, z: f.z }, umbral, f.etiquetas);
  if (esAglomerado) hincharGeometria(geometria, f, factorHinchamiento(f));
  const parametros = esAglomerado ? {
    color: 0x20221f,
    emissive: 0x050605,
    emissiveIntensity: 0.025,
    metalness: 0.015,
    roughness: 0.96,
    transparent: true,
    opacity: 0.82,
    depthWrite: true,
    side: THREE.DoubleSide,
  } : {
    color, emissive: color, emissiveIntensity: 0.09, metalness: 0.05, roughness: 0.42,
    transparent: true, opacity: opacidad, depthWrite: false, side: THREE.DoubleSide,
  };
  const iso = new THREE.Mesh(geometria, materialRecortable(THREE.MeshStandardMaterial, parametros));
  iso.name = esAglomerado ? 'Aglomerado de coque poroso mate' : 'Isosuperficie por marching cubes';
  iso.renderOrder = 6;
  escena.add(iso);
  estado.dinamicos.push(iso);
}

function actualizarIsosuperficies() {
  let isotermaYaVisible = false;
  if ($('#activar-iso').checked) {
    const tipo = $('#campo-iso').value;
    construirIsosuperficie(estado.fotograma[tipo], Number($('#umbral-iso').value), tipo === 'T' ? 0xf2c66d : tipo === 'conversion' ? 0x69b9d5 : 0x20221f, 0.32, tipo);
    isotermaYaVisible = tipo === 'T';
  }
  if ($('#capa-frente-termico').checked && !isotermaYaVisible) construirIsosuperficie(estado.fotograma.T, 900, 0xffa53a, 0.38, 'T');
  if ($('#capa-aglomerado').checked && $('#campo-iso').value !== 'cohesion') construirIsosuperficie(estado.fotograma.cohesion, 0.12, 0x20221f, 0.82, 'cohesion');
}

function construirLineasCorriente() {
  if (!$('#capa-lineas').checked || !estado.lineasDatos.length) return;
  const posicionesParticulas = [];
  estado.lineasDatos.forEach((datos) => {
    const geometria = new THREE.BufferGeometry().setFromPoints(datos.map((p) => new THREE.Vector3(...p)));
    const linea = new THREE.Line(geometria, materialRecortable(THREE.LineBasicMaterial, { color: 0x80d4ef, transparent: true, opacity: 0.48, depthWrite: false }));
    linea.renderOrder = 4;
    escena.add(linea); estado.dinamicos.push(linea);
    posicionesParticulas.push(...datos[0]);
  });
  const geometria = new THREE.BufferGeometry();
  geometria.setAttribute('position', new THREE.Float32BufferAttribute(posicionesParticulas, 3));
  const puntos = new THREE.Points(geometria, materialRecortable(THREE.PointsMaterial, {
    color: 0xd7f5ff, size: 1.15, sizeAttenuation: true, map: texturaPuntoRedondo(),
    alphaTest: 0.04, transparent: true, opacity: 0.92, depthWrite: false,
    blending: THREE.AdditiveBlending,
  }));
  puntos.renderOrder = 7;
  escena.add(puntos); estado.dinamicos.push(puntos); estado.particulas = puntos;
}

function animarParticulas(deltaSegundos) {
  if (!estado.particulas || !$('#capa-lineas').checked) return;
  estado.faseParticulas += deltaSegundos * 0.42;
  const atributo = estado.particulas.geometry.getAttribute('position');
  estado.lineasDatos.forEach((linea, i) => {
    const progreso = ((estado.faseParticulas + i * 0.137) % 1) * (linea.length - 1);
    const q = Math.floor(progreso); const mezcla = progreso - q;
    const a = linea[q]; const b = linea[Math.min(q + 1, linea.length - 1)];
    atributo.setXYZ(i, THREE.MathUtils.lerp(a[0], b[0], mezcla), THREE.MathUtils.lerp(a[1], b[1], mezcla), THREE.MathUtils.lerp(a[2], b[2], mezcla));
  });
  atributo.needsUpdate = true;
}

function liberarDinamicos() {
  estado.particulas = null;
  estado.volumenDinamico = null;
  estado.dinamicos.forEach((objeto) => {
    escena.remove(objeto);
    if (objeto.userData.volumetrico) liberarObjetoVolumetrico(objeto);
    else objeto.traverse((hijo) => {
      hijo.geometry?.dispose();
      const materiales = Array.isArray(hijo.material) ? hijo.material : [hijo.material];
      materiales.filter(Boolean).forEach((material) => material.dispose());
    });
  });
  estado.dinamicos = [];
}

function actualizarLeyenda(campo) {
  const [minimo, maximo] = estado.limites;
  const paleta = campo.divergente ? 'divergente' : $('#paleta').value;
  $('#leyenda-campo').textContent = campo.nombre;
  $('#leyenda-unidad').textContent = campo.unidad;
  $('#leyenda-min').textContent = formatear(minimo);
  $('#leyenda-max').textContent = formatear(maximo);
  $('#barra-color').style.background = gradienteCss(paleta);
  $('#lectura-campo').textContent = campo.nombre;
  $('#lectura-rango').textContent = `${formatear(minimo)} – ${formatear(maximo)} ${campo.unidad}`;
}

function reconstruir() {
  estado.reconstruccionPendiente = false;
  if (!estado.fotograma) return;
  liberarDinamicos();
  const campo = campoSeleccionado();
  estado.campoActivo = campo.valores;
  estado.nombreCampo = campo.nombre;
  estado.unidadCampo = campo.unidad;
  estado.limites = calcularLimites(campo);
  const modoRepresentacion = $('#modo-representacion').value;
  if (modoRepresentacion !== 'particulas') construirCampoVolumetrico(campo);
  if (estado.sistemaParticulas) {
    estado.sistemaParticulas.visible = modoRepresentacion !== 'campos';
    if (modoRepresentacion !== 'campos') estado.sistemaParticulas.actualizar(estado.fotograma);
  }
  construirCapasVectoriales();
  actualizarIsosuperficies();
  construirLineasCorriente();
  actualizarLeyenda(campo);
  actualizarLineaPerfil();
  actualizarRecorte(false);
}

function agendarReconstruccion() {
  if (estado.reconstruccionPendiente) return;
  estado.reconstruccionPendiente = true;
  requestAnimationFrame(reconstruir);
}

function actualizarRepresentacionInterpolada() {
  if (!estado.fotograma) return;
  const campo = campoSeleccionado();
  estado.campoActivo = campo.valores;
  estado.nombreCampo = campo.nombre;
  estado.unidadCampo = campo.unidad;
  if (estado.volumenDinamico) {
    actualizarTexturaVolumetrica(
      estado.volumenDinamico.textura,
      campo.valores,
      estado.fotograma.etiquetas,
      estado.fotograma.forma,
      (valor) => normalizar(valor),
      estado.volumenDinamico.soloLecho,
    );
  }
  if (estado.sistemaParticulas && $('#modo-representacion').value !== 'campos') {
    estado.sistemaParticulas.actualizar(estado.fotograma);
  }
}

function limitesDominio(f) {
  const dx = f.x[1] - f.x[0]; const dy = f.y[1] - f.y[0]; const dz = f.z[1] - f.z[0];
  return new THREE.Box3(new THREE.Vector3(f.x[0] - dx / 2, f.y[0] - dy / 2, f.z[0] - dz / 2), new THREE.Vector3(f.x.at(-1) + dx / 2, f.y.at(-1) + dy / 2, f.z.at(-1) + dz / 2));
}

function indiceCercano(coordenadas, valor) {
  let mejor = 0;
  for (let i = 1; i < coordenadas.length; i += 1) if (Math.abs(coordenadas[i] - valor) < Math.abs(coordenadas[mejor] - valor)) mejor = i;
  return mejor;
}

function celdaBajoPuntero(evento) {
  if (!estado.fotograma) return null;
  const rect = lienzo.getBoundingClientRect();
  puntero.set((evento.clientX - rect.left) / rect.width * 2 - 1, -(evento.clientY - rect.top) / rect.height * 2 + 1);
  raycaster.setFromCamera(puntero, camara);
  const f = estado.fotograma;
  const entrada = raycaster.ray.intersectBox(limitesDominio(f), new THREE.Vector3());
  if (!entrada) return null;
  const paso = Math.min(f.x[1] - f.x[0], f.y[1] - f.y[0], f.z[1] - f.z[0]) * 0.42;
  for (let distancia = 0; distancia < 55; distancia += paso) {
    const punto = entrada.clone().addScaledVector(raycaster.ray.direction, distancia);
    if (!limitesDominio(f).containsPoint(punto)) break;
    if (['x', 'y', 'z'].some((eje) => $(`#activar-corte-${eje}`).checked && punto[eje] > Number($(`#corte-${eje}`).value))) continue;
    const i = indiceCercano(f.x, punto.x); const j = indiceCercano(f.y, punto.y); const k = indiceCercano(f.z, punto.z);
    const q = indice3(i, j, k, f.forma);
    if (f.etiquetas[q] === 3 || f.etiquetas[q] === 4) return { i, j, k, q, punto: new THREE.Vector3(f.x[i], f.y[j], f.z[k]) };
  }
  return null;
}

/**
 * Textura de punto redondo con caída suave.
 *
 * `PointsMaterial` sin mapa dibuja el punto como un CUADRADO opaco: con
 * `size: 7` eso llenaba la vista de cuadrados blancos que no representaban
 * nada físico. Una textura radial con alfa convierte cada trazador en una
 * mota redonda, que es lo que se pretendía dibujar.
 */
let _texturaPuntoRedondo = null;
function texturaPuntoRedondo() {
  if (_texturaPuntoRedondo) return _texturaPuntoRedondo;
  const lado = 64;
  const datos = new Uint8Array(lado * lado * 4);
  for (let y = 0; y < lado; y += 1) {
    for (let x = 0; x < lado; x += 1) {
      const r = Math.hypot(x - (lado - 1) / 2, y - (lado - 1) / 2) / ((lado - 1) / 2);
      const q = (x + lado * y) * 4;
      // Núcleo brillante y halo que se apaga: legible sobre fondo claro y oscuro.
      const nucleo = Math.exp(-Math.pow(r / 0.34, 2));
      const halo = Math.max(0, 1 - r) ** 2;
      const alfa = THREE.MathUtils.clamp(nucleo + 0.45 * halo, 0, 1);
      datos[q] = 255; datos[q + 1] = 252; datos[q + 2] = 236;
      datos[q + 3] = Math.round(255 * alfa);
    }
  }
  const textura = new THREE.DataTexture(datos, lado, lado, THREE.RGBAFormat);
  textura.minFilter = THREE.LinearFilter;
  textura.magFilter = THREE.LinearFilter;
  textura.needsUpdate = true;
  _texturaPuntoRedondo = textura;
  return textura;
}

function texturaMarcador() {
  const lado = 32; const datos = new Uint8Array(lado * lado * 4);
  for (let y = 0; y < lado; y += 1) for (let x = 0; x < lado; x += 1) {
    const r = Math.hypot(x - 15.5, y - 15.5) / 15.5; const q = (x + lado * y) * 4;
    const alfa = r < 0.32 ? 255 : r < 0.85 ? Math.round(255 * (0.85 - r) / 0.53) : 0;
    datos[q] = 255; datos[q + 1] = 245; datos[q + 2] = 180; datos[q + 3] = alfa;
  }
  const textura = new THREE.DataTexture(datos, lado, lado, THREE.RGBAFormat);
  textura.needsUpdate = true;
  return textura;
}

function mostrarLecturaLocal(celda) {
  const f = estado.fotograma; const { i, j, k, q, punto } = celda;
  const rapidez = Math.hypot(f.u[q], f.v[q], f.w[q]);
  const gradienteT = gradiente(f.T, f);
  const especieActiva = $('#especie').value;
  const concentracion = f.c_especies[especieActiva];
  const gradienteC = gradiente(concentracion, f);
  const flujoCalor = 0.6 * Math.hypot(gradienteT[0][q], gradienteT[1][q], gradienteT[2][q]);
  const flujoAdvectivo = rapidez * concentracion[q];
  const flujoDifusivo = 1.6e-5 * Math.hypot(gradienteC[0][q], gradienteC[1][q], gradienteC[2][q]);
  $('#estado-sonda').textContent = `celda ${i}, ${j}, ${k}`;
  $('#lectura-coordenadas').textContent = `(${formatear(punto.x)}, ${formatear(punto.y)}, ${formatear(punto.z)}) mm`;
  $('#lectura-temperatura').textContent = `${formatear(f.T[q])} K`;
  $('#lectura-presion').textContent = `${formatear(f.P[q])} Pa`;
  $('#lectura-velocidad').textContent = `${formatear(rapidez)} m/s`;
  $('#lectura-vector-velocidad').textContent = `(${formatear(f.u[q])}, ${formatear(f.v[q])}, ${formatear(f.w[q])}) m/s`;
  $('#lectura-flujo-calor').textContent = `${formatear(flujoCalor)} W/m²`;
  $('#lectura-porosidad').textContent = formatear(f.eps[q]);
  $('#lectura-conversion').textContent = `${formatear(100 * f.conversion[q])} %`;
  $('#lectura-reduccion').textContent = `${formatear(100 * f.reduccion[q])} %`;
  $('#lectura-cohesion').textContent = `${formatear(100 * f.cohesion[q])} %`;
  $('#lectura-flujos-masa').textContent = `${especieActiva}: ${formatear(flujoAdvectivo)} / ${formatear(flujoDifusivo)} mol/(m²·s)`;
  $('#lectura-especies').textContent = estado.config.especies.map((especie) => `${especie} ${formatear(f.c_especies[especie][q])}`).join(' · ');
  $('#lectura-solidos').textContent = Object.entries(f.solido).map(([fase, valores]) => `${fase} ${formatear(valores[q])}`).join(' · ');
  if (!estado.marcador) {
    estado.marcador = new THREE.Sprite(new THREE.SpriteMaterial({ map: texturaMarcador(), transparent: true, depthTest: false, sizeAttenuation: true }));
    estado.marcador.name = 'Sonda Raycaster';
    estado.marcador.scale.set(2.1, 2.1, 1);
    estado.marcador.renderOrder = 20;
    escena.add(estado.marcador);
  }
  estado.marcador.position.copy(punto);
  estado.marcador.visible = true;
}

function seleccionarEnVolumen(evento) {
  const celda = celdaBajoPuntero(evento);
  if (!celda) return;
  if (estado.modoPerfil) {
    estado.puntosPerfil.push(celda.punto.clone());
    if (estado.puntosPerfil.length === 2) {
      estado.modoPerfil = false;
      $('#instruccion-perfil').classList.add('oculto');
      $('#trazar-linea').classList.remove('activo');
      actualizarLineaPerfil(); dibujarPerfil();
    } else $('#instruccion-perfil').textContent = 'Seleccione el punto final del perfil';
  } else mostrarLecturaLocal(celda);
}

function actualizarLineaPerfil() {
  if (estado.lineaPerfil) {
    escena.remove(estado.lineaPerfil); estado.lineaPerfil.geometry.dispose(); estado.lineaPerfil.material.dispose(); estado.lineaPerfil = null;
  }
  if (estado.puntosPerfil.length !== 2) return;
  const geometria = new THREE.BufferGeometry().setFromPoints(estado.puntosPerfil);
  estado.lineaPerfil = new THREE.Line(geometria, new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
  estado.lineaPerfil.renderOrder = 18;
  escena.add(estado.lineaPerfil);
}

function prepararGrafica(canvas) {
  const ctx = canvas.getContext('2d'); const w = canvas.width; const h = canvas.height;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#0f171c'; ctx.fillRect(0, 0, w, h); ctx.strokeStyle = '#253640';
  for (let i = 0; i < 5; i += 1) { const y = 15 + i * (h - 35) / 4; ctx.beginPath(); ctx.moveTo(34, y); ctx.lineTo(w - 8, y); ctx.stroke(); }
  ctx.fillStyle = '#84959f'; ctx.font = '10px Segoe UI'; return { ctx, w, h, m: { l: 34, r: 8, t: 15, b: 20 } };
}

function trazarSerie(ctx, datos, extraer, color, min, max, m, w, h) {
  ctx.beginPath(); datos.forEach((d, i) => { const x = m.l + i * (w - m.l - m.r) / Math.max(1, datos.length - 1); const y = m.t + (max - extraer(d)) / (max - min || 1) * (h - m.t - m.b); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
  ctx.strokeStyle = color; ctx.lineWidth = 1.7; ctx.stroke();
}

function dibujarSerie() {
  const c = prepararGrafica($('#grafica-serie')); const s = estado.serie;
  if (!s.length) { c.ctx.fillText('Cargando serie…', 105, 95); return; }
  const temperaturas = s.map((d) => d.T_media_K);
  trazarSerie(c.ctx, s, (d) => d.T_media_K, '#e69f00', Math.min(...temperaturas), Math.max(...temperaturas), c.m, c.w, c.h);
  trazarSerie(c.ctx, s, (d) => d.conversion_media, '#56b4e9', 0, 1, c.m, c.w, c.h);
  trazarSerie(c.ctx, s, (d) => d.cohesion_max, '#cc79a7', 0, 1, c.m, c.w, c.h);
  const x = c.m.l + estado.indice * (c.w - c.m.l - c.m.r) / Math.max(1, s.length - 1);
  c.ctx.strokeStyle = '#dbe4e8'; c.ctx.beginPath(); c.ctx.moveTo(x, c.m.t); c.ctx.lineTo(x, c.h - c.m.b); c.ctx.stroke();
}

function dibujarPerfil() {
  const c = prepararGrafica($('#grafica-perfil'));
  if (estado.puntosPerfil.length !== 2 || !estado.fotograma) { c.ctx.fillText('Trace una línea en la vista 3D', 82, 90); return; }
  const [a, b] = estado.puntosPerfil; const f = estado.fotograma; const valores = [];
  for (let s = 0; s < 64; s += 1) {
    const t = s / 63; const p = a.clone().lerp(b, t);
    valores.push(estado.campoActivo[indice3(indiceCercano(f.x, p.x), indiceCercano(f.y, p.y), indiceCercano(f.z, p.z), f.forma)]);
  }
  const min = Math.min(...valores); const max = Math.max(...valores);
  trazarSerie(c.ctx, valores, (d) => d, '#82c8c5', min, max, c.m, c.w, c.h);
  const longitud = a.distanceTo(b); $('#longitud-perfil').textContent = `${formatear(longitud)} mm`;
  $('#ayuda-perfil').textContent = `${estado.nombreCampo} · ${estado.unidadCampo} · predicción no validada`;
}

function actualizarPanelNumeros() {
  const n = estado.fotograma.numeros;
  // El origen lo declara el servidor: «solucionador» si la instantánea trae los
  // números calculados con las propiedades reales de cada celda, «demostración»
  // si son los de la función sintética. Rotularlo evita presentar como medida
  // del modelo lo que no lo es.
  const delSolucionador = n.origen === 'solucionador';
  const sello = delSolucionador ? 'Solucionador' : 'Sintético';
  $('#numero-re').textContent = formatear(n.Re); $('#numero-ra').textContent = formatear(n.Ra); $('#numero-pe').textContent = formatear(n.Pe); $('#numero-da').textContent = formatear(n.Da);
  $('#regimen-re').textContent = `${sello} · ${n.Re < 1 ? 'viscoso / Darcy' : n.Re < 2300 ? 'laminar' : 'transición turbulenta'}`;
  $('#regimen-ra').textContent = `${sello} · ${n.Ra < 1708 ? 'conducción dominante' : 'convección natural activa'}`;
  $('#regimen-pe').textContent = `${sello} · ${n.Pe < 1 ? 'difusión térmica domina' : 'advección térmica domina'}`;
  $('#regimen-da').textContent = `${sello} · ${n.Da < 1 ? 'transporte más rápido' : 'reacción dominante'}`;
  const titulo = $('#origen-numeros');
  if (titulo) {
    titulo.textContent = delSolucionador ? 'del solucionador' : 'campo sintético';
    titulo.title = delSolucionador
      ? 'Calculados por nucleo/momentum.numeros_adimensionales con las propiedades de cada celda'
      : 'Función de demostración: densidad y viscosidad fijas, Damköhler en rampa';
  }
}

function actualizarTiempo() {
  const t = estado.fotograma.t;
  $('#tiempo').value = estado.cursorReproduccion;
  $('#tiempo-actual').textContent = `${formatear(t)} s`;
  $('#estado-fotograma').textContent = `t = ${formatear(t)} s · ${FPS_REPRODUCCION} fps`;
  $('#punto-serie').textContent = `t = ${formatear(t)} s`;
}

function resumirSeriePrecargada() {
  return estado.cacheFotogramas.fotogramas.map((f) => {
    let sumaT = 0; let sumaCO = 0; let activos = 0; let sumaConversion = 0; let lecho = 0; let cohesionMax = 0;
    for (let q = 0; q < f.etiquetas.length; q += 1) {
      if (f.etiquetas[q] === 3 || f.etiquetas[q] === 4) {
        sumaT += f.T[q]; sumaCO += f.c_especies.CO[q]; activos += 1;
      }
      if (f.etiquetas[q] === 3) { sumaConversion += f.conversion[q]; lecho += 1; }
      cohesionMax = Math.max(cohesionMax, f.cohesion[q]);
    }
    return {
      t: f.t,
      T_media_K: sumaT / Math.max(activos, 1),
      CO_medio_mol_m3: sumaCO / Math.max(activos, 1),
      conversion_media: sumaConversion / Math.max(lecho, 1),
      cohesion_max: cohesionMax,
    };
  });
}

function aplicarCursorPrecargado(cursor, forzarReconstruccion = false) {
  const cache = estado.cacheFotogramas;
  if (!cache?.completa) return;
  const ultimo = cache.cantidad - 1;
  const posicion = THREE.MathUtils.clamp(cursor, 0, ultimo);
  const indiceA = Math.min(Math.floor(posicion), ultimo);
  const indiceB = Math.min(indiceA + 1, ultimo);
  const mezcla = posicion - indiceA;
  estado.cursorReproduccion = posicion;
  estado.indice = indiceA;
  estado.fotograma = interpolarFotogramas(
    cache.obtener(indiceA), cache.obtener(indiceB), mezcla, estado.fotogramaInterpolado,
  );
  const cambioFotogramaBase = indiceA !== estado.indiceReconstruido;
  if (cambioFotogramaBase) estado.lineasDatos = cache.obtenerLineas(indiceA);
  if (forzarReconstruccion || cambioFotogramaBase) {
    estado.indiceReconstruido = indiceA;
    reconstruir();
  } else {
    actualizarRepresentacionInterpolada();
  }
  actualizarSuperficiesTermicas(estado.fotograma.termico);
  actualizarPanelNumeros();
  actualizarLeyendaFases(estado.fotograma);
  actualizarTiempo();
  dibujarSerie();
  dibujarPerfil();
}

function seleccionarFotograma(indice) {
  estado.relojReproduccionMs = indice * DURACION_TRAMO_MS;
  estado.acumuladorReproduccionMs = 0;
  aplicarCursorPrecargado(indice, true);
}

function mostrarProgresoPrecarga(completados, total, transporte = '') {
  const contenedor = $('#precarga-serie');
  contenedor.hidden = false;
  contenedor.classList.remove('oculto');
  $('#progreso-precarga').max = total;
  $('#progreso-precarga').value = completados;
  $('#texto-precarga').textContent = `Precargando ${completados} / ${total}`;
  $('#detalle-precarga').textContent = transporte || 'Campos Float32 + trayectorias';
}

function ocultarIndicadoresCarga() {
  for (const selector of ['#mensaje-carga', '#precarga-serie']) {
    const elemento = $(selector);
    elemento.classList.add('oculto');
    elemento.hidden = true;
  }
}

function textoUmbral() {
  const valor = Number($('#umbral-iso').value);
  return $('#campo-iso').value === 'T' ? `${valor.toFixed(0)} K` : `${Math.round(valor * 100)} %`;
}

function configurarUmbralIso() {
  const tipo = $('#campo-iso').value; const control = $('#umbral-iso');
  if (tipo === 'T') { control.min = 300; control.max = 1350; control.step = 5; control.value = 900; }
  else { control.min = 0; control.max = 1; control.step = 0.01; control.value = tipo === 'conversion' ? 0.55 : 0.22; }
  $('#valor-iso').textContent = textoUmbral(); agendarReconstruccion();
}

function vistaPredefinida(tipo) {
  const posiciones = { alzado: [0, -78, 18], planta: [0.01, -0.01, 92], iso: [50, -52, 48], lecho: [12.5, -17.5, 8.5] };
  camara.position.fromArray(posiciones[tipo] || posiciones.iso);
  controles.target.set(0, 0, tipo === 'lecho' ? 3.63 : 16);
  controles.update();
  $$('[data-vista]').forEach((boton) => boton.classList.toggle('activo', boton.dataset.vista === tipo));
}

function descargar(blob, nombre) {
  const url = URL.createObjectURL(blob); const enlace = document.createElement('a'); enlace.href = url; enlace.download = nombre; enlace.click(); setTimeout(() => URL.revokeObjectURL(url), 500);
}

function exportarPng() {
  renderer.render(escena, camara);
  lienzo.toBlob((blob) => blob && descargar(blob, `prediccion_3d_t${Math.round(estado.fotograma.t)}s.png`), 'image/png');
}

function exportarCsv() {
  const cabecera = 'tiempo_s,temperatura_media_K,CO_medio_mol_m3,conversion_media,cohesion_max,estado_validacion\n';
  const filas = estado.serie.map((d) => [d.t, d.T_media_K, d.CO_medio_mol_m3, d.conversion_media, d.cohesion_max, 'PREDICCION_SINTETICA_NO_VALIDADA'].join(',')).join('\n');
  descargar(new Blob([cabecera + filas], { type: 'text/csv;charset=utf-8' }), 'serie_temporal_sintetica_no_validada.csv');
}

function configurarEventos() {
  let inicioPuntero = null;
  lienzo.addEventListener('pointerdown', (evento) => {
    inicioPuntero = [evento.clientX, evento.clientY];
    lienzo.focus({ preventScroll: true });
  });
  // OrbitControls escucha wheel sobre este mismo canvas. El listener explícito
  // impide que un contenedor desplazable robe el gesto de zoom.
  lienzo.addEventListener('wheel', (evento) => evento.preventDefault(), { passive: false });
  lienzo.addEventListener('pointerup', (evento) => {
    if (inicioPuntero && Math.hypot(evento.clientX - inicioPuntero[0], evento.clientY - inicioPuntero[1]) < 4) seleccionarEnVolumen(evento);
    inicioPuntero = null;
  });
  $('#campo').addEventListener('change', () => { agendarReconstruccion(); dibujarPerfil(); });
  $('#especie').addEventListener('change', agendarReconstruccion);
  $('#paleta').addEventListener('change', agendarReconstruccion);
  $('#tipo-escala').addEventListener('change', agendarReconstruccion);
  $('#modo-volumen').addEventListener('change', () => {
    $('#nota-render').textContent = $('#modo-volumen').value === 'raymarch' ? 'Data3DTexture + ShaderMaterial: integra el campo a través del volumen.' : 'Alternativa de DataTexture: menos coste, pero el apilado axial revela bandas al mirar de lado.';
    $('#calidad-volumen').disabled = $('#modo-volumen').value !== 'raymarch'; agendarReconstruccion();
  });
  $('#modo-representacion').addEventListener('change', agendarReconstruccion);
  $('#luz-neutra').addEventListener('change', aplicarIluminacion);
  $('#colorear-fases').addEventListener('change', (evento) => {
    if (estado.sistemaParticulas) estado.sistemaParticulas.colorearPorFase = evento.target.checked;
    agendarReconstruccion();
  });
  $('#calidad-volumen').addEventListener('change', agendarReconstruccion);
  $('#limites-auto').addEventListener('change', (evento) => { $('#limite-min').disabled = evento.target.checked; $('#limite-max').disabled = evento.target.checked; agendarReconstruccion(); });
  $('#limite-min').addEventListener('change', agendarReconstruccion); $('#limite-max').addEventListener('change', agendarReconstruccion);
  $('#opacidad').addEventListener('input', (evento) => { $('#valor-opacidad').textContent = `${evento.target.value} %`; agendarReconstruccion(); });
  $$('.grupo-capas input').forEach((control) => control.addEventListener('change', agendarReconstruccion));
  $('#capa-radiacion').addEventListener('change', () => {
    actualizarFlechasRadiacion(estado.fotograma?.termico || { flujo_radiativo_W_m2: 85000 });
  });
  $('#activar-iso').addEventListener('change', agendarReconstruccion); $('#campo-iso').addEventListener('change', configurarUmbralIso);
  $('#umbral-iso').addEventListener('input', () => { $('#valor-iso').textContent = textoUmbral(); agendarReconstruccion(); });
  ['x', 'y', 'z'].forEach((eje) => {
    $(`#corte-${eje}`).addEventListener('input', () => actualizarRecorte(true));
    $(`#activar-corte-${eje}`).addEventListener('change', () => actualizarRecorte(true));
  });
  $('#mostrar-planos').addEventListener('change', () => actualizarRecorte(false));
  $$('[data-vista]').forEach((boton) => boton.addEventListener('click', () => vistaPredefinida(boton.dataset.vista)));
  $('#encuadrar').addEventListener('click', () => vistaPredefinida('iso'));
  $('#tiempo').addEventListener('input', (evento) => {
    estado.reproduciendo = false;
    $('#reproducir').textContent = '▶';
    aplicarCursorPrecargado(Number(evento.target.value), false);
    estado.relojReproduccionMs = Number(evento.target.value) * DURACION_TRAMO_MS;
  });
  $('#retroceder').addEventListener('click', () => seleccionarFotograma(Math.max(0, Math.floor(estado.cursorReproduccion) - 1)));
  $('#avanzar').addEventListener('click', () => seleccionarFotograma(Math.min(estado.config.n_fotogramas - 1, Math.floor(estado.cursorReproduccion) + 1)));
  $('#reproducir').addEventListener('click', () => {
    if (!estado.cacheFotogramas?.completa) return;
    if (!estado.reproduciendo && estado.cursorReproduccion >= estado.config.n_fotogramas - 1) seleccionarFotograma(0);
    estado.reproduciendo = !estado.reproduciendo;
    estado.ultimoTiempoReproduccion = null;
    $('#reproducir').textContent = estado.reproduciendo ? 'Ⅱ' : '▶';
    $('#reproducir').ariaLabel = estado.reproduciendo ? 'Pausar' : 'Reproducir';
  });
  $('#velocidad-reproduccion').addEventListener('change', (evento) => { estado.velocidad = Number(evento.target.value); });
  $('#trazar-linea').addEventListener('click', () => { estado.modoPerfil = true; estado.puntosPerfil = []; actualizarLineaPerfil(); $('#trazar-linea').classList.add('activo'); $('#instruccion-perfil').textContent = 'Seleccione el punto inicial del perfil'; $('#instruccion-perfil').classList.remove('oculto'); });
  $('#exportar-png').addEventListener('click', exportarPng); $('#exportar-csv').addEventListener('click', exportarCsv);
}

function redimensionar() {
  const rect = visor.getBoundingClientRect(); renderer.setSize(rect.width, rect.height, false); camara.aspect = rect.width / Math.max(1, rect.height); camara.updateProjectionMatrix();
}

function animar(tiempo) {
  requestAnimationFrame(animar);
  const delta = Math.min((tiempo - (animar.anterior || tiempo)) / 1000, 0.1); animar.anterior = tiempo;
  // Reproducir con lo que ya esté precargado.
  //
  // Antes esta condición exigía `estado.cacheFotogramas.completa`, es decir, que
  // TODA la serie estuviera descargada. Con 51 fotogramas del solucionador real
  // esa condición no llegaba a cumplirse y el botón de reproducir no hacía nada:
  // la línea de tiempo se quedaba clavada en t=0 aunque los datos estuvieran
  // disponibles. Basta con tener cargado el fotograma siguiente; si el cursor
  // alcanza el borde de lo precargado, la reproducción espera ahí en vez de
  // abortar.
  // `estado.config` es null hasta que responde /api/config, y el bucle de
  // animación arranca antes: sin el encadenamiento opcional se lanzaba una
  // excepción por fotograma durante toda la carga. No rompía la reproducción
  // —el rAF siguiente ya está pedido cuando salta— pero llenaba la consola y
  // tapaba errores de verdad.
  const listos = estado.cacheFotogramas?.listos ?? estado.config?.n_fotogramas ?? 0;
  if (estado.reproduciendo && !estado.cargando && listos > 1) {
    if (estado.ultimoTiempoReproduccion === null) estado.ultimoTiempoReproduccion = tiempo;
    const transcurrido = Math.min(tiempo - estado.ultimoTiempoReproduccion, 200);
    estado.ultimoTiempoReproduccion = tiempo;
    estado.acumuladorReproduccionMs += transcurrido;
    if (estado.acumuladorReproduccionMs >= PASO_REPRODUCCION_MS) {
      const pasos = Math.floor(estado.acumuladorReproduccionMs / PASO_REPRODUCCION_MS);
      estado.acumuladorReproduccionMs -= pasos * PASO_REPRODUCCION_MS;
      estado.relojReproduccionMs += pasos * PASO_REPRODUCCION_MS * estado.velocidad;
      const ultimo = estado.config.n_fotogramas - 1;
      // no rebasar lo que hay descargado: se detiene en el borde y sigue cuando llega
      const tope = Math.min(ultimo, Math.max(0, listos - 1));
      let cursor = estado.relojReproduccionMs / DURACION_TRAMO_MS;
      if (cursor > tope) {
        cursor = tope;
        estado.relojReproduccionMs = tope * DURACION_TRAMO_MS;
      }
      aplicarCursorPrecargado(Math.min(cursor, ultimo), false);
      if (cursor >= ultimo) {
        estado.reproduciendo = false;
        $('#reproducir').textContent = '▶';
        $('#reproducir').ariaLabel = 'Reproducir';
      }
    }
  } else {
    estado.ultimoTiempoReproduccion = null;
  }
  controles.update(); animarParticulas(delta); animarPulsosRadiacion(delta); renderer.render(escena, camara);
}

async function iniciar() {
  instalarIluminacion(); construirMufla(); configurarAyudantesCorte(); configurarEventos(); redimensionar();
  new ResizeObserver(redimensionar).observe(visor); requestAnimationFrame(animar); dibujarPerfil(); dibujarSerie();
  $('#reproducir').disabled = true;
  $('#tiempo').disabled = true;
  estado.cargando = true;
  mostrarProgresoPrecarga(0, 1, 'Preparando caché local');
  try {
    const respuesta = await fetch(rutaConfig(), { cache: modoCache() });
    if (!respuesta.ok) throw new Error('No se pudo leer la configuración de la simulación');
    estado.config = await respuesta.json();
    construirCrisol(estado.config.geometria); construirSistemaParticulas(); actualizarRecorte(false);
    const conFases = estado.sistemaParticulas.configurarFases(estado.config.fases);
    $('#colorear-fases').disabled = !conFases;
    estado.sistemaParticulas.colorearPorFase = conFases && $('#colorear-fases').checked;
    construirLeyendaFases();
    aplicarIluminacion();
    const hin = estado.config.hinchamiento;
    if (hin?.factor_mezcla) {
      $('#valor-atenuacion').textContent =
        `×${formatear(hin.factor_mezcla)} frente a ×${formatear(hin.factor_carbon_solo)} del carbón solo`;
    }
    const enf = estado.config.magnetismo?.enfriamiento;
    if (enf?.veredicto) {
      $('#valor-temple').textContent =
        `${formatear(enf.velocidad_en_eutectoide_C_min)} °C/min en el eutectoide · ${enf.veredicto.split(':')[0]}`;
      $('#valor-temple').title =
        `${enf.veredicto}. Umbral publicado para suprimir la transformación: ${enf.umbral_supresion_C_min} °C/min.`;
    }
    const p = estado.config.particulas;
    const factorRadio = estado.sistemaParticulas.factorRadioRender;
    $('#resumen-particulas').textContent = `${p.total_muestra.toLocaleString('es-CO')} esferas representan ${p.total_real.toLocaleString('es-CO')} partículas · radio de render ${formatear(factorRadio)}×`;
    $('#factor-radio-render').textContent = `${formatear(factorRadio)}× · exageración visual`;
    $('#tiempo').max = estado.config.n_fotogramas - 1;
    $('#tiempo').step = 0.001;
    estado.cacheFotogramas = new CacheFotogramas(estado.config.n_fotogramas, {
      concurrencia: 3,
      alProgresar: mostrarProgresoPrecarga,
    });
    await estado.cacheFotogramas.precargar();
    estado.serie = resumirSeriePrecargada();
    seleccionarFotograma(0);
    const activos = estado.fotograma.etiquetas.reduce((suma, etiqueta) => suma + Number(etiqueta === 3 || etiqueta === 4), 0);
    $('#estado-malla').textContent = `${activos.toLocaleString('es-CO')} celdas activas`;
    const transporte = estado.cacheFotogramas.obtener(0).metadatos.transporte_cliente;
    $('#estado-aplicacion').textContent = `Caché completa · ${transporte} · reproducción interpolada a ${FPS_REPRODUCCION} fps sin peticiones HTTP`;
    $('#reproducir').disabled = false;
    $('#tiempo').disabled = false;
    estado.cargando = false;
    ocultarIndicadoresCarga();
  } catch (error) {
    estado.cargando = false;
    $('#precarga-serie').classList.add('oculto');
    $('#precarga-serie').hidden = true;
    $('#mensaje-carga').textContent = `No se pudo iniciar: ${error.message}`;
    $('#mensaje-carga').hidden = false;
    $('#mensaje-carga').classList.remove('oculto');
    $('#estado-aplicacion').textContent = 'Interfaz sin conexión con el servidor local'; console.error(error);
  }
}

iniciar();
