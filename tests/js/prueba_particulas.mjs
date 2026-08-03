/**
 * Pruebas de comportamiento del sistema de partículas, ejecutadas en Node.
 *
 * No son comprobaciones de texto sobre el fuente: construyen el objeto real,
 * le pasan una instantánea y miden lo que sale (color de instancia y escala de
 * la matriz). `tests/test_interfaz.py` las lanza y se salta si no hay Node.
 *
 * Uso: node tests/js/prueba_particulas.mjs
 */
import assert from 'node:assert/strict';
import { SistemaParticulasLecho } from '../../interfaz/web/js/particle-system.js';

const FORMA = [8, 8, 8];
const CELDAS = FORMA[0] * FORMA[1] * FORMA[2];

// Paleta equivalente a la que sirve fisica/fases_visuales.paleta_web().
const PALETA = {
  fases: {
    Fe3O4: { nombre: 'Magnetita', color: '#2B2B30', grupo: 'mineral', volumen_molar_cm3_mol: 231.531 / 5.17 },
    Fe: { nombre: 'Hierro metálico', color: '#B7B2AA', grupo: 'mineral', volumen_molar_cm3_mol: 55.845 / 7.874 },
    char: { nombre: 'Char', color: '#141312', grupo: 'carbonoso', volumen_molar_cm3_mol: 12.011 / 1.35 },
    carbon: { nombre: 'Carbón', color: '#1A1815', grupo: 'carbonoso', volumen_molar_cm3_mol: 1 / 1.35 },
    cenizas: { nombre: 'Cenizas', color: '#A9A296', grupo: 'carbonoso', volumen_molar_cm3_mol: 1 / 2.3 },
    aglomerado: { nombre: 'Aglomerado', color: '#2A2320', grupo: 'agregado', volumen_molar_cm3_mol: null },
  },
  campos_solidos: {
    H2O_liq: null, volatil: 'carbon', C: 'char', ceniza: 'cenizas', Fe3O4: 'Fe3O4', Fe: 'Fe',
  },
  orden_leyenda: ['Fe3O4', 'carbon', 'Fe', 'char', 'cenizas', 'aglomerado'],
};

function ejes(inicio, fin, n) {
  return Float32Array.from({ length: n }, (_, i) => inicio + (fin - inicio) * i / (n - 1));
}

function constante(valor) {
  return Float32Array.from({ length: CELDAS }, () => valor);
}

function instantanea({ t = 0, cohesion = 0, solido = {} } = {}) {
  return {
    t,
    forma: FORMA,
    x: ejes(-14, 14, FORMA[0]),
    y: ejes(-14, 14, FORMA[1]),
    z: ejes(0, 8, FORMA[2]),
    T: constante(900),
    cohesion: constante(cohesion),
    solido: Object.fromEntries(Object.entries(solido).map(([k, v]) => [k, constante(v)])),
  };
}

function colorInstancia(malla, indice) {
  const a = malla.instanceColor.array;
  return [a[indice * 3], a[indice * 3 + 1], a[indice * 3 + 2]];
}

function escalaInstancia(malla, indice) {
  // La escala uniforme es la norma de la primera columna de la matriz 4x4.
  const m = malla.instanceMatrix.array;
  const o = indice * 16;
  return Math.hypot(m[o], m[o + 1], m[o + 2]);
}

const pruebas = [];
const prueba = (nombre, fn) => pruebas.push([nombre, fn]);

prueba('la paleta se acepta y el color mineral sigue a la fase presente', () => {
  const sistema = new SistemaParticulasLecho();
  assert.equal(sistema.configurarFases(PALETA), true);
  sistema.colorearPorFase = true;

  sistema.actualizar(instantanea({ solido: { Fe3O4: 1000, volatil: 500, C: 2000 } }));
  const conMagnetita = colorInstancia(sistema.magnetita, 0);

  sistema.actualizar(instantanea({ t: 1, solido: { Fe: 1000, volatil: 500, C: 2000 } }));
  const conHierro = colorInstancia(sistema.magnetita, 0);

  // El hierro metálico (#B7B2AA) es mucho más claro que la magnetita (#2B2B30).
  assert.ok(conHierro[0] > 2 * conMagnetita[0],
    `hierro ${conHierro} debería ser mucho más claro que magnetita ${conMagnetita}`);
  // Y el color de la magnetita debe estar cerca del hexadecimal del mineral,
  // salvo la variación de brillo grano a grano (0,84-1,00).
  const objetivo = 0x2B / 255;
  const variacion = sistema._variacion[sistema.configuracion.carbon];
  assert.ok(Math.abs(conMagnetita[0] - objetivo * variacion) < 0.02,
    `magnetita ${conMagnetita[0]} lejos de ${objetivo * variacion}`);
});

function brilloMedio(malla, cantidad = 4000) {
  const a = malla.instanceColor.array;
  let suma = 0;
  for (let i = 0; i < cantidad; i += 1) suma += a[i * 3];
  return suma / cantidad;
}

prueba('la ceniza aclara la POBLACIÓN de granos carbonosos', () => {
  const sistema = new SistemaParticulasLecho();
  sistema.configurarFases(PALETA);
  sistema.colorearPorFase = true;
  sistema.actualizar(instantanea({ solido: { volatil: 2000, C: 20000, ceniza: 10 } }));
  const carbonoso = brilloMedio(sistema.carbon);
  sistema.actualizar(instantanea({ t: 2, solido: { volatil: 10, C: 2000, ceniza: 40000 } }));
  const cenizoso = brilloMedio(sistema.carbon);
  assert.ok(cenizoso > carbonoso + 0.15,
    `las cenizas (${cenizoso}) deben aclarar frente al carbón (${carbonoso})`);
});

prueba('cada grano es de UNA fase, no del promedio de la celda', () => {
  // Con 50 % de char y 50 % de cenizas deben verse las dos poblaciones, no un
  // gris intermedio uniforme: es lo que permite distinguir las fases.
  const sistema = new SistemaParticulasLecho();
  sistema.configurarFases(PALETA);
  sistema.colorearPorFase = true;
  // Volúmenes iguales: char 12,011/1,35 = 8,90 cm3/mol; cenizas 1/2,3 = 0,435.
  sistema.actualizar(instantanea({ solido: { C: 1000, ceniza: 20460 } }));
  const a = sistema.carbon.instanceColor.array;
  let oscuros = 0; let claros = 0;
  for (let i = 0; i < 4000; i += 1) {
    if (a[i * 3] < 0.15) oscuros += 1;
    else if (a[i * 3] > 0.5) claros += 1;
  }
  assert.ok(oscuros > 800 && claros > 800,
    `deben coexistir granos negros y claros: ${oscuros} oscuros, ${claros} claros`);
});

prueba('una fase minoritaria pinta pocos granos, no tiñe el lecho entero', () => {
  const sistema = new SistemaParticulasLecho();
  sistema.configurarFases(PALETA);
  sistema.colorearPorFase = true;
  // Trazas de hierro metálico entre magnetita: ~2 % del volumen mineral.
  sistema.actualizar(instantanea({ solido: { Fe3O4: 1000, Fe: 45 } }));
  const a = sistema.magnetita.instanceColor.array;
  let brillantes = 0;
  const total = sistema.configuracion.magnetita;
  for (let i = 0; i < total; i += 1) if (a[i * 3] > 0.5) brillantes += 1;
  const fraccion = brillantes / total;
  assert.ok(fraccion > 0.002 && fraccion < 0.10,
    `el hierro traza debe pintar una minoría de granos, no ${(100 * fraccion).toFixed(1)} %`);
});

prueba('sin coloreado por fase el color de instancia vuelve a ser gris', () => {
  const sistema = new SistemaParticulasLecho();
  sistema.configurarFases(PALETA);
  sistema.colorearPorFase = true;
  sistema.actualizar(instantanea({ solido: { Fe: 1000, C: 2000 } }));
  sistema.colorearPorFase = false;
  sistema.actualizar(instantanea({ t: 3, solido: { Fe: 1000, C: 2000 } }));
  const [r, g, b] = colorInstancia(sistema.magnetita, 0);
  assert.ok(Math.abs(r - g) < 1e-6 && Math.abs(g - b) < 1e-6, `debería ser gris, es ${[r, g, b]}`);
});

prueba('la partícula DESAPARECE donde la cohesión supera el umbral', () => {
  const sistema = new SistemaParticulasLecho();
  const radios = [0, 0.3, 0.5, 0.7, 0.95].map((c) => {
    sistema.actualizar(instantanea({ t: 0, cohesion: c }));
    return escalaInstancia(sistema.carbon, 0);
  });
  const [suelto, incipiente, umbral, alto, total] = radios;
  assert.ok(suelto > 0, 'sin cohesión la partícula debe verse');
  assert.ok(incipiente >= umbral, 'el radio no puede crecer con la cohesión');
  // El umbral operativo del aglomerado es 0,5 (fisica/cohesion.aglomerado).
  assert.ok(umbral < 0.75 * suelto, `a c=0,5 ya debe estar desapareciendo: ${umbral} vs ${suelto}`);
  assert.ok(alto < 0.15 * suelto, `a c=0,7 debe quedar casi nada: ${alto} vs ${suelto}`);
  assert.ok(total < 0.05 * suelto, `a c=0,95 debe estar integrada en el aglomerado: ${total}`);
  for (let i = 1; i < radios.length; i += 1) {
    assert.ok(radios[i] <= radios[i - 1] + 1e-9, `radio no monótono: ${radios}`);
  }
});

prueba('la contracción del carbón sigue al volátil real, no al reloj', () => {
  const sistema = new SistemaParticulasLecho();
  sistema.configurarFases(PALETA);
  sistema.actualizar(instantanea({ t: 0, solido: { volatil: 1000, C: 20000 } }));
  const inicial = escalaInstancia(sistema.carbon, 0);
  // Mismo instante, pero ya sin volátil: debe contraerse igualmente.
  sistema.actualizar(instantanea({ t: 0.1, solido: { volatil: 1, C: 20000 } }));
  const devolatilizado = escalaInstancia(sistema.carbon, 0);
  assert.ok(devolatilizado < 0.75 * inicial,
    `el carbón devolatilizado debe encoger: ${devolatilizado} vs ${inicial}`);
});

prueba('una paleta incompleta se rechaza sin romper el render', () => {
  const sistema = new SistemaParticulasLecho();
  assert.equal(sistema.configurarFases(undefined), false);
  assert.equal(sistema.configurarFases({ fases: {} }), false);
  sistema.colorearPorFase = true;
  sistema.actualizar(instantanea({ solido: { Fe3O4: 1000 } }));  // no debe lanzar
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
console.log(`${pruebas.length - fallos}/${pruebas.length} pruebas de partículas correctas`);
process.exit(fallos ? 1 : 0);
