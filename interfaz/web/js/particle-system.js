import * as THREE from './three.module.js';

// Submuestra exacta de 388 bloques 119 C : 10 Fe3O4 = 11,9 : 1.
export const PARTICULAS_REPRESENTATIVAS = Object.freeze({
  carbon: 46172,
  magnetita: 3880,
  total: 50052,
  totalReal: 222824,
  diametroMinUm: 100,
  diametroMaxUm: 250,
  porosidad: 0.54,
  volumenLechoCm3: 1.3593,
  razonNumero: 11.9,
});

// Exageración exclusivamente visual: el mínimo renderizado (0,12 mm de radio)
// ocupa 3,7 % de los 3,26 mm del lecho y sigue dejando visibles granos separados.
export const FACTOR_RADIO_RENDER = 2.40;

function aleatorioSembrado(semilla = 0x5eed60) {
  let estado = semilla >>> 0;
  return () => {
    estado += 0x6d2b79f5;
    let t = estado;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function suave01(valor) {
  const x = THREE.MathUtils.clamp(valor, 0, 1);
  return x * x * (3 - 2 * x);
}

function aLineal(canal) {
  return canal <= 0.04045 ? canal / 12.92 : ((canal + 0.055) / 1.055) ** 2.4;
}

function aSrgb(canal) {
  const c = THREE.MathUtils.clamp(canal, 0, 1);
  return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
}

function linealDeHex(hex) {
  const n = parseInt(String(hex).replace('#', ''), 16);
  return [aLineal(((n >> 16) & 255) / 255), aLineal(((n >> 8) & 255) / 255), aLineal((n & 255) / 255)];
}

function crearMaterialCarbon(clippingPlanes) {
  const material = new THREE.MeshStandardMaterial({
    color: 0x242526,
    roughness: 0.92,
    metalness: 0.025,
    clippingPlanes,
  });
  material.userData.recortable = true;
  return material;
}

function crearMaterialMagnetita(clippingPlanes) {
  const material = new THREE.MeshStandardMaterial({
    color: 0x71808a,
    roughness: 0.18,
    metalness: 0.91,
    clippingPlanes,
  });
  material.userData.recortable = true;
  return material;
}

/**
 * Lecho discreto reproducible. Los diametros nominales se sortean uniformes
 * entre 100 y 250 um porque no existen d10/d50/d90 medidos. El radio de render
 * usa un factor visual explícito. El factor físico por volumen se conserva como
 * diagnóstico, pero 2,40× evita que los granos desaparezcan en la vista global.
 */
export class SistemaParticulasLecho {
  constructor({ clippingPlanes = [], configuracion = PARTICULAS_REPRESENTATIVAS } = {}) {
    this.configuracion = { ...PARTICULAS_REPRESENTATIVAS, ...configuracion };
    this.grupo = new THREE.Group();
    this.grupo.name = 'Vista de particulas: polvo de carbon y magnetita malla 60';
    this._matriz = new THREE.Matrix4();
    this._cuaternion = new THREE.Quaternion();
    this._indicesCampo = null;
    this._formaCampo = '';
    this.factorRadioRender = this.configuracion.factorRadioRender || FACTOR_RADIO_RENDER;
    this._carbonInicial = new THREE.Color(0x242526);
    this._carbonCoque = new THREE.Color(0x343638);
    // Coloreado por fase: apagado hasta que llegue la paleta mineral real.
    this._paletaFases = null;
    this.colorearPorFase = false;
    this._distribucion = null;
    this._tDistribucion = null;
    this._pintadoPrevio = false;
    this._volatilInicial = undefined;
    this._generarParticulas();

    const geometria = new THREE.SphereGeometry(1, 7, 5);
    this.carbon = new THREE.InstancedMesh(
      geometria,
      crearMaterialCarbon(clippingPlanes),
      this.configuracion.carbon,
    );
    this.magnetita = new THREE.InstancedMesh(
      geometria.clone(),
      crearMaterialMagnetita(clippingPlanes),
      this.configuracion.magnetita,
    );
    this.carbon.name = 'Carbon: negro mate que se contrae y coquiza';
    this.magnetita.name = 'Magnetita: gris metalico';
    this.carbon.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.magnetita.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.carbon.frustumCulled = false;
    this.magnetita.frustumCulled = false;
    this.carbon.renderOrder = 4;
    this.magnetita.renderOrder = 5;
    // Variación de brillo grano a grano: un polvo real no es de un solo tono.
    // Se guarda porque también multiplica al color de fase cuando éste se activa.
    this._variacion = new Float32Array(this.configuracion.total);
    for (let n = 0; n < this.configuracion.carbon; n += 1) {
      this._variacion[n] = 0.82 + 0.20 * ((Math.sin(n * 12.9898) * 43758.5453) % 1 + 1) % 1;
    }
    for (let local = 0; local < this.configuracion.magnetita; local += 1) {
      this._variacion[this.configuracion.carbon + local] =
        0.84 + 0.16 * ((Math.sin(local * 7.137 + 4.2) * 15731.743) % 1 + 1) % 1;
    }
    this._aplicarColoresGrises();
    this.grupo.add(this.carbon, this.magnetita);
    this.actualizarMatrices(null);
  }

  _generarParticulas() {
    const cfg = this.configuracion;
    const cantidad = cfg.total;
    this.posiciones = new Float32Array(cantidad * 3);
    this.centrosUnion = new Float32Array(cantidad * 3);
    this.diametrosUm = new Float32Array(cantidad);
    const random = aleatorioSembrado();
    let sumaVolumenNominal = 0;
    const radioUtil = 10.98;
    const zMin = 2.12;
    const zMax = 5.20;
    for (let i = 0; i < cantidad; i += 1) {
      const radio = radioUtil * Math.sqrt(random());
      const angulo = 2 * Math.PI * random();
      const x = radio * Math.cos(angulo);
      const y = radio * Math.sin(angulo);
      const z = zMin + (zMax - zMin) * random();
      const diametroUm = cfg.diametroMinUm + (cfg.diametroMaxUm - cfg.diametroMinUm) * random();
      const diametroMm = diametroUm / 1000;
      this.posiciones.set([x, y, z], i * 3);
      this.diametrosUm[i] = diametroUm;
      sumaVolumenNominal += Math.PI / 6 * diametroMm ** 3;

      // Centros locales irregulares: al superar el umbral de cohesion las
      // particulas convergen a estos nucleos y aparece la masa conectada.
      const paso = 0.72;
      const cx = Math.round((x + 0.16 * Math.sin(y * 1.7)) / paso) * paso;
      const cy = Math.round((y + 0.14 * Math.cos(x * 1.5)) / paso) * paso;
      const cz = Math.round((z - 2.0) / 0.55) * 0.55 + 2.0;
      this.centrosUnion.set([cx, cy, cz], i * 3);
    }
    // Sorteo estable por partícula: fija QUÉ fase le toca dentro de la
    // composición de su celda. Al depender sólo del índice, un grano no salta
    // de mineral entre fotogramas; sólo cambia si cambia la composición.
    this._sorteoFase = new Float32Array(cantidad);
    const sorteo = aleatorioSembrado(0x9e3779b9);
    for (let i = 0; i < cantidad; i += 1) this._sorteoFase[i] = sorteo();
    const volumenSolidoObjetivoMm3 = cfg.volumenLechoCm3 * 1000 * (1 - cfg.porosidad);
    this.factorRadioFisico = Math.cbrt(volumenSolidoObjetivoMm3 / sumaVolumenNominal);
  }

  _aplicarColoresGrises() {
    const color = new THREE.Color();
    const cfg = this.configuracion;
    for (let n = 0; n < cfg.carbon; n += 1) {
      const v = this._variacion[n];
      this.carbon.setColorAt(n, color.setRGB(v, v, v));
    }
    for (let local = 0; local < cfg.magnetita; local += 1) {
      const v = this._variacion[cfg.carbon + local];
      this.magnetita.setColorAt(local, color.setRGB(v, v, v));
    }
    this.carbon.instanceColor.needsUpdate = true;
    this.magnetita.instanceColor.needsUpdate = true;
  }

  /**
   * Paleta mineral servida por `/api/config` (fisica/fases_visuales.paleta_web).
   * Se recibe en vez de codificarla aquí para que la mineralogía viva en un
   * único sitio: color, masa molar y densidad de cada fase.
   */
  configurarFases(paleta) {
    if (!paleta?.fases || !paleta.campos_solidos) return false;
    const grupos = { carbonoso: [], mineral: [] };
    for (const [campo, clave] of Object.entries(paleta.campos_solidos)) {
      if (!clave) continue;                       // H2O_liq: humedad, sin color propio
      const fase = paleta.fases[clave];
      if (!fase || !(fase.volumen_molar_cm3_mol > 0)) continue;
      const destino = grupos[fase.grupo];
      if (!destino) continue;                     // el aglomerado no es fase del inventario
      const n = parseInt(String(fase.color).replace('#', ''), 16);
      destino.push({
        campo, clave, volumenMolar: fase.volumen_molar_cm3_mol,
        lineal: linealDeHex(fase.color),
        srgb: [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255],
      });
    }
    if (!grupos.carbonoso.length && !grupos.mineral.length) return false;
    this._paletaFases = grupos;
    this._distribucion = null;
    this._tDistribucion = null;
    return true;
  }

  /**
   * Distribución de fases por celda: fracción de VOLUMEN acumulada.
   *
   * El campo del solucionador está en mol/m3; multiplicarlo por el volumen
   * molar M/rho da el volumen que ocupa cada fase, que es lo que determina qué
   * proporción de los granos es de cada una. Ponderar por moles daría demasiado
   * peso a las fases ligeras: un mol de Fe ocupa 7,09 cm3 y uno de magnetita
   * 44,8.
   */
  _calcularDistribucionCelda(fotograma) {
    if (!this._paletaFases) return null;
    if (this._tDistribucion === fotograma.t && this._distribucion) return this._distribucion;
    const celdas = fotograma.T.length;
    const distribucion = {};
    for (const grupo of ['carbonoso', 'mineral']) {
      const aportes = this._paletaFases[grupo]
        .map((fase) => ({ ...fase, valores: fotograma.solido?.[fase.campo] }))
        .filter((fase) => fase.valores && fase.valores.length === celdas);
      const n = aportes.length;
      const acumulada = new Float32Array(celdas * Math.max(n, 1));
      for (let q = 0; q < celdas; q += 1) {
        let total = 0;
        for (let i = 0; i < n; i += 1) {
          const volumen = Math.max(aportes[i].valores[q] * aportes[i].volumenMolar, 0);
          total += volumen;
          acumulada[q * n + i] = total;
        }
        if (total > 0) {
          for (let i = 0; i < n; i += 1) acumulada[q * n + i] /= total;
        } else if (n > 0) {
          // Celda sin sólido: se asigna la primera fase para no dejar el grano
          // sin color; en la práctica la partícula ya está desvanecida.
          for (let i = 0; i < n; i += 1) acumulada[q * n + i] = 1;
        }
      }
      distribucion[grupo] = { acumulada, fases: aportes, n };
    }
    this._tDistribucion = fotograma.t;
    this._distribucion = distribucion;
    return distribucion;
  }

  /**
   * Cada grano es de UNA fase, sorteada según la composición de su celda.
   *
   * Antes cada partícula llevaba el color de la MEZCLA de su celda, y entonces
   * todos los granos salían del mismo gris promedio: no se distinguía una fase
   * de otra, que es justo lo que hay que ver. Un lecho real es un mosaico de
   * granos, cada uno de su mineral, así que se sortea la fase de cada partícula
   * con la fracción volumétrica local. El sorteo es determinista —depende sólo
   * del índice de la partícula— para que un grano no cambie de mineral de un
   * fotograma a otro sin motivo; sólo cambia cuando cambia la composición.
   *
   * El efecto buscado: cuando aparece un 0,1 % de hierro metálico, aparecen
   * unos pocos granos brillantes entre los negros, en vez de aclararse todo el
   * lecho imperceptiblemente.
   */
  _aplicarColoresDeFase(fotograma) {
    const distribucion = this._calcularDistribucionCelda(fotograma);
    if (!distribucion) return false;
    const cfg = this.configuracion;
    const color = new THREE.Color();
    const pintar = (malla, inicio, cantidad, grupo) => {
      const { acumulada, fases, n } = distribucion[grupo];
      if (!n) return;
      for (let local = 0; local < cantidad; local += 1) {
        const indice = inicio + local;
        const celda = this._indicesCampo[indice];
        const sorteo = this._sorteoFase[indice];
        let elegida = n - 1;
        for (let i = 0; i < n; i += 1) {
          if (sorteo <= acumulada[celda * n + i]) { elegida = i; break; }
        }
        const rgb = fases[elegida].srgb;
        const v = this._variacion[indice];
        color.setRGB(rgb[0] * v, rgb[1] * v, rgb[2] * v);
        malla.setColorAt(local, color);
      }
      malla.instanceColor.needsUpdate = true;
    };
    // El material pasa a blanco: el color real lo lleva cada instancia.
    this.carbon.material.color.setRGB(1, 1, 1);
    this.magnetita.material.color.setRGB(1, 1, 1);
    pintar(this.carbon, 0, cfg.carbon, 'carbonoso');
    pintar(this.magnetita, cfg.carbon, cfg.magnetita, 'mineral');
    return true;
  }

  _prepararIndicesCampo(fotograma) {
    const clave = fotograma.forma.join('x');
    if (this._indicesCampo && clave === this._formaCampo) return;
    const [nx, ny, nz] = fotograma.forma;
    const dx = fotograma.x[1] - fotograma.x[0];
    const dy = fotograma.y[1] - fotograma.y[0];
    const dz = fotograma.z[1] - fotograma.z[0];
    this._indicesCampo = new Uint32Array(this.configuracion.total);
    for (let n = 0; n < this.configuracion.total; n += 1) {
      const p = n * 3;
      const i = THREE.MathUtils.clamp(Math.round((this.posiciones[p] - fotograma.x[0]) / dx), 0, nx - 1);
      const j = THREE.MathUtils.clamp(Math.round((this.posiciones[p + 1] - fotograma.y[0]) / dy), 0, ny - 1);
      const k = THREE.MathUtils.clamp(Math.round((this.posiciones[p + 2] - fotograma.z[0]) / dz), 0, nz - 1);
      this._indicesCampo[n] = (i * ny + j) * nz + k;
    }
    this._formaCampo = clave;
  }

  /**
   * Avance de la devolatilización, medido sobre el campo si lo hay.
   *
   * Con datos reales del solucionador se usa el volátil que queda en el lecho
   * respecto al inicial, que es la magnitud física; el calendario t/720 sólo
   * queda como respaldo para las instantáneas sintéticas, que no traen
   * `solido.volatil`.
   */
  _avanceDevolatilizacion(fotograma) {
    const volatil = fotograma.solido?.volatil;
    if (!volatil) return THREE.MathUtils.clamp(fotograma.t / 720, 0, 1);
    let suma = 0;
    for (let q = 0; q < volatil.length; q += 1) suma += volatil[q];
    if (this._volatilInicial === undefined || fotograma.t <= 0) this._volatilInicial = suma;
    if (!(this._volatilInicial > 0)) return 0;
    return THREE.MathUtils.clamp(1 - suma / this._volatilInicial, 0, 1);
  }

  actualizar(fotograma) {
    if (!fotograma) return;
    this._prepararIndicesCampo(fotograma);
    const avance = this._avanceDevolatilizacion(fotograma);
    const conCampo = Boolean(fotograma.solido?.volatil);
    const devolatilizacion = conCampo ? avance : suave01((avance - 0.04) / 0.38);
    const coquizacion = conCampo ? suave01((avance - 0.10) / 0.55) : suave01((avance - 0.16) / 0.36);
    const pintado = this.colorearPorFase && this._aplicarColoresDeFase(fotograma);
    if (!pintado) {
      // Vuelta al aspecto genérico: material coloreado e instancias en gris.
      if (this._pintadoPrevio) this._aplicarColoresGrises();
      this.magnetita.material.color.setHex(0x71808a);
      this.carbon.material.color.lerpColors(this._carbonInicial, this._carbonCoque, coquizacion);
    }
    this._pintadoPrevio = pintado;
    this.carbon.material.roughness = THREE.MathUtils.lerp(0.92, 0.68, coquizacion);
    this.carbon.material.metalness = THREE.MathUtils.lerp(0.025, 0.10, coquizacion);
    this.carbon.material.needsUpdate = true;
    this.magnetita.material.needsUpdate = true;
    this.actualizarMatrices(fotograma, 1 - 0.31 * devolatilizacion);
  }

  actualizarMatrices(fotograma, contraccionCarbon = 1) {
    const cfg = this.configuracion;
    const escala = new THREE.Vector3();
    const posicion = new THREE.Vector3();
    const actualizarMalla = (malla, inicio, cantidad, esCarbon) => {
      for (let local = 0; local < cantidad; local += 1) {
        const n = inicio + local;
        const p = n * 3;
        const celda = fotograma ? this._indicesCampo[n] : 0;
        const cohesion = fotograma ? fotograma.cohesion[celda] : 0;
        const union = suave01((cohesion - 0.22) / 0.38);
        // HINCHAMIENTO. El carbón tiene índice 8 y la magnetita lo atenúa; el
        // campo lo calcula `fisica/hinchamiento.py`. La pared del crisol impide
        // la expansión radial, así que la masa sólo puede subir: la altura sobre
        // el suelo del lecho escala con todo el factor volumétrico.
        const hinchado = fotograma?.hinchamiento ? fotograma.hinchamiento[celda] : 1;
        const zBase = 2.12;
        posicion.set(
          THREE.MathUtils.lerp(this.posiciones[p], this.centrosUnion[p], 0.72 * union),
          THREE.MathUtils.lerp(this.posiciones[p + 1], this.centrosUnion[p + 1], 0.72 * union),
          zBase + (THREE.MathUtils.lerp(this.posiciones[p + 2], this.centrosUnion[p + 2], 0.60 * union) - zBase) * hinchado,
        );
        const radioMm = this.diametrosUm[n] * 0.0005 * this.factorRadioRender;
        const factorMaterial = esCarbon ? contraccionCarbon : 1;
        // Desvanecimiento al integrarse en el aglomerado.
        //
        // Antes la particula CRECIA al cohesionar (1 + 0,20*union), lo que daba
        // la impresion contraria a la fisica: el polvo se hinchaba en vez de
        // consumirse. Lo correcto es que desaparezca como grano suelto, porque
        // pasa a formar parte de la masa cohesionada, que ya se dibuja con su
        // propia isosuperficie. Si no, el mismo material se representa dos
        // veces: como particula y como aglomerado.
        //
        // La transicion se reparte en dos tramos para que se vea el proceso:
        // primero el grano se acerca a su centro de union conservando tamano
        // (union < 0,55) y despues se funde en la masa (union -> 1).
        const desvanecido = suave01((union - 0.55) / 0.45);
        const radioRender = radioMm * factorMaterial * (1 - 0.97 * desvanecido);
        escala.setScalar(Math.max(radioRender, 0));
        this._matriz.compose(posicion, this._cuaternion, escala);
        malla.setMatrixAt(local, this._matriz);
      }
      malla.instanceMatrix.needsUpdate = true;
    };
    actualizarMalla(this.carbon, 0, cfg.carbon, true);
    actualizarMalla(this.magnetita, cfg.carbon, cfg.magnetita, false);
  }

  set visible(valor) { this.grupo.visible = Boolean(valor); }
  get visible() { return this.grupo.visible; }

  dispose() {
    this.carbon.geometry.dispose();
    this.magnetita.geometry.dispose();
    this.carbon.material.dispose();
    this.magnetita.material.dispose();
  }
}
