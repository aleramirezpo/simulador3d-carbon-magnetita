"""Pruebas headless de la aplicación de escritorio y su modo demostración."""

from __future__ import annotations

import json
import re
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from interfaz.app import DIRECTORIO_WEB, detener_servidor, iniciar_servidor
from nucleo.salida import cargar_instantanea, guardar_instantanea
from interfaz.datos_sinteticos import (
    ESPECIES,
    PARTICULAS_ENSAYO,
    generar_instantanea_sintetica,
    generar_lineas_corriente,
)


def test_datos_sinteticos_respetan_formas_mac_y_son_solenoidales():
    forma = (10, 10, 12)
    campos = generar_instantanea_sintetica(4, n_fotogramas=9, forma=forma)
    nx, ny, nz = forma
    assert campos["P"].shape == forma
    assert campos["T"].shape == forma
    assert campos["eps"].shape == forma
    assert campos["cohesion"].shape == forma
    assert campos["etiquetas"].shape == forma
    assert campos["u"].shape == (nx + 1, ny, nz)
    assert campos["v"].shape == (nx, ny + 1, nz)
    assert campos["w"].shape == (nx, ny, nz + 1)
    assert set(campos["c_especies"]) == set(ESPECIES)
    assert all(array.shape == forma for array in campos["c_especies"].values())
    assert all(array.shape == forma for array in campos["solido"].values())

    dx = campos["x"][1] - campos["x"][0]
    dy = campos["y"][1] - campos["y"][0]
    dz = campos["z"][1] - campos["z"][0]
    divergencia = (
        np.diff(campos["u"], axis=0) / dx
        + np.diff(campos["v"], axis=1) / dy
        + np.diff(campos["w"], axis=2) / dz
    )
    assert float(np.max(np.abs(divergencia))) < 1e-12


def test_lineas_de_corriente_permanecen_acotadas():
    campos = generar_instantanea_sintetica(8, n_fotogramas=12, forma=(12, 12, 16))
    lineas = generar_lineas_corriente(campos, cantidad=10)
    assert lineas
    minimo = np.array([campos["x"].min(), campos["y"].min(), campos["z"].min()])
    maximo = np.array([campos["x"].max(), campos["y"].max(), campos["z"].max()])
    for linea in lineas:
        assert linea.ndim == 2 and linea.shape[1] == 3 and len(linea) > 1
        assert np.all(linea >= minimo - 1e-12)
        assert np.all(linea <= maximo + 1e-12)
        radios = np.hypot(linea[:, 0], linea[:, 1])
        radios_internos = 11.4 + (13.65 - 11.4) * np.clip(linea[:, 2] / 32.0, 0.0, 1.0)
        assert np.all(radios <= radios_internos + 1e-12)


def test_servidor_headless_sirve_recursos_y_api():
    servidor, hilo, url = iniciar_servidor(0)
    try:
        with urllib.request.urlopen(url, timeout=5) as respuesta:
            html = respuesta.read().decode("utf-8")
            assert respuesta.status == 200
            assert "Datos sintéticos" in html
            assert "sin validación experimental" in html
        with urllib.request.urlopen(url + "api/config", timeout=5) as respuesta:
            config = json.loads(respuesta.read().decode("utf-8"))
            assert config["datos_sinteticos"] is True
            assert config["origen_datos"] == "sinteticos"
            assert config["version_three"] == "0.180.0"
        with urllib.request.urlopen(url + "api/fotograma?indice=0", timeout=10) as respuesta:
            fotograma = json.loads(respuesta.read().decode("utf-8"))
            assert fotograma["forma"] == config["forma"]
            assert len(fotograma["T"]) == int(np.prod(config["forma"]))
            assert set(fotograma["c_especies"]) == set(ESPECIES)
            assert fotograma["metadatos"]["datos_sinteticos"] is True
            assert fotograma["termico"]["T_pared_media_K"] > 298.15
            assert fotograma["termico"]["T_mufla_K"] == 1173.15
        with urllib.request.urlopen(url + "api/lineas?indice=0", timeout=10) as respuesta:
            lineas = json.loads(respuesta.read().decode("utf-8"))["lineas"]
            assert lineas and all(len(linea) > 1 for linea in lineas)
        for recurso in ("css/estilos.css", "js/app.js", "js/three.module.js"):
            with urllib.request.urlopen(url + recurso, timeout=5) as respuesta:
                assert respuesta.status == 200
                assert len(respuesta.read()) > 1000
    finally:
        detener_servidor(servidor, hilo)


def test_interfaz_carga_resultados_reales_y_declara_el_origen(tmp_path):
    campos = generar_instantanea_sintetica(2, n_fotogramas=4, forma=(8, 8, 10))
    campos["metadatos"] = {
        "fuente": "solucionador 3D de prueba",
        "datos_sinteticos": False,
    }
    guardar_instantanea(campos, tmp_path / "instantanea_real.npz")
    servidor, hilo, url = iniciar_servidor(0, datos=tmp_path)
    try:
        with urllib.request.urlopen(url, timeout=5) as respuesta:
            html = respuesta.read().decode("utf-8")
            assert "Resultados del solucionador" in html
            assert "PREDICCIÓN · RESULTADOS DEL SOLUCIONADOR" in html
            assert "sin validación experimental" in html
        with urllib.request.urlopen(url + "api/config", timeout=5) as respuesta:
            config = json.loads(respuesta.read().decode("utf-8"))
            assert config["datos_sinteticos"] is False
            assert config["origen_datos"] == "resultados_del_solucionador"
            assert config["n_fotogramas"] == 1
        with urllib.request.urlopen(url + "api/fotograma?indice=0", timeout=10) as respuesta:
            fotograma = json.loads(respuesta.read().decode("utf-8"))
            assert fotograma["metadatos"]["datos_sinteticos"] is False
    finally:
        detener_servidor(servidor, hilo)


def test_interfaz_cae_a_sinteticos_si_no_hay_instantaneas(tmp_path):
    servidor, hilo, url = iniciar_servidor(0, datos=tmp_path / "ausente")
    try:
        with urllib.request.urlopen(url + "api/config", timeout=5) as respuesta:
            config = json.loads(respuesta.read().decode("utf-8"))
        assert config["datos_sinteticos"] is True
        assert config["origen_datos"] == "sinteticos"
    finally:
        detener_servidor(servidor, hilo)


def test_ejecutable_arranca_y_cierra_en_modo_headless():
    # La consola de Windows usa cp1252 por defecto: sin forzar UTF-8, los
    # acentos de la salida del subproceso llegan mal decodificados y la
    # comparación falla por un motivo que nada tiene que ver con la interfaz.
    entorno = dict(os.environ, PYTHONIOENCODING="utf-8")
    resultado = subprocess.run(
        [sys.executable, "-m", "interfaz.app", "--headless"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=entorno,
        timeout=15,
        check=False,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "Comprobación correcta" in resultado.stdout


# Ficheros de la biblioteca de terceros: se vendorizan tal cual, sin editarlos.
_BIBLIOTECA_VENDORIZADA = {"three.module.js", "three.core.js", "OrbitControls.js"}


def test_codigo_propio_no_contiene_urls_externas():
    """El código escrito para esta aplicación no debe apuntar a la red."""
    patron = re.compile(r"https?://|(?:src|href)\s*=\s*['\"]//", re.IGNORECASE)
    archivos = [DIRECTORIO_WEB / "index.html",
                *(r for r in DIRECTORIO_WEB.rglob("*.js")
                  if r.name not in _BIBLIOTECA_VENDORIZADA)]
    assert archivos
    for ruta in archivos:
        contenido = ruta.read_text(encoding="utf-8")
        assert not patron.search(contenido), f"URL externa encontrada en {ruta}"


def test_la_biblioteca_vendorizada_no_carga_nada_de_la_red():
    """three.js contiene URLs, pero sólo en comentarios de documentación.

    Prohibir la cadena "http" en la biblioteca sería un criterio equivocado: sus
    comentarios JSDoc enlazan a Wikipedia y MDN. Lo que de verdad importa es que
    no haya imports remotos, que es lo que rompería el funcionamiento sin red.
    Las llamadas a fetch/XMLHttpRequest que incluye pertenecen a sus cargadores
    de recursos, que esta aplicación no usa: no carga ningún activo externo.
    """
    for nombre in _BIBLIOTECA_VENDORIZADA:
        ruta = DIRECTORIO_WEB / "js" / nombre
        assert ruta.is_file(), f"falta la biblioteca vendorizada {nombre}"
        contenido = ruta.read_text(encoding="utf-8")
        assert not re.search(r"""from\s+['"]https?://""", contenido), (
            f"{nombre} importa desde la red")
        assert not re.search(r"""import\s*\(\s*['"]https?://""", contenido), (
            f"{nombre} hace un import dinámico remoto")


def test_three_es_la_biblioteca_real_no_un_sustituto():
    """Debe ser three.js auténtico, no una reimplementación de su API.

    Una copia reducida escrita a mano sirve para dibujar cilindros, pero no
    ofrece render volumétrico, materiales transparentes ni raycasting, que es lo
    que necesita esta interfaz. El tamaño es el discriminante: la biblioteca real
    supera el megabyte entre sus dos ficheros.
    """
    modulo = DIRECTORIO_WEB / "js" / "three.module.js"
    core = DIRECTORIO_WEB / "js" / "three.core.js"
    assert modulo.is_file() and core.is_file()
    total = modulo.stat().st_size + core.stat().st_size
    assert total > 1_500_000, (
        f"three.js vendorizado sólo ocupa {total} bytes: parece un sustituto, "
        "no la biblioteca real")
    assert "REVISION = '180'" in core.read_text(encoding="utf-8")
    assert "Three.js Authors" in modulo.read_text(encoding="utf-8")

    app = (DIRECTORIO_WEB / "js" / "app.js").read_text(encoding="utf-8")
    assert "from './three.module.js'" in app
    assert "cdn" not in app.lower()


def test_app_importa_y_usa_orbitcontrols_desde_la_copia_local():
    app = (DIRECTORIO_WEB / "js" / "app.js").read_text(encoding="utf-8")
    assert "from './OrbitControls.js'" in app
    assert re.search(r"new\s+OrbitControls\s*\(", app)
    orbit = (DIRECTORIO_WEB / "js" / "OrbitControls.js").read_text(encoding="utf-8")
    assert "from './three.module.js'" in orbit


def test_app_aprovecha_las_capacidades_3d_reales_requeridas():
    propios = "\n".join(
        ruta.read_text(encoding="utf-8")
        for ruta in (DIRECTORIO_WEB / "js").glob("*.js")
        if ruta.name not in _BIBLIOTECA_VENDORIZADA
    )
    assert "clippingPlanes" in propios
    assert "new THREE.Raycaster" in propios
    assert "new THREE.Data3DTexture" in propios
    assert "new THREE.ShaderMaterial" in propios
    assert "new THREE.InstancedMesh" in propios
    assert "new THREE.Sprite" in propios
    assert "extraerIsosuperficieMarchingCubes" in propios
    assert "DataTexture" in propios and "crearPlanosApilados" in propios


def test_shaders_glsl_tienen_main_y_llaves_balanceadas():
    modulo = (DIRECTORIO_WEB / "js" / "volume-renderer.js").read_text(encoding="utf-8")
    shaders = re.findall(
        r"VOLUME_(?:VERTEX|FRAGMENT)_SHADER\s*=.*?`(.*?)`;",
        modulo,
        flags=re.DOTALL,
    )
    assert len(shaders) == 2
    for shader in shaders:
        assert re.search(r"\bvoid\s+main\s*\(\s*\)", shader)
        nivel = 0
        for caracter in shader:
            if caracter == "{":
                nivel += 1
            elif caracter == "}":
                nivel -= 1
                assert nivel >= 0, "el shader cierra una llave que no abrió"
        assert nivel == 0, "el shader GLSL tiene llaves sin cerrar"


def test_html_declara_avisos_de_prediccion_y_datos_sinteticos():
    html = (DIRECTORIO_WEB / "index.html").read_text(encoding="utf-8")
    assert re.search(r"Datos sintéticos", html, re.IGNORECASE)
    assert re.search(r"Predicción", html, re.IGNORECASE)
    assert "sin validación experimental" in html.lower()
    assert "no usar como medición" in html.lower()


def test_api_expone_el_perfil_fotografico_con_collar():
    servidor, hilo, url = iniciar_servidor(0)
    try:
        with urllib.request.urlopen(url + "api/config", timeout=5) as respuesta:
            geometria = json.loads(respuesta.read().decode("utf-8"))["geometria"]
        assert geometria["tipo"] == "perfil_de_revolucion_con_collar"
        assert geometria["collar_diametro_mm"] == 30.6
        assert geometria["perfil_exterior_mm"] == [
            [0.0, 12.5], [18.0, 14.0], [21.0, 15.3],
            [23.0, 14.75], [32.0, 14.75],
        ]
    finally:
        detener_servidor(servidor, hilo)


def test_planos_de_recorte_arrancan_desactivados():
    html = (DIRECTORIO_WEB / "index.html").read_text(encoding="utf-8")
    for eje in "xyz":
        etiqueta = re.search(
            rf'<input\s+id="activar-corte-{eje}"[^>]*>', html, re.IGNORECASE
        )
        assert etiqueta, f"falta el interruptor de corte {eje}"
        assert not re.search(r"\bchecked\b", etiqueta.group(0), re.IGNORECASE)


def test_submuestreo_conserva_proporcion_numerica_carbon_magnetita():
    p = PARTICULAS_ENSAYO
    razon = p["carbon_muestra"] / p["magnetita_muestra"]
    assert 40_000 <= p["total_muestra"] <= 60_000
    assert p["carbon_muestra"] + p["magnetita_muestra"] == p["total_muestra"]
    assert 11.0 <= razon <= 13.0
    assert razon == p["razon_numero_carbon_magnetita"]


def test_tamanos_nominales_de_particula_estan_en_malla_60():
    p = PARTICULAS_ENSAYO
    rng = np.random.default_rng(60)
    diametros_um = rng.uniform(
        p["diametro_min_um"], p["diametro_max_um"], p["total_muestra"]
    )
    assert float(diametros_um.min()) >= 100.0
    assert float(diametros_um.max()) <= 250.0
    assert p["diametro_max_um"] <= p["apertura_malla_um"]
    assert "no hay d10/d50/d90" in p["distribucion"]


def test_pared_se_calienta_antes_que_centro_del_lecho_en_todo_instante():
    for indice in range(25):
        campos = generar_instantanea_sintetica(indice, n_fotogramas=25)
        pared = np.isin(campos["etiquetas"], (1, 2))
        ix = int(np.argmin(np.abs(campos["x"])))
        iy = int(np.argmin(np.abs(campos["y"])))
        kz = np.flatnonzero(campos["etiquetas"][ix, iy, :] == 3)
        assert np.any(pared) and kz.size
        temperatura_pared = float(np.mean(campos["T"][pared]))
        temperatura_centro = float(np.mean(campos["T"][ix, iy, kz]))
        assert temperatura_pared > temperatura_centro, (
            f"t={campos['t']} s: pared={temperatura_pared}, centro={temperatura_centro}"
        )


def test_cohesion_sintetica_es_un_campo_continuo():
    campos = generar_instantanea_sintetica(12, n_fotogramas=25)
    intermedias = (campos["cohesion"] > 0.01) & (campos["cohesion"] < 0.99)
    assert float(np.mean(intermedias)) > 0.30
    assert np.unique(np.round(campos["cohesion"], decimals=4)).size > 100


def test_cliente_precarga_float32_e_interpola_antes_de_reproducir():
    cache = (DIRECTORIO_WEB / "js" / "frame-cache.js").read_text(encoding="utf-8")
    app = (DIRECTORIO_WEB / "js" / "app.js").read_text(encoding="utf-8")
    assert "class CacheFotogramas" in cache and "async precargar" in cache
    assert "arrayBuffer()" in cache and "Float32Array" in cache
    assert "interpolarFotogramas" in cache
    assert re.search(r"FPS_REPRODUCCION\s*=\s*30", app)
    bloque_animar = app[app.index("function animar(tiempo)"):app.index("async function iniciar()")]
    assert "fetch(" not in bloque_animar


def test_radio_de_render_es_visible_respecto_a_altura_del_lecho():
    particulas = (DIRECTORIO_WEB / "js" / "particle-system.js").read_text(encoding="utf-8")
    coincidencia = re.search(r"FACTOR_RADIO_RENDER\s*=\s*([0-9.]+)", particulas)
    assert coincidencia
    factor = float(coincidencia.group(1))
    radio_minimo_mm = 100.0 * 0.0005 * factor
    assert radio_minimo_mm / 3.26 > 0.03


def test_aglomerado_es_oscuro_mate_y_no_blanco_puro():
    app = (DIRECTORIO_WEB / "js" / "app.js").read_text(encoding="utf-8")
    bloque = app[app.index("const parametros = esAglomerado"):app.index("const iso = new THREE.Mesh", app.index("const parametros = esAglomerado"))]
    assert "color: 0x20221f" in bloque
    assert "roughness: 0.96" in bloque
    assert "0xffffff" not in bloque.lower()


def test_isosuperficie_comparte_vertices_y_promedia_normales():
    marching = (DIRECTORIO_WEB / "js" / "marching-cubes.js").read_text(encoding="utf-8")
    assert "indexarVertices" in marching
    assert "geometria.setIndex" in marching
    assert "geometria.computeVertexNormals()" in marching
    assert "suavizarCampoGaussiano" in marching


def test_configuracion_y_visor_incluyen_tapa_cerrada():
    servidor, hilo, url = iniciar_servidor(0)
    try:
        with urllib.request.urlopen(url + "api/config", timeout=5) as respuesta:
            tapa = json.loads(respuesta.read().decode("utf-8"))["geometria"]["tapa"]
        assert tapa["tipo"] == "cilindro_cerrado"
        assert tapa["radio_mm"] > 0
        assert tapa["espesor_mm"] > 0
    finally:
        detener_servidor(servidor, hilo)
    app = (DIRECTORIO_WEB / "js" / "app.js").read_text(encoding="utf-8")
    assert "Tapa cerrada del crisol del ensayo" in app
    assert "new THREE.CylinderGeometry(rBoca, rBoca" in app


def test_configuracion_sirve_la_paleta_mineral_completa():
    """La leyenda del cliente se construye con lo que manda el servidor."""
    servidor, hilo, url = iniciar_servidor(0)
    try:
        with urllib.request.urlopen(url + "api/config", timeout=5) as respuesta:
            fases = json.loads(respuesta.read().decode("utf-8"))["fases"]
    finally:
        detener_servidor(servidor, hilo)
    assert fases["fases"] and fases["orden_leyenda"]
    assert "campos_solidos" in fases and fases["campos_solidos"]["C"] == "char"
    assert fases["campos_solidos"]["H2O_liq"] is None
    for clave, fase in fases["fases"].items():
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", fase["color"]), clave
        assert fase["grupo"] in {"mineral", "carbonoso", "agregado"}
        if clave != "aglomerado":
            assert fase["volumen_molar_cm3_mol"] > 0, clave
    assert "PREDICCIÓN" in fases["nota"]


def test_la_interfaz_tiene_leyenda_de_fases_y_conmutador_de_color():
    html = (DIRECTORIO_WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="leyenda-fases"' in html
    assert 'id="colorear-fases"' in html
    assert 'id="fases-activas"' in html
    css = (DIRECTORIO_WEB / "css" / "estilos.css").read_text(encoding="utf-8")
    assert ".fase-item" in css and ".muestra-fase" in css and ".fase-item.nueva" in css


def test_el_cliente_no_duplica_la_mineralogia_y_pesa_por_volumen():
    """Los colores minerales viven en Python; el cliente sólo los presenta."""
    app = (DIRECTORIO_WEB / "js" / "app.js").read_text(encoding="utf-8")
    particulas = (DIRECTORIO_WEB / "js" / "particle-system.js").read_text(encoding="utf-8")
    assert "construirLeyendaFases" in app and "actualizarLeyendaFases" in app
    assert "configurarFases" in app and "configurarFases" in particulas
    # La ponderación por volumen es lo que hace fiel la mezcla de colores.
    assert "volumen_molar_cm3_mol" in app and "volumenMolar" in particulas
    # Ningún color de mineral de la paleta puede estar escrito a mano en el
    # cliente: si se cambia en fisica/fases_visuales.py debe cambiar solo.
    from fisica.fases_visuales import FASES
    for clave, fase in FASES.items():
        hexadecimal = fase["color"].lstrip("#").lower()
        assert hexadecimal not in app.lower(), f"{clave} codificado en app.js"
        assert hexadecimal not in particulas.lower(), f"{clave} codificado en particle-system.js"


_GUION_JS = sorted(
    (Path(__file__).resolve().parent / "js").glob("prueba_*.mjs")
)


@pytest.mark.parametrize("guion", _GUION_JS, ids=lambda ruta: ruta.stem)
def test_comportamiento_de_los_modulos_del_navegador(guion):
    """Comprobación del comportamiento real de los módulos del cliente.

    Construyen el objeto, le pasan instantáneas y miden lo que sale (color,
    escala, vértices), en vez de inspeccionar el texto del fuente. Se saltan si
    no hay Node: el resto de la aplicación no lo necesita.
    """
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node no está disponible en este entorno")
    raiz = Path(__file__).resolve().parents[1]
    resultado = subprocess.run(
        [node, str(guion)],
        cwd=raiz,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    assert "FALLO" not in resultado.stdout


def test_transporte_binario_s3df_lo_decodifica_el_cliente_real(tmp_path):
    """El contrato binario se comprueba con el decodificador del navegador.

    145 fotogramas en JSON son ~1,1 GB de números en decimal; en Float32 crudo
    son ~130 MB. El cliente ya trae el decodificador, así que la prueba lo
    ejecuta con Node en vez de reimplementarlo aquí.
    """
    import shutil

    node = shutil.which("node")
    if node is None:
        import pytest

        pytest.skip("Node no está disponible en este entorno")

    from interfaz.app import _codificar_s3df, _estructura_para_web

    campos = generar_instantanea_sintetica(3, n_fotogramas=9, forma=(8, 8, 10))
    campos["termico"] = {"T_mufla_K": 1123.15, "T_pared_media_K": 900.0,
                         "T_tapa_K": 880.0, "flujo_radiativo_W_m2": 1.0}
    estructura = _estructura_para_web(campos)
    binario = tmp_path / "fotograma.bin"
    binario.write_bytes(_codificar_s3df(estructura))
    assert binario.read_bytes()[:4] == b"S3DF"

    raiz = Path(__file__).resolve().parents[1]
    resultado = subprocess.run(
        [node, str(raiz / "tests" / "js" / "decodificar_s3df.mjs"), str(binario)],
        cwd=raiz, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    decodificado = json.loads(resultado.stdout)

    assert decodificado["forma"] == list(estructura["forma"])
    assert abs(decodificado["t"] - float(estructura["t"])) < 1e-6
    assert decodificado["tieneTermico"] and decodificado["tieneNumeros"]

    def comparar(nombre, esperado):
        plano = np.asarray(esperado, dtype=np.float64).ravel(order="C")
        leido = decodificado["campos"][nombre]
        assert leido["n"] == plano.size, nombre
        assert abs(leido["primero"] - plano[0]) <= 1e-4 * max(1.0, abs(plano[0])), nombre
        assert abs(leido["ultimo"] - plano[-1]) <= 1e-4 * max(1.0, abs(plano[-1])), nombre
        # Float32 sobre miles de celdas: se compara la suma en términos relativos.
        assert abs(leido["suma"] - plano.sum()) <= 1e-4 * max(1.0, abs(plano.sum())), nombre

    for nombre in ("x", "y", "z", "u", "v", "w", "P", "T", "eps", "cohesion",
                   "conversion", "reduccion", "etiquetas"):
        comparar(nombre, estructura[nombre])
    for grupo in ("c_especies", "solido"):
        for nombre, valores in estructura[grupo].items():
            comparar(f"{grupo}.{nombre}", valores)


def test_el_servidor_sirve_el_binario_y_pesa_mucho_menos_que_el_json():
    servidor, hilo, url = iniciar_servidor(0)
    try:
        peticion = urllib.request.Request(url + "api/fotograma?indice=0&formato=float32")
        with urllib.request.urlopen(peticion, timeout=15) as respuesta:
            binario = respuesta.read()
            tipo = respuesta.headers["Content-Type"]
        with urllib.request.urlopen(url + "api/fotograma?indice=0", timeout=15) as respuesta:
            texto = respuesta.read()
    finally:
        detener_servidor(servidor, hilo)
    assert binario[:4] == b"S3DF"
    assert "float32" in tipo
    # El binario es más pequeño, pero la ganancia principal no es el tamaño
    # (los campos sólidos son casi todos ceros, y un cero en JSON son 2 bytes):
    # es que el cliente copia memoria en vez de convertir ~250.000 números por
    # fotograma desde texto decimal.
    assert len(binario) < len(texto), (len(binario), len(texto))
    # El JSON debe seguir sirviéndose para clientes que no negocian el binario.
    assert json.loads(texto.decode("utf-8"))["forma"]


def test_el_directorio_predeterminado_es_la_corrida_mas_avanzada(tmp_path):
    from interfaz.app import directorio_predeterminado

    (tmp_path / "corta").mkdir()
    (tmp_path / "larga").mkdir()
    for microsegundos in (0, 5_000_000):
        (tmp_path / "corta" / f"instantanea_{microsegundos:015d}us.npz").write_bytes(b"")
    for microsegundos in (0, 5_000_000, 720_000_000):
        (tmp_path / "larga" / f"instantanea_{microsegundos:015d}us.npz").write_bytes(b"")
    # Una carpeta con una sola instantánea no es una serie reproducible.
    (tmp_path / "suelta").mkdir()
    (tmp_path / "suelta" / "instantanea_000000999000000us.npz").write_bytes(b"")

    assert directorio_predeterminado(tmp_path) == tmp_path / "larga"
    assert directorio_predeterminado(tmp_path / "no_existe") == tmp_path / "no_existe" / "simulacion"


def test_el_directorio_predeterminado_ignora_los_npz_de_corridas_anteriores(tmp_path):
    """Elegir por el NOMBRE de los archivos contradice lo que luego se muestra.

    Un directorio donde se relanzó la simulación conserva los NPZ de la corrida
    vieja: los nombres siguen llegando a 720 s aunque la corrida vigente se
    detuviera a los 100. `EstadoAplicacion` ya descarta lo anterior al t=0 más
    reciente —bien: no mezcla dos físicas en una línea temporal— pero el
    selector no aplicaba ese mismo recorte, así que puntuaba esa carpeta con los
    720 s de la corrida MUERTA y con sus 100 archivos. Prefería la carpeta rancia
    frente a una corrida nueva y coherente, y la interfaz acababa abriendo 100 s
    creyendo haber elegido 720.

    Es el patrón de siempre en este proyecto: dos criterios que no coinciden, y
    el desacuerdo sólo se ve donde nadie mira.
    """
    from interfaz.app import EstadoAplicacion, directorio_predeterminado

    def escribir(carpeta: Path, tiempos_s, instante_ns: int) -> None:
        carpeta.mkdir(parents=True, exist_ok=True)
        for t in tiempos_s:
            campos = generar_instantanea_sintetica(0, n_fotogramas=2, forma=(6, 6, 8))
            campos["t"] = float(t)
            campos["metadatos"] = {"fuente": "prueba", "datos_sinteticos": False}
            ruta = carpeta / f"instantanea_{int(round(t * 1.0e6)):015d}us.npz"
            guardar_instantanea(campos, ruta)
            os.utime(ruta, ns=(instante_ns, instante_ns))

    viejo, nuevo = 1_000_000_000_000_000_000, 2_000_000_000_000_000_000
    mezclada = tmp_path / "mezclada"
    # Corrida vieja: llega a 720 s y deja veinte archivos.
    escribir(mezclada, [t * 40.0 for t in range(1, 19)] + [720.0], viejo)
    # Relanzada: reescribe t=0 y se detiene en 100 s. Es lo único que se verá.
    escribir(mezclada, [0.0, 50.0, 100.0], nuevo)
    # Corrida limpia y completa, con menos archivos que la mezclada.
    escribir(tmp_path / "completa", [0.0, 200.0, 400.0, 600.0, 720.0], nuevo)

    elegido = directorio_predeterminado(tmp_path)
    assert elegido == tmp_path / "completa", (
        f"eligió {elegido.name}: puntúa por el nombre de los archivos y no por "
        "la serie que la interfaz va a mostrar"
    )
    # Y lo elegido tiene que ser, de hecho, lo que más lejos llega al abrirlo.
    alcances = {
        carpeta.name: max(EstadoAplicacion(datos=carpeta).tiempos)
        for carpeta in (mezclada, tmp_path / "completa")
    }
    assert alcances["mezclada"] == 100.0
    assert alcances[elegido.name] == max(alcances.values())


def test_una_instantanea_a_medio_escribir_no_tumba_la_interfaz(tmp_path):
    """La interfaz abre la corrida más avanzada, que puede seguir escribiéndose."""
    campos = generar_instantanea_sintetica(1, n_fotogramas=4, forma=(8, 8, 10))
    campos["metadatos"] = {"fuente": "prueba", "datos_sinteticos": False}
    guardar_instantanea(campos, tmp_path / "instantanea_000000000000000us.npz")
    # NPZ truncado, como el que deja una corrida en marcha.
    completa = (tmp_path / "instantanea_000000000000000us.npz").read_bytes()
    (tmp_path / "instantanea_000000005000000us.npz").write_bytes(completa[: len(completa) // 3])

    servidor, hilo, url = iniciar_servidor(0, datos=tmp_path)
    try:
        with urllib.request.urlopen(url + "api/config", timeout=10) as respuesta:
            config = json.loads(respuesta.read().decode("utf-8"))
        assert config["datos_sinteticos"] is False
        assert config["n_fotogramas"] == 1, "la instantánea incompleta debe descartarse"
        with urllib.request.urlopen(url + "api/fotograma?indice=0", timeout=10) as respuesta:
            assert json.loads(respuesta.read().decode("utf-8"))["forma"] == [8, 8, 10]
    finally:
        detener_servidor(servidor, hilo)


def test_la_conversion_es_cero_al_empezar(tmp_path):
    """La referencia de la conversión es la hematita inicial, no una constante.

    Con la constante de 5.000 mol/m³ que había antes, y una carga cuya hematita
    inicial es 126 mol/m³, la interfaz marcaba **97,6 % de conversión en t=0**.
    """
    from interfaz.app import EstadoAplicacion, _estructura_para_web

    for indice in range(3):
        campos = generar_instantanea_sintetica(indice, n_fotogramas=3, forma=(8, 8, 10))
        campos["metadatos"] = {"fuente": "prueba", "datos_sinteticos": False}
        guardar_instantanea(campos, tmp_path / f"instantanea_{indice:015d}us.npz")

    estado = EstadoAplicacion(datos=tmp_path)
    try:
        primera = estado.cargar(0)
        estructura = _estructura_para_web(primera, estado.Fe2O3_inicial_mol_m3)
        lecho = np.asarray(primera["etiquetas"]) == 3
        conversion = np.asarray(estructura["conversion"])[lecho]
        assert conversion.size
        assert float(np.max(conversion)) < 1.0e-9, (
            f"la conversión en t=0 debe ser cero, es {float(np.max(conversion)):.4f}")
        # Y fuera del lecho no se define conversión de la carga.
        assert float(np.max(np.asarray(estructura["conversion"])[~lecho])) == 0.0
    finally:
        estado.cerrar()


def test_los_numeros_adimensionales_declaran_su_origen(tmp_path):
    """Sin números del solucionador se usa la demo, pero se dice."""
    from interfaz.app import _numeros_adimensionales

    campos = generar_instantanea_sintetica(1, n_fotogramas=3, forma=(8, 8, 10))
    numeros = _numeros_adimensionales(campos)
    assert numeros["origen"] == "demostracion"

    campos["metadatos"] = dict(campos.get("metadatos") or {})
    campos["metadatos"]["numeros_adimensionales"] = {
        "Re_particula": 0.053, "Re_celda": 0.61, "Ra": 188.0,
        "Da": 1.4, "Pe_masico": 1.01, "Pe_termico": 0.98,
    }
    reales = _numeros_adimensionales(campos)
    assert reales["origen"] == "solucionador"
    assert reales["Re"] == 0.053 and reales["Pe"] == 0.98 and reales["Da"] == 1.4

    app = (DIRECTORIO_WEB / "js" / "app.js").read_text(encoding="utf-8")
    assert "origen === 'solucionador'" in app
    assert "'Solucionador'" in app


def test_los_metadatos_conservan_los_diagnosticos_del_caso(tmp_path):
    """`guardar_instantanea` no puede tirar lo que aporta quien la genera."""
    campos = generar_instantanea_sintetica(0, n_fotogramas=2, forma=(6, 6, 8))
    campos["metadatos"] = {
        "fuente": "solucionador",
        "datos_sinteticos": False,
        "numeros_adimensionales": {"Re_particula": 0.053, "Ra": 188.0},
    }
    ruta = guardar_instantanea(campos, tmp_path / "con_diagnosticos.npz")
    recuperado = cargar_instantanea(ruta)
    assert recuperado["metadatos"]["numeros_adimensionales"]["Ra"] == 188.0
    # Las claves propias del formato no se pueden pisar desde fuera.
    assert recuperado["metadatos"]["version_formato"]


def _corrida_de_prueba(directorio: Path, n: int = 3) -> Path:
    directorio.mkdir(parents=True, exist_ok=True)
    for indice in range(n):
        campos = generar_instantanea_sintetica(indice, n_fotogramas=n, forma=(6, 6, 8))
        campos["metadatos"] = {"fuente": "solucionador de prueba", "datos_sinteticos": False}
        micras = int(round(campos["t"] * 1.0e6))
        guardar_instantanea(campos, directorio / f"instantanea_{micras:015d}us.npz")
    return directorio


def test_el_sitio_estatico_sirve_los_mismos_bytes_que_la_api(tmp_path):
    """El sitio publicado y el servidor local no pueden divergir.

    El sitio estático existe para que la simulación se pueda abrir en un PC sin
    Python ni NPZ. Eso sólo vale si lo que se ve ahí es lo mismo que se ve en
    local: si el exportador reimplementara la codificación por su cuenta,
    tendríamos dos verdades y ninguna forma de saber cuál está mirando el
    lector. Aquí se comparan los bytes del fotograma exportado con los que el
    servidor entrega por HTTP para el mismo índice.
    """
    from interfaz.exportar_estatico import exportar

    origen = _corrida_de_prueba(tmp_path / "corrida")
    sitio = tmp_path / "sitio"
    manifiesto = exportar(sitio, origen, registrar=lambda _mensaje: None)
    assert manifiesto["n_fotogramas"] == 3
    # El manifiesto declara lo que de verdad descarga el lector: Pages sirve el
    # Float32 con gzip, así que el peso en disco no es el que sufre nadie.
    assert manifiesto["MB_descarga_gzip"] <= manifiesto["MB_fotogramas"]
    assert manifiesto["KB_por_fotograma_gzip"] > 0.0

    servidor, hilo, url = iniciar_servidor(0, datos=origen)
    try:
        for indice in range(3):
            peticion = urllib.request.Request(
                f"{url}api/fotograma?indice={indice}&formato=float32")
            with urllib.request.urlopen(peticion, timeout=10) as respuesta:
                del_servidor = respuesta.read()
            del_sitio = (sitio / "datos" / f"fotograma_{indice:04d}.bin").read_bytes()
            assert del_sitio == del_servidor, f"el fotograma {indice} diverge del servidor"
        with urllib.request.urlopen(url + "api/config", timeout=10) as respuesta:
            config_servidor = json.loads(respuesta.read().decode("utf-8"))
    finally:
        detener_servidor(servidor, hilo)

    config_sitio = json.loads((sitio / "datos" / "config.json").read_text(encoding="utf-8"))
    # La geometría del crisol y la paleta mineral tienen que llegar enteras: son
    # lo que dibuja el visor y la leyenda de fases.
    assert config_sitio["geometria"] == config_servidor["geometria"]
    assert config_sitio["fases"] == config_servidor["fases"]
    assert config_sitio["tiempos"] == config_servidor["tiempos"]


def test_el_sitio_estatico_no_publica_la_ruta_local_de_quien_lo_generó(tmp_path):
    """`directorio_datos` es una ruta absoluta de la máquina que simuló.

    En el servidor local es información útil; publicada es ruido y una fuga
    gratuita del árbol de archivos del autor. El sitio sólo debe decir QUÉ
    corrida está mostrando.
    """
    from interfaz.exportar_estatico import exportar

    origen = _corrida_de_prueba(tmp_path / "corrida_privada", n=2)
    sitio = tmp_path / "sitio"
    exportar(sitio, origen, registrar=lambda _mensaje: None)

    config = json.loads((sitio / "datos" / "config.json").read_text(encoding="utf-8"))
    assert config["directorio_datos"] == "corrida_privada"
    assert config["corrida"] == "corrida_privada"
    assert str(tmp_path) not in json.dumps(config)
    assert not Path(config["directorio_datos"]).is_absolute()


def test_el_sitio_estatico_se_declara_y_arrastra_todos_sus_recursos(tmp_path):
    """Sin la declaración, el cliente pediría /api/... y no habría servidor."""
    from interfaz.exportar_estatico import exportar

    origen = _corrida_de_prueba(tmp_path / "corrida", n=2)
    sitio = tmp_path / "sitio"
    exportar(sitio, origen, registrar=lambda _mensaje: None)

    html = (sitio / "index.html").read_text(encoding="utf-8")
    assert 'window.SIMULADOR3D_ESTATICO = "datos"' in html
    # La declaración tiene que ir ANTES del módulo que la lee.
    assert html.index("SIMULADOR3D_ESTATICO") < html.index('src="js/app.js"')
    # El servidor reescribía estos rótulos al vuelo; en el sitio no hay nadie
    # que lo haga después, así que se congelan al exportar.
    assert "Resultados del solucionador" in html
    assert "PREDICCIÓN · RESULTADOS DEL SOLUCIONADOR" in html

    for recurso in ("js/app.js", "js/rutas.js", "js/three.module.js", "css/estilos.css"):
        assert (sitio / recurso).is_file(), f"falta {recurso} en el sitio"
    # GitHub Pages pasa el sitio por Jekyll, que descarta lo que empieza por `_`.
    assert (sitio / ".nojekyll").is_file()
    for indice in range(2):
        assert (sitio / "datos" / f"fotograma_{indice:04d}.bin").is_file()
        assert (sitio / "datos" / f"lineas_{indice:04d}.json").is_file()


def test_el_sitio_estatico_no_pide_nada_a_la_red(tmp_path):
    """Publicado o no, la interfaz sigue sin depender de ningún CDN."""
    from interfaz.exportar_estatico import exportar

    origen = _corrida_de_prueba(tmp_path / "corrida", n=2)
    sitio = tmp_path / "sitio"
    exportar(sitio, origen, registrar=lambda _mensaje: None)

    patron = re.compile(r"https?://(?!127\.0\.0\.1|localhost)", re.IGNORECASE)
    for ruta in (*sitio.glob("*.html"), *(sitio / "js").glob("*.js"), *(sitio / "css").glob("*.css")):
        if ruta.name in {"three.module.js", "three.core.js", "OrbitControls.js"}:
            continue  # bibliotecas vendorizadas, ya cubiertas por su propia prueba
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        assert not patron.search(texto), f"{ruta.name} referencia un origen externo"
