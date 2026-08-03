"""Persistencia desacoplada de campos 3D para visualización y posproceso.

El módulo sólo conoce arrays y metadatos. No importa el solucionador ni escribe
al importarse. Las instantáneas NPZ no contienen objetos serializados con
``pickle``: las especies y fases se guardan con prefijos de clave explícitos.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


VERSION_FORMATO = "1.0.0"
PREFIJO_ESPECIE = "c_especies__"
PREFIJO_SOLIDO = "solido__"

UNIDADES_BASE: dict[str, str] = {
    "t": "s",
    "x": "mm",
    "y": "mm",
    "z": "mm",
    "etiquetas": "1",
    "u": "m/s",
    "v": "m/s",
    "w": "m/s",
    "P": "Pa",
    "T": "K",
    "c_especies": "mol/m3",
    "eps": "1",
    "solido": "mol/m3",
    "cohesion": "1",
}


def _obtener(objeto: Any, nombre: str, *alternativas: str) -> Any:
    """Obtiene una clave o atributo con alternativas y un error informativo."""
    nombres = (nombre, *alternativas)
    if isinstance(objeto, Mapping):
        for candidato in nombres:
            if candidato in objeto:
                return objeto[candidato]
    else:
        for candidato in nombres:
            if hasattr(objeto, candidato):
                return getattr(objeto, candidato)
    raise ValueError(f"falta el campo obligatorio '{nombre}'")


def _obtener_opcional(objeto: Any, nombre: str, valor: Any = None) -> Any:
    if isinstance(objeto, Mapping):
        return objeto.get(nombre, valor)
    return getattr(objeto, nombre, valor)


def _geometria(campos: Any, forma: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    """Extrae coordenadas 1-D y etiquetas desde el objeto o su malla."""
    malla = _obtener_opcional(campos, "malla")
    if malla is not None:
        x = np.asarray(malla.x, dtype=np.float64)
        y = np.asarray(malla.y, dtype=np.float64)
        z = np.asarray(malla.z, dtype=np.float64)
    else:
        x = np.asarray(_obtener(campos, "x"), dtype=np.float64)
        y = np.asarray(_obtener(campos, "y"), dtype=np.float64)
        z = np.asarray(_obtener(campos, "z"), dtype=np.float64)
    etiquetas = np.asarray(_obtener(campos, "etiquetas"), dtype=np.uint8)
    if (x.size, y.size, z.size) != forma:
        raise ValueError(
            "las coordenadas no corresponden a la forma de los campos centrales: "
            f"{(x.size, y.size, z.size)} != {forma}"
        )
    if etiquetas.shape != forma:
        raise ValueError(f"etiquetas tiene forma {etiquetas.shape}; se esperaba {forma}")
    return x, y, z, etiquetas


def _normalizar(campos: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convierte un estado por atributos o diccionario a arrays verificables."""
    P = np.asarray(_obtener(campos, "P"), dtype=np.float64)
    if P.ndim != 3:
        raise ValueError("P debe ser un array tridimensional")
    forma = tuple(int(n) for n in P.shape)
    nx, ny, nz = forma
    arrays: dict[str, np.ndarray] = {
        "u": np.asarray(_obtener(campos, "u"), dtype=np.float64),
        "v": np.asarray(_obtener(campos, "v"), dtype=np.float64),
        "w": np.asarray(_obtener(campos, "w"), dtype=np.float64),
        "P": P,
        "T": np.asarray(_obtener(campos, "T"), dtype=np.float64),
        "eps": np.asarray(_obtener(campos, "eps"), dtype=np.float64),
        "cohesion": np.asarray(_obtener(campos, "cohesion"), dtype=np.float64),
    }
    formas_esperadas = {
        "u": (nx + 1, ny, nz),
        "v": (nx, ny + 1, nz),
        "w": (nx, ny, nz + 1),
        "P": forma,
        "T": forma,
        "eps": forma,
        "cohesion": forma,
    }
    for nombre, esperada in formas_esperadas.items():
        if arrays[nombre].shape != esperada:
            raise ValueError(
                f"{nombre} tiene forma {arrays[nombre].shape}; se esperaba {esperada}"
            )

    c_especies = _obtener(campos, "c_especies", "c")
    solido = _obtener(campos, "solido")
    if not isinstance(c_especies, Mapping) or not c_especies:
        raise ValueError("c_especies debe ser un diccionario no vacío")
    if not isinstance(solido, Mapping) or not solido:
        raise ValueError("solido debe ser un diccionario no vacío")

    especies_limpias: dict[str, np.ndarray] = {}
    solidos_limpios: dict[str, np.ndarray] = {}
    for destino, origen, clase in (
        (especies_limpias, c_especies, "especie"),
        (solidos_limpios, solido, "fase sólida"),
    ):
        for nombre, valores in origen.items():
            if not str(nombre) or "__" in str(nombre):
                raise ValueError(f"nombre de {clase} no válido: {nombre!r}")
            array = np.asarray(valores, dtype=np.float64)
            if array.shape != forma:
                raise ValueError(
                    f"{clase} {nombre!r} tiene forma {array.shape}; se esperaba {forma}"
                )
            destino[str(nombre)] = array

    x, y, z, etiquetas = _geometria(campos, forma)
    arrays.update({"x": x, "y": y, "z": z, "etiquetas": etiquetas})
    # Campo OPCIONAL: factor de expansion volumetrica del aglomerado. Se anade
    # solo si el caso lo calcula, para que las series anteriores sigan siendo
    # legibles; al cargarlas vale 1 (sin hinchar) en todas las celdas.
    hinchamiento = _obtener_opcional(campos, "hinchamiento")
    if hinchamiento is not None:
        arreglo = np.asarray(hinchamiento, dtype=np.float64)
        if arreglo.shape != forma:
            raise ValueError(
                f"hinchamiento tiene forma {arreglo.shape}; se esperaba {forma}"
            )
        if np.any(arreglo < 1.0) or not np.all(np.isfinite(arreglo)):
            raise ValueError("hinchamiento debe ser finito y no menor que 1")
        arrays["hinchamiento"] = arreglo
    for nombre, valores in especies_limpias.items():
        arrays[f"{PREFIJO_ESPECIE}{nombre}"] = valores
    for nombre, valores in solidos_limpios.items():
        arrays[f"{PREFIJO_SOLIDO}{nombre}"] = valores

    t = float(_obtener(campos, "t", "tiempo"))
    metadatos_entrada = _obtener_opcional(campos, "metadatos", {}) or {}
    fuente = str(metadatos_entrada.get("fuente", "modelo"))
    sinteticos = bool(metadatos_entrada.get("datos_sinteticos", False))
    campos_meta: dict[str, dict[str, Any]] = {}
    for nombre in ("u", "v", "w", "P", "T", "eps", "cohesion"):
        campos_meta[nombre] = {
            "unidades": UNIDADES_BASE[nombre],
            "validado": False,
        }
    for nombre in especies_limpias:
        campos_meta[f"c_especies/{nombre}"] = {
            "unidades": UNIDADES_BASE["c_especies"],
            "validado": False,
        }
    for nombre in solidos_limpios:
        campos_meta[f"solido/{nombre}"] = {
            "unidades": UNIDADES_BASE["solido"],
            "validado": False,
        }
    metadatos = {
        "version_formato": VERSION_FORMATO,
        "unidades_geometria": "mm",
        "orden_indices": "ijk=xyz",
        "malla_velocidad": "MAC",
        "fuente": fuente,
        "datos_sinteticos": sinteticos,
        "validacion_global": "sin validación experimental",
        "campos": campos_meta,
    }
    # Diagnosticos escalares que aporte quien genera la instantanea (numeros
    # adimensionales, balances...). Se conservan tal cual para que el consumidor
    # no tenga que recalcularlos con propiedades que el NPZ no lleva. Las claves
    # propias del formato no se pueden pisar.
    for clave, valor in metadatos_entrada.items():
        if clave not in metadatos and clave not in {"campos"}:
            metadatos[clave] = valor
    return arrays, {"t": t, "metadatos": metadatos}


def guardar_instantanea(campos: Any, ruta: str | Path, formato: str = "npz") -> Path:
    """Guarda geometría, etiquetas, tiempo y todos los campos del estado.

    ``campos`` puede ser un diccionario o un objeto con atributos. Para NPZ se
    exige el esquema MAC del contrato. ``formato='vtk'`` delega en
    :func:`exportar_vtk`.
    """
    formato = formato.lower().lstrip(".")
    if formato in {"vtk", "vti"}:
        return exportar_vtk(campos, ruta)
    if formato != "npz":
        raise ValueError("formato no admitido; use 'npz' o 'vtk'")

    destino = Path(ruta)
    if destino.suffix.lower() != ".npz":
        destino = destino.with_suffix(".npz")
    destino.parent.mkdir(parents=True, exist_ok=True)
    arrays, extra = _normalizar(campos)
    contenido: dict[str, np.ndarray] = dict(arrays)
    contenido["t"] = np.asarray(extra["t"], dtype=np.float64)
    contenido["_metadatos_json"] = np.asarray(
        json.dumps(extra["metadatos"], ensure_ascii=False, sort_keys=True)
    )
    np.savez_compressed(destino, **contenido)
    return destino


def cargar_instantanea(ruta: str | Path) -> dict[str, Any]:
    """Carga una instantánea NPZ segura y reconstruye especies y sólidos."""
    origen = Path(ruta)
    with np.load(origen, allow_pickle=False) as datos:
        obligatorias = {
            "t", "x", "y", "z", "etiquetas", "u", "v", "w",
            "P", "T", "eps", "cohesion", "_metadatos_json",
        }
        faltantes = sorted(obligatorias.difference(datos.files))
        if faltantes:
            raise ValueError(f"instantánea incompleta {origen}: faltan {faltantes}")
        resultado: dict[str, Any] = {
            "t": float(np.asarray(datos["t"]).item()),
            "c_especies": {},
            "solido": {},
        }
        for nombre in ("x", "y", "z", "etiquetas", "u", "v", "w", "P", "T", "eps", "cohesion"):
            resultado[nombre] = np.array(datos[nombre], copy=True)
        # Series anteriores a la incorporacion del hinchamiento no lo traen:
        # 1 significa "sin hinchar" y las deja perfectamente utilizables.
        resultado["hinchamiento"] = (
            np.array(datos["hinchamiento"], copy=True) if "hinchamiento" in datos.files
            else np.ones_like(resultado["T"])
        )
        for clave in datos.files:
            if clave.startswith(PREFIJO_ESPECIE):
                resultado["c_especies"][clave[len(PREFIJO_ESPECIE):]] = np.array(
                    datos[clave], copy=True
                )
            elif clave.startswith(PREFIJO_SOLIDO):
                resultado["solido"][clave[len(PREFIJO_SOLIDO):]] = np.array(
                    datos[clave], copy=True
                )
        resultado["metadatos"] = json.loads(str(np.asarray(datos["_metadatos_json"]).item()))
    if not resultado["c_especies"] or not resultado["solido"]:
        raise ValueError(f"instantánea incompleta {origen}: faltan especies o fases sólidas")
    return resultado


def cargar_serie(directorio: str | Path) -> list[dict[str, Any]]:
    """Devuelve un índice de instantáneas NPZ ordenado por tiempo físico."""
    base = Path(directorio)
    indice: list[dict[str, Any]] = []
    for ruta in base.glob("*.npz"):
        instantanea = cargar_instantanea(ruta)
        indice.append({
            "ruta": ruta,
            "nombre": ruta.name,
            "t": instantanea["t"],
            "forma": tuple(int(n) for n in instantanea["P"].shape),
            "metadatos": instantanea["metadatos"],
        })
    indice.sort(key=lambda elemento: (elemento["t"], elemento["nombre"]))
    return indice


def _nombre_vtk(nombre: str) -> str:
    limpio = re.sub(r"[^0-9A-Za-z_]+", "_", nombre).strip("_")
    return limpio or "campo"


def _escribir_valores(archivo: Any, valores: np.ndarray, por_linea: int = 9) -> None:
    planos = np.nan_to_num(np.asarray(valores, dtype=np.float64), copy=False).ravel(order="F")
    for inicio in range(0, planos.size, por_linea):
        archivo.write(" ".join(f"{valor:.10g}" for valor in planos[inicio:inicio + por_linea]))
        archivo.write("\n")


def exportar_vtk(campos: Any, ruta: str | Path) -> Path:
    """Exporta VTK legacy ASCII ``RECTILINEAR_GRID`` sin dependencias externas."""
    destino = Path(ruta)
    if destino.suffix.lower() != ".vtk":
        destino = destino.with_suffix(".vtk")
    destino.parent.mkdir(parents=True, exist_ok=True)
    arrays, extra = _normalizar(campos)
    x, y, z = arrays["x"], arrays["y"], arrays["z"]
    nx, ny, nz = len(x), len(y), len(z)
    u_c = 0.5 * (arrays["u"][:-1, :, :] + arrays["u"][1:, :, :])
    v_c = 0.5 * (arrays["v"][:, :-1, :] + arrays["v"][:, 1:, :])
    w_c = 0.5 * (arrays["w"][:, :, :-1] + arrays["w"][:, :, 1:])

    escalares: list[tuple[str, np.ndarray]] = [
        ("etiquetas", arrays["etiquetas"]),
        ("presion_Pa", arrays["P"]),
        ("temperatura_K", arrays["T"]),
        ("porosidad", arrays["eps"]),
        ("cohesion", arrays["cohesion"]),
    ]
    escalares.extend(
        (f"concentracion_{clave[len(PREFIJO_ESPECIE):]}_mol_m3", valor)
        for clave, valor in arrays.items() if clave.startswith(PREFIJO_ESPECIE)
    )
    escalares.extend(
        (f"solido_{clave[len(PREFIJO_SOLIDO):]}_mol_m3", valor)
        for clave, valor in arrays.items() if clave.startswith(PREFIJO_SOLIDO)
    )

    with destino.open("w", encoding="ascii", newline="\n") as archivo:
        archivo.write("# vtk DataFile Version 3.0\n")
        archivo.write(
            f"simulador3d formato {VERSION_FORMATO}; prediccion sin validacion; t={extra['t']:.9g} s\n"
        )
        archivo.write("ASCII\n")
        archivo.write("DATASET RECTILINEAR_GRID\n")
        archivo.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        for eje, coordenadas in (("X", x), ("Y", y), ("Z", z)):
            archivo.write(f"{eje}_COORDINATES {coordenadas.size} double\n")
            _escribir_valores(archivo, coordenadas)
        archivo.write(f"POINT_DATA {nx * ny * nz}\n")
        archivo.write("VECTORS velocidad_m_s double\n")
        vectores = np.stack((u_c, v_c, w_c), axis=-1).reshape((-1, 3), order="F")
        for vector in np.nan_to_num(vectores, copy=False):
            archivo.write(" ".join(f"{valor:.10g}" for valor in vector) + "\n")
        for nombre, valores in escalares:
            archivo.write(f"SCALARS {_nombre_vtk(nombre)} double 1\n")
            archivo.write("LOOKUP_TABLE default\n")
            _escribir_valores(archivo, valores)
    return destino


__all__ = [
    "VERSION_FORMATO",
    "UNIDADES_BASE",
    "guardar_instantanea",
    "cargar_instantanea",
    "cargar_serie",
    "exportar_vtk",
]
