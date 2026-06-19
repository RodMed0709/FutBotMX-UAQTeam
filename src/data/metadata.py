"""Gestion y organizacion de metadatos del dataset (tarea csv_dataset_metadata).

Genera y mantiene ``assets/db_metadata.csv``: un manifiesto tabular de los videos
``.MOV`` ubicados bajo ``dataset_dir``, con sus metadatos (duracion, resolucion,
fps) y una particion reproducible en *splits*.

El CSV es un asset estatico (manifiesto + splits), no un cache de metadatos del
pipeline: ``extract_frames`` y el pipeline no lo consumen. Su valor es habilitar el
trabajo sobre subconjuntos reproducibles (en especial *testing*) y el analisis
tabular del dataset sin abrir cada video.

API publica:
- ``build_metadata_csv``: orquesta descubrir -> extraer -> asignar splits -> validar
  -> escribir el CSV (idempotente).
- ``validate_metadata_schema``: handler independiente de validacion de esquema.

Columnas del CSV (orden fijo): ``id, ruta, nombre, duracion, ancho, alto,
fps_average, split``.
"""

from __future__ import annotations

import json
from pathlib import Path

import decord
import numpy as np
import pandas as pd

from src.utils import PROJECT_ROOT, get_abs_path

# Bridge nativo: decord devuelve arreglos NumPy (sin dependencia de torch).
decord.bridge.set_bridge("native")

# --- Esquema (fuente unica de verdad; mutable desde un solo lugar) ------------
COLUMNS = ["id", "ruta", "nombre", "duracion", "ancho", "alto", "fps_average", "split"]
VIDEO_EXTENSIONS = {".mov"}  # comparacion en minusculas (case-insensitive)

SPLIT_RESERVE = 0  # reserva (resto de los videos)
SPLIT_FINETUNING = 1  # fine-tuning
SPLIT_TESTING = 2  # testing
# Conteos fijos por split (el resto cae en SPLIT_RESERVE).
SPLIT_SIZES = {SPLIT_FINETUNING: 23, SPLIT_TESTING: 20}


def _load_env(env_path: Path) -> dict[str, str]:
    """Parseo simple de un archivo .env (KEY = value), aplicando strip()."""
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _load_metadata_config() -> tuple[str, str, int, list[str]]:
    """Lee la configuracion del manifiesto desde la configuracion global.

    El nombre del archivo de configuracion se toma de CONFIG_FILENAME en el .env.

    Returns:
        Tupla ``(dataset_dir, metadata_csv, split_seed, forced_testing)``, donde
        ``forced_testing`` es la lista de rutas (relativas a PROJECT_ROOT) que deben
        quedar siempre en el split de testing; vacia si la clave no existe.

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si faltan ``working_dirs.dataset_dir``,
            ``working_dirs.metadata_csv`` o ``seeds.split``.
        FileNotFoundError: si el archivo de configuracion no existe.
    """
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")

    config_path = get_abs_path(f"configs/{config_filename}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    working_dirs = config.get("working_dirs", {})
    if "dataset_dir" not in working_dirs:
        raise KeyError("Falta 'working_dirs.dataset_dir' en la configuracion.")
    if "metadata_csv" not in working_dirs:
        raise KeyError("Falta 'working_dirs.metadata_csv' en la configuracion.")

    seeds = config.get("seeds", {})
    if "split" not in seeds:
        raise KeyError("Falta 'seeds.split' en la configuracion.")

    # Opcional: videos fijados a testing (vacio si la seccion/clave no existe).
    forced_testing = config.get("splits", {}).get("forced_testing", [])

    return (
        working_dirs["dataset_dir"],
        working_dirs["metadata_csv"],
        int(seeds["split"]),
        list(forced_testing),
    )


def _discover_videos(dataset_dir: str) -> list[Path]:
    """Descubre los videos bajo dataset_dir de forma recursiva y determinista.

    Busca ``.MOV`` (case-insensitive) con ``rglob`` y los ordena alfabeticamente por
    su ruta POSIX relativa a PROJECT_ROOT, de modo que el ``id`` asignado sea estable
    e independiente del sistema de archivos o del directorio de trabajo.

    Args:
        dataset_dir: ruta relativa a PROJECT_ROOT del directorio de videos.

    Returns:
        Lista de rutas absolutas a los videos, en orden determinista.

    Raises:
        ValueError / FileNotFoundError: si dataset_dir no resuelve (via get_abs_path).
    """
    base = get_abs_path(dataset_dir)
    videos = [
        p
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda p: p.relative_to(PROJECT_ROOT).as_posix())


def _extract_video_metadata(abs_path: Path) -> dict:
    """Extrae los metadatos de un video con decord (solo lectura de metadatos).

    Lee fps promedio y conteo de frames del contenedor; las dimensiones se obtienen
    del primer frame (decord no expone alto/ancho sin decodificar un frame). La
    duracion se deriva como ``n_frames / fps_average``.

    Args:
        abs_path: ruta absoluta a un video valido.

    Returns:
        dict con claves ``duracion`` (float, s), ``ancho`` (int), ``alto`` (int) y
        ``fps_average`` (float).
    """
    reader = decord.VideoReader(str(abs_path))
    n_frames = len(reader)
    fps = float(reader.get_avg_fps())
    frame0 = reader[0].asnumpy()  # (H, W, 3)
    alto, ancho = int(frame0.shape[0]), int(frame0.shape[1])
    duracion = float(n_frames / fps) if fps > 0 else 0.0
    return {"duracion": duracion, "ancho": ancho, "alto": alto, "fps_average": fps}


def _assign_splits(
    n: int, seed: int, forced_testing_idx: list[int] | None = None
) -> list[int]:
    """Asigna un split a cada uno de los ``n`` videos de forma reproducible.

    Los indices en ``forced_testing_idx`` quedan **siempre** en testing; las plazas
    restantes de testing (``20 - nº fijados``) y las de fine-tuning (23) se reparten
    al azar (con la seed) entre los videos **no fijados**, mediante una unica
    permutacion y cortes contiguos. Asi los splits son disjuntos y sin reemplazo por
    construccion, y la fijacion no altera los conteos por split. El orden de salida
    esta alineado con el orden determinista de ``_discover_videos`` (indice -> split).

    Args:
        n: numero de videos.
        seed: semilla para reproducibilidad.
        forced_testing_idx: indices (sobre el orden de ``_discover_videos``) que se
            fijan a testing; ``None``/vacio replica el reparto totalmente aleatorio.

    Returns:
        Lista de longitud ``n`` con el split (0/1/2) de cada video.

    Raises:
        ValueError: si ``n`` es menor que la suma de los conteos fijos de splits, o si
            hay mas videos fijados que plazas de testing.
    """
    required = sum(SPLIT_SIZES.values())
    if n < required:
        raise ValueError(
            f"Videos insuficientes para los splits: se requieren al menos {required}, "
            f"hay {n}."
        )

    forced = set(forced_testing_idx or ())
    n_testing = SPLIT_SIZES[SPLIT_TESTING]
    if len(forced) > n_testing:
        raise ValueError(
            f"Hay {len(forced)} videos fijados a testing, pero solo {n_testing} "
            "plazas de testing."
        )

    splits = [SPLIT_RESERVE] * n
    for idx in forced:
        splits[idx] = SPLIT_TESTING

    # La aleatoriedad actua solo sobre los no fijados (reproducible con la seed).
    pool = [i for i in range(n) if i not in forced]
    rng = np.random.default_rng(seed)
    shuffled = [pool[int(i)] for i in rng.permutation(len(pool))]

    cursor = 0
    remaining_testing = n_testing - len(forced)
    for idx in shuffled[cursor : cursor + remaining_testing]:
        splits[idx] = SPLIT_TESTING
    cursor += remaining_testing
    for idx in shuffled[cursor : cursor + SPLIT_SIZES[SPLIT_FINETUNING]]:
        splits[idx] = SPLIT_FINETUNING
    return splits


def validate_metadata_schema(csv_path: Path) -> bool:
    """Handler de validacion: True si el CSV existe y su esquema coincide con COLUMNS.

    Comprueba solo la estructura (existencia + conjunto y orden de columnas), no el
    contenido fila a fila. Esta encapsulado aparte de la generacion porque el esquema
    es mutable a futuro: actualizar ``COLUMNS`` actualiza tambien esta validacion.

    Args:
        csv_path: ruta al archivo CSV a validar.

    Returns:
        ``True`` si el archivo existe y sus columnas son exactamente ``COLUMNS`` (en
        ese orden); ``False`` en caso contrario (incluido CSV ausente o ilegible).
    """
    if not csv_path.exists():
        return False
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        return False
    return list(header.columns) == COLUMNS


def build_metadata_csv(force: bool = False) -> pd.DataFrame:
    """Genera (o reutiliza) ``assets/db_metadata.csv`` y devuelve su DataFrame.

    Orquesta: descubrir videos -> extraer metadatos -> asignar splits -> escribir el
    CSV. La ruta del CSV y la semilla de los splits provienen de la configuracion
    global (``working_dirs.metadata_csv`` y ``seeds.split``).

    Idempotente: si el CSV ya existe, pasa el handler de validacion y ``force=False``,
    se devuelve sin reescribir. En cualquier otro caso (ausente, esquema invalido o
    ``force=True``) se regenera y sobrescribe por completo.

    Args:
        force: si es True, regenera y sobrescribe aunque exista un CSV valido.

    Returns:
        ``pandas.DataFrame`` con el contenido del CSV (columnas ``COLUMNS``).

    Raises:
        ValueError / KeyError / FileNotFoundError: ver ``_load_metadata_config``.
        ValueError: si no hay videos suficientes para los splits (``_assign_splits``).
    """
    dataset_dir, metadata_csv, seed, forced_testing = _load_metadata_config()
    # No usamos get_abs_path: el CSV puede no existir aun (lo estamos creando).
    csv_path = PROJECT_ROOT / metadata_csv

    if not force and csv_path.exists() and validate_metadata_schema(csv_path):
        return pd.read_csv(csv_path)

    videos = _discover_videos(dataset_dir)

    # Mapa ruta-relativa -> indice, para resolver los videos fijados a testing.
    ruta_a_idx = {
        p.relative_to(PROJECT_ROOT).as_posix(): idx for idx, p in enumerate(videos)
    }
    forced_testing_idx = []
    for ruta in forced_testing:
        if ruta not in ruta_a_idx:
            raise ValueError(
                f"Video fijado a testing no encontrado en el dataset: {ruta!r}"
            )
        forced_testing_idx.append(ruta_a_idx[ruta])

    splits = _assign_splits(len(videos), seed, forced_testing_idx)

    rows = []
    for idx, abs_path in enumerate(videos):
        meta = _extract_video_metadata(abs_path)
        rows.append(
            {
                "id": idx,
                "ruta": abs_path.relative_to(PROJECT_ROOT).as_posix(),
                "nombre": abs_path.name,
                **meta,
                "split": splits[idx],
            }
        )

    df = pd.DataFrame(rows, columns=COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
