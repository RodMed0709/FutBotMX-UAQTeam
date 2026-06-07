"""Exportacion del set de frames de evaluacion (tarea eval_frame_export).

Congela un conjunto reproducible de frames de los videos del split *testing* para
poder (a) anotar el ground-truth y (b) correr el pipeline sobre exactamente los
mismos frames. Sobre los videos con ``split == 2`` en ``db_metadata.csv``:

- extrae los frames de cuota reusando ``src.core.frame_extraction.extract_frames``,
- persiste cada frame como una imagen PNG bajo ``working_dirs.testing_frames_dir``
  (``data/testing_frames/``, git-ignored),
- genera un CSV de control **versionado** en ``working_dirs.testing_frames_csv``
  (``assets/testing_frames.csv``) con la procedencia y el grupo de cada frame.

El CSV alinea por ``(video_id, frame_index)`` (clave para emparejar GT y prediccion)
y registra ademas ``frame_original`` (indice del frame en el video fuente, via
``get_frame_indices``) para trazabilidad exacta. ``cv2`` se importa de forma
perezosa para que ``import src.data`` no lo arrastre.

API publica:
- ``export_testing_frames``: orquesta extraer -> escribir imagenes -> escribir CSV
  (idempotente).
- ``validate_testing_frames_schema``: handler independiente de validacion de esquema.

Columnas del CSV (orden fijo): ``id, video_id, video_ruta, frame_index,
frame_original, imagen, grupo``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.frame_extraction import extract_frames, get_frame_indices
from src.utils import PROJECT_ROOT, get_abs_path

# --- Esquema (fuente unica de verdad; mutable desde un solo lugar) ------------
COLUMNS = [
    "id",
    "video_id",
    "video_ruta",
    "frame_index",
    "frame_original",
    "imagen",
    "grupo",
]
TESTING_SPLIT = 2  # split de testing en db_metadata.csv
GROUP_RANDOM = "aleatorio"
GROUP_CENITAL = "cenital"  # videos de camara superior (forced_testing)
IMAGE_EXT = ".png"


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


def _load_eval_frames_config() -> tuple[str, str, str, list[str]]:
    """Lee la configuracion del set de evaluacion desde la configuracion global.

    El nombre del archivo de configuracion se toma de CONFIG_FILENAME en el .env.

    Returns:
        Tupla ``(metadata_csv, testing_frames_dir, testing_frames_csv,
        forced_testing)``, donde ``forced_testing`` es la lista de rutas (relativas
        a PROJECT_ROOT) fijadas a testing; vacia si la clave no existe.

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si faltan ``working_dirs.metadata_csv``,
            ``working_dirs.testing_frames_dir`` o ``working_dirs.testing_frames_csv``.
        FileNotFoundError: si el archivo de configuracion no existe.
    """
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")

    config_path = get_abs_path(f"configs/{config_filename}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    working_dirs = config.get("working_dirs", {})
    for key in ("metadata_csv", "testing_frames_dir", "testing_frames_csv"):
        if key not in working_dirs:
            raise KeyError(f"Falta 'working_dirs.{key}' en la configuracion.")

    forced_testing = config.get("splits", {}).get("forced_testing", [])

    return (
        working_dirs["metadata_csv"],
        working_dirs["testing_frames_dir"],
        working_dirs["testing_frames_csv"],
        list(forced_testing),
    )


def _load_testing_videos(metadata_csv: str) -> pd.DataFrame:
    """Devuelve las filas de db_metadata.csv con split == TESTING_SPLIT.

    Args:
        metadata_csv: ruta relativa a PROJECT_ROOT del manifiesto del dataset.

    Returns:
        ``DataFrame`` con columnas ``id`` y ``ruta`` de los videos de testing.

    Raises:
        ValueError / FileNotFoundError: si el manifiesto no resuelve (debe existir;
            esta tarea no lo genera).
    """
    csv_path = get_abs_path(metadata_csv)
    df = pd.read_csv(csv_path)
    testing = df[df["split"] == TESTING_SPLIT][["id", "ruta"]]
    return testing.reset_index(drop=True)


def _group_for(ruta: str, forced_testing: set[str]) -> str:
    """Grupo del frame segun su video de origen: cenital si esta fijado, si no aleatorio."""
    return GROUP_CENITAL if ruta in forced_testing else GROUP_RANDOM


def _write_frame_image(frame_rgb: np.ndarray, dest: Path) -> None:
    """Escribe un frame RGB como PNG (sin perdida) en ``dest``.

    ``extract_frames`` devuelve RGB; OpenCV escribe BGR, de ahi la conversion.
    """
    import cv2

    cv2.imwrite(str(dest), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def validate_testing_frames_schema(csv_path: Path) -> bool:
    """True si el CSV existe y su esquema coincide con COLUMNS (orden incluido).

    Comprueba solo la estructura (existencia + conjunto y orden de columnas), no el
    contenido fila a fila. Encapsulado aparte de la generacion porque el esquema es
    mutable a futuro: actualizar ``COLUMNS`` actualiza tambien esta validacion.

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


def export_testing_frames(force: bool = False) -> pd.DataFrame:
    """Exporta los frames de cuota de los videos de testing y devuelve el CSV.

    Orquesta: leer videos de testing -> por video extraer frames + indices ->
    escribir imagenes PNG -> escribir el CSV de control. La cuota la aplica
    ``extract_frames`` (lee ``preprocess.frame_quota``); las rutas salen de la
    configuracion global.

    Idempotente: si el CSV ya existe, pasa el handler de validacion y ``force=False``,
    se devuelve sin re-extraer. En cualquier otro caso (ausente, esquema invalido o
    ``force=True``) se regenera y sobrescribe por completo.

    Args:
        force: si es True, regenera y sobrescribe aunque exista un CSV valido.

    Returns:
        ``pandas.DataFrame`` con el contenido del CSV (columnas ``COLUMNS``).

    Raises:
        ValueError / KeyError / FileNotFoundError: ver ``_load_eval_frames_config``
            y ``_load_testing_videos``.
    """
    metadata_csv, frames_dir, frames_csv, forced = _load_eval_frames_config()
    # No usamos get_abs_path: el CSV versionado puede no existir aun (lo creamos).
    csv_path = PROJECT_ROOT / frames_csv

    if not force and csv_path.exists() and validate_testing_frames_schema(csv_path):
        return pd.read_csv(csv_path)

    forced_set = set(forced)
    videos = _load_testing_videos(metadata_csv)
    out_dir = PROJECT_ROOT / frames_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fid = 0
    for video_id, ruta in videos.itertuples(index=False):
        frames = extract_frames(Path(ruta), all_frames=False)
        originals = get_frame_indices(Path(ruta), all_frames=False)
        grupo = _group_for(ruta, forced_set)
        for frame_index, frame in enumerate(frames):
            img_name = f"{int(video_id):04d}_{frame_index:04d}{IMAGE_EXT}"
            _write_frame_image(frame, out_dir / img_name)
            rows.append(
                {
                    "id": fid,
                    "video_id": int(video_id),
                    "video_ruta": ruta,
                    "frame_index": frame_index,
                    "frame_original": int(originals[frame_index]),
                    "imagen": (Path(frames_dir) / img_name).as_posix(),
                    "grupo": grupo,
                }
            )
            fid += 1

    df = pd.DataFrame(rows, columns=COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
