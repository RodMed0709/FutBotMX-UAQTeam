"""Extraccion de frames de un video (tarea frame_extraction).

Define ``extract_frames``, que extrae frames de un unico video en dos modos:

- Modo cuota (por defecto): devuelve una cantidad fija de frames repartidos de
  forma uniforme en el tiempo. La cuota se lee del archivo de configuracion .json
  del proyecto (clave ``preprocess.frame_quota``), nunca se incrusta en el codigo.
- Modo completo: devuelve todos los frames disponibles del video.

La ruta del video se verifica reutilizando ``src.utils.get_abs_path``. Los frames
se devuelven en memoria como un arreglo de NumPy con forma ``(N, H, W, 3)``; esta
funcion no escribe nada a disco.
"""

from __future__ import annotations

import json
from pathlib import Path

import decord
import numpy as np

from src.utils import PROJECT_ROOT, get_abs_path

# Bridge nativo: decord devuelve arreglos NumPy (sin dependencia de torch).
decord.bridge.set_bridge("native")


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


def _load_frame_quota() -> int:
    """Lee la cuota de frames desde el archivo de configuracion del proyecto.

    El nombre del archivo de configuracion se toma de CONFIG_FILENAME en el .env;
    la cuota se lee de preprocess.frame_quota y debe ser un entero positivo.

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env o la cuota no es un
            entero positivo.
        KeyError: si falta la clave preprocess.frame_quota en la configuracion.
    """
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")

    config_path = get_abs_path(f"configs/{config_filename}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    preprocess = config.get("preprocess", {})
    if "frame_quota" not in preprocess:
        raise KeyError(
            "Falta la clave 'frame_quota' en 'preprocess' del archivo de configuracion."
        )

    quota = preprocess["frame_quota"]
    if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
        raise ValueError(
            f"'frame_quota' debe ser un entero positivo, se recibio: {quota!r}"
        )
    return quota


def _resolve_video_path(video_path: Path) -> Path:
    """Verifica la ruta del video y devuelve su ruta absoluta verificada.

    Acepta dos tipos de ruta:

    - **Ruta relativa** (respecto a PROJECT_ROOT): se delega en ``get_abs_path``,
      que la resuelve contra la raiz del proyecto y verifica su existencia. Es el
      camino para los videos que viven bajo ``dataset_dir`` (p. ej. data/raw).
    - **Ruta absoluta**: se acepta siempre que apunte a un **archivo existente y
      valido**, sin exigir que este bajo PROJECT_ROOT. Esto habilita videos
      ubicados en montajes o ubicaciones externas al proyecto. ``Path.is_file()``
      cubre a la vez "existe" y "es archivo" (un directorio se rechaza) y sigue
      symlinks cuyo destino exista.

    La validez del contenido como video no se comprueba aqui; la determina la capa
    de lectura (decord) al abrir el archivo.

    Raises:
        ValueError: si ``video_path`` no es de tipo Path.
        FileNotFoundError: si una ruta relativa no resuelve (via get_abs_path), o
            si una ruta absoluta no existe o no es un archivo.
    """
    if not isinstance(video_path, Path):
        raise ValueError(
            f"Se esperaba una ruta de tipo Path, se recibio: {type(video_path).__name__}"
        )

    if not video_path.is_absolute():
        return get_abs_path(str(video_path))

    if not video_path.is_file():
        raise FileNotFoundError(
            f"La ruta del video no existe o no es un archivo: {video_path}"
        )

    return video_path.resolve()


def extract_frames(video_path: Path, all_frames: bool = False) -> np.ndarray:
    """Extrae frames de un video como un arreglo de NumPy.

    Args:
        video_path: ruta del video (Path). Puede ser **relativa** a PROJECT_ROOT
            (se resuelve contra la raiz del proyecto) o **absoluta** a un archivo
            valido en cualquier ubicacion del sistema, incluso fuera del proyecto
            (p. ej. montajes o ubicaciones externas).
        all_frames: si es True, devuelve todos los frames disponibles; si es
            False (por defecto), devuelve una cuota de frames repartidos de forma
            uniforme en el tiempo. La cuota proviene de la configuracion.

    Returns:
        np.ndarray con forma ``(N, H, W, 3)`` (frames RGB) en memoria.

    Raises:
        ValueError: entrada invalida (video_path no es Path, CONFIG_FILENAME
            ausente o cuota invalida).
        FileNotFoundError: si la ruta del video no existe o no es un archivo.
        KeyError: si falta la cuota en la configuracion (modo cuota).
    """
    abs_path = _resolve_video_path(video_path)

    reader = decord.VideoReader(str(abs_path))
    total = len(reader)

    if all_frames:
        indices = np.arange(total)
    else:
        quota = _load_frame_quota()
        if total <= quota:
            # La cuota es un maximo: si el video tiene menos frames, se toman todos.
            indices = np.arange(total)
        else:
            # Indices equiespaciados en el tiempo a lo largo del video.
            indices = np.unique(np.linspace(0, total - 1, quota).round().astype(int))

    frames = reader.get_batch(indices.tolist()).asnumpy()
    return frames


def get_video_fps(video_path: Path) -> float:
    """Devuelve el fps promedio de un video.

    Abre el video solo para leer sus metadatos (no decodifica frames). Util para
    que el pipeline escriba el video de salida a la velocidad real de la fuente en
    modo completo.

    Args:
        video_path: ruta del video (Path). Puede ser relativa a PROJECT_ROOT o
            absoluta a un archivo valido, igual que en ``extract_frames``.

    Returns:
        El fps promedio del video como ``float``.

    Raises:
        ValueError: si ``video_path`` no es de tipo Path.
        FileNotFoundError: si la ruta del video no existe o no es un archivo.
    """
    abs_path = _resolve_video_path(video_path)
    reader = decord.VideoReader(str(abs_path))
    return float(reader.get_avg_fps())
