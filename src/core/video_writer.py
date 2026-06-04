"""Escritor de video (tarea video_writer).

Define ``write_video``, que escribe una secuencia de frames a un archivo mp4 en
disco. Es la unica pieza del MVP por-frame que persiste salida (las demas son
in-memory o display-only).

- Usa ``imageio`` con backend ffmpeg (RGB-nativo; los frames del proyecto son RGB).
- El fps por defecto se lee de la configuracion (``visualization.output_fps``) y
  puede sobreescribirse por parametro (el pipeline pasa el fps real de la fuente
  en modo "video completo").
- Crea el directorio de salida si no existe (``get_abs_path`` no sirve aqui porque
  exige que la ruta exista).

``imageio`` se importa de forma perezosa dentro de la funcion para no encarecer
``import src.core``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils import PROJECT_ROOT, get_abs_path


def _load_output_fps() -> float:
    """Lee el fps de salida por defecto desde el archivo de configuracion.

    Returns:
        ``visualization.output_fps`` como float.

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si falta ``visualization.output_fps`` en la configuracion.
        FileNotFoundError: si el archivo de configuracion no existe.
    """
    env_path = PROJECT_ROOT / ".env"
    config_filename = None
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "CONFIG_FILENAME":
                config_filename = value.strip()
                break
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")

    config_path = get_abs_path(f"configs/{config_filename}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    visualization = config.get("visualization", {})
    if "output_fps" not in visualization:
        raise KeyError("Falta la clave 'visualization.output_fps' en la configuracion.")
    return float(visualization["output_fps"])


def _validate_frames(frames: np.ndarray) -> None:
    """Valida que ``frames`` sea un arreglo ``(N, H, W, 3) uint8`` no vacio.

    Raises:
        ValueError: si el tipo, la forma, la cantidad o el dtype no son validos.
    """
    if not isinstance(frames, np.ndarray):
        raise ValueError(
            f"Se esperaba un numpy.ndarray, se recibio: {type(frames).__name__}"
        )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Se esperaba un arreglo (N, H, W, 3), se recibio forma: {frames.shape}"
        )
    if frames.shape[0] == 0:
        raise ValueError("No hay frames que escribir (N = 0).")
    if frames.dtype != np.uint8:
        raise ValueError(f"Se esperaba dtype uint8 (0-255), se recibio: {frames.dtype}")


def write_video(
    frames: np.ndarray,
    output_path: Path | str,
    fps: float | None = None,
) -> Path:
    """Escribe una secuencia de frames a un archivo mp4.

    Args:
        frames: ``np.ndarray (N, H, W, 3) uint8`` RGB (p. ej. la salida del overlay
            o de ``extract_frames``).
        output_path: ruta completa del mp4 a escribir. Si la carpeta no existe, se
            crea.
        fps: cuadros por segundo del video. Si es ``None``, se usa
            ``visualization.output_fps`` de la configuracion.

    Returns:
        La ruta (``Path``) del archivo escrito.

    Raises:
        ValueError: si ``frames`` no es ``(N, H, W, 3) uint8`` o esta vacio.
    """
    import imageio

    _validate_frames(frames)
    fps = fps if fps is not None else _load_output_fps()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    return output_path
