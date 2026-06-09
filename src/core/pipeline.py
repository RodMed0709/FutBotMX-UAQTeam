"""Primer pipeline ejecutable del MVP SAM3-only (tarea pipeline_runner).

Orquesta el flujo por-frame de punta a punta:

    video -> extract_frames -> (por frame: detect_classes_in_frame ->
    overlay_detections) -> write_video

y escribe ademas un JSON de detecciones (sin mascaras). Carga el modelo SAM3 una
sola vez y lee la configuracion una sola vez.

Modos:
- ``all_frames=False`` (cuota): testeo / generacion de frames para fine-tuning.
- ``all_frames=True`` (completo): uso real. El fps real de la fuente se cableara en
  una tarea posterior; por ahora el modo completo usa el fps de configuracion.
- ``mode="per_frame"`` es el unico implementado; ``mode`` queda preparado para
  conectar el tracking (tarea video_tracking) sin rediseñar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.frame_extraction import (
    extract_frames,
    get_frame_indices,
    get_video_fps,
)
from src.core.inference_schema import (
    build_header,
    frame_record,
    inference_paths,
    write_inference_json,
)
from src.core.overlay import overlay_detections
from src.core.sam3_loader import load_sam3
from src.core.segmentation import detect_classes_in_frame
from src.core.video_writer import write_video
from src.utils import PROJECT_ROOT, get_abs_path


def _load_pipeline_config() -> tuple[list[dict], str, float, dict]:
    """Lee (classes, outputs_dir, output_fps, config) de la configuracion.

    El ``config`` completo se devuelve para embeberlo como snapshot en el entregado
    (auto-descripcion del esquema de inferencia).

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si faltan ``classes``, ``working_dirs.outputs_dir`` o
            ``visualization.output_fps``.
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

    if "classes" not in config:
        raise KeyError("Falta la clave 'classes' en el archivo de configuracion.")
    working_dirs = config.get("working_dirs", {})
    if "outputs_dir" not in working_dirs:
        raise KeyError("Falta la clave 'working_dirs.outputs_dir' en la configuracion.")
    visualization = config.get("visualization", {})
    if "output_fps" not in visualization:
        raise KeyError("Falta la clave 'visualization.output_fps' en la configuracion.")

    return (
        config["classes"],
        working_dirs["outputs_dir"],
        float(visualization["output_fps"]),
        config,
    )


def run_pipeline(
    video_path: Path | str,
    output_path: Path | None = None,
    all_frames: bool = False,
    mode: str = "per_frame",
    include_masks: bool = False,
    render_video: bool = True,
) -> dict[str, Path | None]:
    """Ejecuta el pipeline por-frame y genera el JSON del esquema (+ mp4 opcional).

    El JSON sigue el **esquema comun de inferencia** (ver
    ``src.core.inference_schema``): cabecera con metadatos auto-descriptivos +
    ``frames`` con geometria por deteccion y, opcionalmente, las mascaras en
    COCO-RLE. Aqui el ``obj_id`` de cada deteccion es **inestable** (per-frame): solo
    es estable en el modo tracking.

    El **JSON es el entregable y se escribe siempre**; el mp4 anotado es **opcional**
    (``render_video``). Con ``render_video=False`` no se compone el overlay ni se
    escribe video (ahorro de CPU/IO para lotes/evaluacion).

    Args:
        video_path: ruta del video (relativa a PROJECT_ROOT o absoluta).
        output_path: ruta del mp4 de salida. Si es ``None``, se ubica bajo
            ``working_dirs.outputs_dir`` como ``inference/<stem>/<stem>.mp4`` y el
            JSON junto a el. Con ``render_video=False`` solo se usa para derivar la
            ruta del JSON (no se escribe video).
        all_frames: ``False`` (cuota, por defecto) o ``True`` (todos los frames).
        mode: solo ``"per_frame"`` esta implementado.
        include_masks: si ``True``, cada deteccion incluye su mascara en COCO-RLE
            (requiere ``pycocotools``). Por defecto ``False`` (JSON ligero).
        render_video: si ``True`` (por defecto, uso de un solo video) genera el mp4
            anotado; si ``False`` solo escribe el JSON. Ortogonal a ``mode`` y a
            ``include_masks``.

    Returns:
        ``{"json": <ruta_json>, "video": <ruta_mp4_o_None>}``. La clave ``"video"``
        siempre esta presente: vale la ruta del mp4 si ``render_video=True`` y
        ``None`` si ``render_video=False``.

    Raises:
        NotImplementedError: si ``mode`` no es ``"per_frame"``.
        FileNotFoundError / ValueError: si el video o la config no resuelven.
    """
    if mode != "per_frame":
        raise NotImplementedError(
            f"mode '{mode}' no soportado (solo 'per_frame' por ahora)."
        )

    # La firma acepta str|Path; extract_frames/get_frame_indices/get_video_fps
    # exigen Path, asi que normalizamos aqui.
    video_path = Path(video_path)
    classes, outputs_dir, config_fps, config = _load_pipeline_config()

    # fps de salida: en modo completo, el fps real de la fuente; en cuota
    # (frames muestreados), el fps de configuracion (slideshow).
    fps = get_video_fps(video_path) if all_frames else config_fps

    # Rutas de salida (carpeta por video bajo outputs/inference/).
    stem = Path(video_path).stem
    if output_path is not None:
        mp4_path = Path(output_path)
        json_path = mp4_path.with_name(f"{mp4_path.stem}.json")
    else:
        json_path, mp4_path = inference_paths(stem, outputs_dir)

    # Modelo una sola vez.
    bundle = load_sam3()

    frames = extract_frames(video_path, all_frames=all_frames)
    total = len(frames)
    # Indices REALES en el video fuente (alineados por posicion con extract_frames).
    source_indices = get_frame_indices(Path(video_path), all_frames=all_frames)

    composed: list[np.ndarray] = []
    records: list[dict] = []
    for i, frame in enumerate(frames):
        print(f"  frame {i + 1}/{total}")
        dets = detect_classes_in_frame(frame, classes=classes, bundle=bundle)
        if render_video:
            composed.append(overlay_detections(frame, dets, classes=classes))
        records.append(
            frame_record(int(source_indices[i]), dets, include_masks=include_masks)
        )

    # El mp4 es opcional: con render_video=False no se compone overlay ni se escribe.
    mp4_out = (
        write_video(np.stack(composed), mp4_path, fps=fps) if render_video else None
    )

    header = build_header(
        video=video_path,
        mode="segmentation",
        fps=fps,
        resolution=(frames.shape[1], frames.shape[2]),
        num_frames=total,
        classes=classes,
        include_masks=include_masks,
        config=config,
    )
    json_path = write_inference_json(header, records, json_path)

    return {"json": json_path, "video": mp4_out}
