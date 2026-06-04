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

from src.core.frame_extraction import extract_frames, get_video_fps
from src.core.overlay import overlay_detections
from src.core.sam3_loader import load_sam3
from src.core.segmentation import detect_classes_in_frame
from src.core.video_writer import write_video
from src.utils import PROJECT_ROOT, get_abs_path


def _load_pipeline_config() -> tuple[list[dict], str, float]:
    """Lee (classes, outputs_dir, output_fps) de la configuracion en una lectura.

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
    )


def run_pipeline(
    video_path: Path | str,
    output_path: Path | None = None,
    all_frames: bool = False,
    mode: str = "per_frame",
) -> dict[str, Path]:
    """Ejecuta el pipeline por-frame y genera un mp4 anotado + un JSON.

    Args:
        video_path: ruta del video (relativa a PROJECT_ROOT o absoluta).
        output_path: ruta del mp4 de salida. Si es ``None``, se auto-nombra bajo
            ``working_dirs.outputs_dir`` como ``<stem>_annotated.mp4``.
        all_frames: ``False`` (cuota, por defecto) o ``True`` (todos los frames).
        mode: solo ``"per_frame"`` esta implementado.

    Returns:
        ``{"video": <ruta_mp4>, "detections": <ruta_json>}``.

    Raises:
        NotImplementedError: si ``mode`` no es ``"per_frame"``.
        FileNotFoundError / ValueError: si el video o la config no resuelven.
    """
    if mode != "per_frame":
        raise NotImplementedError(
            f"mode '{mode}' no soportado (solo 'per_frame' por ahora)."
        )

    classes, outputs_dir, config_fps = _load_pipeline_config()

    # fps de salida: en modo completo, el fps real de la fuente; en cuota
    # (frames muestreados), el fps de configuracion (slideshow).
    fps = get_video_fps(video_path) if all_frames else config_fps

    # Composicion de rutas de salida.
    stem = Path(video_path).stem
    if output_path is not None:
        mp4_path = Path(output_path)
        json_path = mp4_path.with_name(f"{mp4_path.stem}_detections.json")
    else:
        base = PROJECT_ROOT / outputs_dir
        mp4_path = base / f"{stem}_annotated.mp4"
        json_path = base / f"{stem}_detections.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Modelo una sola vez.
    bundle = load_sam3()

    frames = extract_frames(video_path, all_frames=all_frames)
    total = len(frames)

    composed: list[np.ndarray] = []
    records: list[dict] = []
    for i, frame in enumerate(frames):
        print(f"  frame {i + 1}/{total}")
        dets = detect_classes_in_frame(frame, classes=classes, bundle=bundle)
        composed.append(overlay_detections(frame, dets, classes=classes))
        records.append(
            {
                "index": i,
                "detections": {
                    name: [{"obj_id": d.obj_id, "score": d.score} for d in cdets]
                    for name, cdets in dets.items()
                },
            }
        )

    mp4_path = write_video(np.stack(composed), mp4_path, fps=fps)

    payload = {
        "video": str(video_path),
        "mode": mode,
        "all_frames": all_frames,
        "fps": fps,
        "num_frames": total,
        "classes": [c["name"] for c in classes],
        "frames": records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {"video": mp4_path, "detections": json_path}
