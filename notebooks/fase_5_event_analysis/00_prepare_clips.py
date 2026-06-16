"""Preparación del banco de pruebas de fase_5 (análisis de eventos). Corre en el POD/GPU.

Para los DOS videos de cámara superior (los únicos con homografía válida), genera el
**primer minuto** como clip independiente y deja TODO listo para las tareas de eventos:

1. **Clip cortado** (primeros ~60 s) → ``outputs/fase5_clips/<stem>_min1.mp4``.
2. **Detección + segmentación + tracking** en un **JSON extendido con máscaras**
   (``run_inference`` mode=tracking, ``yolo_sam3+bytetrack``, ``include_masks=True``) +
   video de tracking. → ``outputs/inference/fase5_clips/<stem>_min1/…``.
3. **Homografía**: minimap (``render_minimap_video`` ``detector="yolo"``) reusando los
   objetos del JSON de tracking. → ``outputs/fase5_clips/<stem>_min1_minimap.mp4``.

Es el primer minuto (desde el frame 0), así que basta ``max_frames`` (no hace falta
``start_frame``). Para minutos posteriores se usará ``start_frame`` cuando se exponga en
``track_video``/``run_inference``.

    python notebooks/fase_5_event_analysis/00_prepare_clips.py
"""

from pathlib import Path

from src.core.frame_extraction import get_video_fps
from src.utils import PROJECT_ROOT

CLIP_SECONDS = 60
RUN_LABEL = "fase5_clips"
OUT_DIR = PROJECT_ROOT / "outputs" / "fase5_clips"
VIDEOS = [
    PROJECT_ROOT / "data/raw/18abril/Camara_superior/IMG_9933.MOV",
    PROJECT_ROOT / "data/raw/18abril/Camara_superior/IMG_9938.MOV",
]


def cut_clip(src: Path, dst: Path, n_frames: int) -> Path:
    """Escribe los primeros ``n_frames`` de ``src`` a ``dst`` (mp4)."""
    import cv2
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(src))
    fps = float(vr.get_avg_fps())
    h, w = vr[0].shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(min(n_frames, len(vr))):
        writer.write(cv2.cvtColor(vr[i].asnumpy(), cv2.COLOR_RGB2BGR))
    writer.release()
    return dst


def _config_outputs_dir() -> str:
    """``working_dirs.outputs_dir`` de la config activa (para ubicar el JSON de tracking)."""
    import json as _json

    from src.core.detectors.yolo_boxes import _load_env

    cfg_name = _load_env(PROJECT_ROOT / ".env").get("CONFIG_FILENAME", "").strip()
    cfg = _json.loads((PROJECT_ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))
    return cfg["working_dirs"]["outputs_dir"]


def prepare(video: Path) -> dict:
    """Genera clip + tracking(+máscaras) + homografía para el primer minuto de ``video``.

    **Idempotente**: salta cada paso (corte / tracking / homografía) si su salida ya existe,
    así que interrumpir y re-correr reanuda sin rehacer lo costoso. Para forzar un paso,
    borra su archivo de salida.
    """
    from src.core.inference import run_inference
    from src.core.inference_schema import inference_paths
    from src.core.minimap_pipeline import render_minimap_video

    stem = f"{video.stem}_min1"
    clip = OUT_DIR / f"{stem}.mp4"
    minimap = OUT_DIR / f"{stem}_minimap.mp4"
    json_path, _ = inference_paths(stem, _config_outputs_dir(), RUN_LABEL)

    # 1) Corte del clip.
    if clip.exists():
        print(f"[{stem}] clip ya existe -> skip corte")
    else:
        n = round(CLIP_SECONDS * get_video_fps(video))
        print(f"[{stem}] cortando {CLIP_SECONDS}s = {n} frames -> {clip}")
        cut_clip(video, clip, n)

    # 2) Detección + segmentación + tracking (JSON extendido con máscaras).
    if json_path.exists():
        print(f"[{stem}] tracking JSON ya existe -> skip inferencia")
    else:
        print(f"[{stem}] detección+segmentación+tracking (yolo_sam3+bytetrack, masks)…")
        json_path = Path(run_inference(
            clip, mode="tracking", detector="yolo_sam3", tracker="bytetrack",
            include_masks=True, render_video=True, run_label=RUN_LABEL,
        )["json"])

    # 3) Homografía (minimap, objetos del JSON).
    if minimap.exists():
        print(f"[{stem}] minimap ya existe -> skip homografía")
    else:
        print(f"[{stem}] homografía (minimap, detector=yolo, objetos del JSON)…")
        render_minimap_video(
            clip, tracks_json=json_path, detector="yolo",
            draw_overlay=True, output_path=minimap,
        )

    return {"clip": clip, "tracking_json": json_path, "minimap": minimap}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for video in VIDEOS:
        if not video.exists():
            print(f"AVISO: no existe {video}; se omite.")
            continue
        res = prepare(video)
        print(f"[{video.stem}] LISTO:")
        for k, v in res.items():
            print(f"    {k}: {v}")
    print("\nBanco de pruebas fase_5 en:", OUT_DIR)


if __name__ == "__main__":
    main()
