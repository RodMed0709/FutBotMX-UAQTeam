"""Preparación del banco de pruebas de fase_5 (análisis de eventos). Corre en el POD/GPU.

Para clips curados de los videos de cámara superior (los únicos con homografía válida),
genera cada clip como video independiente y deja TODO listo para las tareas de eventos:

1. **Clip cortado** (tramo ``[inicio, inicio+dur)``) → ``outputs/fase5_clips/<stem>.mp4``.
2. **Detección + segmentación + tracking** en un **JSON extendido con máscaras**
   (``run_inference`` mode=tracking, ``yolo_sam3+bytetrack``, ``include_masks=True``) +
   video de tracking. → ``outputs/inference/fase5_clips/<stem>/…``.
3. **Homografía**: minimap (``render_minimap_video`` ``detector="yolo"``) reusando los
   objetos del JSON de tracking. → ``outputs/fase5_clips/<stem>_minimap.mp4``.

El clip se **corta físicamente primero** (con offset ``inicio``), así que el tracking y la
homografía operan sobre el clip desde su frame 0: no hace falta exponer ``start_frame`` en
``track_video``/``run_inference``.

Los clips a generar se declaran en ``CLIPS`` (video, etiqueta, inicio_s, duración_s).

    python notebooks/fase_5_event_analysis/00_prepare_clips.py
"""

from pathlib import Path

from src.core.frame_extraction import get_video_fps
from src.utils import PROJECT_ROOT

RUN_LABEL = "fase5_clips"
OUT_DIR = PROJECT_ROOT / "outputs" / "fase5_clips"

_SUPERIOR = PROJECT_ROOT / "data/raw/18abril/Camara_superior"
# Clips curados: (video, etiqueta, inicio_s, duración_s). El stem de salida es
# "<video.stem>_<etiqueta>"; NO cambiar etiquetas ya generadas (rompería la idempotencia).
CLIPS = [
    (_SUPERIOR / "IMG_9933.MOV", "min1", 0, 60),  # primer minuto (mayormente preparación)
    (_SUPERIOR / "IMG_9938.MOV", "min1", 0, 60),
    (_SUPERIOR / "IMG_9933.MOV", "5m30", 330, 60),  # 5:30–6:30: jugada con gol, más movido
]


def cut_clip(src: Path, dst: Path, n_frames: int, start_frame: int = 0) -> Path:
    """Escribe ``n_frames`` de ``src`` a ``dst`` (mp4), arrancando en ``start_frame``."""
    import cv2
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(src))
    fps = float(vr.get_avg_fps())
    h, w = vr[0].shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    end = min(start_frame + n_frames, len(vr))
    for i in range(start_frame, end):
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


def prepare(video: Path, label: str, start_s: int, dur_s: int) -> dict:
    """Genera clip + tracking(+máscaras) + homografía para el tramo ``[start_s, start_s+dur_s)``.

    **Idempotente**: salta cada paso (corte / tracking / homografía) si su salida ya existe,
    así que interrumpir y re-correr reanuda sin rehacer lo costoso. Para forzar un paso,
    borra su archivo de salida.
    """
    from src.core.inference import run_inference
    from src.core.inference_schema import inference_paths
    from src.core.minimap_pipeline import render_minimap_video

    stem = f"{video.stem}_{label}"
    clip = OUT_DIR / f"{stem}.mp4"
    minimap = OUT_DIR / f"{stem}_minimap.mp4"
    json_path, _ = inference_paths(stem, _config_outputs_dir(), RUN_LABEL)

    # 1) Corte del clip.
    if clip.exists():
        print(f"[{stem}] clip ya existe -> skip corte")
    else:
        fps = get_video_fps(video)
        start_frame = round(start_s * fps)
        n = round(dur_s * fps)
        print(f"[{stem}] cortando {dur_s}s desde {start_s}s "
              f"(frames {start_frame}..{start_frame + n}) -> {clip}")
        cut_clip(video, clip, n, start_frame=start_frame)

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
    for video, label, start_s, dur_s in CLIPS:
        if not video.exists():
            print(f"AVISO: no existe {video}; se omite.")
            continue
        res = prepare(video, label, start_s, dur_s)
        print(f"[{video.stem}_{label}] LISTO:")
        for k, v in res.items():
            print(f"    {k}: {v}")
    print("\nBanco de pruebas fase_5 en:", OUT_DIR)


if __name__ == "__main__":
    main()
