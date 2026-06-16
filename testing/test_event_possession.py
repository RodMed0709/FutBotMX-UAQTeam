"""Harness de análisis de eventos sobre un clip (fase_5). Reutilizable por las tareas
de eventos posteriores.

Analiza la posesión sobre un **JSON de tracking**. Dos modos:

    # 1) Analizar un JSON existente (CPU local, sin GPU)
    python testing/test_event_possession.py [ruta/al/tracks.json]

    # 2) Generar el tracking de un clip de CLIP_SECONDS y analizarlo (POD/GPU)
    python testing/test_event_possession.py --video data/raw/.../IMG_9933.MOV

El análisis consume un JSON, así que para un clip de ~35 s hace falta un JSON de ~35 s:
el modo (2) lo genera con `run_inference(mode="tracking", max_frames=35*fps)`.
(Generar arranca en el frame 0: `track_video` aún no expone `start_frame` — pendiente de
la tarea `frame_window_sampling`.)
"""

import json
import sys
from pathlib import Path

from src.core.events import (
    FrameObject,
    compute_possession,
    load_frame_objects,
    write_possession_json,
)
from src.utils import PROJECT_ROOT

# --- Configuración del clip ---
CLIP_SECONDS = 35  # duración objetivo del clip a analizar (300 frames ≈ 10 s no alcanzaba)
DEFAULT_TRACKS = PROJECT_ROOT / "outputs/inference/fase3_eventos/IMG_9780/IMG_9780.json"
GEN_DETECTOR, GEN_TRACKER = "yolo_sam3", "bytetrack"  # para el modo --video (pod)


def generate_tracks(video: str, seconds: int) -> Path:
    """Genera un JSON de tracking del clip (primeros `seconds`) vía run_inference. POD/GPU."""
    from src.core.frame_extraction import get_video_fps
    from src.core.inference import run_inference

    fps = get_video_fps(Path(video))
    max_frames = round(seconds * fps)
    print(f"[gen] tracking {seconds}s = {max_frames} frames ({GEN_DETECTOR}+{GEN_TRACKER})…")
    res = run_inference(
        video, mode="tracking", detector=GEN_DETECTOR, tracker=GEN_TRACKER,
        max_frames=max_frames, render_video=False, run_label="fase5_eventos",
    )
    return Path(res["json"])


def resolve_tracks() -> Path:
    """Resuelve el JSON a analizar según los argumentos."""
    args = sys.argv[1:]
    if "--video" in args:
        return generate_tracks(args[args.index("--video") + 1], CLIP_SECONDS)
    if args:
        return Path(args[0])
    return DEFAULT_TRACKS


def _plot_timeline(result, png_path, fps) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = sorted(result.por_frame)
    owners = [result.por_frame[f] for f in frames]
    ids = sorted({o for o in owners if o is not None})
    y_of = {oid: i for i, oid in enumerate(ids)}
    xs = [f for f, o in zip(frames, owners) if o is not None]
    ys = [y_of[o] for o in owners if o is not None]

    fig, ax = plt.subplots(figsize=(12, max(2.0, len(ids) * 0.4)))
    ax.scatter(xs, ys, s=8)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"robot #{i}" for i in ids])
    ax.set_xlabel("frame")
    ax.set_title("Posesión por frame (event_possession)")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("timeline:", png_path)


def _edge_cases(fps) -> None:
    assert compute_possession({}, fps=fps).resumen["n_frames"] == 0  # vacío
    sin_robot = {0: [FrameObject(9, "orange_ball", (10, 10, 4, 4), (12, 12), 0.9)]}
    assert compute_possession(sin_robot, min_frames=1).por_frame[0] is None
    sin_balon = {0: [FrameObject(0, "robot", (0, 0, 50, 50), (25, 25), 0.9)]}
    assert compute_possession(sin_balon, min_frames=1).por_frame[0] is None
    pegado = {
        0: [
            FrameObject(7, "robot", (0, 0, 50, 50), (25, 25), 0.9),
            FrameObject(9, "orange_ball", (24, 24, 4, 4), (25, 25), 0.9),
        ]
    }
    assert compute_possession(pegado, min_frames=1).por_frame[0] == 7
    print("casos borde OK")


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    meta = json.loads(tracks.read_text(encoding="utf-8"))
    fps = meta.get("fps")
    by_frame = load_frame_objects(tracks)
    n = meta.get("num_frames") or len(by_frame)
    dur = (n / fps) if fps else 0.0
    print(f"video: {meta.get('video')}")
    print(f"json:  {tracks}")
    print(f"frames={n} fps={fps} duración={dur:.1f}s")
    if dur + 0.5 < CLIP_SECONDS:
        print(f"AVISO: el clip dura {dur:.1f}s (< objetivo {CLIP_SECONDS}s). "
              f"Para uno más largo: --video <ruta> (genera tracking en el pod).")

    result = compute_possession(by_frame, fps=fps)
    print("resumen:\n" + json.dumps(result.resumen, indent=2, ensure_ascii=False))

    # --- invariantes ---
    r = result.resumen
    total = r["pct_controlado"] + r["pct_libre"] + r["pct_no_visible"]
    assert abs(total - 100.0) <= 0.3, f"porcentajes no suman ~100: {total}"
    n_owned = sum(1 for v in result.por_frame.values() if v is not None)
    assert sum(o["frames"] for o in r["posesion_por_obj"].values()) == n_owned
    print("invariantes OK")

    _edge_cases(fps)

    stem = tracks.stem
    _plot_timeline(result, PROJECT_ROOT / "outputs" / f"event_possession_{stem}.png", fps)
    out_json = write_possession_json(
        result, PROJECT_ROOT / "outputs" / f"event_possession_{stem}.json"
    )
    print("escrito:", out_json)
    print("OK")


if __name__ == "__main__":
    main()
