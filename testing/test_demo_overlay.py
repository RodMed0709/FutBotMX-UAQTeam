"""Harness de T7 `event_overlay_narrative` (demo). Corre en CPU local (sin GPU).

Ensambla el video demo (mosaico + métricas + banner de gol) y escribe un frame de muestra.

    python testing/test_demo_overlay.py [ruta/al/tracks.json]
"""

import sys
from pathlib import Path

from src.core.demo_overlay import compose_demo
from src.core.events_schema import events_paths
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT
    / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _sample_png(mp4: Path, png: Path, frame_idx: int = 850) -> None:
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if ok:
        png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png), img)
        print("frame de muestra:", png)


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    # Sin output_path: compose_demo deriva el default vía events_paths.
    mp4 = compose_demo(tracks, max_seconds=120.0)

    # --- invariantes ---
    assert mp4.exists() and mp4.stat().st_size > 0, "el demo no se escribió"
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert n > 0, "el demo no tiene frames"
    assert n / max(fps, 1e-6) <= 121.0, f"el demo excede 2 min: {n / fps:.1f}s"
    print(f"demo: {n} frames @ {fps:.2f}fps = {n / fps:.1f}s")
    print("invariantes OK")

    _sample_png(mp4, events_paths(tracks.stem, "demo_sample", "png"))
    print("OK")


if __name__ == "__main__":
    main()
