"""Harness del overlay de espectador (ronda de entregable de eventos). CPU local (sin GPU).

Genera el video de espectador (marcador, banner de gol, minimapa, heatmap, métricas, lista de
eventos) con ambos layouts y la fuente de gol estricta/geométrica. Para ir rápido, renderiza
solo la **ventana del gol** (`start_frame`/`max_frames`) y verifica que el mp4 se crea/abre, el
marcador final cuadra con la fuente de gol y se exporta un PNG de muestra.

    python testing/test_event_broadcast_overlay.py [ruta/al/tracks.json]
"""

import sys
from pathlib import Path

from src.core.event_broadcast_overlay import render_broadcast_overlay
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)
# ventana que cubre el gol real (f1167-1246) para validar marcador + banner sin renderizar todo
GOAL_WINDOW = dict(start_frame=1120, max_frames=180)


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _frame_count(mp4: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")
    is_ref = tracks.stem == "IMG_9933_5m30"

    # --- layout 2 (default) + gol estricto ---
    r2 = render_broadcast_overlay(tracks, layout=2, goal_source="strict",
                                  progress=False, **GOAL_WINDOW)
    print("layout2 strict:", r2.resumen)
    assert r2.video.exists(), "no se escribió el mp4 (layout 2)"
    assert _frame_count(r2.video) > 0, "el mp4 (layout 2) no abre / 0 frames"
    assert r2.sample_png is not None and r2.sample_png.exists(), "falta el PNG de muestra"
    assert r2.resumen["overlay_degradado"] is False, "se esperaba homografía disponible"
    if is_ref:
        assert r2.resumen["marcador_final"]["blue"] == 1, "gol estricto: azul debería ser 1"

    # --- layout 1 (mismo contenido, otra disposición) ---
    r1 = render_broadcast_overlay(tracks, layout=1, goal_source="strict",
                                  progress=False, **GOAL_WINDOW)
    print("layout1 strict:", r1.resumen)
    assert r1.video.exists() and _frame_count(r1.video) > 0, "el mp4 (layout 1) no abre"

    # --- fuente de gol geométrica (más laxa) ---
    rg = render_broadcast_overlay(tracks, layout=2, goal_source="geometric",
                                  progress=False, **GOAL_WINDOW)
    print("layout2 geometric:", rg.resumen)
    if is_ref:
        assert rg.resumen["marcador_final"]["blue"] == 3, "gol geométrico: azul debería ser 3"

    print("PNG de muestra:", r2.sample_png)
    print("OK")


if __name__ == "__main__":
    main()
