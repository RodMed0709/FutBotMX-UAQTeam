"""Smoke de ``event_goal_zone`` (fase_5, Capa A · T2). Corre en LOCAL sin GPU.

Reusa el harness de T1 para resolver el JSON (acepta ruta o ``--video`` para generar el
clip en el pod).

    python testing/test_event_goal_zone.py [ruta/al/tracks.json]
    python testing/test_event_goal_zone.py --video data/raw/.../IMG_9933.MOV
"""

import json

from test_event_possession import resolve_tracks  # harness reusable de T1 (mismo dir)

from src.core.event_goals import compute_goal_zone_events, write_goal_events_json
from src.core.events_core import FrameObject, load_frame_objects
from src.core.events_schema import events_paths

_COLORS = {"yellow": "#d4b106", "blue": "#1f77b4"}


def _plot_timeline(result, png_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zonas = result.resumen["zonas_presentes"] or ["(ninguna)"]
    y_of = {z: i for i, z in enumerate(zonas)}
    fig, ax = plt.subplots(figsize=(12, max(1.5, len(zonas) * 0.7)))
    for e in result.eventos:
        ax.broken_barh(
            [(e.frame_inicio, max(1, e.dur_frames))],
            (y_of.get(e.zona, 0) - 0.3, 0.6),
            facecolors=_COLORS.get(e.zona, "gray"),
        )
    ax.set_yticks(range(len(zonas)))
    ax.set_yticklabels([f"zona {z}" for z in zonas])
    ax.set_xlabel("frame")
    ax.set_title("Balón en zona de gol (candidatos a gol)")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("timeline:", png_path)


def _edge_cases() -> None:
    zone = (100.0, 100.0, 50.0, 50.0)  # bbox de zona: x 100..150, y 100..150

    def frames(ball_centroid_xy):
        bx, by = ball_centroid_xy
        return {
            f: [
                FrameObject(1, "yellow_zone", zone, (125, 125), 0.9),
                FrameObject(9, "orange_ball", (bx, by, 4, 4), (bx + 2, by + 2), 0.9),
            ]
            for f in range(10)
        }

    # balón nunca en la zona -> 0 eventos
    assert compute_goal_zone_events(frames((10, 10)), min_frames=2).resumen["total_eventos"] == 0
    # balón dentro sostenido -> 1 evento; zona azul ausente no rompe
    r = compute_goal_zone_events(frames((123, 123)), min_frames=2, exit_frames=2)
    assert r.resumen["total_eventos"] == 1, r.resumen
    assert r.resumen["zonas_presentes"] == ["yellow"]
    print("casos borde OK")


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    meta = json.loads(tracks.read_text(encoding="utf-8"))
    fps = meta.get("fps")
    by_frame = load_frame_objects(tracks)
    print(f"video: {meta.get('video')} | json: {tracks}")
    print(f"frames={len(by_frame)} fps={fps}")

    result = compute_goal_zone_events(by_frame, fps=fps)
    print("resumen:\n" + json.dumps(result.resumen, indent=2, ensure_ascii=False))
    for e in result.eventos:
        print(f"  evento {e.zona}: frames {e.frame_inicio}-{e.frame_fin} ({e.dur_s}s)")

    _edge_cases()

    stem = tracks.stem
    _plot_timeline(result, events_paths(stem, "goal_zone", "png"))
    out_json = write_goal_events_json(
        result, events_paths(stem, "goal_zone", "json")
    )
    print("escrito:", out_json)
    print("OK")


if __name__ == "__main__":
    main()
