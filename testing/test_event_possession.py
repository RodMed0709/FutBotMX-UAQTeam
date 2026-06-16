"""Smoke manual de ``event_possession`` (fase_5, Capa A). Corre en LOCAL sin GPU.

Sobre un JSON de tracking ya existente: calcula la posesión, valida invariantes, ejerce
casos borde y genera una **línea de tiempo** de posesión (matplotlib → png).

    python testing/test_event_possession.py
"""

import json

from src.core.events import (
    FrameObject,
    compute_possession,
    load_frame_objects,
    write_possession_json,
)
from src.utils import PROJECT_ROOT

TRACKS = PROJECT_ROOT / "outputs/inference/fase3_eventos/IMG_9780/IMG_9780.json"


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


def main() -> None:
    if not TRACKS.exists():
        raise FileNotFoundError(f"No hay JSON de tracking de prueba: {TRACKS}")
    fps = json.loads(TRACKS.read_text(encoding="utf-8")).get("fps")
    by_frame = load_frame_objects(TRACKS)
    print(f"frames={len(by_frame)} fps={fps}")

    result = compute_possession(by_frame, fps=fps)
    print("resumen:\n" + json.dumps(result.resumen, indent=2, ensure_ascii=False))

    # --- invariantes ---
    r = result.resumen
    total = r["pct_controlado"] + r["pct_libre"] + r["pct_no_visible"]
    assert abs(total - 100.0) <= 0.3, f"porcentajes no suman ~100: {total}"
    n_owned = sum(1 for v in result.por_frame.values() if v is not None)
    assert sum(o["frames"] for o in r["posesion_por_obj"].values()) == n_owned
    print("invariantes OK")

    # --- casos borde ---
    assert compute_possession({}, fps=fps).resumen["n_frames"] == 0  # vacío
    # frame con balón pero sin robots -> None; frame sin balón -> None
    sin_robot = {0: [FrameObject(9, "orange_ball", (10, 10, 4, 4), (12, 12), 0.9)]}
    assert compute_possession(sin_robot, min_frames=1).por_frame[0] is None
    sin_balon = {0: [FrameObject(0, "robot", (0, 0, 50, 50), (25, 25), 0.9)]}
    assert compute_possession(sin_balon, min_frames=1).por_frame[0] is None
    # robot pegado al balón (dist 0 < gate) -> lo posee
    pegado = {
        0: [
            FrameObject(7, "robot", (0, 0, 50, 50), (25, 25), 0.9),
            FrameObject(9, "orange_ball", (24, 24, 4, 4), (25, 25), 0.9),
        ]
    }
    assert compute_possession(pegado, min_frames=1).por_frame[0] == 7
    print("casos borde OK")

    # --- visualización de validación ---
    _plot_timeline(result, PROJECT_ROOT / "outputs" / "event_possession_timeline.png", fps)
    out_json = write_possession_json(result, PROJECT_ROOT / "outputs" / "event_possession.json")
    print("escrito:", out_json)
    print("OK")


if __name__ == "__main__":
    main()
