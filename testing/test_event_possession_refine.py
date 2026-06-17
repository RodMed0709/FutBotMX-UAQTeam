"""Harness de posesión vs control (ronda de entregable de eventos). Corre en CPU local (sin GPU).

Separa **posesión** (robot más cercano al balón, histéresis adaptativa) de **control** (balón
en movimiento con el poseedor). Valida invariantes (`control ⊆ posesión`, cobertura de frames),
un caso de coherencia (balón quieto = posesión sin control) y dibuja una línea de tiempo de dos
filas (posesión / control), color por `obj_id`.

    python testing/test_event_possession_refine.py [ruta/al/tracks.json]
"""

import json
import sys
from pathlib import Path

from src.core.event_possession_refine import (
    PossessionRefineResult,
    compute_possession_refine,
    load_frame_objects,
    write_possession_refine_json,
)
from src.core.events_core import FrameObject
from src.core.events_schema import events_paths
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _video_fps(tracks: Path) -> float | None:
    return json.loads(tracks.read_text(encoding="utf-8")).get("fps")


def _print(result: PossessionRefineResult) -> None:
    s = result.resumen
    print(f"posesión total={s['pct_posesion_total']}%  control total={s['pct_control_total']}%  "
          f"libre={s['pct_libre']}%  no_visible={s['pct_no_visible']}%")
    print(f"cambios: posesión={s['cambios_de_posesion']}  control={s['cambios_de_control']}")
    print("por robot (pct posesión -> control):")
    for oid, p in s["posesion_por_obj"].items():
        c = s["control_por_obj"].get(oid, {}).get("pct", 0.0)
        print(f"  #{oid}: {p['pct']}% -> {c}%")


def _invariants(result: PossessionRefineResult) -> None:
    pos, ctrl = result.posesion_por_frame, result.control_por_frame
    # control ⊆ posesión por frame
    for f in pos:
        assert ctrl[f] is None or ctrl[f] == pos[f], f"control fuera de posesión en f{f}"
    # cobertura de frames: control + posesión-sin-control + libre + no_visible = total
    s = result.resumen
    n = s["n_frames"]
    n_ctrl = sum(v["frames"] for v in s["control_por_obj"].values())
    n_pos = sum(v["frames"] for v in s["posesion_por_obj"].values())
    n_libre = round(s["pct_libre"] / 100 * n)
    n_novis = round(s["pct_no_visible"] / 100 * n)
    assert n_ctrl <= n_pos, "control total > posesión total"
    # posesión + libre + no_visible debe cubrir todo (con tolerancia de redondeo)
    assert abs((n_pos + n_libre + n_novis) - n) <= 2, "la cobertura de frames no cierra"
    # por robot: control <= posesión
    for oid, p in s["posesion_por_obj"].items():
        c = s["control_por_obj"].get(oid, {}).get("pct", 0.0)
        assert c <= p["pct"] + 1e-6, f"#{oid}: control {c}% > posesión {p['pct']}%"
    print("invariantes OK (control ⊆ posesión, cobertura, control ≤ posesión por robot)")


def _coherence(result: PossessionRefineResult) -> None:
    """Debe existir un robot con posesión > control (balón quieto junto a él = no engañoso)."""
    s = result.resumen
    gap = [
        (oid, p["pct"], s["control_por_obj"].get(oid, {}).get("pct", 0.0))
        for oid, p in s["posesion_por_obj"].items()
    ]
    assert any(p > c for _, p, c in gap), "ningún robot muestra posesión > control"
    oid, p, c = max(gap, key=lambda t: t[1] - t[2])
    print(f"coherencia OK: #{oid} posee {p}% pero solo controla {c}% (posesión pasiva)")


def _synthetic_static_ball() -> None:
    """Balón quieto junto a un robot ⇒ posesión sin control."""
    objs = lambda: [  # noqa: E731
        FrameObject(1, "robot", (100, 100, 80, 80), (140, 140), 0.9),
        FrameObject(99, "orange_ball", (150, 150, 20, 20), (160, 160), 0.9),  # quieto
    ]
    by_frame = {f: objs() for f in range(20)}
    r = compute_possession_refine(by_frame, fps=30)
    assert r.resumen["pct_posesion_total"] > 0, "debería haber posesión (robot junto al balón)"
    assert r.resumen["pct_control_total"] == 0.0, "balón quieto no debería dar control"
    print("sintético OK (balón quieto = posesión sin control)")


def _plot(result: PossessionRefineResult, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    frames = sorted(result.posesion_por_frame)
    oids = sorted({v for v in result.posesion_por_frame.values() if v is not None})
    idx = {oid: i + 1 for i, oid in enumerate(oids)}  # 0 = sin poseedor (blanco)
    cmap_base = plt.get_cmap("tab10")
    colors = ["#ffffff"] + [cmap_base(i % 10) for i in range(len(oids))]
    cmap = ListedColormap(colors)

    def row(serie):
        return [idx.get(serie[f], 0) for f in frames]

    arr = [row(result.posesion_por_frame), row(result.control_por_frame)]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=len(oids),
              extent=[frames[0], frames[-1], 1.5, -0.5], interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["posesión", "control"])
    ax.set_xlabel("frame")
    ax.set_title("Posesión vs control — línea de tiempo")
    ax.legend(handles=[Patch(color=colors[idx[o]], label=f"#{o}") for o in oids],
              loc="upper right", ncol=len(oids), fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print("viz:", png_path)


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    by_frame = load_frame_objects(tracks)
    result = compute_possession_refine(by_frame, fps=_video_fps(tracks))
    _print(result)
    _invariants(result)
    _coherence(result)
    _synthetic_static_ball()

    stem = tracks.stem
    _plot(result, events_paths(stem, "possession_refine", "png"))
    out = write_possession_refine_json(result, events_paths(stem, "possession_refine", "json"))
    print("escrito:", out)
    print("OK")


if __name__ == "__main__":
    main()
