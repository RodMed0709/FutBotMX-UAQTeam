"""Harness de T3 `metric_positions` (fase_5 · Capa B). Corre en CPU local (sin GPU).

Proyecta robots/balón a cm desde un JSON de tracking extendido (``include_masks=True``) y
valida: invariantes (posiciones dentro de la cancha salvo margen, conteos coherentes), casos
borde, y una **viz** de trayectorias en cm sobre la cancha canónica.

    python testing/test_metric_positions.py [ruta/al/tracks.json]

Por defecto usa el clip del gol de cámara superior (``IMG_9933_5m30``), que tiene homografía
fiable y jugada real. El clip (.mp4) debe estar junto al JSON.
"""

import json
import sys
from pathlib import Path

from src.core import field_template as ft
from src.core.metric_positions import (
    compute_metric_positions,
    write_metric_positions_json,
)
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT
    / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)
# Margen tolerado fuera de la cancha (ruido de H / objetos sobre la pared).
MARGIN_CM = 40.0


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _color_bgr(cls: str) -> tuple[int, int, int]:
    if cls in ("orange_ball", "ball"):
        return (0, 140, 255)  # naranja (BGR)
    return (60, 60, 60)  # robots: gris oscuro


def _plot_trajectories(result, png_path: Path) -> None:
    import cv2

    canvas, to_px = ft.render_field(scale=2.6, margin_cm=10.0)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    # agrupar por obj_id, en orden de frame
    by_obj: dict[int, list] = {}
    for p in result.posiciones:
        if p.xy_cm is not None:
            by_obj.setdefault(p.obj_id, []).append((p.frame_index, p.cls, p.xy_cm))
    for obj_id, pts in by_obj.items():
        pts.sort()
        cls = pts[0][1]
        poly = [to_px(xy) for _, _, xy in pts]
        for a, b in zip(poly, poly[1:]):
            cv2.line(canvas, a, b, _color_bgr(cls), 1, cv2.LINE_AA)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), canvas)
    print("trayectorias:", png_path)


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    result = compute_metric_positions(tracks)
    r = result.resumen
    print("resumen:\n" + json.dumps(r, indent=2, ensure_ascii=False))

    # --- invariantes ---
    assert r["n_estimated"] + r["n_propagated"] + r["n_rejected"] == r["n_frames"], \
        "los estados de H no suman n_frames"
    assert r["n_con_cm"] <= r["n_posiciones"]
    assert r["pct_H_valida"] > 0.0, "ninguna H válida: la homografía no ancló"

    # posiciones dentro de la cancha (salvo margen) en la mayoría
    dentro = fuera = 0
    for p in result.posiciones:
        if p.xy_cm is None:
            continue
        x, y = p.xy_cm
        if (-MARGIN_CM <= x <= ft.LENGTH_CM + MARGIN_CM
                and -MARGIN_CM <= y <= ft.WIDTH_CM + MARGIN_CM):
            dentro += 1
        else:
            fuera += 1
    pct_dentro = 100.0 * dentro / max(1, dentro + fuera)
    print(f"posiciones dentro de la cancha (±{MARGIN_CM:.0f}cm): {pct_dentro:.1f}% "
          f"({dentro}/{dentro + fuera})")
    assert pct_dentro >= 80.0, f"demasiadas posiciones fuera de la cancha: {pct_dentro:.1f}%"
    print("invariantes OK")

    # --- casos borde ---
    _edge_cases()

    stem = tracks.stem
    _plot_trajectories(result, PROJECT_ROOT / "outputs" / f"metric_positions_{stem}.png")
    out_json = write_metric_positions_json(
        result, PROJECT_ROOT / "outputs" / f"metric_positions_{stem}.json"
    )
    print("escrito:", out_json)
    print("OK")


def _edge_cases() -> None:
    """JSON sin include_masks -> error claro; sin frames -> anchors vacío (no rompe)."""
    from src.core.metric_positions import _load_field_anchors

    try:
        _load_field_anchors({"include_masks": False, "frames": []})
    except ValueError:
        pass
    else:
        raise AssertionError("debió fallar sin include_masks")

    assert _load_field_anchors({"include_masks": True, "frames": []}) == {}
    print("casos borde OK")


if __name__ == "__main__":
    main()
