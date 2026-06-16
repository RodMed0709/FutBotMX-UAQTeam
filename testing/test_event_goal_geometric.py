"""Harness del gol geométrico (fase_5 · Capa B). Corre en CPU local (sin GPU).

Detecta goles en cm (balón cruzando la línea de gol dentro de la boca) sobre la salida métrica
de T3, valida invariantes + casos borde + lo compara con T2, y dibuja una viz (timeline +
posición de entrada sobre la cancha).

    python testing/test_event_goal_geometric.py [ruta/al/tracks.json]
"""

import json
import sys
from pathlib import Path

from src.core import field_template as ft
from src.core.event_goal_geometric import (
    GeometricGoalResult,
    _in_goal,
    compute_geometric_goals,
    write_geometric_goals_json,
)
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT
    / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _t2_events(tracks: Path) -> dict:
    """Eventos de T2 (Capa A, píxeles) para comparar."""
    from src.core.event_goals import compute_goal_zone_events
    from src.core.events_core import load_frame_objects

    by_frame = load_frame_objects(tracks)
    r = compute_goal_zone_events(by_frame)
    return r.resumen.get("eventos_por_zona", {})


def _plot(result: GeometricGoalResult, png_path: Path) -> None:
    import cv2

    canvas, to_px = ft.render_field(scale=2.6, margin_cm=10.0)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for e in result.eventos:
        if e.xy_cm is not None:
            px = to_px(e.xy_cm)
            cv2.circle(canvas, px, 7, (0, 0, 255), -1)
            cv2.putText(canvas, f"{e.zona} f{e.frame_inicio}", (px[0] + 8, px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), canvas)
    print("viz:", png_path)


def _edge_cases() -> None:
    """Geometría de la región: dentro de la boca azul vs fuera."""
    m = 8.0
    # balón dentro de la portería azul (x=237, y=91 centro) -> True azul, False amarilla
    assert _in_goal((237.0, 91.0), "blue", m) is True
    assert _in_goal((237.0, 91.0), "yellow", m) is False
    # balón en x azul pero fuera de la boca (y=20) -> False
    assert _in_goal((237.0, 20.0), "blue", m) is False
    # balón en el centro del campo -> False en ambas
    assert _in_goal((121.0, 91.0), "blue", m) is False
    assert _in_goal((121.0, 91.0), "yellow", m) is False
    # gol amarillo (x=6, y en boca)
    assert _in_goal((6.0, 91.0), "yellow", m) is True
    print("casos borde OK")


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    result = compute_geometric_goals(tracks)
    print("resumen:\n" + json.dumps(result.resumen, indent=2, ensure_ascii=False))
    for e in result.eventos:
        xy = f"({e.xy_cm[0]:.0f},{e.xy_cm[1]:.0f})cm" if e.xy_cm else "?"
        print(f"  gol {e.zona}: frames {e.frame_inicio}-{e.frame_fin} "
              f"({e.dur_s}s) entrada {xy}")

    # --- invariantes ---
    for e in result.eventos:
        assert e.frame_fin >= e.frame_inicio
        assert e.zona in ("yellow", "blue")
    assert result.resumen["total_eventos"] == sum(
        result.resumen["eventos_por_zona"].values()
    )
    print("invariantes OK")

    # --- comparación con T2 ---
    t2 = _t2_events(tracks)
    print(f"comparación: T2 (píxeles) eventos_por_zona={t2} | "
          f"geométrico (cm) eventos_por_zona={result.resumen['eventos_por_zona']}")

    _edge_cases()

    stem = tracks.stem
    _plot(result, PROJECT_ROOT / "outputs" / f"goal_geometric_{stem}.png")
    out = write_geometric_goals_json(
        result, PROJECT_ROOT / "outputs" / f"goal_geometric_{stem}.json"
    )
    print("escrito:", out)
    print("OK")


if __name__ == "__main__":
    main()
