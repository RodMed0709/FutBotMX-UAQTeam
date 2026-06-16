"""T7 (fase_5) — ensamble del video demo (convocatoria 3.5.3).

**Compone** un video demo a partir de componentes ya renderizados (no reimplementa overlays):
- **Original**: el clip crudo de cámara superior.
- **Segmentación**: ``<stem>_seg.mp4`` (máscaras por clase; generado en el pod).
- **Tracking**: ``<stem>_obj_id.mp4`` (caja + ``nombre #id`` + estela; T7 lo genera en local con
  ``track_overlay.render_obj_id_overlay`` — dibujo puro desde el JSON, sin modelo).
- **Minimap**: ``<stem>_minimap.mp4`` (trails en cm; generado en el pod).

Sobre el mosaico añade un **panel de métricas** (posesión T1, velocidad/distancia T4, zona T6) y
un **banner de gol** (gol geométrico) cuando corresponde. Corre en **CPU local** (todo viene del
JSON; no re-infiere). Salida: un mp4 ≤ 2 min.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core import field_template as ft
from src.core.event_goal_geometric import compute_geometric_goals
from src.core.events import compute_possession
from src.core.events_core import BALL_CLASSES, load_frame_objects
from src.core.frame_extraction import get_video_fps
from src.core.metric_field_zones import compute_field_zones
from src.core.metric_heatmap import render_heatmap
from src.core.metric_kinematics import compute_kinematics
from src.core.metric_positions import MetricResult, compute_metric_positions

HEATMAP_BIN_CM = 5.0

CELL_H = 540  # alto de cada panel del mosaico (px)
METRICS_H = 120  # alto de la barra de métricas (px)
_WHITE = (255, 255, 255)
_RED = (0, 0, 255)


@dataclass
class _Panel:
    label: str
    path: Path


def _resolve_components(tracks_json: Path, components: dict | None) -> list[_Panel]:
    """Ubica los componentes junto al JSON; genera ``_obj_id.mp4`` si falta. Degrada con aviso."""
    stem = tracks_json.stem
    parent = tracks_json.parent
    clip = Path(components["clip"]) if components and "clip" in components else parent / f"{stem}.mp4"

    panels: list[_Panel] = []
    if clip.exists():
        panels.append(_Panel("Original", clip))
    else:
        print(f"AVISO: no está el clip original ({clip}); se omite el panel Original.")

    seg = parent / f"{stem}_seg.mp4"
    if seg.exists():
        panels.append(_Panel("Segmentacion", seg))
    else:
        print(f"AVISO: no está {seg.name} (se genera en el pod); se omite Segmentación.")

    obj_id = parent / f"{stem}_obj_id.mp4"
    if not obj_id.exists() and clip.exists():
        from src.core.track_overlay import render_obj_id_overlay

        print("[T7] generando overlay de tracking (_obj_id.mp4) en local…")
        obj_id = render_obj_id_overlay(tracks_json, video_path=clip, output_path=obj_id)
    if obj_id.exists():
        panels.append(_Panel("Tracking", obj_id))

    mini = parent / f"{stem}_minimap.mp4"
    if mini.exists():
        panels.append(_Panel("Minimap", mini))
    else:
        print(f"AVISO: no está {mini.name} (se genera en el pod); se omite Minimap.")

    return panels


def _collect_metrics(tracks_json: Path, metric: MetricResult, fps: float | None) -> dict:
    """Agregados del clip (T1/T4/T6 + gol geométrico) + intervalos de gol para el banner."""
    fps = fps or metric.resumen.get("fps")
    by_frame = load_frame_objects(tracks_json)
    pos = compute_possession(by_frame, fps=fps)
    kin = compute_kinematics(metric, fps=fps)
    zones = compute_field_zones(tracks_json, metric=metric, fps=fps)
    goals = compute_geometric_goals(metric, fps=fps)

    top_owner = max(pos.resumen["posesion_por_obj"].items(),
                    key=lambda kv: kv[1]["frames"], default=(None, {"segundos": 0}))
    top_dist = kin.por_obj[0] if kin.por_obj else None
    ball_vmax = max((o.v_max_cms for o in kin.por_obj if o.cls in ("orange_ball", "ball")),
                    default=0.0)
    azul = zones.por_esquema["mitades"]["presencia"]["ball"].get("azul", 0.0)

    lines = [
        f"Posesion: #{top_owner[0]} {top_owner[1]['segundos']}s | "
        f"controlado {pos.resumen['pct_controlado']}%",
        (f"Dist max: #{top_dist.obj_id} {top_dist.dist_cm:.0f}cm | "
         f"v_max balon {ball_vmax:.0f}cm/s") if top_dist else "Dist: -",
        f"Balon mitad azul: {azul:.0f}% | goles: {goals.resumen['total_eventos']}",
    ]
    goal_spans = [(e.frame_inicio, e.frame_fin, e.zona) for e in goals.eventos]
    return {"lines": lines, "goal_spans": goal_spans, "fps": fps}


def _live_heatmap_state(metric: MetricResult, bin_cm: float) -> tuple[np.ndarray, dict[int, list]]:
    """Rejilla vacía + posiciones (cm) de robots y balón por frame, para el heatmap en vivo."""
    cols = int(np.ceil(ft.LENGTH_CM / bin_cm))
    rows = int(np.ceil(ft.WIDTH_CM / bin_cm))
    grid = np.zeros((rows, cols), dtype=float)
    pos_by_frame: dict[int, list] = {}
    for p in metric.posiciones:
        if p.xy_cm is None or (p.cls not in BALL_CLASSES and p.cls != "robot"):
            continue
        pos_by_frame.setdefault(p.frame_index, []).append(p.xy_cm)
    return grid, pos_by_frame


def _accumulate(grid: np.ndarray, pts: list, bin_cm: float) -> None:
    """Suma las posiciones (cm) de un frame a la rejilla (clip a la cancha)."""
    for x, y in pts:
        cx = min(max(x, 0.0), ft.LENGTH_CM - 1e-6)
        cy = min(max(y, 0.0), ft.WIDTH_CM - 1e-6)
        grid[int(cy / bin_cm), int(cx / bin_cm)] += 1.0


def _fit(frame: np.ndarray, h: int) -> np.ndarray:
    """Redimensiona un frame a alto ``h`` preservando aspecto."""
    import cv2

    sh, sw = frame.shape[:2]
    w = max(1, int(round(h * sw / sh)))
    return cv2.resize(frame, (w, h))


def _label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1, cv2.LINE_AA)
    return img


def _metrics_bar(width: int, lines: list[str], goal: str | None) -> np.ndarray:
    import cv2

    bar = np.full((METRICS_H, width, 3), (30, 30, 30), dtype=np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(bar, ln, (10, 28 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1,
                    cv2.LINE_AA)
    if goal:
        cv2.rectangle(bar, (0, 0), (width, METRICS_H), _RED, 6)
        cv2.putText(bar, f"GOL - porteria {goal}", (width // 2 - 120, METRICS_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _RED, 2, cv2.LINE_AA)
    return bar


def compose_demo(
    tracks_json: str | Path,
    *,
    output_path: str | Path | None = None,
    max_seconds: float = 120.0,
    components: dict | None = None,
) -> Path:
    """Ensambla el video demo (mosaico de componentes + panel de métricas + banner de gol)."""
    import cv2

    from src.core.video_writer import open_video_writer

    tracks_json = Path(tracks_json)
    panels = _resolve_components(tracks_json, components)
    if not panels:
        raise FileNotFoundError("no hay ningún componente para componer el demo")
    metric = compute_metric_positions(tracks_json)
    metrics = _collect_metrics(tracks_json, metric, fps=None)
    fps = metrics["fps"] or get_video_fps(panels[0].path)
    max_frames = int(round(max_seconds * fps))
    grid, pos_by_frame = _live_heatmap_state(metric, HEATMAP_BIN_CM)

    caps = [cv2.VideoCapture(str(p.path)) for p in panels]
    out = Path(output_path) if output_path else tracks_json.parent / f"{tracks_json.stem}_demo.mp4"
    n = 0
    try:
        with open_video_writer(out, fps=fps) as append:
            while n < max_frames:
                cells = []
                ok_all = True
                for cap, panel in zip(caps, panels):
                    ok, frame = cap.read()
                    if not ok:
                        ok_all = False
                        break
                    cells.append(_label(_fit(frame, CELL_H), panel.label))
                if not ok_all:
                    break
                # Heatmap en vivo (acumula hasta el frame n) — misma cancha que el minimap.
                _accumulate(grid, pos_by_frame.get(n, []), HEATMAP_BIN_CM)
                hm = render_heatmap(grid, HEATMAP_BIN_CM)
                cells.append(_label(_fit(hm, CELL_H), "Heatmap (vivo)"))
                row = cv2.hconcat(cells)
                goal = next((z for s, e, z in metrics["goal_spans"] if s <= n <= e), None)
                bar = _metrics_bar(row.shape[1], metrics["lines"], goal)
                canvas = cv2.vconcat([row, bar])
                append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                n += 1
    finally:
        for cap in caps:
            cap.release()
    print(f"[T7] demo escrito: {out} ({n} frames, {n / fps:.1f}s, paneles={[p.label for p in panels]})")
    return out
