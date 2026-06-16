"""Gol geométrico (fase_5 · Capa B) — refinamiento en cm de T2.

Mientras T2 (`event_goal_zone`, Capa A) marca "candidato a gol" cuando el balón entra al
**bbox** de una zona en píxeles, aquí se usan las **posiciones en cm** de T3
(`metric_positions`) y la **línea de gol real** + la **boca** sobre la cancha canónica
(`field_template`). Reusa el **mismo motor de estados de T2** (`_events_from_series`):
solo cambia cómo se construye el booleano por frame (línea en cm vs bbox en píxeles).

Solo aplica a **cámara superior** (donde T3 tiene posiciones fiables). Corre en **CPU local**.
Las cifras son indicativas (limitadas por la detección del balón y la homografía).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.core import field_template as ft
from src.core.event_goals import _events_from_series
from src.core.events_core import BALL_CLASSES
from src.core.metric_positions import MetricResult, compute_metric_positions

DEFAULT_MARGIN_CM = 8.0
DEFAULT_MIN_FRAMES = 3
DEFAULT_EXIT_FRAMES = 3
DEFAULT_COOLDOWN = 15


@dataclass
class GoalEventGeo:
    zona: str  # "yellow" | "blue"
    frame_inicio: int
    frame_fin: int
    dur_frames: int
    dur_s: float | None
    xy_cm: tuple[float, float] | None  # posición del balón al inicio del evento


@dataclass
class GeometricGoalResult:
    eventos: list[GoalEventGeo]
    resumen: dict


def _in_goal(xy: tuple[float, float], zona: str, margin: float) -> bool:
    """¿El balón (cm) cruzó la línea de gol dentro de la boca, para ``zona``?"""
    x, y = xy
    in_mouth = (ft._GOAL_TOP_Y_CM - margin) <= y <= (ft._GOAL_BOTTOM_Y_CM + margin)
    if not in_mouth:
        return False
    if zona == "yellow":
        return x <= ft.GOAL_LINE_X_LEFT_CM + margin
    return x >= ft.GOAL_LINE_X_RIGHT_CM - margin  # blue


def _ball_by_frame(result: MetricResult) -> dict[int, list[tuple[float, float]]]:
    """Posiciones del balón en cm por frame (puede haber varias por ID-switch)."""
    out: dict[int, list[tuple[float, float]]] = {}
    for p in result.posiciones:
        if p.cls in BALL_CLASSES and p.xy_cm is not None:
            out.setdefault(p.frame_index, []).append(p.xy_cm)
    return out


def _series(ball_by_frame: dict[int, list], frames: list[int], zona: str,
            margin: float) -> list[tuple[int, bool]]:
    """Serie ``[(frame, dentro)]``: alguna muestra del balón en la región de ``zona``."""
    return [
        (f, any(_in_goal(xy, zona, margin) for xy in ball_by_frame.get(f, [])))
        for f in frames
    ]


def _entry_xy(ball_by_frame: dict[int, list], zona: str, frame: int,
              margin: float) -> tuple[float, float] | None:
    """Posición (cm) de la muestra del balón en la región, en ``frame`` (o la 1ª si ninguna)."""
    samples = ball_by_frame.get(frame, [])
    for xy in samples:
        if _in_goal(xy, zona, margin):
            return xy
    return samples[0] if samples else None


def compute_geometric_goals(
    source: str | Path | MetricResult,
    *,
    margin_cm: float = DEFAULT_MARGIN_CM,
    min_frames: int = DEFAULT_MIN_FRAMES,
    exit_frames: int = DEFAULT_EXIT_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN,
    fps: float | None = None,
) -> GeometricGoalResult:
    """Detecta goles geométricos (balón cruzando la línea de gol en cm) por portería.

    ``source`` = ruta a tracks_json (llama a T3) o un ``MetricResult`` ya calculado.
    """
    result = source if isinstance(source, MetricResult) else compute_metric_positions(Path(source))
    fps = fps or result.resumen.get("fps")

    ball_by_frame = _ball_by_frame(result)
    eventos: list[GoalEventGeo] = []
    if ball_by_frame:
        lo, hi = min(ball_by_frame), max(ball_by_frame)
        frames = list(range(lo, hi + 1))  # timeline contiguo (huecos = fuera)
        for zona in ("yellow", "blue"):
            serie = _series(ball_by_frame, frames, zona, margin_cm)
            for ev in _events_from_series(
                serie, zona, min_frames=min_frames, exit_frames=exit_frames,
                cooldown=cooldown_frames, fps=fps,
            ):
                eventos.append(GoalEventGeo(
                    zona=ev.zona, frame_inicio=ev.frame_inicio, frame_fin=ev.frame_fin,
                    dur_frames=ev.dur_frames, dur_s=ev.dur_s,
                    xy_cm=_entry_xy(ball_by_frame, zona, ev.frame_inicio, margin_cm),
                ))
    eventos.sort(key=lambda e: e.frame_inicio)

    por_zona: dict[str, int] = {}
    for e in eventos:
        por_zona[e.zona] = por_zona.get(e.zona, 0) + 1
    resumen = {
        "fps": fps,
        "zonas_evaluadas": ["yellow", "blue"],
        "eventos_por_zona": por_zona,
        "total_eventos": len(eventos),
        "params": {
            "margin_cm": margin_cm, "min_frames": min_frames,
            "exit_frames": exit_frames, "cooldown_frames": cooldown_frames,
        },
        "nota": "gol geométrico en cm (cámara superior); cifras indicativas",
    }
    return GeometricGoalResult(eventos=eventos, resumen=resumen)


def write_geometric_goals_json(result: GeometricGoalResult, path: str | Path) -> Path:
    """Escribe el resultado a JSON (resumen + eventos)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "eventos": [
            {
                "zona": e.zona,
                "frame_inicio": e.frame_inicio,
                "frame_fin": e.frame_fin,
                "dur_frames": e.dur_frames,
                "dur_s": e.dur_s,
                "xy_cm": list(e.xy_cm) if e.xy_cm is not None else None,
            }
            for e in result.eventos
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
