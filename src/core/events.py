"""Posesión por cercanía sobre el JSON de tracking (fase_5, Capa A · tarea event_possession).

Capa **relacional (en píxeles), UNIVERSAL**: funciona sobre el JSON de **cualquier** video
sin homografía ni GPU. ``compute_possession`` asigna el balón al robot más cercano dentro de
un gate (relativo al tamaño del robot), con histéresis, y resume métricas temporales.

La base compartida (``FrameObject``/``load_frame_objects``/``ball_centroid``) vive en
``events_core`` y se re-exporta aquí por compatibilidad.

``numpy`` solo para distancias; sin GPU, sin homografía.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.events_core import (
    BALL_CLASSES,
    ROBOT_CLASS,
    FrameObject,
    ball_centroid,
    load_frame_objects,
)

__all__ = [
    "BALL_CLASSES",
    "ROBOT_CLASS",
    "FrameObject",
    "load_frame_objects",
    "ball_centroid",
    "PossessionResult",
    "compute_possession",
    "write_possession_json",
]

# Defaults configurables por parámetro.
DEFAULT_GATE_K = 1.5     # gate de cercanía = gate_k * diagonal del bbox del robot
DEFAULT_MIN_FRAMES = 3   # frames consecutivos para confirmar un cambio de posesión


@dataclass
class PossessionResult:
    """Resultado de ``compute_possession``."""

    por_frame: dict[int, int | None]  # frame_index -> obj_id poseedor (None = libre/no visible)
    resumen: dict


def _bbox_diagonal(bbox: tuple[float, float, float, float]) -> float:
    """Diagonal del bbox ``[x,y,w,h]`` (escala del robot para el gate)."""
    return float(np.hypot(bbox[2], bbox[3]))


def _nearest_robot(
    ball_xy: tuple[float, float], robots: list[FrameObject]
) -> tuple[int, float, float] | None:
    """``(obj_id, distancia, diagonal_bbox)`` del robot más cercano, o ``None``."""
    best: tuple[int, float, float] | None = None
    for r in robots:
        d = float(np.hypot(r.centroid[0] - ball_xy[0], r.centroid[1] - ball_xy[1]))
        if best is None or d < best[1]:
            best = (r.obj_id, d, _bbox_diagonal(r.bbox))
    return best


def _raw_possession(by_frame: dict[int, list[FrameObject]], gate_k: float) -> dict[int, int | None]:
    """Poseedor **crudo** por frame (sin histéresis)."""
    raw: dict[int, int | None] = {}
    for idx, objs in by_frame.items():
        ball = ball_centroid(objs)
        if ball is None:
            raw[idx] = None
            continue
        robots = [o for o in objs if o.class_name == ROBOT_CLASS]
        near = _nearest_robot(ball, robots)
        if near is None:
            raw[idx] = None
            continue
        oid, dist, diag = near
        raw[idx] = oid if dist <= gate_k * diag else None
    return raw


def _apply_hysteresis(raw: dict[int, int | None], min_frames: int) -> dict[int, int | None]:
    """Confirma un cambio de poseedor solo tras ``min_frames`` consecutivos del nuevo valor.

    Suaviza el parpadeo frame-a-frame: la posesión persiste hasta que otro valor se
    sostiene ``min_frames`` frames (incluido un hueco breve del balón).
    """
    out: dict[int, int | None] = {}
    current: int | None = None
    pending: int | None = None
    streak = 0
    for f in sorted(raw):
        v = raw[f]
        if v == current:
            pending, streak = None, 0
        else:
            if v == pending:
                streak += 1
            else:
                pending, streak = v, 1
            if streak >= min_frames:
                current, pending, streak = pending, None, 0
        out[f] = current
    return out


def _summarize(
    por_frame: dict[int, int | None],
    ball_visible: dict[int, bool],
    fps: float | None,
) -> dict:
    """Resumen de métricas temporales a partir de la serie por-frame."""
    n_frames = len(por_frame)
    counts = Counter(v for v in por_frame.values() if v is not None)
    n_controlado = sum(counts.values())
    no_owner = [f for f in por_frame if por_frame[f] is None]
    n_no_visible = sum(1 for f in no_owner if not ball_visible.get(f, False))
    n_libre = len(no_owner) - n_no_visible

    serie = [por_frame[f] for f in sorted(por_frame)]
    cambios = sum(1 for a, b in zip(serie, serie[1:]) if a != b)

    def pct(n: int) -> float:
        return round(100.0 * n / n_frames, 1) if n_frames else 0.0

    def secs(n: int) -> float | None:
        return round(n / fps, 2) if fps else None

    return {
        "n_frames": n_frames,
        "posesion_por_obj": {
            str(oid): {"frames": n, "segundos": secs(n)} for oid, n in sorted(counts.items())
        },
        "cambios_de_posesion": cambios,
        "pct_controlado": pct(n_controlado),
        "pct_libre": pct(n_libre),
        "pct_no_visible": pct(n_no_visible),
        "fps": fps,
    }


def compute_possession(
    by_frame: dict[int, list[FrameObject]],
    *,
    gate_k: float = DEFAULT_GATE_K,
    min_frames: int = DEFAULT_MIN_FRAMES,
    fps: float | None = None,
) -> PossessionResult:
    """Posesión por cercanía: robot más cercano al balón dentro del gate, con histéresis.

    Args:
        by_frame: salida de ``load_frame_objects``.
        gate_k: gate de cercanía = ``gate_k * diagonal del bbox del robot``.
        min_frames: frames consecutivos para confirmar un cambio de posesión.
        fps: para convertir frames a segundos en el resumen (``None`` ⇒ solo frames).

    Returns:
        ``PossessionResult`` con ``por_frame`` (poseedor por frame) y ``resumen``.
    """
    raw = _raw_possession(by_frame, gate_k)
    por_frame = _apply_hysteresis(raw, min_frames)
    ball_visible = {idx: ball_centroid(objs) is not None for idx, objs in by_frame.items()}
    return PossessionResult(por_frame=por_frame, resumen=_summarize(por_frame, ball_visible, fps))


def write_possession_json(result: PossessionResult, path: str | Path) -> Path:
    """Persiste el resultado a un JSON (``resumen`` + ``por_frame``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "por_frame": {str(k): result.por_frame[k] for k in sorted(result.por_frame)},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
