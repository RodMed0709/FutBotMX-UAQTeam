"""Posesión vs control (ronda de entregable de eventos) — refina la posesión por cercanía.

Separa dos nociones que ``events.compute_possession`` mezclaba en una sola:

- **posesión**: qué robot está con el balón (el más cercano dentro del gate), con una
  histéresis **adaptativa** — se actualiza de inmediato ante un **cambio radical** (el balón
  sale disparado o cambia el robot más cercano con margen claro) y conserva la histéresis de
  ``min_frames`` para cambios ambiguos (parpadeo);
- **control**: qué robot **conduce** el balón — posesión **y** balón en movimiento dentro de
  una ventana. Un balón muerto junto a un robot es posesión **sin** control.

Así el resumen deja de ser engañoso: un robot parado junto a un balón quieto no figura como
dominante. Capa **A (en píxeles), universal**: corre en **CPU local** sobre el JSON de
tracking, sin GPU ni homografía. **Consume** la base de ``events``/``events_core`` (no la
reescribe): ``_raw_possession``, ``_nearest_robot``, ``_bbox_diagonal``, ``ball_centroid``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.core.events import _bbox_diagonal, _nearest_robot, _raw_possession
from src.core.events_core import ROBOT_CLASS, FrameObject, ball_centroid, load_frame_objects

DEFAULT_GATE_K = 1.5  # heredado de events (gate de cercanía = gate_k * diagonal del robot)
DEFAULT_MIN_FRAMES = 3  # histéresis para cambios ambiguos
DEFAULT_CONTROL_WINDOW = 5  # frames para medir movimiento del balón (control)
DEFAULT_MOVE_K = 0.15  # umbral de movimiento (·diagonal del balón / frame)
DEFAULT_JUMP_K = 0.5  # salto radical del balón (·diagonal del balón en 1 frame)
DEFAULT_CLEAR_FACTOR = 0.6  # cambio radical por cercanía clara


@dataclass
class PossessionRefineResult:
    posesion_por_frame: dict[int, int | None]  # obj_id poseedor (histéresis adaptativa)
    control_por_frame: dict[int, int | None]  # obj_id que controla (None si balón quieto)
    resumen: dict


# --- helpers -------------------------------------------------------------------

def _robot_by_id(objs: list[FrameObject], oid: int) -> FrameObject | None:
    for o in objs:
        if o.class_name == ROBOT_CLASS and o.obj_id == oid:
            return o
    return None


def _ball_diag(objs: list[FrameObject]) -> float | None:
    """Diagonal del bbox del balón (mayor score) ese frame — escala del movimiento."""
    from src.core.events_core import BALL_CLASSES

    balls = [o for o in objs if o.class_name in BALL_CLASSES]
    if not balls:
        return None
    return _bbox_diagonal(max(balls, key=lambda o: o.score).bbox)


def _nearest_by_frame(
    by_frame: dict[int, list[FrameObject]]
) -> dict[int, tuple[int, float, float] | None]:
    """``{frame: (obj_id, dist, diagonal)}`` del robot más cercano al balón (o ``None``)."""
    out: dict[int, tuple[int, float, float] | None] = {}
    for f, objs in by_frame.items():
        ball = ball_centroid(objs)
        if ball is None:
            out[f] = None
            continue
        robots = [o for o in objs if o.class_name == ROBOT_CLASS]
        out[f] = _nearest_robot(ball, robots)
    return out


def _count_changes(serie: list[int | None]) -> int:
    return sum(1 for a, b in zip(serie, serie[1:]) if a != b)


# --- posesión con histéresis adaptativa ----------------------------------------

def _is_radical(
    f: int,
    new_oid: int,
    current: int | None,
    ball: dict[int, tuple[float, float] | None],
    ball_diag: dict[int, float | None],
    nearest: dict[int, tuple[int, float, float] | None],
    by_frame: dict[int, list[FrameObject]],
    prev_ball: tuple[float, float] | None,
    *,
    jump_k: float,
    clear_factor: float,
) -> bool:
    """¿El cambio de poseedor a ``new_oid`` es **radical** (actualización inmediata)?"""
    n = nearest.get(f)
    if n is None:
        return False
    _, new_dist, _ = n
    bxy = ball.get(f)
    bdiag = ball_diag.get(f)
    # (a) salto grande del balón respecto al frame visible anterior.
    if bxy is not None and prev_ball is not None and bdiag is not None:
        if math.hypot(bxy[0] - prev_ball[0], bxy[1] - prev_ball[1]) > jump_k * bdiag:
            return True
    # (b) cercanía clara: el nuevo poseedor mucho más cerca que el actual.
    if current is not None and bxy is not None:
        cur = _robot_by_id(by_frame.get(f, []), current)
        if cur is not None:
            cur_dist = math.hypot(cur.centroid[0] - bxy[0], cur.centroid[1] - bxy[1])
            if new_dist < clear_factor * cur_dist:
                return True
    return False


def _adaptive_possession(
    raw: dict[int, int | None],
    ball: dict[int, tuple[float, float] | None],
    ball_diag: dict[int, float | None],
    nearest: dict[int, tuple[int, float, float] | None],
    by_frame: dict[int, list[FrameObject]],
    *,
    min_frames: int,
    jump_k: float,
    clear_factor: float,
) -> dict[int, int | None]:
    """Confirma la posesión: inmediata si el cambio es radical, si no racha ``min_frames``."""
    out: dict[int, int | None] = {}
    current: int | None = None
    pending: int | None = None
    streak = 0
    prev_ball: tuple[float, float] | None = None

    for f in sorted(raw):
        v = raw[f]
        if v == current:
            pending, streak = None, 0
        elif v is not None and _is_radical(
            f, v, current, ball, ball_diag, nearest, by_frame, prev_ball,
            jump_k=jump_k, clear_factor=clear_factor,
        ):
            current, pending, streak = v, None, 0
        else:
            if v == pending:
                streak += 1
            else:
                pending, streak = v, 1
            if streak >= min_frames:
                current, pending, streak = pending, None, 0
        out[f] = current
        if ball.get(f) is not None:
            prev_ball = ball[f]
    return out


# --- capa de control -----------------------------------------------------------

def _ball_moved(
    ball: dict[int, tuple[float, float] | None],
    frames: list[int],
    idx: int,
    window: int,
    thresh: float,
) -> bool:
    """¿El balón se mueve (paso medio ≥ ``thresh``) en la ventana ``[idx, idx+window)``?"""
    pts = [
        ball[frames[j]]
        for j in range(idx, min(idx + window, len(frames)))
        if ball.get(frames[j]) is not None
    ]
    if len(pts) < 2:
        return False
    total = sum(
        math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
        for k in range(len(pts) - 1)
    )
    return total / (len(pts) - 1) >= thresh


def _control_series(
    posesion: dict[int, int | None],
    ball: dict[int, tuple[float, float] | None],
    ball_diag: dict[int, float | None],
    *,
    control_window: int,
    move_k: float,
) -> dict[int, int | None]:
    """``control[f] = oid`` si hay poseedor **y** el balón se mueve en la ventana; si no None.

    El umbral de movimiento se escala con el **tamaño del balón** (``move_k * diagonal``), no
    con el del robot: el balón se desplaza pocos px/frame frente a robots grandes.
    """
    frames = sorted(posesion)
    control: dict[int, int | None] = {}
    for idx, f in enumerate(frames):
        oid = posesion[f]
        bdiag = ball_diag.get(f)
        if oid is None or bdiag is None:
            control[f] = None
            continue
        if _ball_moved(ball, frames, idx, control_window, move_k * bdiag):
            control[f] = oid
        else:
            control[f] = None
    return control


# --- resumen -------------------------------------------------------------------

def _summarize(
    posesion: dict[int, int | None],
    control: dict[int, int | None],
    ball_visible: dict[int, bool],
    fps: float | None,
    params: dict,
) -> dict:
    n_frames = len(posesion)
    pos_counts = Counter(v for v in posesion.values() if v is not None)
    ctrl_counts = Counter(v for v in control.values() if v is not None)
    n_pos = sum(pos_counts.values())
    n_ctrl = sum(ctrl_counts.values())
    no_owner = [f for f in posesion if posesion[f] is None]
    n_no_visible = sum(1 for f in no_owner if not ball_visible.get(f, False))
    n_libre = len(no_owner) - n_no_visible

    def pct(n: int) -> float:
        return round(100.0 * n / n_frames, 1) if n_frames else 0.0

    def secs(n: int) -> float | None:
        return round(n / fps, 2) if fps else None

    def por_obj(counts: Counter) -> dict:
        return {
            str(oid): {"frames": n, "segundos": secs(n), "pct": pct(n)}
            for oid, n in sorted(counts.items())
        }

    return {
        "n_frames": n_frames,
        "posesion_por_obj": por_obj(pos_counts),
        "control_por_obj": por_obj(ctrl_counts),
        "pct_posesion_total": pct(n_pos),
        "pct_control_total": pct(n_ctrl),
        "pct_libre": pct(n_libre),
        "pct_no_visible": pct(n_no_visible),
        "cambios_de_posesion": _count_changes([posesion[f] for f in sorted(posesion)]),
        "cambios_de_control": _count_changes([control[f] for f in sorted(control)]),
        "fps": fps,
        "params": params,
        "nota": "control ⊆ posesión; balón muerto junto a un robot = posesión sin control",
    }


# --- API pública ---------------------------------------------------------------

def compute_possession_refine(
    by_frame: dict[int, list[FrameObject]],
    *,
    gate_k: float = DEFAULT_GATE_K,
    min_frames: int = DEFAULT_MIN_FRAMES,
    control_window: int = DEFAULT_CONTROL_WINDOW,
    move_k: float = DEFAULT_MOVE_K,
    jump_k: float = DEFAULT_JUMP_K,
    clear_factor: float = DEFAULT_CLEAR_FACTOR,
    fps: float | None = None,
) -> PossessionRefineResult:
    """Posesión (histéresis adaptativa) + control (balón en movimiento) por frame.

    Args:
        by_frame: salida de ``load_frame_objects``.
        gate_k: gate de cercanía = ``gate_k * diagonal del bbox del robot`` (heredado).
        min_frames: frames consecutivos para confirmar un cambio **ambiguo** de posesión.
        control_window: ventana (frames) para medir el movimiento del balón (control).
        move_k: umbral de movimiento del balón = ``move_k * diagonal`` por frame.
        jump_k: salto radical del balón = ``jump_k * diagonal`` (cambio inmediato).
        clear_factor: el nuevo poseedor a ``< clear_factor *`` la distancia del actual ⇒
            cambio radical inmediato.
        fps: para convertir frames a segundos (``None`` ⇒ solo frames/pct).

    Returns:
        ``PossessionRefineResult`` con ``posesion_por_frame``, ``control_por_frame`` y
        ``resumen``. Por construcción ``control ⊆ posesión``.
    """
    raw = _raw_possession(by_frame, gate_k)
    ball = {f: ball_centroid(objs) for f, objs in by_frame.items()}
    ball_diag = {f: _ball_diag(objs) for f, objs in by_frame.items()}
    nearest = _nearest_by_frame(by_frame)

    posesion = _adaptive_possession(
        raw, ball, ball_diag, nearest, by_frame,
        min_frames=min_frames, jump_k=jump_k, clear_factor=clear_factor,
    )
    control = _control_series(
        posesion, ball, ball_diag, control_window=control_window, move_k=move_k,
    )
    ball_visible = {f: ball[f] is not None for f in by_frame}
    params = {
        "gate_k": gate_k, "min_frames": min_frames, "control_window": control_window,
        "move_k": move_k, "jump_k": jump_k, "clear_factor": clear_factor,
    }
    resumen = _summarize(posesion, control, ball_visible, fps, params)
    return PossessionRefineResult(
        posesion_por_frame=posesion, control_por_frame=control, resumen=resumen,
    )


def write_possession_refine_json(result: PossessionRefineResult, path: str | Path) -> Path:
    """Persiste el resultado a un JSON (``resumen`` + series por frame)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "posesion_por_frame": {
            str(k): result.posesion_por_frame[k] for k in sorted(result.posesion_por_frame)
        },
        "control_por_frame": {
            str(k): result.control_por_frame[k] for k in sorted(result.control_por_frame)
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = [
    "PossessionRefineResult",
    "compute_possession_refine",
    "write_possession_refine_json",
    "load_frame_objects",
]
