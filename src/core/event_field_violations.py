"""Violaciones de campo (ronda de entregable de eventos) — fuera, lack-of-progress, pushing.

Tres detectores que comparten el motor de estados ``_events_from_series`` (de ``event_goals``):

- **``fuera``** (Capa B, cm) — el centroide de un robot sale del rectángulo de líneas blancas
  (``causa="salida_campo"``) **o** entra al área chica (``causa="area_chica"``; entrar al área
  chica *también* es fuera). Excepción: cruzar la línea de gol **dentro de la boca** (yendo al
  gol) no cuenta. Geométrico ⇒ ``probabilidad=1.0``.
- **``lack_of_progress``** (Capa A, px, **probabilístico**) — el balón casi inmóvil durante una
  ventana larga. La vía 100% fiable sería el audio; aquí es una **confianza heurística**.
- **``pushing``** (Capa A, px, **probabilístico**) — dos robots en contacto sostenido **dentro
  del área chica**, con desplazamiento del empujado. Un empuje fuera del área chica no cuenta.

``fuera`` usa ``compute_metric_positions`` (robots en cm) + geometría de ``field_template``;
los otros dos usan el JSON en píxeles (``load_frame_objects``). Si la homografía no es fiable,
``fuera`` se omite (``fuera_disponible=False``) y los probabilísticos siguen disponibles.
Corre en **CPU local**, sin GPU. ``cv2`` se importa de forma perezosa (punto-en-polígono).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.core import field_template as ft
from src.core.event_goals import _events_from_series
from src.core.events import _bbox_diagonal
from src.core.events_core import (
    BALL_CLASSES,
    ROBOT_CLASS,
    FrameObject,
    ball_centroid,
    load_frame_objects,
)
from src.core.metric_positions import MetricResult, compute_metric_positions

# --- geometría del campo (cm) --------------------------------------------------
FIELD_X0 = ft.LINE_BORDER_CM  # 12
FIELD_X1 = ft.LENGTH_CM - ft.LINE_BORDER_CM  # 231
FIELD_Y0 = ft.LINE_BORDER_CM  # 12
FIELD_Y1 = ft.WIDTH_CM - ft.LINE_BORDER_CM  # 170
_MOUTH_TOP = ft._GOAL_TOP_Y_CM  # 61
_MOUTH_BOT = ft._GOAL_BOTTOM_Y_CM  # 121

DEFAULT_LINE_MARGIN_CM = 3.0
DEFAULT_LOP_WINDOW = 60  # ventana de lack-of-progress (frames)
DEFAULT_LOP_MOVE_K = 0.10  # paso medio del balón ·diagonal del balón
DEFAULT_PUSH_IOU = 0.05
DEFAULT_PUSH_K = 1.0  # cercanía = push_k·(radio_i + radio_j)
DEFAULT_PUSH_MOVE_K = 0.15  # desplazamiento del empujado ·diagonal del robot
DEFAULT_MIN_FRAMES = 3
DEFAULT_EXIT_FRAMES = 3
DEFAULT_COOLDOWN = 15
DEFAULT_GAP_FRAMES = 20
_PUSH_DISP_WINDOW = 5  # ventana (frames) para medir el desplazamiento del empujado
_FALLBACK_FPS = 30.0


@dataclass
class FieldViolationEvent:
    tipo: str  # "fuera" | "lack_of_progress" | "pushing"
    causa: str | None  # fuera: "salida_campo" | "area_chica"; otros: None
    obj_ids: list[int]  # robot(s) involucrado(s); [] si no aplica (balón)
    zona: str | None  # "yellow" | "blue" para area_chica/pushing; None si no aplica
    frame_inicio: int
    frame_fin: int
    dur_frames: int
    dur_s: float | None
    ref: tuple[float, float] | None  # cm (fuera) o px (lack/pushing)
    probabilidad: float  # 1.0 geométrico; (0,1) probabilístico


@dataclass
class FieldViolationsResult:
    eventos: list[FieldViolationEvent]
    resumen: dict


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# --- geometría: punto-en-polígono y predicados de fuera ------------------------

def _penalty_polys():
    """Contornos cerrados (cm) del área chica, por zona, para ``cv2.pointPolygonTest``."""
    import numpy as np

    yellow = ft._penalty_outline_cm(FIELD_X0, FIELD_X0 + ft.PENALTY_DEPTH_CM)
    blue = ft._penalty_outline_cm(FIELD_X1, FIELD_X1 - ft.PENALTY_DEPTH_CM)
    return {
        "yellow": np.array(yellow, dtype="float32"),
        "blue": np.array(blue, dtype="float32"),
    }


def _in_penalty(xy: tuple[float, float], polys, margin: float) -> str | None:
    """Zona del área chica que contiene ``xy`` (cm), o ``None``."""
    import cv2

    pt = (float(xy[0]), float(xy[1]))
    for zona, poly in polys.items():
        if cv2.pointPolygonTest(poly, pt, True) >= -margin:
            return zona
    return None


def _out_of_field(xy: tuple[float, float], margin: float) -> bool:
    """¿El centroide (cm) cruzó la línea blanca del campo (salvo por la boca de portería)?"""
    x, y = xy
    if FIELD_X0 - margin <= x <= FIELD_X1 + margin and FIELD_Y0 - margin <= y <= FIELD_Y1 + margin:
        return False
    # excepción: cruzar la línea de gol dentro de la boca (entrando al gol) no es fuera.
    if (x < FIELD_X0 or x > FIELD_X1) and _MOUTH_TOP <= y <= _MOUTH_BOT:
        return False
    return True


def _classify_fuera(
    xy: tuple[float, float], polys, margin: float
) -> tuple[str | None, str | None]:
    """``(causa, zona)``: area_chica tiene prioridad sobre salida_campo; ``(None, None)`` si legal."""
    zona = _in_penalty(xy, polys, margin)
    if zona is not None:
        return "area_chica", zona
    if _out_of_field(xy, margin):
        return "salida_campo", None
    return None, None


# --- relleno de huecos (compartido) --------------------------------------------

def _fill_gaps(flags: list[tuple[int, bool, bool]], gap_frames: int) -> list[tuple[int, bool]]:
    """``[(f, present, region)]`` → serie booleana; sostiene ``region`` en ausencias ≤ gap."""
    serie: list[tuple[int, bool]] = []
    held: bool | None = None
    gap = 0
    for f, present, region in flags:
        if present:
            val = region
            held = val
            gap = 0
        elif held is not None and gap < gap_frames:
            val = held
            gap += 1
        else:
            val = False
            held = None
        serie.append((f, val))
    return serie


def _avg_step(points: dict[int, tuple[float, float] | None], frames: list[int],
              idx: int, window: int) -> float | None:
    """Paso medio (px o cm) en la ventana ``[idx, idx+window)`` con las muestras visibles."""
    pts = [
        points[frames[j]]
        for j in range(idx, min(idx + window, len(frames)))
        if points.get(frames[j]) is not None
    ]
    if len(pts) < 2:
        return None
    total = sum(
        math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
        for k in range(len(pts) - 1)
    )
    return total / (len(pts) - 1)


# --- detector `fuera` (Capa B, cm) ---------------------------------------------

def _robots_cm(result: MetricResult) -> dict[int, dict[int, tuple[float, float] | None]]:
    """``{obj_id: {frame: xy_cm}}`` de los robots."""
    out: dict[int, dict[int, tuple[float, float] | None]] = {}
    for p in result.posiciones:
        if p.cls == ROBOT_CLASS:
            out.setdefault(p.obj_id, {})[p.frame_index] = p.xy_cm
    return out


def _detect_fuera(
    robots: dict[int, dict[int, tuple[float, float] | None]], polys, *,
    margin, min_frames, exit_frames, cooldown, fps,
) -> list[FieldViolationEvent]:
    eventos: list[FieldViolationEvent] = []
    for oid, frame_xy in robots.items():
        present = [f for f, xy in frame_xy.items() if xy is not None]
        if not present:
            continue
        lo, hi = min(present), max(present)
        serie: list[tuple[int, bool]] = []
        info: dict[int, tuple[str, str | None, tuple[float, float]]] = {}
        for f in range(lo, hi + 1):
            xy = frame_xy.get(f)
            flag = False
            if xy is not None:
                causa, zona = _classify_fuera(xy, polys, margin)
                if causa is not None:
                    flag = True
                    info[f] = (causa, zona, xy)
            serie.append((f, flag))
        for ev in _events_from_series(
            serie, str(oid), min_frames=min_frames, exit_frames=exit_frames,
            cooldown=cooldown, fps=fps,
        ):
            opening = next((f for f in range(ev.frame_inicio, ev.frame_fin + 1) if f in info),
                           ev.frame_inicio)
            causa, zona, ref = info.get(opening, (None, None, None))
            eventos.append(FieldViolationEvent(
                tipo="fuera", causa=causa, obj_ids=[oid], zona=zona,
                frame_inicio=ev.frame_inicio, frame_fin=ev.frame_fin,
                dur_frames=ev.dur_frames, dur_s=ev.dur_s, ref=ref, probabilidad=1.0,
            ))
    return eventos


# --- detector `lack_of_progress` (Capa A, px, prob.) ---------------------------

def _ball_diag(objs: list[FrameObject]) -> float | None:
    balls = [o for o in objs if o.class_name in BALL_CLASSES]
    if not balls:
        return None
    return _bbox_diagonal(max(balls, key=lambda o: o.score).bbox)


def _detect_lack_of_progress(
    by_frame: dict[int, list[FrameObject]], *,
    lop_window, lop_move_k, gap_frames, min_frames, exit_frames, cooldown, fps,
) -> list[FieldViolationEvent]:
    frames = sorted(by_frame)
    ball = {f: ball_centroid(by_frame[f]) for f in frames}
    bdiag = {f: _ball_diag(by_frame[f]) for f in frames}
    flags: list[tuple[int, bool, bool]] = []
    for idx, f in enumerate(frames):
        present = ball[f] is not None and bdiag[f] is not None
        stalled = False
        if present:
            avg = _avg_step(ball, frames, idx, lop_window)
            stalled = avg is not None and avg < lop_move_k * bdiag[f]
        flags.append((f, present, stalled))
    near = _fill_gaps(flags, gap_frames)

    eventos: list[FieldViolationEvent] = []
    for ev in _events_from_series(
        near, "ball", min_frames=min_frames, exit_frames=exit_frames,
        cooldown=cooldown, fps=fps,
    ):
        prob = min(0.95, _clamp01(0.5 + 0.5 * ev.dur_frames / (2 * lop_window)))
        ref = next((ball[f] for f in range(ev.frame_inicio, ev.frame_fin + 1)
                    if ball.get(f) is not None), None)
        eventos.append(FieldViolationEvent(
            tipo="lack_of_progress", causa=None, obj_ids=[], zona=None,
            frame_inicio=ev.frame_inicio, frame_fin=ev.frame_fin,
            dur_frames=ev.dur_frames, dur_s=ev.dur_s, ref=ref, probabilidad=round(prob, 2),
        ))
    return eventos


# --- detector `pushing` (Capa A, px, prob.) — solo en área chica ---------------

def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _point_in_zone_px(pt, objs: list[FrameObject]) -> str | None:
    """Proxy px: ¿el punto cae en el bbox de ``yellow_zone``/``blue_zone``?"""
    for o in objs:
        if o.class_name in ("yellow_zone", "blue_zone"):
            x, y, w, h = o.bbox
            if x <= pt[0] <= x + w and y <= pt[1] <= y + h:
                return "yellow" if o.class_name == "yellow_zone" else "blue"
    return None


def _contact_zone(
    f, oid_i, oid_j, ci, cj, cm_lookup, by_frame, polys, margin,
) -> str | None:
    """Zona de área chica donde ocurre el contacto (cm si hay, si no proxy px), o ``None``."""
    if cm_lookup is not None:
        pi, pj = cm_lookup.get((f, oid_i)), cm_lookup.get((f, oid_j))
        if pi is not None and pj is not None:
            mid = ((pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2)
            return _in_penalty(mid, polys, margin)
        return None
    mid_px = ((ci[0] + cj[0]) / 2, (ci[1] + cj[1]) / 2)
    return _point_in_zone_px(mid_px, by_frame[f])


def _detect_pushing(
    by_frame: dict[int, list[FrameObject]], cm_lookup, polys, *,
    push_iou, push_k, push_move_k, margin, min_frames, exit_frames, cooldown, fps,
) -> list[FieldViolationEvent]:
    frames = sorted(by_frame)
    robot_px: dict[int, dict[int, tuple[float, float]]] = {}  # oid -> {f: centroid}
    robot_diag: dict[tuple[int, int], float] = {}
    for f in frames:
        for o in by_frame[f]:
            if o.class_name == ROBOT_CLASS:
                robot_px.setdefault(o.obj_id, {})[f] = o.centroid
                robot_diag[(f, o.obj_id)] = _bbox_diagonal(o.bbox)

    # contacto en área chica por par, por frame. `strength` ∈ [0,1] = solape (IoU) o, si solo
    # hay cercanía, qué tan cerca están (1 = centroides juntos). El contacto NO exige
    # desplazamiento: en un empujón tipo "sumo" los robots quedan trabados casi sin moverse.
    pair_zone: dict[tuple[int, int], dict[int, tuple[float, str]]] = {}  # (i,j)->{f:(strength,zona)}
    for f in frames:
        robs = [o for o in by_frame[f] if o.class_name == ROBOT_CLASS]
        for a in range(len(robs)):
            for b in range(a + 1, len(robs)):
                ri, rj = robs[a], robs[b]
                iou = _iou(ri.bbox, rj.bbox)
                dist = math.hypot(ri.centroid[0] - rj.centroid[0], ri.centroid[1] - rj.centroid[1])
                radii = 0.5 * (_bbox_diagonal(ri.bbox) + _bbox_diagonal(rj.bbox))
                if iou <= push_iou and dist >= push_k * radii:
                    continue
                zona = _contact_zone(f, ri.obj_id, rj.obj_id, ri.centroid, rj.centroid,
                                     cm_lookup, by_frame, polys, margin)
                if zona is None:
                    continue
                strength = max(iou, _clamp01(1.0 - dist / (push_k * radii)) if radii > 0 else 0.0)
                key = (min(ri.obj_id, rj.obj_id), max(ri.obj_id, rj.obj_id))
                pair_zone.setdefault(key, {})[f] = (strength, zona)

    eventos: list[FieldViolationEvent] = []
    for (i, j), fz in pair_zone.items():
        cf = sorted(fz)
        lo, hi = cf[0], cf[-1]
        si = robot_px.get(i, {})
        sj = robot_px.get(j, {})
        si_frames, sj_frames = sorted(si), sorted(sj)
        serie = [(f, f in fz) for f in range(lo, hi + 1)]
        for ev in _events_from_series(
            serie, f"{i}-{j}", min_frames=min_frames, exit_frames=exit_frames,
            cooldown=cooldown, fps=fps,
        ):
            interval = [f for f in range(ev.frame_inicio, ev.frame_fin + 1) if f in fz]
            zona = fz[interval[0]][1] if interval else None
            mean_str = sum(fz[f][0] for f in interval) / len(interval) if interval else 0.0
            # bonus si el empujado se desplaza (evidencia extra; no requisito).
            moved = 0
            for f in interval:
                di = _avg_step(si, si_frames, si_frames.index(f), _PUSH_DISP_WINDOW) if f in si else None
                dj = _avg_step(sj, sj_frames, sj_frames.index(f), _PUSH_DISP_WINDOW) if f in sj else None
                diag = robot_diag.get((f, i)) or robot_diag.get((f, j)) or 1.0
                if max(di or 0.0, dj or 0.0) >= push_move_k * diag:
                    moved += 1
            moved_frac = moved / len(interval) if interval else 0.0
            prob = min(0.95, _clamp01(0.35 + 0.35 * mean_str + 0.2 * min(1.0, ev.dur_frames / 30.0)
                                      + 0.1 * moved_frac))
            opening = interval[0] if interval else ev.frame_inicio
            ci, cj = si.get(opening), sj.get(opening)
            ref = (((ci[0] + cj[0]) / 2, (ci[1] + cj[1]) / 2) if ci and cj else None)
            eventos.append(FieldViolationEvent(
                tipo="pushing", causa=None, obj_ids=[i, j], zona=zona,
                frame_inicio=ev.frame_inicio, frame_fin=ev.frame_fin,
                dur_frames=ev.dur_frames, dur_s=ev.dur_s, ref=ref, probabilidad=round(prob, 2),
            ))
    return eventos


# --- API pública ---------------------------------------------------------------

def _video_fps(tracks_json: Path) -> float | None:
    return json.loads(Path(tracks_json).read_text(encoding="utf-8")).get("fps")


def compute_field_violations(
    source: str | Path,
    *,
    line_margin_cm: float = DEFAULT_LINE_MARGIN_CM,
    lop_window: int = DEFAULT_LOP_WINDOW,
    lop_move_thresh_k: float = DEFAULT_LOP_MOVE_K,
    push_iou: float = DEFAULT_PUSH_IOU,
    push_k: float = DEFAULT_PUSH_K,
    push_move_k: float = DEFAULT_PUSH_MOVE_K,
    min_frames: int = DEFAULT_MIN_FRAMES,
    exit_frames: int = DEFAULT_EXIT_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN,
    gap_frames: int = DEFAULT_GAP_FRAMES,
    fps: float | None = None,
) -> FieldViolationsResult:
    """Detecta ``fuera`` (cm), ``lack_of_progress`` y ``pushing`` (px, prob.) sobre un tracks_json.

    ``fuera`` requiere homografía fiable (``compute_metric_positions``); si no hay cm se omite y
    ``resumen["fuera_disponible"]`` queda en ``False`` (los probabilísticos siguen disponibles).
    """
    tracks_json = Path(source)
    by_frame = load_frame_objects(tracks_json)
    fps = fps or _video_fps(tracks_json)
    polys = _penalty_polys()

    eventos: list[FieldViolationEvent] = []
    fuera_disponible = False
    cm_lookup: dict[tuple[int, int], tuple[float, float]] | None = None
    try:
        mr = compute_metric_positions(tracks_json)
        robots = _robots_cm(mr)
        fuera_disponible = any(xy is not None for d in robots.values() for xy in d.values())
        fps = fps or mr.resumen.get("fps")
    except Exception:  # homografía/insumos no disponibles: degradar a solo px
        robots = {}

    if fuera_disponible:
        cm_lookup = {
            (f, oid): xy for oid, d in robots.items() for f, xy in d.items() if xy is not None
        }
        eventos += _detect_fuera(
            robots, polys, margin=line_margin_cm, min_frames=min_frames,
            exit_frames=exit_frames, cooldown=cooldown_frames, fps=fps,
        )

    eventos += _detect_lack_of_progress(
        by_frame, lop_window=lop_window, lop_move_k=lop_move_thresh_k, gap_frames=gap_frames,
        min_frames=min_frames, exit_frames=exit_frames, cooldown=cooldown_frames, fps=fps,
    )
    eventos += _detect_pushing(
        by_frame, cm_lookup, polys, push_iou=push_iou, push_k=push_k, push_move_k=push_move_k,
        margin=line_margin_cm, min_frames=min_frames, exit_frames=exit_frames,
        cooldown=cooldown_frames, fps=fps,
    )

    eventos.sort(key=lambda e: e.frame_inicio)
    params = {
        "line_margin_cm": line_margin_cm, "lop_window": lop_window,
        "lop_move_thresh_k": lop_move_thresh_k, "push_iou": push_iou, "push_k": push_k,
        "push_move_k": push_move_k, "min_frames": min_frames, "exit_frames": exit_frames,
        "cooldown_frames": cooldown_frames, "gap_frames": gap_frames,
    }
    return FieldViolationsResult(
        eventos=eventos, resumen=_summarize(eventos, fuera_disponible, fps, params),
    )


def _summarize(eventos, fuera_disponible, fps, params) -> dict:
    fuera = {"salida_campo": 0, "area_chica": 0}
    n_lop = n_push = 0
    for e in eventos:
        if e.tipo == "fuera":
            fuera[e.causa] = fuera.get(e.causa, 0) + 1
        elif e.tipo == "lack_of_progress":
            n_lop += 1
        elif e.tipo == "pushing":
            n_push += 1
    return {
        "fuera_disponible": fuera_disponible,
        "conteo": {"fuera": fuera, "lack_of_progress": n_lop, "pushing": n_push},
        "total_eventos": len(eventos),
        "fps": fps,
        "params": params,
        "nota": "fuera = geométrico (cm); lack_of_progress/pushing = probabilístico (px, indicativo)",
    }


def write_field_violations_json(result: FieldViolationsResult, path: str | Path) -> Path:
    """Escribe el resultado a JSON (resumen + eventos)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "eventos": [
            {
                "tipo": e.tipo, "causa": e.causa, "obj_ids": e.obj_ids, "zona": e.zona,
                "frame_inicio": e.frame_inicio, "frame_fin": e.frame_fin,
                "dur_frames": e.dur_frames, "dur_s": e.dur_s,
                "ref": list(e.ref) if e.ref is not None else None,
                "probabilidad": e.probabilidad,
            }
            for e in result.eventos
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
