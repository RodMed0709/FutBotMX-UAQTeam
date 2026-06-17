"""Tiro a gol vs gol (ronda de entregable de eventos) — clasifica cada lance cerca de una
portería como ``"tiro"`` o ``"gol"``.

Refina la detección de gol. La idea central, validada contra el video real: **un gol exige
que el balón cruce la línea de gol real Y caiga dentro de la boca real** (no basta con tocar
el bbox de la portería ni acercarse a la línea). Se separa:

- **gol**: el centroide del balón **cruza la línea de gol** (``x ≥ GOAL_LINE_X_RIGHT_CM`` azul /
  ``x ≤ GOAL_LINE_X_LEFT_CM`` amarilla) **dentro de la boca real** ``y∈[_GOAL_TOP_Y_CM,
  _GOAL_BOTTOM_Y_CM]`` — sin ensanchar la boca ni correr la línea hacia el campo;
- **tiro**: el balón entra a una **banda frente a la portería** (``tiro_depth_cm`` antes de la
  línea) dentro de la boca **±``side_cm``** (tolerancia para tiros al poste), **sin** cumplir
  el test estricto de gol.

Detalles aprendidos del caso real (``IMG_9933_5m30``): los falsos goles venían de (a) correr
la línea hacia adentro con un margen y (b) ensanchar la boca — un tiro al poste (fuera de la
boca) o un tiro que se queda corto se contaban como gol. Además el balón se detecta de forma
**intermitente**: un balón parado frente a la portería parpadea y se fragmentaba en muchos
lances; por eso se **fusionan huecos de detección** de hasta ``gap_frames`` frames. La
**dirección** dejó de exigirse: un tiro al poste se queda estático y aun así es un tiro.

Dos rutas, misma salida (``ShotGoalEvent``):

- **cm (cámara superior, autoridad):** posiciones del balón en cm de ``metric_positions`` +
  geometría de ``field_template``. Es la fiable.
- **px (universal, proxy):** centroide vs bbox de la zona, con bbox encogido y regla de 3/4
  hacia la pared. Para tomas parciales sin cm fiable; **subdetecta** (es indicativa).

No reimplementa el motor de estados: reutiliza ``_events_from_series`` (de ``event_goals``)
para segmentar los lances (sobre una serie con huecos rellenados) y añade un **post-paso de
clasificación** del intervalo. Corre en **CPU local** sobre el JSON de tracking (+ T3
``metric_positions`` para cm); sin GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.core import field_template as ft
from src.core.event_goal_geometric import _ball_by_frame
from src.core.event_goals import _events_from_series, _zones_present, _ZONE_LABEL
from src.core.events_core import BALL_CLASSES, ball_centroid, load_frame_objects
from src.core.metric_positions import MetricResult, compute_metric_positions

DEFAULT_TIRO_DEPTH_CM = 15.0  # banda (cm) frente a la línea que cuenta como tiro
DEFAULT_SIDE_CM = 12.0  # tolerancia lateral de la boca (tiros al poste)
DEFAULT_GOAL_MARGIN_CM = 0.0  # penetración exigida más allá de la línea para gol
DEFAULT_GAP_FRAMES = 20  # huecos de detección a fusionar (balón parpadeante)
DEFAULT_THREE_QUARTER_FRAC = 0.75  # ruta px
DEFAULT_MARGIN_PX = 0.0  # ruta px (inset del bbox)
DEFAULT_MIN_FRAMES = 3
DEFAULT_EXIT_FRAMES = 3
DEFAULT_COOLDOWN = 15
_FALLBACK_FPS = 30.0


@dataclass
class ShotGoalEvent:
    tipo: str  # "tiro" | "gol"
    zona: str  # "yellow" | "blue"
    frame_inicio: int
    frame_fin: int
    dur_frames: int
    dur_s: float | None
    xy_cm: tuple[float, float] | None  # ruta px ⇒ None


@dataclass
class ShotGoalResult:
    eventos: list[ShotGoalEvent]
    resumen: dict


# --- relleno de huecos de detección -------------------------------------------

def _fill_gaps(
    flags: list[tuple[int, bool, bool]], gap_frames: int
) -> list[tuple[int, bool]]:
    """Convierte ``[(f, present, region)]`` en una serie booleana con huecos fusionados.

    Mientras el balón está **ausente**, se sostiene el último valor de ``region`` hasta
    ``gap_frames`` frames (puente sobre el parpadeo de detección); pasado ese límite se cierra.
    Un balón **presente pero fuera** de la región sí cuenta como salida real.
    """
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


# --- geometría en cm -----------------------------------------------------------

def _in_mouth(y: float) -> bool:
    """¿``y`` cae dentro de la boca **real** de la portería (sin tolerancia)?"""
    return ft._GOAL_TOP_Y_CM <= y <= ft._GOAL_BOTTOM_Y_CM


def _crossed_cm(xy: tuple[float, float], zona: str, goal_margin: float) -> bool:
    """¿El balón (cm) cruzó la línea de gol **dentro de la boca real**, para ``zona``?

    Estricto: la boca no se ensancha y la línea no se corre hacia el campo. ``goal_margin``
    exige penetración adicional más allá de la línea (default 0 = justo la línea).
    """
    x, y = xy
    if not _in_mouth(y):
        return False
    if zona == "yellow":
        return x <= ft.GOAL_LINE_X_LEFT_CM - goal_margin
    return x >= ft.GOAL_LINE_X_RIGHT_CM + goal_margin


def _in_approach(xy: tuple[float, float], zona: str, depth: float, side: float) -> bool:
    """¿El balón (cm) está en la zona de aproximación a la portería (tiro)?

    Banda de ``depth`` cm frente a la línea (y cualquier punto ya pasado), dentro de la boca
    ±``side`` (tolerancia para tiros al poste).
    """
    x, y = xy
    if not (ft._GOAL_TOP_Y_CM - side <= y <= ft._GOAL_BOTTOM_Y_CM + side):
        return False
    if zona == "yellow":
        return x <= ft.GOAL_LINE_X_LEFT_CM + depth
    return x >= ft.GOAL_LINE_X_RIGHT_CM - depth


def _nearest_to_goal(
    samples: list[tuple[float, float]], zona: str
) -> tuple[float, float] | None:
    """Muestra del balón más cercana a la portería (min x yellow / max x blue)."""
    if not samples:
        return None
    return min(samples, key=lambda xy: xy[0]) if zona == "yellow" else max(
        samples, key=lambda xy: xy[0]
    )


def _predicted_only_ball_frames(result: MetricResult) -> set[int]:
    """Frames donde el balón existe pero **todas** sus muestras son predichas (oclusión Kalman).

    Una posición cuenta como *medida* si su ``source`` es ``None`` (cruda, T3), ``"measured"`` o
    ``"gated"``; solo ``"predicted"`` es predicha. Con posiciones crudas (flag apagado) ``source``
    es siempre ``None`` → el conjunto queda **vacío** y el comportamiento es idéntico al actual.
    """
    has_measured: set[int] = set()
    has_any: set[int] = set()
    for p in result.posiciones:
        if p.cls not in BALL_CLASSES or p.xy_cm is None:
            continue
        has_any.add(p.frame_index)
        if p.source != "predicted":
            has_measured.add(p.frame_index)
    return has_any - has_measured


def _cm_series(
    ball_by_frame: dict[int, list[tuple[float, float]]],
    frames: list[int],
    zona: str,
    *,
    tiro_depth: float,
    side: float,
    goal_margin: float,
    gap_frames: int,
    predicted_only: set[int] = frozenset(),
) -> tuple[list[tuple[int, bool]], dict[int, bool], dict[int, tuple[float, float] | None]]:
    """Serie ``near`` (en aproximación, con huecos fusionados), ``goal`` por frame y la muestra.

    ``near`` alimenta ``_events_from_series`` (segmenta lances); ``goal`` clasifica el intervalo
    (gol si alguna muestra cruzó la línea dentro de la boca); ``ref`` da la posición de cm.

    ``predicted_only``: frames cuyo balón es solo-predicho (oclusión); el gol se **suprime** ahí
    (conservador) — no se declara gol sobre un balón que no se vio. Vacío = sin efecto.
    """
    flags: list[tuple[int, bool, bool]] = []
    goal_by_frame: dict[int, bool] = {}
    ref: dict[int, tuple[float, float] | None] = {}

    for f in frames:
        n = _nearest_to_goal(ball_by_frame.get(f, []), zona)
        ref[f] = n
        present = n is not None
        is_goal = present and f not in predicted_only and _crossed_cm(n, zona, goal_margin)
        # un gol implica aproximación; si no, se evalúa la banda de tiro.
        in_app = present and (is_goal or _in_approach(n, zona, tiro_depth, side))
        goal_by_frame[f] = is_goal
        flags.append((f, present, in_app))

    near = _fill_gaps(flags, gap_frames)
    return near, goal_by_frame, ref


def _classify(
    near: list[tuple[int, bool]],
    goal_by_frame: dict[int, bool],
    ref: dict[int, tuple[float, float] | None],
    zona: str,
    *,
    min_frames: int,
    exit_frames: int,
    cooldown: int,
    fps: float | None,
) -> list[ShotGoalEvent]:
    """Segmenta lances con ``_events_from_series`` y clasifica cada intervalo tiro/gol."""
    eventos: list[ShotGoalEvent] = []
    for ev in _events_from_series(
        near, zona, min_frames=min_frames, exit_frames=exit_frames,
        cooldown=cooldown, fps=fps,
    ):
        interval = range(ev.frame_inicio, ev.frame_fin + 1)
        goal_frame = next((f for f in interval if goal_by_frame.get(f, False)), None)
        if goal_frame is not None:
            tipo, xy = "gol", ref.get(goal_frame)
        else:
            tipo = "tiro"
            xy = next((ref[f] for f in interval if ref.get(f) is not None), None)
        eventos.append(ShotGoalEvent(
            tipo=tipo, zona=ev.zona, frame_inicio=ev.frame_inicio, frame_fin=ev.frame_fin,
            dur_frames=ev.dur_frames, dur_s=ev.dur_s, xy_cm=xy,
        ))
    return eventos


def _compute_cm(
    result: MetricResult, *, tiro_depth_cm, side_cm, goal_margin_cm, gap_frames,
    min_frames, exit_frames, cooldown_frames, fps,
) -> list[ShotGoalEvent]:
    ball_by_frame = _ball_by_frame(result)
    if not ball_by_frame:
        return []
    predicted_only = _predicted_only_ball_frames(result)
    lo, hi = min(ball_by_frame), max(ball_by_frame)
    frames = list(range(lo, hi + 1))  # timeline contiguo (huecos = fuera)
    eventos: list[ShotGoalEvent] = []
    for zona in ("yellow", "blue"):
        near, goal_by_frame, ref = _cm_series(
            ball_by_frame, frames, zona,
            tiro_depth=tiro_depth_cm, side=side_cm, goal_margin=goal_margin_cm,
            gap_frames=gap_frames, predicted_only=predicted_only,
        )
        eventos += _classify(
            near, goal_by_frame, ref, zona,
            min_frames=min_frames, exit_frames=exit_frames, cooldown=cooldown_frames, fps=fps,
        )
    return eventos


# --- geometría en píxeles (ruta px, proxy) -------------------------------------

def _wall_axis(bbox, W: float, H: float) -> tuple[int, float]:
    """Eje de profundidad (0=x, 1=y) y signo hacia la pared para una zona.

    La pared (línea de gol) es el lado de la zona **más alejado del centro de la imagen**.
    Se elige el eje con mayor excentricidad normalizada (robusto a tomas vertical/horizontal).
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    ex = abs(cx - W / 2) / (W / 2) if W else 0.0
    ey = abs(cy - H / 2) / (H / 2) if H else 0.0
    if ex >= ey:
        return 0, (1.0 if cx >= W / 2 else -1.0)
    return 1, (1.0 if cy >= H / 2 else -1.0)


def _pick_zone_bbox(objs, zone_class: str, ball: tuple[float, float]):
    """bbox de ``zone_class`` con centro más cercano al balón (o ``None``)."""
    bx, by = ball
    zbs = [o.bbox for o in objs if o.class_name == zone_class]
    if not zbs:
        return None
    return min(zbs, key=lambda b: (b[0] + b[2] / 2 - bx) ** 2 + (b[1] + b[3] / 2 - by) ** 2)


def _depth_frac(
    ball: tuple[float, float], bbox, axis: int, sign: float, margin_px: float
) -> tuple[float | None, bool, float | None]:
    """Fracción de profundidad del balón hacia la pared (0=borde interno, 1=pared).

    Devuelve ``(frac, within_other, along)`` sobre el bbox **encogido** por ``margin_px``.
    ``within_other`` = el balón cae dentro del bbox en el eje transversal. ``frac`` puede ser
    >1 si el balón rebasó la pared, <0 si está antes del borde interno (fuera de la zona).
    """
    x, y, w, h = bbox
    ix, iy = x + margin_px, y + margin_px
    iw, ih = w - 2 * margin_px, h - 2 * margin_px
    if iw <= 0 or ih <= 0:
        return None, False, None
    bx, by = ball
    if axis == 0:
        lo, length, along = ix, iw, bx
        other_lo, other_hi, other = iy, iy + ih, by
    else:
        lo, length, along = iy, ih, by
        other_lo, other_hi, other = ix, ix + iw, bx
    within_other = other_lo <= other <= other_hi
    frac = (along - lo) / length if sign > 0 else (lo + length - along) / length
    return frac, within_other, along


def _px_series(
    by_frame: dict[int, list],
    frames: list[int],
    zone_class: str,
    *,
    margin_px: float,
    frac_gol: float,
    gap_frames: int,
    W: float,
    H: float,
) -> tuple[list[tuple[int, bool]], dict[int, bool], dict[int, tuple[float, float] | None]]:
    """Análogo px de ``_cm_series``: ``near`` (con huecos fusionados) y ``goal`` por frame.

    ``goal`` = el centroide rebasa ``frac_gol`` de la profundidad del bbox encogido hacia la
    pared (gol px, conservador); ``in_zone`` (tiro px) = dentro del bbox encogido sin alcanzar
    ``frac_gol``. ``ref`` es ``None`` (la ruta px no da cm).
    """
    flags: list[tuple[int, bool, bool]] = []
    goal_by_frame: dict[int, bool] = {}
    ref: dict[int, tuple[float, float] | None] = {}

    for f in frames:
        objs = by_frame.get(f, [])
        ball = ball_centroid(objs)
        present = ball is not None
        in_region = False
        is_goal = False
        if present:
            zb = _pick_zone_bbox(objs, zone_class, ball)
            if zb is not None:
                axis, sign = _wall_axis(zb, W, H)
                frac, within_other, _ = _depth_frac(ball, zb, axis, sign, margin_px)
                if frac is not None and within_other and frac >= 0:
                    is_goal = frac >= frac_gol
                    in_region = True  # dentro del bbox (tiro) o ya cruzó 3/4 (gol)
            else:
                present = False  # sin bbox de zona ese frame: trátalo como ausencia
        goal_by_frame[f] = is_goal
        ref[f] = None
        flags.append((f, present, in_region))

    near = _fill_gaps(flags, gap_frames)
    return near, goal_by_frame, ref


def _compute_px(
    by_frame: dict[int, list], W: float, H: float, *, margin_px, three_quarter_frac,
    gap_frames, min_frames, exit_frames, cooldown_frames, fps,
) -> list[ShotGoalEvent]:
    if not by_frame:
        return []
    frames = sorted(by_frame)
    eventos: list[ShotGoalEvent] = []
    for zone_class in _zones_present(by_frame):
        near, goal_by_frame, ref = _px_series(
            by_frame, frames, zone_class,
            margin_px=margin_px, frac_gol=three_quarter_frac, gap_frames=gap_frames, W=W, H=H,
        )
        eventos += _classify(
            near, goal_by_frame, ref, _ZONE_LABEL[zone_class],
            min_frames=min_frames, exit_frames=exit_frames, cooldown=cooldown_frames, fps=fps,
        )
    return eventos


def _video_meta(tracks_json: Path) -> tuple[float | None, float, float]:
    """``(fps, width, height)`` del JSON de tracking (para la ruta px)."""
    data = json.loads(Path(tracks_json).read_text(encoding="utf-8"))
    res = data.get("resolution", {}) or {}
    return data.get("fps"), float(res.get("width", 0) or 0), float(res.get("height", 0) or 0)


# --- API pública ---------------------------------------------------------------

def compute_shot_vs_goal(
    source: str | Path | MetricResult,
    *,
    route: str = "cm",
    tiro_depth_cm: float = DEFAULT_TIRO_DEPTH_CM,
    side_cm: float = DEFAULT_SIDE_CM,
    goal_margin_cm: float = DEFAULT_GOAL_MARGIN_CM,
    three_quarter_frac: float = DEFAULT_THREE_QUARTER_FRAC,
    margin_px: float = DEFAULT_MARGIN_PX,
    gap_frames: int = DEFAULT_GAP_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
    exit_frames: int = DEFAULT_EXIT_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN,
    fps: float | None = None,
) -> ShotGoalResult:
    """Clasifica lances cerca de cada portería como ``"tiro"`` o ``"gol"``.

    ``route="cm"``: ``source`` = ruta a tracks_json (llama a ``compute_metric_positions``) o
    un ``MetricResult`` ya calculado (autoridad, cámara superior). Gol = cruce de la línea de
    gol real dentro de la boca real; tiro = banda de ``tiro_depth_cm`` frente a la línea (boca
    ±``side_cm``) sin cruzar.

    ``route="px"``: ``source`` = ruta a tracks_json (proxy universal en píxeles; bbox encogido
    + regla de 3/4 hacia la pared). ``xy_cm`` queda en ``None``; subdetecta goles.
    """
    if route == "cm":
        result = source if isinstance(source, MetricResult) else compute_metric_positions(Path(source))
        fps = fps or result.resumen.get("fps")
        eventos = _compute_cm(
            result, tiro_depth_cm=tiro_depth_cm, side_cm=side_cm, goal_margin_cm=goal_margin_cm,
            gap_frames=gap_frames, min_frames=min_frames, exit_frames=exit_frames,
            cooldown_frames=cooldown_frames, fps=fps,
        )
    elif route == "px":
        if isinstance(source, MetricResult):
            raise ValueError("route='px' requiere una ruta a tracks_json, no un MetricResult")
        by_frame = load_frame_objects(Path(source))
        meta_fps, W, H = _video_meta(Path(source))
        fps = fps or meta_fps
        eventos = _compute_px(
            by_frame, W, H, margin_px=margin_px, three_quarter_frac=three_quarter_frac,
            gap_frames=gap_frames, min_frames=min_frames, exit_frames=exit_frames,
            cooldown_frames=cooldown_frames, fps=fps,
        )
    else:
        raise ValueError(f"route inválido: {route!r} (usa 'cm' o 'px')")

    eventos.sort(key=lambda e: e.frame_inicio)
    return ShotGoalResult(eventos=eventos, resumen=_summarize(eventos, route, fps, locals()))


def _summarize(eventos: list[ShotGoalEvent], route: str, fps: float | None, params: dict) -> dict:
    conteo: dict[str, dict[str, int]] = {"tiro": {}, "gol": {}}
    for e in eventos:
        conteo[e.tipo][e.zona] = conteo[e.tipo].get(e.zona, 0) + 1
    return {
        "route": route,
        "fps": fps,
        "tiros": {"total": sum(conteo["tiro"].values()), "por_zona": conteo["tiro"]},
        "goles": {"total": sum(conteo["gol"].values()), "por_zona": conteo["gol"]},
        "total_eventos": len(eventos),
        "params": {
            "tiro_depth_cm": params.get("tiro_depth_cm"),
            "side_cm": params.get("side_cm"),
            "goal_margin_cm": params.get("goal_margin_cm"),
            "three_quarter_frac": params.get("three_quarter_frac"),
            "margin_px": params.get("margin_px"),
            "gap_frames": params.get("gap_frames"),
            "min_frames": params.get("min_frames"),
            "exit_frames": params.get("exit_frames"),
            "cooldown_frames": params.get("cooldown_frames"),
        },
        "nota": "cm = autoridad (cámara superior); px = proxy conservador (subdetecta)",
    }


def write_shot_vs_goal_json(result: ShotGoalResult, path: str | Path) -> Path:
    """Escribe el resultado a JSON (resumen + eventos)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "eventos": [
            {
                "tipo": e.tipo, "zona": e.zona,
                "frame_inicio": e.frame_inicio, "frame_fin": e.frame_fin,
                "dur_frames": e.dur_frames, "dur_s": e.dur_s,
                "xy_cm": list(e.xy_cm) if e.xy_cm is not None else None,
            }
            for e in result.eventos
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
