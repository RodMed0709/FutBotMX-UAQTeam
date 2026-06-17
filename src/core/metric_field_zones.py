"""T6 (fase_5 · Capa B) — zonas del campo en cm: presencia y posesión.

Divide la cancha canónica (`field_template`) en **zonas en cm** y mide **presencia** (tiempo de
balón/robots por zona) y **posesión por zona** (combinando T1 `compute_possession` con el balón
en cm de T3 `metric_positions`). Resuelve el "medio campo real" que en píxeles era impreciso
(`W//2` partía la imagen sin perspectiva). Solo cámara superior; CPU local.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.core import field_template as ft
from src.core.events import compute_possession
from src.core.events_core import BALL_CLASSES, load_frame_objects
from src.core.metric_positions import MetricResult, compute_metric_positions

_L = ft.LENGTH_CM


def _mitades(x: float, _y: float) -> str:
    return "amarilla" if x < _L / 2.0 else "azul"


def _tercios(x: float, _y: float) -> str:
    if x < _L / 3.0:
        return "amarillo"
    if x < 2.0 * _L / 3.0:
        return "medio"
    return "azul"


# Esquema = (labels ordenadas, fn(x, y) -> label). Añadir uno = añadir entrada (config-driven).
_SCHEMES: dict[str, tuple[tuple[str, ...], Callable[[float, float], str]]] = {
    "mitades": (("amarilla", "azul"), _mitades),
    "tercios": (("amarillo", "medio", "azul"), _tercios),
}


@dataclass
class FieldZonesResult:
    por_esquema: dict[str, dict]  # esquema -> {"presencia": {...}, "posesion": {...}}
    resumen: dict


def _clip(xy: tuple[float, float]) -> tuple[float, float]:
    x = min(max(xy[0], 0.0), _L - 1e-6)
    y = min(max(xy[1], 0.0), ft.WIDTH_CM - 1e-6)
    return (x, y)


def _pct(counter: Counter, labels: tuple[str, ...]) -> dict[str, float]:
    total = sum(counter.values())
    return {z: (round(100.0 * counter.get(z, 0) / total, 1) if total else 0.0) for z in labels}


def _presence(metric: MetricResult, labels, fn) -> dict[str, dict[str, float]]:
    cnt = {"ball": Counter(), "robot": Counter()}
    for p in metric.posiciones:
        if p.xy_cm is None:
            continue
        cat = "ball" if p.cls in BALL_CLASSES else ("robot" if p.cls == "robot" else None)
        if cat is None:
            continue
        x, y = _clip(p.xy_cm)
        cnt[cat][fn(x, y)] += 1
    return {cat: _pct(cnt[cat], labels) for cat in ("ball", "robot")}


def _ball_cm_by_frame(metric: MetricResult) -> dict[int, tuple[float, float]]:
    """Primera muestra del balón (cm) por frame."""
    out: dict[int, tuple[float, float]] = {}
    for p in metric.posiciones:
        if p.cls in BALL_CLASSES and p.xy_cm is not None and p.frame_index not in out:
            out[p.frame_index] = p.xy_cm
    return out


def _possession_by_zone(possession, ball_cm, labels, fn) -> tuple[dict[str, float], int]:
    cnt: Counter = Counter()
    used = 0
    for f, owner in possession.por_frame.items():
        if owner is None or f not in ball_cm:
            continue
        x, y = _clip(ball_cm[f])
        cnt[fn(x, y)] += 1
        used += 1
    return _pct(cnt, labels), used


def compute_field_zones(
    tracks_json: str | Path,
    *,
    schemes: tuple[str, ...] = ("mitades", "tercios"),
    fps: float | None = None,
    metric: MetricResult | None = None,
) -> FieldZonesResult:
    """Presencia y posesión por zona. Combina T3 (cm) y T1 (posesión)."""
    for s in schemes:
        if s not in _SCHEMES:
            raise ValueError(f"esquema desconocido: {s!r} (válidos: {list(_SCHEMES)})")
    tracks_json = Path(tracks_json)
    metric = metric or compute_metric_positions(tracks_json)
    fps = fps or metric.resumen.get("fps")

    by_frame = load_frame_objects(tracks_json)
    possession = compute_possession(by_frame, fps=fps)
    ball_cm = _ball_cm_by_frame(metric)

    por_esquema: dict[str, dict] = {}
    pos_frames_used: dict[str, int] = {}
    for s in schemes:
        labels, fn = _SCHEMES[s]
        posesion, used = _possession_by_zone(possession, ball_cm, labels, fn)
        por_esquema[s] = {"presencia": _presence(metric, labels, fn), "posesion": posesion}
        pos_frames_used[s] = used

    resumen = {
        "fps": fps,
        "esquemas": list(schemes),
        "frames_posesion_usados": pos_frames_used,
        "nota": "zonas en cm (cámara superior); cifras indicativas (tracking/H)",
    }
    return FieldZonesResult(por_esquema=por_esquema, resumen=resumen)


def render_zones(
    scheme: str,
    presence_ball: dict[str, float],
    posesion: dict[str, float],
    *,
    scale: float = 2.6,
    margin_cm: float = 10.0,
) -> "object":
    """Cancha con fronteras del esquema + % de presencia(balón)/posesión por zona (BGR)."""
    import cv2

    labels, _fn = _SCHEMES[scheme]
    canvas, to_px = ft.render_field(scale=scale, margin_cm=margin_cm)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    n = len(labels)
    for i in range(1, n):  # fronteras verticales entre zonas (en x)
        xb = _L * i / n
        cv2.line(canvas, to_px((xb, 0.0)), to_px((xb, ft.WIDTH_CM)), (0, 0, 255), 2)
    for i, z in enumerate(labels):
        cx = _L * (i + 0.5) / n
        px = to_px((cx, 18.0))
        cv2.putText(canvas, f"{z}", (px[0] - 30, px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        py = to_px((cx, ft.WIDTH_CM - 14.0))
        cv2.putText(canvas, f"bal {presence_ball.get(z, 0):.0f}% pos {posesion.get(z, 0):.0f}%",
                    (py[0] - 55, py[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return canvas


def write_zones_png(result: FieldZonesResult, scheme: str, path: str | Path) -> Path:
    """Renderiza y escribe el PNG de un esquema."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result.por_esquema[scheme]
    img = render_zones(scheme, data["presencia"]["ball"], data["posesion"])
    cv2.imwrite(str(path), img)
    return path


def write_field_zones_json(result: FieldZonesResult, path: str | Path) -> Path:
    """Persiste el resultado a JSON."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"resumen": result.resumen, "por_esquema": result.por_esquema}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
