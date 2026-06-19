"""Eventos de "balón en zona de gol" sobre el JSON de tracking (fase_5, Capa A · T2).

Capa **relacional (en píxeles), UNIVERSAL**: compara el centroide del balón con el bbox de
las zonas de gol (``yellow_zone``/``blue_zone``) en el mismo frame, así que funciona sobre
el JSON de **cualquier** video sin homografía ni GPU.

``compute_goal_zone_events`` detecta **candidatos a gol** como entradas sostenidas del balón
en una zona, con debounce (apertura), cierre por salida sostenida y cooldown (evita doble
conteo del mismo lance). Es un proxy en píxeles: el gol geométrico real (línea de gol en cm)
es Capa B (cámara superior + homografía).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.core.events_core import FrameObject, ball_centroid

YELLOW_ZONE = "yellow_zone"
BLUE_ZONE = "blue_zone"
# Clase del JSON -> etiqueta corta del evento.
_ZONE_LABEL = {YELLOW_ZONE: "yellow", BLUE_ZONE: "blue"}

# Defaults configurables por parámetro.
DEFAULT_MARGIN = 0.0       # holgura (px) al test punto-en-bbox
DEFAULT_MIN_FRAMES = 3     # frames dentro para ABRIR un evento
DEFAULT_EXIT_FRAMES = 3    # frames fuera para CERRAR un evento
DEFAULT_COOLDOWN = 15      # frames de refractario tras cerrar (no recontar el mismo lance)


@dataclass
class GoalEvent:
    """Un candidato a gol: entrada sostenida del balón en una zona."""

    zona: str            # "yellow" | "blue"
    frame_inicio: int
    frame_fin: int
    dur_frames: int
    dur_s: float | None


@dataclass
class GoalZoneResult:
    """Resultado de ``compute_goal_zone_events``."""

    eventos: list[GoalEvent]
    resumen: dict


def _point_in_bbox(pt: tuple[float, float], bbox, margin: float) -> bool:
    """¿El punto cae dentro del bbox ``[x,y,w,h]`` (± margen)?"""
    x, y, w, h = bbox
    return (x - margin) <= pt[0] <= (x + w + margin) and (y - margin) <= pt[1] <= (y + h + margin)


def _ball_in_zone(ball_xy, objs: list[FrameObject], zone_class: str, margin: float) -> bool:
    """¿El balón cae dentro de **cualquier** track de ``zone_class`` ese frame?"""
    return any(
        _point_in_bbox(ball_xy, o.bbox, margin)
        for o in objs
        if o.class_name == zone_class
    )


def _zones_present(by_frame: dict[int, list[FrameObject]]) -> list[str]:
    """Zonas de gol (clases) que aparecen en el JSON (puede faltar la azul)."""
    seen = {o.class_name for objs in by_frame.values() for o in objs}
    return [z for z in (YELLOW_ZONE, BLUE_ZONE) if z in seen]


def _events_from_series(
    serie: list[tuple[int, bool]],
    zona: str,
    *,
    min_frames: int,
    exit_frames: int,
    cooldown: int,
    fps: float | None,
) -> list[GoalEvent]:
    """Extrae eventos de una serie ``[(frame_index, dentro)]`` con debounce/cierre/cooldown."""
    events: list[GoalEvent] = []
    in_event = False
    start: int | None = None
    last_inside: int | None = None
    inside_streak = 0
    outside_streak = 0
    cooldown_left = 0

    for f, inside in serie:
        if cooldown_left > 0 and not in_event:
            cooldown_left -= 1
        if not in_event:
            inside_streak = inside_streak + 1 if inside else 0
            if inside_streak >= min_frames and cooldown_left == 0:
                in_event = True
                start = f - min_frames + 1  # primer frame dentro de la racha
                last_inside = f
                outside_streak = 0
        else:
            if inside:
                last_inside = f
                outside_streak = 0
            else:
                outside_streak += 1
                if outside_streak >= exit_frames:
                    dur = last_inside - start + 1
                    events.append(
                        GoalEvent(
                            zona=zona,
                            frame_inicio=start,
                            frame_fin=last_inside,
                            dur_frames=dur,
                            dur_s=round(dur / fps, 2) if fps else None,
                        )
                    )
                    in_event = False
                    inside_streak = 0
                    cooldown_left = cooldown

    if in_event and start is not None and last_inside is not None:  # evento abierto al final
        dur = last_inside - start + 1
        events.append(
            GoalEvent(
                zona=zona,
                frame_inicio=start,
                frame_fin=last_inside,
                dur_frames=dur,
                dur_s=round(dur / fps, 2) if fps else None,
            )
        )
    return events


def compute_goal_zone_events(
    by_frame: dict[int, list[FrameObject]],
    *,
    margin: float = DEFAULT_MARGIN,
    min_frames: int = DEFAULT_MIN_FRAMES,
    exit_frames: int = DEFAULT_EXIT_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN,
    fps: float | None = None,
) -> GoalZoneResult:
    """Detecta candidatos a gol (balón en zona) como eventos discretos.

    Args:
        by_frame: salida de ``load_frame_objects``.
        margin: holgura (px) al test punto-en-bbox de la zona.
        min_frames: frames dentro para abrir un evento (debounce).
        exit_frames: frames fuera para cerrar un evento.
        cooldown_frames: refractario tras cerrar (no recontar el mismo lance).
        fps: para reportar duraciones en segundos (``None`` ⇒ solo frames).

    Returns:
        ``GoalZoneResult`` con ``eventos`` (ordenados por frame) y ``resumen``.
    """
    frames = sorted(by_frame)
    zones = _zones_present(by_frame)

    eventos: list[GoalEvent] = []
    for zone_class in zones:
        serie = [
            (f, _ball_in_zone(ball_centroid(by_frame[f]), by_frame[f], zone_class, margin)
             if ball_centroid(by_frame[f]) is not None else False)
            for f in frames
        ]
        eventos += _events_from_series(
            serie, _ZONE_LABEL[zone_class],
            min_frames=min_frames, exit_frames=exit_frames, cooldown=cooldown_frames, fps=fps,
        )

    eventos.sort(key=lambda e: e.frame_inicio)
    por_zona: dict[str, int] = {}
    for e in eventos:
        por_zona[e.zona] = por_zona.get(e.zona, 0) + 1

    resumen = {
        "n_frames": len(frames),
        "zonas_presentes": [_ZONE_LABEL[z] for z in zones],
        "eventos_por_zona": por_zona,
        "total_eventos": len(eventos),
        "fps": fps,
    }
    return GoalZoneResult(eventos=eventos, resumen=resumen)


def write_goal_events_json(result: GoalZoneResult, path: str | Path) -> Path:
    """Persiste el resultado a un JSON (``resumen`` + ``eventos``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "eventos": [vars(e) for e in result.eventos],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
