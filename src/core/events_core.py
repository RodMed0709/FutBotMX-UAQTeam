"""Base compartida del análisis de eventos del partido (fase_5).

Carga el JSON de tracking a una estructura **por frame**, reutilizada por las tareas de
eventos (posesión, zona de gol, …). Capa A (en píxeles), universal: sin GPU, sin homografía.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROBOT_CLASS = "robot"
BALL_CLASSES = {"orange_ball", "ball"}


@dataclass
class FrameObject:
    """Un objeto detectado en un frame (cualquier clase)."""

    obj_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # [x, y, w, h]
    centroid: tuple[float, float]            # [x, y]
    score: float


def load_frame_objects(tracks_json: str | Path) -> dict[int, list[FrameObject]]:
    """Invierte el JSON de tracking a una estructura **por frame**.

    Returns:
        ``{frame_index: [FrameObject, ...]}`` con todas las clases. Base reusable por las
        tareas de eventos.

    Raises:
        ValueError: si ``bbox`` no es ``[x,y,w,h]`` o ``centroid`` no es ``[x,y]``.
    """
    data = json.loads(Path(tracks_json).read_text(encoding="utf-8"))
    by_frame: dict[int, list[FrameObject]] = {}
    for tr in data.get("tracks", []):
        cls = tr["class"]
        oid = int(tr["obj_id"])
        for obs in tr["observations"]:
            bbox = obs["bbox"]
            centroid = obs["centroid"]
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                raise ValueError(f"bbox debe ser [x,y,w,h]; se recibió: {bbox!r}")
            if not (isinstance(centroid, (list, tuple)) and len(centroid) == 2):
                raise ValueError(f"centroid debe ser [x,y]; se recibió: {centroid!r}")
            by_frame.setdefault(int(obs["frame_index"]), []).append(
                FrameObject(
                    obj_id=oid,
                    class_name=cls,
                    bbox=tuple(float(v) for v in bbox),
                    centroid=(float(centroid[0]), float(centroid[1])),
                    score=float(obs.get("score", 0.0)),
                )
            )
    return by_frame


def ball_centroid(objs: list[FrameObject]) -> tuple[float, float] | None:
    """Centroide del balón (clase de ``BALL_CLASSES`` con mayor score) o ``None``."""
    balls = [o for o in objs if o.class_name in BALL_CLASSES]
    if not balls:
        return None
    return max(balls, key=lambda o: o.score).centroid
