"""Adaptador ByteTrack (tarea botsort_tracker).

Envuelve el ``ByteTrackTracker`` de Roboflow (paquete ``trackers``) tras el factory
común de ``src/core/trackers``. Es el tracker **actual** del proyecto: este módulo
solo lo expone con la misma firma que el resto de trackers, **sin cambiar su
comportamiento** (mismos kwargs de ``tracking.*``), para garantizar la no-regresión.

``trackers`` se importa de forma **perezosa**.
"""

from __future__ import annotations

from typing import Any


def make_bytetrack(frame_rate: float, kwargs: dict) -> Any:
    """Crea un ``ByteTrackTracker`` listo para ``.update(detections, frame)``.

    Args:
        frame_rate: fps del video (para el buffer de tracks de ByteTrack).
        kwargs: parámetros de ByteTrack (de la sección ``tracking`` del config).

    Returns:
        Una instancia de ``ByteTrackTracker`` (interfaz ``.update`` ya compatible).
    """
    from trackers import ByteTrackTracker

    return ByteTrackTracker(frame_rate=frame_rate, **kwargs)
