"""Trackers intercambiables del tracking (subpaquete trackers, tarea botsort_tracker).

Cada tracker expone la misma interfaz que ``track_video`` consume:
``.update(detections, frame) -> detections`` con ``tracker_id`` y ``data["src"]``
preservado. Hay dos: ``bytetrack`` (el actual, Roboflow) y ``botsort`` (BoT-SORT de
ultralytics, con GMC). El factory ``get_tracker`` los resuelve por nombre.

Importar este subpaquete es **barato**: ``ultralytics``/``supervision``/``trackers``
se importan dentro de los ``make_*``/``update``, no a nivel de módulo.
"""

from __future__ import annotations

from typing import Any

KNOWN_TRACKERS = ("bytetrack", "botsort")


def get_tracker(
    name: str,
    frame_rate: float,
    *,
    bytetrack_kwargs: dict | None = None,
    botsort_config: dict | None = None,
) -> Any:
    """Resuelve y construye un tracker por nombre.

    Args:
        name: ``"bytetrack"`` (Roboflow, actual) o ``"botsort"`` (ultralytics + GMC).
        frame_rate: fps del video (buffer de tracks).
        bytetrack_kwargs: parámetros de ByteTrack (sección ``tracking``).
        botsort_config: parámetros de BoT-SORT (sección ``botsort``).

    Returns:
        Un tracker con interfaz ``.update(detections, frame)``.

    Raises:
        ValueError: si ``name`` no está en ``KNOWN_TRACKERS``.
    """
    if name == "bytetrack":
        from src.core.trackers.bytetrack import make_bytetrack

        return make_bytetrack(frame_rate, bytetrack_kwargs or {})
    if name == "botsort":
        from src.core.trackers.botsort import make_botsort

        return make_botsort(frame_rate, botsort_config or {})
    raise ValueError(
        f"tracker '{name}' no soportado (usa uno de {list(KNOWN_TRACKERS)})."
    )


__all__ = ["KNOWN_TRACKERS", "get_tracker"]
