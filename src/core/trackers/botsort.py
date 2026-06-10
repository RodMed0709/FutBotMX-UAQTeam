"""Adaptador BoT-SORT (tarea botsort_tracker).

Envuelve ``ultralytics.trackers.BOTSORT`` (que trae **GMC**, compensación de
movimiento de cámara) tras la interfaz común de tracker del proyecto:
``.update(detections, frame) -> detections`` con ``tracker_id`` y el ``data["src"]``
preservado (índice de la detección de entrada, para recuperar la máscara aguas
abajo). Así BoT-SORT es intercambiable con ByteTrack sin tocar el bucle de
``track_video``.

``ultralytics`` y ``supervision`` se importan de forma **perezosa**. BoT-SORT viene
con ``ultralytics`` (ya usado para YOLO): no es una dependencia nueva.
"""

from __future__ import annotations

import numpy as np

# Defaults espejo de botsort.yaml de ultralytics (GMC activo, ReID desactivado).
_BOTSORT_DEFAULTS: dict = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "with_reid": False,
}


def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """Convierte cajas xyxy a xywh (centro + ancho/alto)."""
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    cx = xyxy[:, 0] + w / 2.0
    cy = xyxy[:, 1] + h / 2.0
    return np.stack([cx, cy, w, h], axis=1)


class _Det:
    """Objeto "results" mínimo que ``BOTSORT.update`` lee (conf/cls/xyxy/xywh)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        self.xywh = _xyxy_to_xywh(self.xyxy)


class BotSortTracker:
    """BoT-SORT (ultralytics) tras la interfaz común de tracker del proyecto."""

    def __init__(self, frame_rate: float, config: dict) -> None:
        import inspect
        from types import SimpleNamespace

        from ultralytics.trackers import BOTSORT

        args = SimpleNamespace(**{**_BOTSORT_DEFAULTS, **(config or {})})
        # La firma de BOTSORT varia entre versiones de ultralytics: algunas aceptan
        # frame_rate, otras lo derivan de args. Se pasa solo si existe el parametro.
        if "frame_rate" in inspect.signature(BOTSORT.__init__).parameters:
            self._tracker = BOTSORT(args, frame_rate=int(round(frame_rate)))
        else:
            self._tracker = BOTSORT(args)

    def update(self, detections, frame: np.ndarray):
        """Asocia ``detections`` (sv.Detections) con BoT-SORT y devuelve los tracks.

        Devuelve ``sv.Detections`` con ``tracker_id`` y ``data["src"]`` mapeado por la
        columna ``idx`` que emite ultralytics (índice de la detección de entrada).
        """
        import supervision as sv

        n = len(detections)
        xyxy = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        conf = (
            np.asarray(detections.confidence, dtype=np.float32).reshape(-1)
            if detections.confidence is not None
            else np.ones(n, dtype=np.float32)
        )
        # Un tracker por clase: la clase es única, se usa 0 como etiqueta.
        cls = np.zeros(n, dtype=np.float32)

        out = self._tracker.update(_Det(xyxy, conf, cls), frame)
        if out is None or len(out) == 0:
            return sv.Detections.empty()

        out = np.asarray(out, dtype=np.float32)
        # Filas: [x1, y1, x2, y2, track_id, score, cls, idx].
        idx = out[:, 7].astype(int)
        src_in = detections.data.get("src") if detections.data else None
        src = np.asarray(src_in)[idx] if src_in is not None else idx
        return sv.Detections(
            xyxy=out[:, :4].astype(float),
            confidence=out[:, 5].astype(float),
            tracker_id=out[:, 4].astype(int),
            data={"src": np.asarray(src)},
        )


def make_botsort(frame_rate: float, config: dict) -> BotSortTracker:
    """Crea un ``BotSortTracker`` con la sección ``botsort`` del config."""
    return BotSortTracker(frame_rate=frame_rate, config=config)
