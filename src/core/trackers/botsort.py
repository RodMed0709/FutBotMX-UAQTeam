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

def _botsort_yaml_path():
    """Ruta del ``botsort.yaml`` que trae la versión instalada de ultralytics.

    Usa el resolvedor oficial (``check_yaml``) y, si su API cambió, cae a la ruta
    dentro del paquete. Evita depender de utilidades de ultralytics que se renombran
    entre versiones (p. ej. ``yaml_load``).
    """
    from pathlib import Path

    try:
        from ultralytics.utils.checks import check_yaml

        return Path(check_yaml("botsort.yaml"))
    except Exception:  # noqa: BLE001 - fallback a la ruta del paquete
        import ultralytics

        return (
            Path(ultralytics.__file__).resolve().parent
            / "cfg"
            / "trackers"
            / "botsort.yaml"
        )


def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """Convierte cajas xyxy a xywh (centro + ancho/alto)."""
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    cx = xyxy[:, 0] + w / 2.0
    cy = xyxy[:, 1] + h / 2.0
    return np.stack([cx, cy, w, h], axis=1)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz IoU (M,N) entre cajas xyxy ``a`` (M,4) y ``b`` (N,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


class _Det:
    """Objeto "results" que ``BOTSORT.update`` consume.

    La API de tracking de ultralytics exige que ``results`` exponga ``.conf``,
    ``.cls``, ``.xywh`` y ``.xyxy`` (np.ndarray) y que sea **indexable** por máscara
    booleana (``results[mask]`` -> subconjunto del mismo tipo) y soporte ``len()``.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        self.xywh = _xyxy_to_xywh(self.xyxy)

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, index) -> "_Det":
        return _Det(self.xyxy[index], self.conf[index], self.cls[index])


class BotSortTracker:
    """BoT-SORT (ultralytics) tras la interfaz común de tracker del proyecto."""

    def __init__(self, frame_rate: float, config: dict) -> None:
        import inspect
        from types import SimpleNamespace

        import yaml
        from ultralytics.trackers import BOTSORT

        # Base: el botsort.yaml que trae la version INSTALADA de ultralytics, para que
        # 'args' tenga TODOS los campos que esa version espera (model, with_reid, etc.,
        # que cambian entre versiones). Encima se aplican los overrides de la config
        # del proyecto (gmc_method, thresholds, with_reid=false, ...).
        base = yaml.safe_load(_botsort_yaml_path().read_text(encoding="utf-8"))
        base.update(config or {})
        args = SimpleNamespace(**base)
        fr = int(round(frame_rate)) or 30
        # La firma de BOTSORT varia entre versiones de ultralytics: algunas aceptan
        # frame_rate, otras lo derivan de args. Se pasa solo si existe el parametro.
        if "frame_rate" in inspect.signature(BOTSORT.__init__).parameters:
            self._tracker = BOTSORT(args, frame_rate=fr)
        else:
            self._tracker = BOTSORT(args)

        # El buffer de tracks (max_time_lost) depende del fps: track_buffer esta en
        # frames y debe escalar con el fps real para que la ventana de "coasting" sea
        # constante EN TIEMPO (1 s a 30 o 60 fps). Se recalcula aqui para no depender
        # de como cada version de ultralytics maneje frame_rate (p. ej. asumir 30).
        buffer = int(round(fr / 30.0 * args.track_buffer))
        for attr in ("max_time_lost", "buffer_size"):
            if hasattr(self._tracker, attr):
                setattr(self._tracker, attr, buffer)

    def update(self, detections, frame: np.ndarray):
        """Asocia ``detections`` (sv.Detections) con BoT-SORT y devuelve los tracks.

        Devuelve ``sv.Detections`` con ``tracker_id`` y ``data["src"]`` preservado. El
        ``src`` se recupera mapeando cada track de salida a su detección de entrada por
        **IoU** (la columna ``idx`` de ultralytics es relativa al subconjunto filtrado
        por umbral, no al índice original, así que no sirve para el mapeo).
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
        out_xyxy = out[:, :4]
        src_in = detections.data.get("src") if detections.data else None

        # Mapear cada track de salida a la deteccion de entrada por mayor IoU. Los
        # tracks confirmados de este frame coinciden con una caja de entrada; los que
        # no superan un IoU minimo (coasting/KF puro) se descartan (no hay mascara que
        # asociar este frame).
        iou = _iou_matrix(out_xyxy, xyxy)
        best = iou.argmax(axis=1) if n else np.zeros(len(out), dtype=int)
        best_iou = iou[np.arange(len(out)), best] if n else np.zeros(len(out))
        keep = best_iou >= 0.1
        if not keep.any():
            return sv.Detections.empty()

        out = out[keep]
        best = best[keep]
        src = np.asarray(src_in)[best] if src_in is not None else best
        return sv.Detections(
            xyxy=out[:, :4].astype(float),
            confidence=out[:, 5].astype(float),
            tracker_id=out[:, 4].astype(int),
            data={"src": np.asarray(src)},
        )


def make_botsort(frame_rate: float, config: dict) -> BotSortTracker:
    """Crea un ``BotSortTracker`` con la sección ``botsort`` del config."""
    return BotSortTracker(frame_rate=frame_rate, config=config)
