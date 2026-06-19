"""Detectores/segmentadores intercambiables del pipeline (subpaquete detectors).

Cada modulo aqui es una **primitiva de inferencia** que produce ``Detection`` a
partir de un frame: segmentacion por texto (SAM3), por caja (SAM3 box-prompt) o
deteccion rapida (YOLO, tarea posterior). Estas piezas se componen en el tracking sin
reimplementar el bucle.

Piezas: ``boxes_to_masks`` (box-prompt, tarea sam3_box_prompt); el detector de cajas
YOLO ``detect_boxes`` / ``load_yolo`` / ``BoxDetection`` (tarea yolo_detector); y los
detectores **inyectables** del tracking con su registro ``get_detector`` (tarea
detector_strategy).

Un detector es un callable ``detect(frame, classes=None, bundle=None) ->
{nombre: [Detection]}``. El registro mapea un nombre a su implementación, para
seleccionarlo por config o por parámetro en ``track_video``/``run_inference``.
"""

from __future__ import annotations

from src.core.detectors import sam3_text, yolo_sam3
from src.core.detectors.box_prompt import boxes_to_masks
from src.core.detectors.yolo_boxes import BoxDetection, detect_boxes, load_yolo

# Registro nombre -> detector (callable con el contrato del tracking).
_DETECTORS = {
    "sam3_text": sam3_text.detect,
    "yolo_sam3": yolo_sam3.detect,
}


def get_detector(name: str):
    """Resuelve un detector por nombre.

    Args:
        name: ``"sam3_text"`` (SAM3 por texto, actual) o ``"yolo_sam3"`` (YOLO →
            SAM3 box-prompt + green_floor por texto).

    Returns:
        El callable del detector.

    Raises:
        ValueError: si ``name`` no está registrado.
    """
    if name not in _DETECTORS:
        raise ValueError(
            f"detector '{name}' no soportado (usa uno de {sorted(_DETECTORS)})."
        )
    return _DETECTORS[name]


__all__ = [
    "boxes_to_masks",
    "BoxDetection",
    "detect_boxes",
    "load_yolo",
    "get_detector",
]
