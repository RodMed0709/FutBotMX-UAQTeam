"""Detectores/segmentadores intercambiables del pipeline (subpaquete detectors).

Cada modulo aqui es una **primitiva de inferencia** que produce ``Detection`` a
partir de un frame: segmentacion por texto (SAM3), por caja (SAM3 box-prompt) o
deteccion rapida (YOLO, tarea posterior). Estas piezas se componen en el tracking sin
reimplementar el bucle.

Pieza actual: ``boxes_to_masks`` (box-prompt, tarea sam3_box_prompt).
"""

from __future__ import annotations

from src.core.detectors.box_prompt import boxes_to_masks

__all__ = ["boxes_to_masks"]
