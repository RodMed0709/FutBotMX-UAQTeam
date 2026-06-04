"""Submodulo core del proyecto FutBotMX.

Contiene la logica central del pipeline. Piezas actuales: extraccion de frames de
un video (tarea frame_extraction), carga del modelo SAM3 (tarea sam3_loader) y
segmentacion por texto (tarea text_segmentation).
"""

from __future__ import annotations

from src.core.frame_extraction import extract_frames
from src.core.sam3_loader import Sam3Bundle, load_sam3
from src.core.segmentation import (
    Detection,
    detect_classes_in_frame,
    segment_with_text,
)

__all__ = [
    "extract_frames",
    "load_sam3",
    "Sam3Bundle",
    "Detection",
    "segment_with_text",
    "detect_classes_in_frame",
]
