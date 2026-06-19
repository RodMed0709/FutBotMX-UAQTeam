"""Detector SAM3-text como estrategia inyectable (tarea detector_strategy).

Adaptador **delgado** que expone el detector por **texto** existente
(``segmentation.detect_classes_in_frame``) bajo el contrato de detector del
tracking: ``detect(frame, classes=None, bundle=None) -> {nombre: [Detection]}``.

La implementación canónica **se queda** en ``segmentation.py`` (no se mueve, para no
romper imports existentes); aquí solo se le da un punto de entrada con el nombre del
detector, para registrarlo junto a ``yolo_sam3``.
"""

from __future__ import annotations

import numpy as np

from src.core.sam3_loader import Sam3Bundle
from src.core.segmentation import Detection, detect_classes_in_frame


def detect(
    frame: np.ndarray,
    classes: list[dict] | None = None,
    bundle: Sam3Bundle | None = None,
) -> dict[str, list[Detection]]:
    """Detector SAM3-text: una sesión de texto por clase (camino actual)."""
    return detect_classes_in_frame(frame, classes=classes, bundle=bundle)
