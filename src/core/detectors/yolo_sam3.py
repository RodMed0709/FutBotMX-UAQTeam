"""Detector YOLO + SAM3 box-prompt como estrategia inyectable (tarea detector_strategy).

Composer SAM3-céntrico: para las clases con ``yolo_id`` localiza con **YOLO**
(``detect_boxes``) y convierte cada caja en máscara fina con **SAM3 box-prompt**
(``boxes_to_masks``); las clases **sin** ``yolo_id`` (``green_floor``) se segmentan
por **text-prompt** como hoy (``segment_with_text``). SAM3 sigue siendo el centro;
YOLO solo acelera la localización.

Cumple el contrato de detector del tracking
(``detect(frame, classes=None, bundle=None) -> {nombre: [Detection]}``), así que
``track_video`` lo usa sin cambiar el resto del bucle: ByteTrack le pone el
``obj_id`` estable, y el JSON/overlay se reutilizan.

YOLO se carga internamente vía ``load_yolo`` (cacheado); el box-prompt y green_floor
usan el ``bundle`` SAM3 recibido. ``torch``/``ultralytics`` siguen siendo imports
perezosos (dentro de ``detect_boxes``/``boxes_to_masks``).
"""

from __future__ import annotations

import numpy as np

from src.core.detectors.box_prompt import boxes_to_masks
from src.core.detectors.yolo_boxes import detect_boxes
from src.core.sam3_loader import Sam3Bundle, load_sam3
from src.core.segmentation import Detection, _load_classes, segment_with_text


def detect(
    frame: np.ndarray,
    classes: list[dict] | None = None,
    bundle: Sam3Bundle | None = None,
) -> dict[str, list[Detection]]:
    """Detector YOLO+SAM3: cajas YOLO -> máscaras box-prompt; green_floor por texto.

    Args:
        frame: imagen ``(H, W, 3)`` RGB.
        classes: clases del repo. ``None`` ⇒ las de la configuración. Las que tengan
            ``yolo_id`` van por YOLO→box-prompt; el resto por text-prompt.
        bundle: SAM3 cargado. ``None`` ⇒ ``load_sam3()``.

    Returns:
        ``{nombre_clase: [Detection con máscara]}`` (misma forma que el detector
        SAM3-text), listo para alimentar al tracking.
    """
    classes = classes if classes is not None else _load_classes()
    bundle = bundle or load_sam3()

    yolo_classes = [c for c in classes if "yolo_id" in c]
    text_classes = [c for c in classes if "yolo_id" not in c]

    result: dict[str, list[Detection]] = {}

    # Clases YOLO: cajas (detect_boxes) -> máscaras finas (SAM3 box-prompt).
    if yolo_classes:
        boxes_by_class = detect_boxes(frame, classes=yolo_classes)
        for cls in yolo_classes:
            name = cls["name"]
            bds = boxes_by_class.get(name, [])
            boxes = [bd.bbox for bd in bds]
            scores = [bd.score for bd in bds]
            result[name] = boxes_to_masks(frame, boxes, bundle=bundle, scores=scores)

    # Clases sin yolo_id (green_floor): text-prompt como hoy.
    for cls in text_classes:
        result[cls["name"]] = segment_with_text(frame, cls["sam3_prompts"][0], bundle)

    return result
