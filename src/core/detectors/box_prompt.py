"""Segmentacion por caja con SAM3 (tarea sam3_box_prompt).

Building block **box-prompt**: dado un frame y un conjunto de **cajas**, usa la 2a
cara de SAM3 (``Sam3TrackerModel``, ``input_boxes``) para producir una **mascara
fina por caja**, empaquetada en la moneda comun ``Detection`` y alineada **1:1** con
las cajas de entrada.

Es el corazon del pipeline SAM3-centrico: un detector rapido (YOLO, tarea posterior)
solo tiene que **localizar** con cajas y SAM3 -que sigue siendo el centro- convierte
cada caja en mascara, sin recorrer toda la imagen con text-prompt. La caja no necesita
ser perfecta; SAM3 hace la mascara buena.

Sustituye el ``boxes_to_masks`` suelto y ad-hoc de
``notebooks/fase_2_YOLO_SAM3/pipeline_yolo_sam3.py`` por una pieza config-driven que
reutiliza ``sam3_loader`` (carga) y ``Detection`` (moneda comun).

``torch`` y ``PIL`` se importan de forma **perezosa** dentro de la funcion, para que
``import src.core`` no los arrastre.

Nota sobre ``obj_id``: en este building block es **posicional** (indice de la caja en
el frame), por tanto **inestable** entre frames. La identidad estable la asigna el
tracker en una tarea posterior.
"""

from __future__ import annotations

import numpy as np

from src.core.sam3_loader import Sam3Bundle, ensure_tracker_loaded, load_sam3
from src.core.segmentation import Detection


def boxes_to_masks(
    frame: np.ndarray,
    boxes: list[tuple[float, float, float, float]] | list[list[float]],
    bundle: Sam3Bundle | None = None,
    scores: list[float] | None = None,
) -> list[Detection]:
    """Convierte cajas en mascaras finas via SAM3 box-prompt.

    Args:
        frame: imagen ``(H, W, 3)`` RGB (convencion del repo).
        boxes: lista de cajas **xyxy en pixeles absolutos** (``[x1, y1, x2, y2]``).
        bundle: SAM3 ya cargado. Si es ``None`` se obtiene con ``load_sam3()``. En
            ambos casos se garantiza la 2a cara con ``ensure_tracker_loaded``.
        scores: score por caja (el del detector que las produjo). Si es ``None``, cada
            ``Detection`` recibe ``score=1.0``.

    Returns:
        ``list[Detection]`` con un elemento por caja, **1:1 y en el mismo orden**.
        Cada ``Detection`` lleva la mascara booleana ``(H, W)``, el ``score``
        correspondiente y un ``obj_id`` posicional (per-frame, inestable).

    Notas:
        - Con ``boxes`` vacio devuelve ``[]`` **sin** invocar al modelo.
        - Las mascaras vacias/degeneradas **no** se filtran (se preserva el 1:1); su
          descarte es responsabilidad del consumidor (p. ej. ``inference_schema``).
    """
    # Caso vacio: sin cajas no hay nada que segmentar (no se carga ni invoca al
    # modelo, ni siquiera torch).
    if not boxes:
        return []

    import torch
    from PIL import Image

    bundle = bundle or load_sam3()
    ensure_tracker_loaded(bundle)

    box_list = [list(b) for b in boxes]

    with torch.no_grad():
        img = Image.fromarray(frame)
        inp = bundle.tracker_processor(
            images=[img], input_boxes=[box_list], return_tensors="pt"
        ).to(bundle.device)
        # Castear a bfloat16 SOLO los tensores flotantes (no indices/cajas enteras).
        inp2 = {
            k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
            for k, v in inp.items()
        }
        out = bundle.tracker_model(**inp2, multimask_output=False)
        masks = bundle.tracker_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"]
        )

    m = np.array(masks[0])
    if m.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
        m = m[:, 0]
    m = m.astype(bool)

    detections: list[Detection] = []
    for i, mask in enumerate(m):
        score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        detections.append(Detection(obj_id=i, mask=mask, score=score))
    return detections
