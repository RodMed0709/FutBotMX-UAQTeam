"""Nucleo de segmentacion por texto (tarea text_segmentation).

Define el nucleo de inferencia por-frame del MVP SAM3-only:

- ``Detection``: dataclass con (obj_id, mask, score) de una deteccion.
- ``segment_with_text(frame, prompt)``: segmenta un frame con un prompt de texto
  y devuelve una lista de ``Detection`` con mascaras booleanas a tamano del frame.
- ``detect_classes_in_frame(frame, classes)``: aplica todas las clases del
  proyecto (de la configuracion) a un frame y devuelve ``{nombre_clase: [Detection]}``.

El modelo se obtiene de ``src.core.sam3_loader.load_sam3`` (sin globals) y las
clases de ``configs/<CONFIG_FILENAME>``. Es modo **por-frame**: cada frame/clase
se segmenta por separado; la identidad estable de objetos entre frames es de la
tarea ``video_tracking``.

``torch``, ``cv2`` y ``PIL`` se importan de forma **perezosa** dentro de las
funciones para que ``import src.core`` no los arrastre.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.core.sam3_loader import Sam3Bundle, load_sam3
from src.utils import PROJECT_ROOT, get_abs_path


@dataclass
class Detection:
    """Una deteccion de SAM3 sobre un frame.

    Attributes:
        obj_id: identificador del objeto dentro de la sesion SAM3 (por-sesion; no
            es una identidad estable entre clases ni entre frames).
        mask: mascara booleana con forma ``(H, W)`` del frame de entrada.
        score: score de la deteccion.
    """

    obj_id: int
    mask: np.ndarray
    score: float


def _validate_frame(frame: np.ndarray) -> tuple[int, int]:
    """Valida el frame de entrada y devuelve ``(H, W)``.

    Raises:
        ValueError: si ``frame`` no es un ``np.ndarray`` 3D ``(H, W, 3)``.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError(
            f"Se esperaba un frame numpy.ndarray, se recibio: {type(frame).__name__}"
        )
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            f"Se esperaba un frame con forma (H, W, 3), se recibio: {frame.shape}"
        )
    return frame.shape[0], frame.shape[1]


def _mask_from_logits(logits: np.ndarray, w: int, h: int) -> np.ndarray:
    """Logits del modelo -> upscale BILINEAR a (h, w) -> umbral -> mascara bool.

    El upscale bilinear sobre los logits (antes del umbral) da un borde suave
    sub-pixel, a diferencia de escalar una mascara ya booleana.
    """
    import cv2

    lo = logits.astype(np.float32)
    if lo.shape != (h, w):
        lo = cv2.resize(lo, (w, h), interpolation=cv2.INTER_LINEAR)
    return lo > 0.0


def _load_classes() -> list[dict]:
    """Lee la lista de clases desde el archivo de configuracion del proyecto.

    El nombre del archivo se toma de CONFIG_FILENAME en el .env; las clases se
    leen de la clave ``classes``.

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si falta la clave ``classes`` en la configuracion.
        FileNotFoundError: si el archivo de configuracion no existe.
    """
    env_path = PROJECT_ROOT / ".env"
    config_filename = None
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "CONFIG_FILENAME":
                config_filename = value.strip()
                break
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")

    config_path = get_abs_path(f"configs/{config_filename}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "classes" not in config:
        raise KeyError("Falta la clave 'classes' en el archivo de configuracion.")
    return config["classes"]


def segment_with_text(
    frame: np.ndarray,
    prompt: str,
    bundle: Sam3Bundle | None = None,
) -> list[Detection]:
    """Segmenta un frame con un prompt de texto via SAM3.

    Args:
        frame: frame RGB como ``np.ndarray`` con forma ``(H, W, 3)``.
        prompt: prompt de texto para SAM3.
        bundle: modelo cargado (``Sam3Bundle``). Si es ``None`` se obtiene con
            ``load_sam3()``.

    Returns:
        Lista de ``Detection`` (vacia si no hay objetos). Cada mascara es booleana
        con forma ``(H, W)`` del frame.

    Raises:
        ValueError: si ``frame`` no tiene forma ``(H, W, 3)``.
    """
    import torch
    from PIL import Image

    h, w = _validate_frame(frame)
    bundle = bundle or load_sam3()
    img = Image.fromarray(frame)

    with torch.no_grad():
        session = bundle.processor.init_video_session(
            video=[img],
            inference_device=bundle.device,
            dtype=torch.bfloat16,
        )
        session = bundle.processor.add_text_prompt(session, text=prompt)
        out = bundle.model(inference_session=session, frame_idx=0)

        detections: list[Detection] = []
        for oid in out.object_ids:
            m = out.obj_id_to_mask[oid].detach().cpu().float().numpy()
            if m.ndim == 4:
                m = m[0, 0]
            elif m.ndim == 3:
                m = m[0]
            detections.append(
                Detection(
                    obj_id=int(oid),
                    mask=_mask_from_logits(m, w, h),
                    score=float(out.obj_id_to_score.get(oid, 0.0)),
                )
            )
    return detections


def detect_classes_in_frame(
    frame: np.ndarray,
    classes: list[dict] | None = None,
    bundle: Sam3Bundle | None = None,
) -> dict[str, list[Detection]]:
    """Aplica todas las clases del proyecto a un frame.

    Args:
        frame: frame RGB como ``np.ndarray`` con forma ``(H, W, 3)``.
        classes: lista de clases (cada una con ``name`` y ``sam3_prompts``). Si es
            ``None`` se leen de la configuracion.
        bundle: modelo cargado. Si es ``None`` se obtiene con ``load_sam3()`` una
            sola vez y se reutiliza para todas las clases.

    Returns:
        Diccionario ``{nombre_clase: [Detection, ...]}`` usando el prompt activo
        (``sam3_prompts[0]``) de cada clase.
    """
    bundle = bundle or load_sam3()
    classes = classes if classes is not None else _load_classes()

    result: dict[str, list[Detection]] = {}
    for cls in classes:
        prompt = cls["sam3_prompts"][0]
        result[cls["name"]] = segment_with_text(frame, prompt, bundle)
    return result
