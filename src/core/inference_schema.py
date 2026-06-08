"""Esquema comun del entregable de inferencia (tarea inference_schema).

Centraliza el formato del JSON que emiten los dos caminos de inferencia
(``run_pipeline`` per-frame y ``track_video`` tracking): geometria por deteccion
(mascara -> caja/centroide), codificacion **COCO-RLE** opcional de las mascaras,
cabecera con metadatos auto-descriptivos y la convencion de rutas de salida
(``outputs/inference/<stem>/``).

El **dato estructurado es el producto** del pipeline; las mascaras (COCO-RLE) son
**opcionales** (``include_masks``) porque son pesadas y solo se necesitan para
evaluacion o depuracion. RLE es una codificacion **sin perdida**: permite
reconstruir/visualizar las mascaras desde el JSON **sin volver a invocar el
modelo** (basta el JSON + el video real).

``pycocotools`` y ``cv2`` se importan de forma **perezosa** (estilo del repo): el
caso por defecto (``include_masks=False``) **no** arrastra ``pycocotools``.

Nota sobre ``obj_id``: es **inestable** en modo per-frame (segmentacion) y
**estable** en modo tracking. El campo es el mismo; la semantica depende del modo.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import PROJECT_ROOT

SCHEMA_VERSION = "1.0"


def mask_to_bbox_centroid(
    mask: np.ndarray,
) -> tuple[list[int], list[float]] | None:
    """Caja envolvente y centroide de una mascara booleana.

    Args:
        mask: mascara booleana con forma ``(H, W)``.

    Returns:
        Tupla ``(bbox, centroid)`` con ``bbox = [x, y, w, h]`` (pixeles absolutos) y
        ``centroid = [cx, cy]`` (centro de la caja), o ``None`` si la mascara es
        vacia.
    """
    import cv2

    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w == 0 or h == 0:
        return None
    return [int(x), int(y), int(w), int(h)], [x + w / 2.0, y + h / 2.0]


def encode_rle(mask: np.ndarray) -> dict:
    """Codifica una mascara booleana a **COCO-RLE** JSON-serializable.

    Usa ``pycocotools.mask.encode`` sobre un arreglo ``uint8`` Fortran-contiguo. El
    campo ``counts`` (bytes) se decodifica a ``str`` ascii para que el resultado sea
    serializable a JSON.

    Args:
        mask: mascara booleana con forma ``(H, W)``.

    Returns:
        ``{"size": [H, W], "counts": "<str>"}`` (COCO-RLE).
    """
    from pycocotools import mask as mask_utils

    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode("ascii")
    rle["size"] = [int(s) for s in rle["size"]]
    return rle


def decode_rle(rle: dict) -> np.ndarray:
    """Inverso de :func:`encode_rle`: COCO-RLE -> mascara booleana ``(H, W)``.

    La ida-vuelta es **sin perdida** (``decode_rle(encode_rle(m)) == m``).

    Args:
        rle: dict COCO-RLE ``{"size": [H, W], "counts": "<str>"}``.

    Returns:
        Mascara booleana con forma ``(H, W)``.
    """
    from pycocotools import mask as mask_utils

    r = {
        "size": [int(s) for s in rle["size"]],
        "counts": rle["counts"].encode("ascii"),
    }
    return mask_utils.decode(r).astype(bool)


def detection_record(det: Any, include_masks: bool) -> dict | None:
    """Convierte una ``Detection`` a su registro en el esquema.

    Args:
        det: ``Detection`` (con ``obj_id``, ``mask`` booleana y ``score``).
        include_masks: si ``True``, embebe la mascara como ``rle`` (COCO-RLE).

    Returns:
        ``{"obj_id", "bbox", "centroid", "score", ["rle"]}`` o ``None`` si la
        mascara es vacia (no se puede derivar caja).
    """
    geom = mask_to_bbox_centroid(det.mask)
    if geom is None:
        return None
    bbox, centroid = geom
    record: dict = {
        "obj_id": int(det.obj_id),
        "bbox": bbox,
        "centroid": centroid,
        "score": float(det.score),
    }
    if include_masks:
        record["rle"] = encode_rle(det.mask)
    return record


def frame_record(
    frame_index: int,
    dets_by_class: dict[str, list],
    include_masks: bool,
) -> dict:
    """Ensambla el registro frame-indexed de un frame.

    Args:
        frame_index: indice **real** del frame en el video fuente.
        dets_by_class: ``{nombre_clase: [Detection, ...]}`` (salida de
            ``detect_classes_in_frame`` / del tracking).
        include_masks: propaga la inclusion de ``rle`` en cada deteccion.

    Returns:
        ``{"frame_index", "detections": {clase: [detection_record, ...]}}``. Las
        detecciones con mascara vacia se omiten.
    """
    detections: dict[str, list[dict]] = {}
    for name, dets in dets_by_class.items():
        recs: list[dict] = []
        for det in dets:
            rec = detection_record(det, include_masks)
            if rec is not None:
                recs.append(rec)
        detections[name] = recs
    return {"frame_index": int(frame_index), "detections": detections}


def _package_versions(names: list[str]) -> dict[str, str]:
    """Versiones instaladas de ``names`` (best-effort; omite las no resueltas)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            continue
    return out


def _model_version(config: dict) -> dict:
    """Identificador best-effort del modelo: puntero al checkpoint + versiones."""
    sam3_dir = config.get("working_dirs", {}).get("sam3_dir")
    return {
        "sam3_dir": sam3_dir,
        "packages": _package_versions(["sam3", "transformers"]),
    }


def build_header(
    *,
    video: Path | str,
    mode: str,
    fps: float,
    resolution: tuple[int, int],
    num_frames: int,
    classes: list[dict],
    include_masks: bool,
    config: dict,
) -> dict:
    """Construye la cabecera (metadatos de corrida) del entregable.

    Args:
        video: ruta del video fuente.
        mode: ``"segmentation"`` o ``"tracking"``.
        fps: fps real de la fuente.
        resolution: ``(H, W)`` del frame.
        num_frames: numero de frames procesados.
        classes: lista de clases del config (se emiten sus ``name``).
        include_masks: si las detecciones llevan ``rle``.
        config: snapshot **completo** de la config activa (auto-descripcion).

    Returns:
        Dict con las claves de cabecera del esquema (sin ``frames``/``tracks``).
    """
    h, w = resolution
    return {
        "schema_version": SCHEMA_VERSION,
        "video": str(video),
        "mode": mode,
        "model_version": _model_version(config),
        "timestamp": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "fps": float(fps),
        "resolution": {"height": int(h), "width": int(w)},
        "num_frames": int(num_frames),
        "include_masks": bool(include_masks),
        "classes": [c["name"] for c in classes],
        "config": config,
    }


def inference_paths(video_stem: str, outputs_dir: str) -> tuple[Path, Path]:
    """Rutas de salida por video: ``outputs_dir/inference/<stem>/<stem>.{json,mp4}``.

    Args:
        video_stem: nombre base del video (``Path.stem``).
        outputs_dir: directorio de salidas (relativo a ``PROJECT_ROOT``).

    Returns:
        Tupla ``(json_path, mp4_path)``. **No** crea las carpetas (lo hace el
        escritor correspondiente).
    """
    base = PROJECT_ROOT / outputs_dir / "inference" / video_stem
    return base / f"{video_stem}.json", base / f"{video_stem}.mp4"


def write_inference_json(
    header: dict,
    frames: list[dict],
    json_path: Path | str,
    tracks: list[dict] | None = None,
) -> Path:
    """Compone y escribe el JSON unificado del entregable.

    Args:
        header: cabecera de :func:`build_header`.
        frames: lista frame-indexed de :func:`frame_record`.
        json_path: ruta del JSON a escribir (se crea la carpeta padre).
        tracks: indice de tracks (solo modo tracking); si es ``None`` se omite.

    Returns:
        La ruta (``Path``) del archivo escrito.
    """
    payload = dict(header)
    payload["frames"] = frames
    if tracks is not None:
        payload["tracks"] = tracks

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path
