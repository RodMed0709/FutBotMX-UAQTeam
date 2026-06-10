"""Detector de cajas YOLO (tarea yolo_detector).

Building block que carga el detector **YOLO11** ya entrenado (``best.pt``, destilado
de SAM3 en fase_1) y, dado un frame, devuelve las **cajas por clase del repo**
(estructura ligera caja+score, sin mascara). Esas cajas alimentan al box-prompt de
SAM3 (``boxes_to_masks``) para producir las mascaras finas: SAM3 sigue siendo el
centro, YOLO solo acelera la localizacion.

Sustituye la inferencia YOLO suelta y hardcoded de
``notebooks/fase_2_YOLO_SAM3/pipeline_yolo_sam3.py``:
- ruta de ``best.pt`` desde la configuracion (``working_dirs.yolo_weights``), no
  junto al notebook;
- mapeo de clase YOLO -> clase del repo via el campo ``yolo_id`` de la config;
- ``conf``/``imgsz`` desde la seccion ``yolo`` de la config.

``ultralytics``, ``torch`` y ``PIL`` se importan de forma **perezosa** dentro de las
funciones, para que ``import src.core`` no los arrastre.

IMPORTANTE: ``best.pt`` es un artefacto pesado que vive en el **pod** (lo produjo el
entrenamiento de fase_1) y NO en el repo. La inferencia/los tests YOLO se corren en
el pod; en local fallan por falta del peso (y de ``ultralytics``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.core.segmentation import _load_classes
from src.utils import PROJECT_ROOT, get_abs_path

# Defaults de inferencia si la seccion `yolo` no los define.
_YOLO_DEFAULTS = {"conf": 0.4, "imgsz": 960}


@dataclass
class BoxDetection:
    """Una caja detectada por YOLO (sin mascara).

    Attributes:
        bbox: caja ``(x1, y1, x2, y2)`` en pixeles absolutos del frame (xyxy).
        score: confianza de la deteccion (``boxes.conf`` de YOLO).
    """

    bbox: tuple[float, float, float, float]
    score: float


def _load_env(env_path: Path) -> dict[str, str]:
    """Parseo simple de un archivo .env (KEY = value), aplicando strip()."""
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _load_yolo_config() -> tuple[str, float, int]:
    """Lee de la configuracion la ruta de pesos y los parametros de inferencia YOLO.

    Returns:
        Tupla ``(yolo_weights_rel, conf, imgsz)``: ruta **relativa** del ``best.pt``
        y los parametros (con defaults si la seccion ``yolo`` no los define).

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si falta ``working_dirs.yolo_weights``.
        FileNotFoundError: si el archivo de configuracion no existe.
    """
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")

    config = json.loads(get_abs_path(f"configs/{config_filename}").read_text("utf-8"))

    working_dirs = config.get("working_dirs", {})
    if "yolo_weights" not in working_dirs:
        raise KeyError("Falta 'working_dirs.yolo_weights' en la configuracion.")

    yolo = config.get("yolo", {})
    conf = float(yolo.get("conf", _YOLO_DEFAULTS["conf"]))
    imgsz = int(yolo.get("imgsz", _YOLO_DEFAULTS["imgsz"]))
    return working_dirs["yolo_weights"], conf, imgsz


def _resolve_weights(weights: str | Path | None) -> str:
    """Resuelve la ruta absoluta del ``best.pt``, con error claro si falta.

    Si ``weights`` es ``None`` se lee ``working_dirs.yolo_weights`` de la config.

    Raises:
        FileNotFoundError: si el peso no existe en disco, con un mensaje que explica
            que es un artefacto del pod (fase_1) y que la inferencia YOLO corre alli.
    """
    weights_rel = str(weights) if weights is not None else _load_yolo_config()[0]
    try:
        return str(get_abs_path(weights_rel))
    except FileNotFoundError as exc:
        abs_guess = PROJECT_ROOT / weights_rel
        raise FileNotFoundError(
            f"No se encontro el modelo YOLO en '{abs_guess}' "
            f"(working_dirs.yolo_weights='{weights_rel}'). "
            "El 'best.pt' es un artefacto pesado producido por el entrenamiento de "
            "fase_1 y vive en el POD, no en el repo. Coloca el archivo en "
            "'assets/yolo/best.pt' (git-ignored). La inferencia/los tests YOLO se "
            "ejecutan en el pod; en local fallaran por falta del peso."
        ) from exc


@lru_cache(maxsize=1)
def _cached_load_yolo(weights_abs: str) -> Any:
    """Carga cacheada (singleton) del modelo YOLO desde una ruta absoluta."""
    from ultralytics import YOLO

    return YOLO(weights_abs)


def load_yolo(weights: str | Path | None = None, device: str | None = None) -> Any:
    """Carga el detector YOLO (``best.pt``) listo para inferir.

    La ruta de pesos se resuelve sola desde la configuracion
    (``working_dirs.yolo_weights``) si no se pasa ``weights``. El modelo se **cachea**
    (singleton) por ruta; la carga es independiente del ``Sam3Bundle``.

    Args:
        weights: ruta del ``best.pt`` (relativa a PROJECT_ROOT o absoluta). ``None``
            ⇒ se lee de la configuracion.
        device: aceptado por simetria con ``load_sam3``; ultralytics mueve el modelo
            al device en ``predict(device=...)``, asi que aqui no se fija.

    Returns:
        El modelo ``YOLO`` de ultralytics.

    Raises:
        FileNotFoundError: si el peso no existe (mensaje claro, ver ``_resolve_weights``).
        ValueError / KeyError: si la configuracion es invalida.
    """
    return _cached_load_yolo(_resolve_weights(weights))


def detect_boxes(
    frame: np.ndarray,
    model: Any = None,
    classes: list[dict] | None = None,
    conf: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
) -> dict[str, list[BoxDetection]]:
    """Detecta cajas en un frame y las agrupa por nombre de clase del repo.

    Args:
        frame: imagen ``(H, W, 3)`` RGB (convencion del repo).
        model: modelo YOLO ya cargado. ``None`` ⇒ ``load_yolo()``.
        classes: lista de clases del repo. ``None`` ⇒ las de la configuracion. Solo
            las que tengan ``yolo_id`` participan (``green_floor`` queda fuera).
        conf: umbral de confianza. ``None`` ⇒ seccion ``yolo`` de la config / default.
        imgsz: tamano de inferencia. ``None`` ⇒ seccion ``yolo`` de la config / default.
        device: device de ``predict``. ``None`` ⇒ auto (``cuda`` si disponible).

    Returns:
        ``{nombre_clase: [BoxDetection, ...]}`` con una entrada por clase con
        ``yolo_id`` (lista vacia si no hubo detecciones de esa clase).
    """
    import torch
    from PIL import Image

    model = model if model is not None else load_yolo()
    classes = classes if classes is not None else _load_classes()

    # Mapa yolo_id -> nombre de clase del repo (solo clases con yolo_id).
    id_to_name = {c["yolo_id"]: c["name"] for c in classes if "yolo_id" in c}

    # Parametros de inferencia: argumento -> config -> default.
    if conf is None or imgsz is None:
        _, cfg_conf, cfg_imgsz = _load_yolo_config()
        conf = cfg_conf if conf is None else conf
        imgsz = cfg_imgsz if imgsz is None else imgsz
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # PIL (RGB) evita la ambiguedad RGB/BGR de ultralytics con arrays numpy.
    img = Image.fromarray(frame)
    res = model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]

    out: dict[str, list[BoxDetection]] = {name: [] for name in id_to_name.values()}
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    scores = res.boxes.conf.cpu().numpy()
    for box, cls_id, score in zip(xyxy, cls, scores):
        name = id_to_name.get(int(cls_id))
        if name is None:  # clase YOLO sin mapeo en config: se descarta (defensivo)
            continue
        bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        out[name].append(BoxDetection(bbox=bbox, score=float(score)))
    return out
