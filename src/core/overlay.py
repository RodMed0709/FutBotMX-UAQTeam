"""Visualizacion multi-clase de detecciones (tarea segmentation_overlay).

Dos piezas:

- ``overlay_detections(frame, detections_by_class)``: pinta las mascaras de cada
  clase sobre el frame (color por clase, mezcla alpha) y **devuelve** el frame
  compuesto como ``np.ndarray uint8 (H, W, 3)``. Es la pieza reutilizable que
  alimenta tanto el display como el futuro escritor de video.
- ``show_overlay(...)``: compone y **muestra** el resultado con leyenda
  (matplotlib). Display-only: no escribe a disco ni devuelve el array.

Los colores de cada clase y el alpha por defecto provienen de la configuracion
(``classes[].color`` y ``visualization.overlay_alpha``), con override por
parametro. ``overlay_detections`` es numpy-only; matplotlib se importa de forma
perezosa dentro de ``show_overlay`` para no encarecer ``import src.core``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from src.utils import PROJECT_ROOT, get_abs_path


def _load_overlay_config() -> tuple[list[dict], float]:
    """Lee (clases, alpha por defecto) desde el archivo de configuracion.

    Returns:
        Una tupla ``(classes, default_alpha)`` con la lista de clases
        (``classes``) y ``visualization.overlay_alpha``.

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si faltan ``classes`` o ``visualization.overlay_alpha``.
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
    visualization = config.get("visualization", {})
    if "overlay_alpha" not in visualization:
        raise KeyError(
            "Falta la clave 'visualization.overlay_alpha' en la configuracion."
        )
    return config["classes"], float(visualization["overlay_alpha"])


def _validate_frame(frame: np.ndarray) -> tuple[int, int]:
    """Valida el frame y devuelve ``(H, W)``.

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


def _resolve_classes_alpha(
    classes: list[dict] | None, alpha: float | None
) -> tuple[dict[str, tuple], float]:
    """Resuelve el mapa ``name -> color`` y el alpha, con prioridad del parametro."""
    if classes is None or alpha is None:
        cfg_classes, cfg_alpha = _load_overlay_config()
        classes = classes if classes is not None else cfg_classes
        alpha = alpha if alpha is not None else cfg_alpha
    color_map = {cls["name"]: tuple(cls["color"]) for cls in classes}
    return color_map, float(alpha)


def overlay_detections(
    frame: np.ndarray,
    detections_by_class: dict[str, list],
    classes: list[dict] | None = None,
    alpha: float | None = None,
) -> np.ndarray:
    """Pinta las detecciones de todas las clases sobre el frame.

    Args:
        frame: frame RGB como ``np.ndarray uint8`` con forma ``(H, W, 3)``.
        detections_by_class: ``{nombre_clase: [Detection, ...]}`` (salida de
            ``detect_classes_in_frame``). Solo se usa ``det.mask`` (booleana,
            ``(H, W)``).
        classes: lista de clases (cada una con ``name`` y ``color``). Si es
            ``None``, se lee de la configuracion.
        alpha: transparencia de la mezcla (0-1). Si es ``None``, se usa
            ``visualization.overlay_alpha`` de la configuracion.

    Returns:
        El frame compuesto como ``np.ndarray uint8 (H, W, 3)`` RGB. No muta la
        entrada ni escribe a disco.

    Raises:
        ValueError: si ``frame`` no tiene forma ``(H, W, 3)``.
        KeyError: si una clase con detecciones no tiene color en la config.
    """
    h, w = _validate_frame(frame)
    color_map, alpha = _resolve_classes_alpha(classes, alpha)

    out = frame.astype(np.float32) / 255.0  # copia en float (no muta la entrada)
    for name, dets in detections_by_class.items():
        if not dets:
            continue
        color01 = np.array(color_map[name], dtype=np.float32) / 255.0
        for det in dets:
            mask = det.mask
            if mask.shape != (h, w):
                warnings.warn(
                    f"Mascara de '{name}' con forma {mask.shape} != frame {(h, w)};"
                    " se omite."
                )
                continue
            out[mask] = (1.0 - alpha) * out[mask] + alpha * color01

    return (out * 255.0).round().clip(0, 255).astype(np.uint8)


def show_overlay(
    frame: np.ndarray,
    detections_by_class: dict[str, list],
    classes: list[dict] | None = None,
    alpha: float | None = None,
) -> None:
    """Muestra el frame compuesto con leyenda (color <-> clase). Display-only.

    Args:
        frame: frame RGB ``np.ndarray uint8 (H, W, 3)``.
        detections_by_class: ``{nombre_clase: [Detection, ...]}``.
        classes: lista de clases; si es ``None``, de la configuracion.
        alpha: transparencia; si es ``None``, de la configuracion.

    Returns:
        ``None``. Su efecto es el render via matplotlib; no escribe a disco.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    composed = overlay_detections(frame, detections_by_class, classes, alpha)
    color_map, _ = _resolve_classes_alpha(classes, alpha)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(composed)
    ax.axis("off")

    # Leyenda solo de las clases presentes en las detecciones.
    handles = [
        mpatches.Patch(color=np.array(color_map[name]) / 255.0, label=name)
        for name in detections_by_class
        if name in color_map
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    plt.show()
