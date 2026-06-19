"""Fachada única de inferencia por video (tarea unified_inference).

Una sola puerta de entrada, ``run_inference``, que recibe el ``mode`` de inferencia
(``"segmentation"`` o ``"tracking"``) y enruta a las implementaciones existentes sin
reimplementar el bucle de inferencia:

- ``mode="segmentation"`` -> ``src.core.pipeline.run_pipeline`` (per-frame, ``obj_id``
  inestable). Camino más barato; es el modo por defecto.
- ``mode="tracking"`` -> ``src.core.tracking.track_video`` (per-frame + ByteTrack,
  ``obj_id`` estable). Cablea el modo que antes era un stub en ``pipeline.py``.

La fachada:

- resuelve el **muestreo por modo** vía un selector ``sampling`` (``"auto"`` deja que
  el modo decida) y **valida** la combinación inválida (cuota + tracking);
- **unifica el retorno** a ``{"json", "video", "index"}`` (``"index"`` es ``None`` en
  segmentación, donde no hay tracks estables);
- hereda ``include_masks`` y ``render_video`` (tareas inference_schema y
  optional_render) como parámetros sobreescribibles, ortogonales al modo;
- propaga un ``bundle`` SAM3 ya cargado a **ambos** modos (carga única en lotes).

Los imports pesados (torch/cv2/supervision/trackers) viven dentro de las
implementaciones, no aquí; este módulo solo enruta.
"""

from __future__ import annotations

from pathlib import Path

from src.core.pipeline import run_pipeline
from src.core.sam3_loader import Sam3Bundle
from src.core.tracking import track_video

# Estrategias de muestreo válidas a nivel de fachada.
_VALID_SAMPLING = ("auto", "quota", "all", "contiguous")


def _resolve_segmentation_sampling(sampling: str) -> bool:
    """Traduce ``sampling`` al ``all_frames`` de ``run_pipeline`` (segmentación).

    Returns:
        ``all_frames``: ``False`` (cuota equiespaciada) o ``True`` (video completo).

    Raises:
        ValueError: si ``sampling="contiguous"`` (segmentación no hace muestreo de
            prefijo contiguo) o si ``sampling`` no es válido.
    """
    if sampling in ("auto", "quota"):
        return False
    if sampling == "all":
        return True
    if sampling == "contiguous":
        raise ValueError(
            "sampling='contiguous' no es compatible con mode='segmentation' "
            "(el muestreo de prefijo contiguo es propio de tracking)."
        )
    raise ValueError(
        f"sampling '{sampling}' no soportado (usa uno de {_VALID_SAMPLING})."
    )


def _resolve_tracking_sampling(sampling: str, max_frames: int | None) -> int | None:
    """Traduce ``sampling`` al ``max_frames`` efectivo de ``track_video`` (tracking).

    Returns:
        ``max_frames`` efectivo: el tope contiguo recibido (``"auto"``/``"contiguous"``)
        o ``None`` para video completo (``"all"``).

    Raises:
        ValueError: si ``sampling="quota"`` (ByteTrack requiere frames contiguos) o si
            ``sampling`` no es válido.
    """
    if sampling in ("auto", "contiguous"):
        return max_frames
    if sampling == "all":
        return None
    if sampling == "quota":
        raise ValueError(
            "sampling='quota' no es compatible con mode='tracking' "
            "(ByteTrack requiere frames contiguos)."
        )
    raise ValueError(
        f"sampling '{sampling}' no soportado (usa uno de {_VALID_SAMPLING})."
    )


def run_inference(
    video_path: Path | str,
    mode: str = "segmentation",
    output_path: Path | None = None,
    classes: list[dict] | None = None,
    sampling: str = "auto",
    max_frames: int | None = None,
    bundle: Sam3Bundle | None = None,
    include_masks: bool = False,
    render_video: bool = True,
    detector: str | None = None,
    tracker: str | None = None,
    run_label: str | None = None,
    progress: bool = True,
) -> dict:
    """Ejecuta la inferencia de un video por la puerta única de la fachada.

    Enruta al camino correcto según ``mode`` y devuelve siempre la misma forma de
    resultado, para que la capa de lotes la consuma sin ramificar por modo.

    Args:
        video_path: ruta del video (relativa a PROJECT_ROOT o absoluta).
        mode: ``"segmentation"`` (por defecto, per-frame, ``obj_id`` inestable) o
            ``"tracking"`` (per-frame + ByteTrack, ``obj_id`` estable). Otro valor
            levanta ``ValueError``.
        output_path: ruta del mp4 de salida; si es ``None``, se ubica bajo
            ``working_dirs.outputs_dir`` y el JSON junto a él (ver implementaciones).
        classes: lista de clases a procesar. ``None`` (por defecto) usa las del config.
        sampling: estrategia de muestreo de frames. ``"auto"`` (por defecto) deja que
            el **modo** decida (segmentación → cuota equiespaciada; tracking → prefijo
            contiguo completo). Valores explícitos:

            ===========  =============================  ===============================
            sampling     segmentation                   tracking
            ===========  =============================  ===============================
            auto         cuota (``all_frames=False``)   contiguo (respeta ``max_frames``)
            quota        cuota (``all_frames=False``)   ``ValueError``
            all          completo (``all_frames=True``) completo (``max_frames=None``)
            contiguous   ``ValueError``                 contiguo (respeta ``max_frames``)
            ===========  =============================  ===============================

        max_frames: tope de frames **contiguos** (solo aplica en tracking; en
            segmentación se **ignora** —el conteo lo fija ``preprocess.frame_quota``—).
        bundle: modelo SAM3 precargado. Si es ``None`` cada implementación lo carga;
            pasarlo permite reutilizar el modelo entre videos (carga única en lotes).
        include_masks: si ``True``, las detecciones incluyen máscara en COCO-RLE.
            Por defecto ``False``. Ortogonal al modo y a ``render_video``.
        render_video: si ``True`` (por defecto, uso de un solo video) genera el mp4
            anotado; si ``False`` solo escribe el JSON. Ortogonal al modo y a
            ``include_masks``.
        detector: estrategia de detección por frame (``"sam3_text"`` | ``"yolo_sam3"``).
            ``None`` ⇒ la config (clave ``detector``) o ``"sam3_text"``. **Ortogonal al
            modo**: aplica en **ambos** modos. En ``mode="segmentation"`` selecciona la
            estrategia por-frame (``obj_id`` inestable, sin asociación temporal); en
            ``mode="tracking"`` además alimenta al tracker. Un nombre inválido levanta
            ``ValueError`` antes de cargar SAM3.
        tracker: tracker para ``mode="tracking"`` (``"bytetrack"`` | ``"botsort"``).
            ``None`` ⇒ la config (``tracking.tracker``) o ``"bytetrack"``. Ortogonal a
            ``detector``. En ``mode="segmentation"`` se **ignora**.
        run_label: subcarpeta opcional por config para las salidas derivadas por
            defecto (``inference/<run_label>/<stem>/…``); evita que varias configs se
            pisen al correr sobre los mismos videos. ``None`` ⇒ ruta plana actual. Se
            **ignora** si se pasa ``output_path``.
        progress: si ``True`` (por defecto) muestra una barra de progreso ``tqdm`` por
            video (ETA + frames/s); ``False`` la silencia. Aplica a ambos modos.

    Returns:
        ``{"json": Path, "video": Path | None, "index": dict | None}``. ``"video"`` es
        la ruta del mp4 o ``None`` (``render_video=False``). ``"index"`` es el índice
        ``obj_id->Track`` en tracking o ``None`` en segmentación.

    Raises:
        ValueError: ``mode`` desconocido; ``sampling`` inválido para el modo
            (``"quota"``+tracking o ``"contiguous"``+segmentación); ``sampling``
            desconocido. La validación ocurre **antes** de cargar SAM3.
        FileNotFoundError: si el video o la configuración no resuelven (ver
            implementaciones).
    """
    if mode == "segmentation":
        all_frames = _resolve_segmentation_sampling(sampling)
        res = run_pipeline(
            video_path,
            output_path=output_path,
            all_frames=all_frames,
            mode="per_frame",
            classes=classes,
            bundle=bundle,
            include_masks=include_masks,
            render_video=render_video,
            detector=detector,
            run_label=run_label,
            progress=progress,
        )
        return {"json": res["json"], "video": res["video"], "index": None}

    if mode == "tracking":
        eff_max_frames = _resolve_tracking_sampling(sampling, max_frames)
        return track_video(
            video_path,
            output_path=output_path,
            classes=classes,
            max_frames=eff_max_frames,
            bundle=bundle,
            include_masks=include_masks,
            render_video=render_video,
            detector=detector,
            tracker=tracker,
            run_label=run_label,
            progress=progress,
        )

    raise ValueError(f"mode '{mode}' no soportado (usa 'segmentation' o 'tracking').")
