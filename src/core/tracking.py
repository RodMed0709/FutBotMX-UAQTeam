"""Tracking por detección per-frame + ByteTrack (tarea video_tracking).

Asocia las detecciones per-frame de SAM3 en trayectorias con ``obj_id`` estables a
lo largo de un video, usando **ByteTrack** (paquete ``trackers``) como capa de
asociación por cajas. Reutiliza el pipeline per-frame existente
(``detect_classes_in_frame``) y no abre una sesión SAM3-video: por eso soporta
**video completo en streaming** sin acumular features ni frames en memoria.

Flujo (por frame, en streaming):

    iter_frames -> detect_classes_in_frame -> (por clase: máscara->caja ->
    ByteTrack -> obj_id estable) -> overlay -> escribir mp4 incremental

API pública:
- ``track_video``: orquesta el tracking de un video → mp4 + JSON + índice de tracks.
- ``get_trajectories``: centroides por ``obj_id`` en el tiempo (utilidad derivada).
- ``Track`` / ``TrackObservation``: modelo de datos agnóstico al tracker.

El índice de tracks **no** guarda máscaras (memoria acotada). ``supervision``,
``trackers`` y ``cv2`` se importan de forma perezosa.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.frame_extraction import get_video_fps, iter_frames
from src.core.inference_schema import (
    build_header,
    frame_record,
    inference_paths,
    mask_to_bbox_centroid,
    write_inference_json,
)
from src.core.overlay import overlay_detections
from src.core.sam3_loader import Sam3Bundle, load_sam3
from src.core.segmentation import _load_classes, detect_classes_in_frame
from src.core.video_writer import open_video_writer
from src.utils import PROJECT_ROOT, get_abs_path

# Claves de ByteTrack (paquete `trackers`) que se leen de la config (con defaults).
_BYTETRACK_DEFAULTS: dict[str, float | int] = {
    "track_activation_threshold": 0.4,
    "lost_track_buffer": 30,
    "minimum_consecutive_frames": 1,
    "minimum_iou_threshold": 0.2,
}


@dataclass
class TrackObservation:
    """Una observación de un track en un frame (sin máscara).

    Attributes:
        frame_index: índice del frame en el video fuente.
        bbox: caja ``(x, y, w, h)`` del track en ese frame.
        centroid: centro ``(cx, cy)`` de la caja.
        score: score de la detección que originó la observación.
    """

    frame_index: int
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    score: float


@dataclass
class Track:
    """Una trayectoria con identidad estable a lo largo del video.

    Attributes:
        obj_id: identificador estable y **globalmente único** entre clases.
        class_name: clase del objeto (por construcción: el tracker de esa clase).
        observations: lista de ``TrackObservation`` en orden temporal.
    """

    obj_id: int
    class_name: str
    observations: list[TrackObservation]


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


def _load_tracking_config() -> tuple[dict, int | None, str, dict]:
    """Lee la configuración de tracking desde la configuración global.

    Returns:
        Tupla ``(bytetrack_kwargs, max_frames, outputs_dir, config)``: los parámetros
        de ByteTrack (con defaults), el tope de frames (``None`` = video completo), el
        directorio de salidas y el ``config`` completo (para embeberlo como snapshot).

    Raises:
        ValueError: si CONFIG_FILENAME no está en el .env.
        KeyError: si falta ``working_dirs.outputs_dir``.
        FileNotFoundError: si el archivo de configuración no existe.
    """
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")

    config = json.loads(get_abs_path(f"configs/{config_filename}").read_text("utf-8"))

    working_dirs = config.get("working_dirs", {})
    if "outputs_dir" not in working_dirs:
        raise KeyError("Falta 'working_dirs.outputs_dir' en la configuracion.")

    tracking = config.get("tracking", {})
    bytetrack_kwargs = {
        key: tracking.get(key, default) for key, default in _BYTETRACK_DEFAULTS.items()
    }
    max_frames = tracking.get("max_frames", None)
    if max_frames is not None:
        max_frames = int(max_frames)

    return bytetrack_kwargs, max_frames, working_dirs["outputs_dir"], config


def get_trajectories(
    tracks: dict[int, Track],
) -> dict[int, list[tuple[int, float, float]]]:
    """Devuelve, por ``obj_id``, la lista ``[(frame_index, cx, cy), ...]``.

    Utilidad derivada del índice de tracks (a partir de los centroides de cada
    observación), para graficar o analizar trayectorias.
    """
    return {
        oid: [(o.frame_index, o.centroid[0], o.centroid[1]) for o in t.observations]
        for oid, t in tracks.items()
    }


def _tracks_payload(tracks: dict[int, Track]) -> list[dict]:
    """Serializa el índice de tracks a la sección ``tracks`` del esquema (sin máscaras).

    Devuelve la lista de tracks (ordenados por ``obj_id``) que se funde en el JSON
    unificado del entregable; ya no se escribe un archivo ``_tracks.json`` aparte.
    """
    return [
        {
            "obj_id": t.obj_id,
            "class": t.class_name,
            "observations": [
                {
                    "frame_index": o.frame_index,
                    "bbox": list(o.bbox),
                    "centroid": list(o.centroid),
                    "score": o.score,
                }
                for o in t.observations
            ],
        }
        for t in sorted(tracks.values(), key=lambda t: t.obj_id)
    ]


def track_video(
    video_path: Path | str,
    output_path: Path | None = None,
    classes: list[dict] | None = None,
    max_frames: int | None = None,
    bundle: Sam3Bundle | None = None,
    include_masks: bool = False,
    render_video: bool = True,
) -> dict:
    """Trackea un video con detección per-frame (SAM3) + ByteTrack por clase.

    Recorre el video en streaming; por frame detecta con ``detect_classes_in_frame``,
    convierte cada máscara a caja y la asocia con un ByteTrack por clase, asignando
    ``obj_id`` **estables y globalmente únicos** (a diferencia del modo per-frame,
    donde el ``obj_id`` es inestable). Escribe un **JSON unificado** (esquema común de
    inferencia) con la vista frame-indexed (``frames``) y el índice de tracks
    (``tracks``) en el mismo archivo.

    El **JSON es el entregable y se escribe siempre**; el mp4 con overlay es
    **opcional** (``render_video``). Con ``render_video=False`` no se abre el escritor
    de video ni se compone el overlay (ahorro de CPU/IO para lotes/evaluacion).

    Args:
        video_path: ruta del video (relativa a PROJECT_ROOT o absoluta).
        output_path: ruta del mp4 de salida. Si es ``None``, se ubica bajo
            ``working_dirs.outputs_dir`` como ``inference/<stem>/<stem>.mp4`` y el
            JSON junto a él. Con ``render_video=False`` solo se usa para derivar la
            ruta del JSON (no se escribe video).
        classes: lista de clases a trackear. Si es ``None``, todas las del config.
        max_frames: tope de frames (clip). Si es ``None``, usa el valor de la config
            (``tracking.max_frames``); si ese también es ``None``, recorre todo el video.
        bundle: modelo SAM3 cargado. Si es ``None`` se obtiene con ``load_sam3()``.
        include_masks: si ``True``, cada detección de la vista frame-indexed incluye
            su máscara en COCO-RLE (requiere ``pycocotools``). Por defecto ``False``.
        render_video: si ``True`` (por defecto, uso de un solo video) genera el mp4
            con overlay; si ``False`` solo escribe el JSON. Ortogonal a
            ``include_masks``.

    Returns:
        ``{"json": <ruta_json>, "video": <ruta_mp4_o_None>, "index": <dict
        obj_id->Track>}``. La clave ``"video"`` siempre está presente: vale la ruta
        del mp4 si ``render_video=True`` y ``None`` si ``render_video=False``.

    Raises:
        ValueError / KeyError / FileNotFoundError: ver ``_load_tracking_config``.
    """
    import supervision as sv
    from trackers import ByteTrackTracker

    # La firma acepta str|Path; el resto del flujo (get_video_fps, iter_frames)
    # exige Path, asi que normalizamos aqui.
    video_path = Path(video_path)
    classes = classes if classes is not None else _load_classes()
    bundle = bundle or load_sam3()
    bytetrack_kwargs, cfg_max_frames, outputs_dir, config = _load_tracking_config()
    max_frames = max_frames if max_frames is not None else cfg_max_frames

    fps = get_video_fps(video_path)

    # Rutas de salida (carpeta por video bajo outputs/inference/).
    stem = Path(video_path).stem
    if output_path is not None:
        mp4_path = Path(output_path)
        json_path = mp4_path.with_name(f"{mp4_path.stem}.json")
    else:
        json_path, mp4_path = inference_paths(stem, outputs_dir)

    # Un tracker ByteTrack por clase (la clase queda determinada por construcción).
    trackers = {
        cls["name"]: ByteTrackTracker(frame_rate=fps, **bytetrack_kwargs)
        for cls in classes
    }

    global_id: dict[tuple[str, int], int] = {}  # (clase, tracker_id) -> obj_id
    next_obj_id = 0
    tracks: dict[int, Track] = {}
    frames_records: list[dict] = []
    resolution: tuple[int, int] | None = None
    num_frames = 0

    # El mp4 es opcional: con render_video=False usamos nullcontext (no abre imageio
    # ni crea archivo) y el mismo bucle corre sin escribir video.
    writer_cm = (
        open_video_writer(mp4_path, fps=fps) if render_video else nullcontext(None)
    )
    with writer_cm as append:
        for frame_index, frame in iter_frames(video_path, max_frames):
            if resolution is None:
                resolution = (frame.shape[0], frame.shape[1])
            num_frames += 1
            dets = detect_classes_in_frame(frame, classes=classes, bundle=bundle)
            per_frame: dict[str, list] = {}

            for cls in classes:
                name = cls["name"]
                cdets = dets.get(name, [])

                # Cajas (no vacías) de esta clase para alimentar ByteTrack.
                boxes, scores, srcs = [], [], []
                for idx, det in enumerate(cdets):
                    geom = mask_to_bbox_centroid(det.mask)
                    if geom is None:
                        continue
                    (x, y, w, h), _ = geom
                    boxes.append((x, y, x + w, y + h))  # xyxy para ByteTrack
                    scores.append(float(det.score))
                    srcs.append(idx)

                if boxes:
                    sv_det = sv.Detections(
                        xyxy=np.array(boxes, dtype=float),
                        confidence=np.array(scores, dtype=float),
                        data={"src": np.array(srcs)},
                    )
                else:
                    sv_det = sv.Detections.empty()

                tracked = trackers[name].update(sv_det, frame)

                out_list = []
                src_arr = tracked.data.get("src") if tracked.data else None
                for i in range(len(tracked)):
                    if src_arr is None or i >= len(src_arr):
                        continue  # no se puede mapear de vuelta a la máscara
                    det = cdets[int(src_arr[i])]
                    tid = int(tracked.tracker_id[i])
                    if tid < 0:
                        # Warm-up de ByteTrack: aún sin identidad estable este frame.
                        det.obj_id = -1
                        out_list.append(det)
                        continue

                    key = (name, tid)
                    if key not in global_id:
                        global_id[key] = next_obj_id
                        next_obj_id += 1
                    obj_id = global_id[key]
                    det.obj_id = obj_id

                    x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    tracks.setdefault(
                        obj_id, Track(obj_id=obj_id, class_name=name, observations=[])
                    ).observations.append(
                        TrackObservation(
                            frame_index=frame_index,
                            bbox=bbox,
                            centroid=centroid,
                            score=float(det.score),
                        )
                    )
                    out_list.append(det)

                per_frame[name] = out_list

            if render_video:
                composed = overlay_detections(frame, per_frame, classes=classes)
                append(composed)

            # Registro frame-indexed (reusa las máscaras vivas antes de descartarlas).
            frames_records.append(
                frame_record(frame_index, per_frame, include_masks=include_masks)
            )

    header = build_header(
        video=video_path,
        mode="tracking",
        fps=fps,
        resolution=resolution if resolution is not None else (0, 0),
        num_frames=num_frames,
        classes=classes,
        include_masks=include_masks,
        config=config,
    )
    json_path = write_inference_json(
        header, frames_records, json_path, tracks=_tracks_payload(tracks)
    )
    return {
        "json": json_path,
        "video": mp4_path if render_video else None,
        "index": tracks,
    }
