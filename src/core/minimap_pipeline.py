"""Driver de fase_4: video -> homografia por frame -> minimap con trayectorias.

Orquesta las piezas de fase_4, **agnostico al pipeline de segmentacion**: las
detecciones (anclas y objetos) salen del **detector pluggable** del repo
(``get_detector(detector)`` -> ``sam3_text`` | ``yolo_sam3``), no de llamadas SAM3
hardcodeadas. Asi se puede comparar la homografia entre ambos pipelines flipeando un
parametro, y el modulo reusa la misma abstraccion que ``track_video``/``run_inference``.

1. **Anclas de homografia** (por frame): del dict de detecciones se toma la mascara de
   ``green_floor`` (alfombra) y los centroides de ``yellow_zone``/``blue_zone``
   (orientacion). Con ``yolo_sam3`` YOLO localiza **las dos porterias** (azul incluida)
   -> orientacion firme; con ``sam3_text`` la azul suele faltar.
2. **Homografia** (camino C): ``src.core.auto_homography.VideoHomography.update_masks``
   estima ``H`` (img->cm) sobre el rectangulo interior de lineas + orientacion por
   porteria, con lock de arranque, gate de consistencia temporal, EMA y propagacion.
3. **Objetos/trayectorias** de robots y balon: de un JSON de tracking ya generado
   (``tracks_json``, salida de cualquier config 2x2) o, si no se da, del mismo dict de
   detecciones seguido con un tracker greedy por vecino mas cercano (autocontenido). De
   cada objeto se toma el **punto de contacto con el piso** (centro-inferior de la caja
   para robots; centroide para el balon).
4. **Render**: se proyectan las posiciones al campo y se dibujan como trails en el
   minimap (``src.core.minimap``), compuesto sobre cada frame y escrito a un mp4.

El JSON de tracking y los frames del video se recorren con el **mismo** muestreo
contiguo (``iter_frames``), por lo que los ``frame_index`` coinciden.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.auto_homography import VideoHomography  # camino C
from src.core.detectors import get_detector
from src.core.frame_extraction import get_frame_count, get_video_fps, iter_frames
from src.core.homography import mask_centroid, project_points
from src.core.inference_schema import mask_to_bbox_centroid
from src.core.minimap import MinimapRenderer, draw_field_overlay
from src.core.sam3_loader import Sam3Bundle, load_sam3
from src.core.segmentation import _load_classes
from src.core.video_writer import open_video_writer
from src.utils import PROJECT_ROOT

FIELD_CLASS = "green_floor"
YELLOW_CLASS = "yellow_zone"
BLUE_CLASS = "blue_zone"
ROBOT_CLASS = "robot"
BALL_CLASS = "orange_ball"
BALL_CLASSES = {"orange_ball", "ball"}


def _largest_mask(dets: list) -> np.ndarray | None:
    """Mascara booleana de la deteccion de mayor area, o ``None`` si no hay."""
    best, best_area = None, 0
    for d in dets:
        area = int(d.mask.sum())
        if area > best_area:
            best, best_area = d.mask, area
    return best


def _foot_point(class_name: str, bbox) -> tuple[float, float]:
    """Punto de contacto con el piso: centro-inferior (robot) o centro (balon).

    ``bbox`` es ``[x, y, w, h]``.
    """
    x, y, w, h = bbox
    if class_name in BALL_CLASSES:
        return (x + w / 2.0, y + h / 2.0)
    return (x + w / 2.0, y + h)


class _GreedyTracker:
    """Tracker minimo por vecino mas cercano (por clase) para ids estables.

    Asocia cada deteccion al track activo de su misma clase mas cercano dentro de
    un radio (``gate_px``); si no hay, abre un track nuevo. Suficiente para trazar
    trayectorias suaves cuando el movimiento entre frames es pequeno; evita
    depender de un paquete de tracking externo.
    """

    def __init__(self, gate_px: float = 180.0, max_age: int = 12) -> None:
        self._gate = gate_px
        self._max_age = max_age
        self._tracks: dict[int, dict] = {}  # id -> {class, pt, age}
        self._next = 0

    def update(self, dets: list[tuple[str, tuple[float, float]]]) -> list[tuple[int, str, tuple[float, float]]]:
        """Asocia ``dets`` (``(class, foot_xy)``) y devuelve ``(obj_id, class, foot_xy)``."""
        for t in self._tracks.values():
            t["age"] += 1

        out: list[tuple[int, str, tuple[float, float]]] = []
        used: set[int] = set()
        for cls, pt in dets:
            best_id, best_d = None, self._gate
            for tid, t in self._tracks.items():
                if tid in used or t["class"] != cls:
                    continue
                d = float(np.hypot(t["pt"][0] - pt[0], t["pt"][1] - pt[1]))
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self._next
                self._next += 1
            used.add(best_id)
            self._tracks[best_id] = {"class": cls, "pt": pt, "age": 0}
            out.append((best_id, cls, pt))

        # Purga de tracks viejos (no vistos hace > max_age frames).
        for tid in [k for k, t in self._tracks.items() if t["age"] > self._max_age]:
            del self._tracks[tid]
        return out


def _objects_from_dets(dets_by_class: dict) -> list[tuple[str, tuple[float, float]]]:
    """Robots y balon de un dict ``{clase: [Detection]}`` -> lista ``(class, foot_xy)``."""
    out: list[tuple[str, tuple[float, float]]] = []
    for cls in (ROBOT_CLASS, BALL_CLASS):
        for d in dets_by_class.get(cls, []):
            geom = mask_to_bbox_centroid(d.mask)
            if geom is not None:
                out.append((cls, _foot_point(cls, geom[0])))
    return out


def _load_tracks_from_json(tracks_json: Path) -> tuple[dict[int, list], int]:
    """Indexa un JSON de tracking por frame.

    Returns:
        ``(frame_to_objs, max_index)`` con ``frame_to_objs[idx]`` = lista de
        ``(obj_id, class_name, foot_xy)`` de robots/balon, y ``max_index`` el
        ultimo ``frame_index`` (-1 si vacio).
    """
    data = json.loads(Path(tracks_json).read_text(encoding="utf-8"))
    frame_to_objs: dict[int, list] = {}
    max_index = -1
    for tr in data.get("tracks", []):
        cls = tr["class"]
        if cls != ROBOT_CLASS and cls not in BALL_CLASSES:
            continue
        for obs in tr["observations"]:
            bbox = obs["bbox"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(
                    f"bbox debe ser [x, y, w, h] (4 valores); se recibio: {bbox!r}"
                )
            idx = int(obs["frame_index"])
            foot = _foot_point(cls, bbox)
            frame_to_objs.setdefault(idx, []).append((int(tr["obj_id"]), cls, foot))
            max_index = max(max_index, idx)
    return frame_to_objs, max_index


def render_minimap_video(
    video_path: str | Path,
    tracks_json: str | Path | None = None,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
    start_frame: int = 0,
    frame_step: int = 1,
    detector: str = "sam3_text",
    conf: float | None = None,
    bundle: Sam3Bundle | None = None,
    draw_overlay: bool = False,
    smooth_beta: float = 0.4,
    progress: bool = True,
) -> dict:
    """Genera el video con minimap de trayectorias proyectadas por homografia.

    Args:
        video_path: ruta del video (relativa a PROJECT_ROOT o absoluta).
        tracks_json: JSON de tracking ya generado (robots/balon). Si es ``None``,
            se detectan y siguen los objetos en el mismo paso (autocontenido).
        detector: pipeline de segmentacion para anclas (y objetos si no hay
            ``tracks_json``): ``"sam3_text"`` (SAM3 por texto) o ``"yolo_sam3"`` (YOLO
            localiza objetos/porterias + ``green_floor`` por SAM3-texto; reproduce
            ``pod_minimap_sam3``). Permite comparar la homografia entre ambos pipelines.
        conf: umbral de confianza de YOLO (solo aplica con ``detector="yolo_sam3"``;
            ignorado con ``sam3_text``). ``None`` ⇒ el de la config. Bajalo (p. ej.
            ``0.25``) para detectar mas objetos/porterias, como la demo.
        output_path: ruta del mp4 de salida. Si es ``None`` se escribe en
            ``notebooks/fase_4_homografia/outputs/<stem>_minimap.mp4``.
        max_frames: tope de **frames procesados** (cantidad). Con ``tracks_json`` y sin
            recorte (``start_frame=0``/``frame_step=1``) se usa el rango cubierto por los
            tracks si es ``None``; en cualquier otro caso ``None`` recorre hasta el final.
        start_frame: frame fuente donde empezar (0 = inicio). Para renderizar un tramo
            concreto (p. ej. una jugada) sin procesar todo el video.
        frame_step: paso de muestreo (``1`` todos, ``2`` 1 de cada 2…). Abarata el costo
            proporcionalmente. Los ``frame_index`` de ``tracks_json`` siguen casando
            porque son índices del video fuente.
        bundle: modelo SAM3 cargado; si ``None`` se carga con ``load_sam3()``.
        draw_overlay: si ``True`` dibuja las mascaras de anclas sobre el frame
            (depuracion de la homografia).
        smooth_beta: suavizado temporal de la homografia.
        progress: barra de progreso ``tqdm``.

    Returns:
        ``{"video", "n_frames", "homography": {"estimated", "propagated", "rejected"},
        "sample_frame"}``.
    """
    from tqdm.auto import tqdm

    video_path = Path(video_path)
    classes = _load_classes()
    bundle = bundle or load_sam3()
    detect = get_detector(detector)
    # `conf` solo lo entiende el detector YOLO; con sam3_text se ignora.
    detect_kwargs = {"conf": conf} if (conf is not None and detector == "yolo_sam3") else {}

    frame_to_objs = None
    tracker = None
    if tracks_json is not None:
        frame_to_objs, max_index = _load_tracks_from_json(Path(tracks_json))
        # El default derivado de los tracks solo aplica al recorrido completo desde 0;
        # con recorte (start_frame/frame_step) max_frames es una cuenta, no un rango.
        if max_frames is None and max_index >= 0 and start_frame == 0 and frame_step == 1:
            max_frames = max_index + 1
    else:
        tracker = _GreedyTracker()

    # Solo se detectan las clases que se usan: anclas siempre; objetos solo si no hay
    # tracks_json (con tracks_json los objetos vienen del JSON, no del detector).
    wanted = {FIELD_CLASS, YELLOW_CLASS, BLUE_CLASS}
    if tracks_json is None:
        wanted |= {ROBOT_CLASS, BALL_CLASS}
    needed_classes = [c for c in classes if c["name"] in wanted]

    fps = get_video_fps(video_path)
    if output_path is None:
        out_dir = PROJECT_ROOT / "notebooks" / "fase_4_homografia" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{video_path.stem}_minimap.mp4"
    output_path = Path(output_path)

    # Frames a procesar tras aplicar start_frame/frame_step (para dimensionar la barra).
    total_src = get_frame_count(video_path)
    strided = max(0, (total_src - int(start_frame) + int(frame_step) - 1) // int(frame_step))
    n_total = strided if max_frames is None else min(int(max_frames), strided)

    vh = VideoHomography(smooth_beta=smooth_beta)
    # El camino C hornea la orientación en H (amarilla siempre a x<L/2), así que el
    # minimap canónico ya queda orientado: NO se llama a renderer.orient_once.
    renderer = MinimapRenderer()
    last_composed = None
    n_frames = 0

    with open_video_writer(output_path, fps=fps) as append:
        for frame_index, frame in tqdm(
            iter_frames(video_path, max_frames, start_frame=start_frame, frame_step=frame_step),
            total=n_total,
            desc=f"minimap {video_path.stem}", unit="frame", leave=False, disable=not progress,
        ):
            n_frames += 1

            # Una sola llamada al detector pluggable devuelve todas las clases pedidas
            # (anclas + objetos). Con yolo_sam3 esto es 1 YOLO + box-prompts + green texto.
            dets = detect(frame, classes=needed_classes, bundle=bundle, **detect_kwargs)
            field_mask = _largest_mask(dets.get(FIELD_CLASS, []))
            yellow_mask = _largest_mask(dets.get(YELLOW_CLASS, []))
            blue_mask = _largest_mask(dets.get(BLUE_CLASS, []))
            # Centroides de portería (en la imagen) para fijar la orientación. Con
            # sam3_text la azul suele faltar (bc=None, basta la amarilla); yolo_sam3 da
            # ambas -> orientación más firme.
            yc = mask_centroid(yellow_mask)
            bc = mask_centroid(blue_mask)
            # Camino C: H sobre la alfombra + centroides de portería. solve_masks solo
            # usa color para detectar BLANCO (hue-agnóstico), así que el frame RGB de
            # iter_frames sirve sin convertir a BGR. Si no hay alfombra, se propaga.
            if field_mask is not None:
                H, _status = vh.update_masks(frame, field_mask, yc, bc)
            else:
                vh.n_propagated += 1
                H, _status = vh.prev_H, "propagated"

            # Objetos (id estable, clase, punto-pie): del JSON 2×2 o del mismo detector.
            if frame_to_objs is not None:
                objs = frame_to_objs.get(frame_index, [])
            else:
                objs = tracker.update(_objects_from_dets(dets))

            projected: list[tuple[int, str, float, float]] = []
            if H is not None and objs:
                feet = np.array([foot for _, _, foot in objs], dtype=np.float32)
                cm = project_points(feet, H)
                for (obj_id, cls, _), (x_cm, y_cm) in zip(objs, cm):
                    projected.append((obj_id, cls, float(x_cm), float(y_cm)))
            renderer.update(projected)

            base = frame
            if draw_overlay:
                # Reproyecta la cancha (rectangulo + circulo) y, si el frame fue
                # estimado con anclas, las 4 esquinas detectadas (como en la demo).
                corners = vh.last_corners if _status == "anchors" else None
                base = draw_field_overlay(frame, H, corners)

            composed = renderer.composite(base)
            last_composed = composed
            append(composed)

    sample_path = None
    if last_composed is not None:
        import imageio

        sample_path = output_path.with_name(f"{output_path.stem}_sample.jpg")
        imageio.imwrite(str(sample_path), last_composed)

    return {
        "video": output_path,
        "n_frames": n_frames,
        "homography": {
            "estimated": vh.n_estimated,
            "propagated": vh.n_propagated,
            "rejected": vh.n_rejected,
        },
        "sample_frame": sample_path,
    }
