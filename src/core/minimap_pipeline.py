"""Driver de fase_4: video -> homografia por frame -> minimap con trayectorias.

Orquesta las piezas de fase_4:

1. **Trayectorias** de robots y balon: se reusan de un JSON de tracking ya
   generado (``tracks_json``, salida de ``track_video``) o, si no se da, se
   detectan y se siguen **en el mismo paso** con un tracker greedy por vecino mas
   cercano (autocontenido, sin dependencias externas de tracking). De cada objeto
   se toma el **punto de contacto con el piso** (centro-inferior de la caja para
   robots; centroide para el balon).
2. **Homografia por frame** (camino C): se segmenta con SAM3-texto la alfombra
   (``green_floor``) y se toman los centroides de portería (``yellow_zone``/
   ``blue_zone``); ``src.core.auto_homography.VideoHomography.update_masks`` estima
   ``H`` (img->cm) sobre el rectángulo interior de líneas + orientación por color de
   portería, con lock de arranque, gate de consistencia temporal, EMA y propagación.
3. **Render**: se proyectan las posiciones al campo y se dibujan como trails en el
   minimap (``src.core.minimap``), compuesto sobre cada frame y escrito a un mp4.

El JSON de tracking y los frames del video se recorren con el **mismo** muestreo
contiguo (``iter_frames``), por lo que los ``frame_index`` coinciden.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.auto_homography import VideoHomography  # camino C
from src.core.frame_extraction import get_frame_count, get_video_fps, iter_frames
from src.core.homography import mask_centroid, project_points
from src.core.inference_schema import mask_to_bbox_centroid
from src.core.minimap import MinimapRenderer
from src.core.overlay import overlay_detections
from src.core.sam3_loader import Sam3Bundle, load_sam3
from src.core.segmentation import _load_classes, segment_with_text
from src.core.video_writer import open_video_writer
from src.utils import PROJECT_ROOT

FIELD_CLASS = "green_floor"
YELLOW_CLASS = "yellow_zone"
BLUE_CLASS = "blue_zone"
ROBOT_CLASS = "robot"
BALL_CLASSES = {"orange_ball", "ball"}

_DEFAULT_PROMPTS = {
    FIELD_CLASS: "green playing surface with lines",
    YELLOW_CLASS: "yellow zone",
    BLUE_CLASS: "blue zone",
    ROBOT_CLASS: "robot",
    "orange_ball": "orange ball",
}


def _prompt_for(class_name: str, classes: list[dict]) -> str:
    """Prompt SAM3 de una clase: el de la config si existe, si no el default."""
    for c in classes:
        if c["name"] == class_name and c.get("sam3_prompts"):
            return c["sam3_prompts"][0]
    return _DEFAULT_PROMPTS.get(class_name, class_name)


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


def _detect_objects(frame, robot_prompt, ball_prompt, bundle):
    """Detecta robots y balon en un frame -> lista de ``(class, foot_xy)``."""
    dets: list[tuple[str, tuple[float, float]]] = []
    for d in segment_with_text(frame, robot_prompt, bundle):
        geom = mask_to_bbox_centroid(d.mask)
        if geom is not None:
            dets.append((ROBOT_CLASS, _foot_point(ROBOT_CLASS, geom[0])))
    for d in segment_with_text(frame, ball_prompt, bundle):
        geom = mask_to_bbox_centroid(d.mask)
        if geom is not None:
            dets.append(("orange_ball", _foot_point("orange_ball", geom[0])))
    return dets


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
        output_path: ruta del mp4 de salida. Si es ``None`` se escribe en
            ``notebooks/fase_4_homografia/outputs/<stem>_minimap.mp4``.
        max_frames: tope de frames. Con ``tracks_json`` se usa el rango cubierto por
            los tracks si es ``None``; sin tracks, ``None`` recorre todo el video.
        bundle: modelo SAM3 cargado; si ``None`` se carga con ``load_sam3()``.
        draw_overlay: si ``True`` dibuja las mascaras de anclas sobre el frame
            (depuracion de la homografia).
        smooth_beta: suavizado temporal de la homografia.
        progress: barra de progreso ``tqdm``.

    Returns:
        ``{"video", "n_frames", "homography": {"estimated", "propagated"}, "sample_frame"}``.
    """
    from tqdm.auto import tqdm

    video_path = Path(video_path)
    classes = _load_classes()
    bundle = bundle or load_sam3()

    frame_to_objs = None
    tracker = None
    if tracks_json is not None:
        frame_to_objs, max_index = _load_tracks_from_json(Path(tracks_json))
        if max_frames is None and max_index >= 0:
            max_frames = max_index + 1
    else:
        tracker = _GreedyTracker()

    p_field = _prompt_for(FIELD_CLASS, classes)
    p_yellow = _prompt_for(YELLOW_CLASS, classes)
    p_blue = _prompt_for(BLUE_CLASS, classes)
    p_robot = _prompt_for(ROBOT_CLASS, classes)
    p_ball = _prompt_for("orange_ball", classes)
    anchor_classes = [c for c in classes if c["name"] in (FIELD_CLASS, YELLOW_CLASS, BLUE_CLASS)]

    fps = get_video_fps(video_path)
    if output_path is None:
        out_dir = PROJECT_ROOT / "notebooks" / "fase_4_homografia" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{video_path.stem}_minimap.mp4"
    output_path = Path(output_path)

    n_total = get_frame_count(video_path)
    if max_frames is not None:
        n_total = min(int(max_frames), n_total)

    vh = VideoHomography(smooth_beta=smooth_beta)
    # El camino C hornea la orientación en H (amarilla siempre a x<L/2), así que el
    # minimap canónico ya queda orientado: NO se llama a renderer.orient_once.
    renderer = MinimapRenderer()
    last_composed = None
    n_frames = 0

    with open_video_writer(output_path, fps=fps) as append:
        for frame_index, frame in tqdm(
            iter_frames(video_path, max_frames), total=n_total,
            desc=f"minimap {video_path.stem}", unit="frame", leave=False, disable=not progress,
        ):
            n_frames += 1

            # Segmentar las anclas una sola vez (SAM3 es lo caro); reusar para la
            # homografia y, si aplica, para el overlay de depuracion.
            f_dets = segment_with_text(frame, p_field, bundle)
            y_dets = segment_with_text(frame, p_yellow, bundle)
            b_dets = segment_with_text(frame, p_blue, bundle)
            field_mask = _largest_mask(f_dets)
            yellow_mask = _largest_mask(y_dets)
            blue_mask = _largest_mask(b_dets)
            # Centroides de portería (en la imagen) para fijar la orientación. La azul
            # suele venir vacía (SAM3 no la segmenta) -> bc=None; basta la amarilla.
            yc = mask_centroid(yellow_mask)
            bc = mask_centroid(blue_mask)
            # Camino C: H sobre la alfombra (SAM3) + centroides de portería. solve_masks
            # solo usa color para detectar BLANCO (hue-agnóstico), así que el frame RGB
            # de iter_frames sirve sin convertir a BGR. Si no hay alfombra, se propaga.
            if field_mask is not None:
                H, _status = vh.update_masks(frame, field_mask, yc, bc)
            else:
                vh.n_propagated += 1
                H = vh.prev_H

            # Objetos (id estable, clase, punto-pie).
            if frame_to_objs is not None:
                objs = frame_to_objs.get(frame_index, [])
            else:
                objs = tracker.update(_detect_objects(frame, p_robot, p_ball, bundle))

            projected: list[tuple[int, str, float, float]] = []
            if H is not None and objs:
                feet = np.array([foot for _, _, foot in objs], dtype=np.float32)
                cm = project_points(feet, H)
                for (obj_id, cls, _), (x_cm, y_cm) in zip(objs, cm):
                    projected.append((obj_id, cls, float(x_cm), float(y_cm)))
            renderer.update(projected)

            base = frame
            if draw_overlay:
                dets = {FIELD_CLASS: f_dets, YELLOW_CLASS: y_dets, BLUE_CLASS: b_dets}
                base = overlay_detections(frame, dets, classes=anchor_classes)

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
