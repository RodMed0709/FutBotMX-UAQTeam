"""Overlay por ``obj_id`` para hacer visible el tracking (tarea obj_id_overlay).

Post-pase **desacoplado**: dado un **JSON de tracking** (esquema de
``inference_schema``) y el **video fuente**, escribe un **mp4 nuevo** donde la
identidad de cada objeto es visible:

- una **caja** por objeto (color **por clase**, de ``config.classes``);
- una **etiqueta** ``nombre #id`` (solo ``nombre`` en warm-up, ``obj_id == -1``);
- la **trayectoria** de cada ``obj_id`` (centroides de la vista ``tracks``), color de
  su clase, acotada a una **ventana deslizante** de los últimos N frames;
- **opcionalmente** el relleno de máscara (si el JSON trae ``rle`` y ``draw_masks``).

No re-infiere ni carga SAM3, no toca ``track_video`` ni ``overlay_detections``. Las
clases excluidas (default ``green_floor``) no se dibujan. ``cv2`` se importa de forma
perezosa; el recorrido es en streaming (memoria acotada).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from src.core.frame_extraction import iter_frames
from src.core.video_writer import open_video_writer
from src.utils import PROJECT_ROOT, get_abs_path

# Defaults de estilo (sobreescribibles por config/parametro).
_DEFAULTS = {
    "trajectory_window": 60,
    "overlay_excluded_classes": ["green_floor"],
    "overlay_alpha": 0.55,
    "overlay_line_scale": 0.0020,  # * max(H, W) -> grosor de linea
    "overlay_font_scale": 0.0011,  # * max(H, W) -> escala de fuente
}


def _load_env(env_path: Path) -> dict[str, str]:
    """Parseo simple de un .env (KEY = value), aplicando strip()."""
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


def _load_overlay_settings() -> dict:
    """Lee los ajustes de estilo del overlay de la config activa (best-effort).

    Si no hay ``.env``/config disponibles, devuelve los defaults de código (el
    post-pase no debe depender de la config para funcionar con parámetros explícitos).
    """
    settings = dict(_DEFAULTS)
    try:
        env = _load_env(PROJECT_ROOT / ".env")
        config_filename = env.get("CONFIG_FILENAME")
        if not config_filename:
            return settings
        config = json.loads(
            get_abs_path(f"configs/{config_filename}").read_text("utf-8")
        )
        viz = config.get("visualization", {})
    except (ValueError, FileNotFoundError, json.JSONDecodeError):
        return settings
    for key in settings:
        if key in viz:
            settings[key] = viz[key]
    return settings


def _color_map(payload: dict) -> dict[str, tuple[int, int, int]]:
    """Mapa ``clase -> color RGB`` desde el snapshot de config embebido en el JSON."""
    classes = payload.get("config", {}).get("classes", [])
    cmap: dict[str, tuple[int, int, int]] = {}
    for cls in classes:
        color = cls.get("color")
        if color is not None:
            cmap[cls["name"]] = tuple(int(c) for c in color)
    return cmap


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Negro o blanco segun la luminancia del color de fondo (RGB)."""
    r, g, b = bg
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if lum >= 128 else (255, 255, 255)


def _label(cls_name: str, obj_id: int) -> str:
    """Texto de la etiqueta: ``nombre #id`` o solo ``nombre`` en warm-up (id < 0)."""
    return cls_name if obj_id < 0 else f"{cls_name} #{obj_id}"


def _trajectories_by_obj(payload: dict) -> dict[int, dict]:
    """Por ``obj_id``: ``{"class", "points": [(frame_index, (cx, cy)), ...]}`` ordenado."""
    out: dict[int, dict] = {}
    for trk in payload.get("tracks", []):
        oid = int(trk["obj_id"])
        pts = [
            (int(o["frame_index"]), (float(o["centroid"][0]), float(o["centroid"][1])))
            for o in trk["observations"]
        ]
        pts.sort(key=lambda p: p[0])
        out[oid] = {"class": trk["class"], "points": pts}
    return out


def _compose_frame(
    frame: np.ndarray,
    frame_index: int,
    frame_by_index: dict[int, dict],
    traj_by_obj: dict[int, dict],
    color_map: dict[str, tuple[int, int, int]],
    *,
    excluded: set[str],
    window: int,
    alpha: float,
    thickness: int,
    font_scale: float,
    draw_masks: bool,
) -> np.ndarray:
    """Dibuja el overlay de un frame (máscara opcional -> trayectorias -> cajas/etiquetas).

    No muta ``frame`` (trabaja sobre una copia contigua) y devuelve el frame compuesto
    RGB ``uint8``. Las clases en ``excluded`` se omiten por completo.
    """
    import cv2

    out = np.ascontiguousarray(frame.copy())
    record = frame_by_index.get(frame_index)

    # 1) Relleno de mascara (opcional, debajo de todo).
    if draw_masks and record is not None:
        from src.core.inference_schema import decode_rle

        for cls_name, dets in record["detections"].items():
            if cls_name in excluded:
                continue
            color = color_map.get(cls_name, (200, 200, 200))
            for det in dets:
                if "rle" not in det:
                    warnings.warn(
                        "draw_masks=True pero el JSON no trae 'rle'; "
                        "se dibujan solo cajas/estela."
                    )
                    continue
                mask = decode_rle(det["rle"])
                if mask.shape != out.shape[:2]:
                    continue
                region = out[mask].astype(np.float32)
                blend = (1.0 - alpha) * region + alpha * np.array(
                    color, dtype=np.float32
                )
                out[mask] = blend.round().clip(0, 255).astype(np.uint8)

    # 2) Trayectorias (centroides en la ventana (f-N, f]).
    for trk in traj_by_obj.values():
        cls_name = trk["class"]
        if cls_name in excluded:
            continue
        color = color_map.get(cls_name, (200, 200, 200))
        pts = [
            (int(round(cx)), int(round(cy)))
            for fi, (cx, cy) in trk["points"]
            if frame_index - window < fi <= frame_index
        ]
        if len(pts) >= 2:
            cv2.polylines(out, [np.array(pts, dtype=np.int32)], False, color, thickness)

    # 3) Cajas + 4) etiquetas (encima).
    if record is not None:
        for cls_name, dets in record["detections"].items():
            if cls_name in excluded:
                continue
            color = color_map.get(cls_name, (200, 200, 200))
            for det in dets:
                x, y, w, h = (int(v) for v in det["bbox"])
                cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
                label = _label(cls_name, int(det["obj_id"]))
                (tw, th), base = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness)
                )
                ytop = max(0, y - th - base)
                cv2.rectangle(out, (x, ytop), (x + tw, y), color, -1)
                cv2.putText(
                    out,
                    label,
                    (x, max(th, y - base)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    _text_color(color),
                    max(1, thickness),
                    cv2.LINE_AA,
                )
    return out


def render_obj_id_overlay(
    json_path: Path | str,
    video_path: Path | str | None = None,
    output_path: Path | None = None,
    draw_masks: bool = False,
    trajectory_window: int | None = None,
    excluded_classes: list[str] | None = None,
) -> Path:
    """Escribe un mp4 con el overlay por ``obj_id`` a partir de un JSON de tracking.

    Args:
        json_path: JSON de tracking de entrada (debe ser ``mode="tracking"``).
        video_path: video fuente; si es ``None``, se toma de la cabecera del JSON.
        output_path: mp4 de salida; si es ``None``, ``<json_stem>_obj_id.mp4`` junto al
            JSON (no pisa el mp4 de inferencia).
        draw_masks: si ``True`` y el JSON trae ``rle``, además rellena la máscara
            (color de clase); sin ``rle`` se avisa y se degrada a cajas/estela.
        trajectory_window: N (frames) de la ventana de estela; ``None`` ⇒ config.
        excluded_classes: clases a no dibujar; ``None`` ⇒ config (default
            ``["green_floor"]``).

    Returns:
        La ruta (``Path``) del mp4 escrito.

    Raises:
        ValueError: si el JSON no es de ``mode="tracking"``.
        FileNotFoundError: si el JSON o el video no resuelven.
    """
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if payload.get("mode") != "tracking":
        raise ValueError(
            "render_obj_id_overlay requiere un JSON de mode='tracking' "
            f"(se recibio mode={payload.get('mode')!r})."
        )

    settings = _load_overlay_settings()
    window = (
        int(trajectory_window)
        if trajectory_window is not None
        else int(settings["trajectory_window"])
    )
    excluded = set(
        excluded_classes
        if excluded_classes is not None
        else settings["overlay_excluded_classes"]
    )
    alpha = float(settings["overlay_alpha"])

    video = Path(video_path) if video_path is not None else Path(payload["video"])
    fps = float(payload["fps"])

    if output_path is None:
        json_p = Path(json_path)
        output_path = json_p.with_name(f"{json_p.stem}_obj_id.mp4")
    output_path = Path(output_path)

    color_map = _color_map(payload)
    frame_by_index = {f["frame_index"]: f for f in payload.get("frames", [])}
    traj_by_obj = _trajectories_by_obj(payload)

    # Estilo derivado de la resolucion.
    res = payload.get("resolution", {})
    ref = max(int(res.get("height", 1080)), int(res.get("width", 1920)))
    thickness = max(1, round(float(settings["overlay_line_scale"]) * ref))
    font_scale = max(0.4, float(settings["overlay_font_scale"]) * ref)

    with open_video_writer(output_path, fps=fps) as append:
        for frame_index, frame in iter_frames(video):
            append(
                _compose_frame(
                    frame,
                    frame_index,
                    frame_by_index,
                    traj_by_obj,
                    color_map,
                    excluded=excluded,
                    window=window,
                    alpha=alpha,
                    thickness=thickness,
                    font_scale=font_scale,
                    draw_masks=draw_masks,
                )
            )

    return output_path
