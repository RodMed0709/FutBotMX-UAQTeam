"""T3 (fase_5 · Capa B) — posiciones métricas en cm.

Proyecta robots y balón de un **JSON de tracking extendido** (``include_masks=True``) a
**centímetros** sobre la cancha canónica (``field_template``), usando por defecto la homografía
**por líneas** (``homography_multifeature.VideoHomographyLines``) — la consolidada y más fiable
(error ~9–23 cm). El camino viejo **por máscaras** (``auto_homography.VideoHomography``) queda
disponible como ``homography="masks"`` (legacy).

Es la base de toda la Capa B (velocidad/distancia, heatmap, zonas, gol geométrico). NO
re-infiere modelos: la alfombra (``green_floor``) y los centroides de portería se leen del
propio JSON (vista ``frames[].detections`` con ``rle``); solo se leen los **píxeles del clip**
para detectar las líneas blancas dentro de la alfombra. En el camino por líneas el frame se
**redimensiona a la resolución de la máscara de alfombra** antes de estimar la H (así los foot
points del JSON, en esa misma resolución, se proyectan bien). Corre en **CPU local** (I/O de
video + ``pycocotools`` + ``cv2``; sin SAM3/YOLO).

Insumo de referencia: ``outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json``.
Solo aplica a video de **cámara superior** (homografía fiable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.core.auto_homography import VideoHomography
from src.core.homography import project_points
from src.core.homography_multifeature import VideoHomographyLines
from src.core.inference_schema import decode_rle
from src.core.minimap_pipeline import (
    BLUE_CLASS,
    FIELD_CLASS,
    YELLOW_CLASS,
    _largest_component,
    _load_tracks_from_json,
)


@dataclass
class MetricPosition:
    """Posición en cm de un objeto en un frame (``xy_cm`` es ``None`` si no hubo H)."""

    obj_id: int
    cls: str
    frame_index: int
    xy_cm: tuple[float, float] | None
    status_H: str


@dataclass
class MetricResult:
    posiciones: list[MetricPosition]
    resumen: dict
    # H px->cm usada en cada frame (None si no hubo H). Solo en memoria; no se serializa.
    H_por_frame: dict[int, "np.ndarray | None"] = field(default_factory=dict)


def _bbox_area(bbox) -> float:
    """Área de un bbox ``[x, y, w, h]``."""
    return float(bbox[2]) * float(bbox[3])


def _largest_green_rle(dets: list) -> dict | None:
    """``rle`` de la detección de ``green_floor`` de mayor bbox (o ``None`` si no hay)."""
    best, best_area = None, 0.0
    for d in dets:
        a = _bbox_area(d["bbox"])
        if d.get("rle") is not None and a > best_area:
            best, best_area = d["rle"], a
    return best


def _largest_centroid(dets: list) -> tuple[float, float] | None:
    """Centroide (campo ``centroid``) de la detección de mayor bbox, o ``None``."""
    best, best_area = None, 0.0
    for d in dets:
        a = _bbox_area(d["bbox"])
        if d.get("centroid") is not None and a > best_area:
            c = d["centroid"]
            best, best_area = (float(c[0]), float(c[1])), a
    return best


def _load_field_anchors(data: dict) -> dict[int, dict]:
    """Por frame: ``rle`` de ``green_floor`` + centroides de portería, del JSON extendido."""
    if not data.get("include_masks"):
        raise ValueError(
            "el JSON debe haberse generado con include_masks=True "
            "(green_floor con rle en frames[].detections)"
        )
    anchors: dict[int, dict] = {}
    for fr in data.get("frames", []):
        idx = int(fr["frame_index"])
        det = fr.get("detections", {})
        anchors[idx] = {
            "carpet_rle": _largest_green_rle(det.get(FIELD_CLASS, [])),
            "yc": _largest_centroid(det.get(YELLOW_CLASS, [])),
            "bc": _largest_centroid(det.get(BLUE_CLASS, [])),
        }
    return anchors


def _resolve_clip(tracks_json: Path, data: dict, video: str | Path | None) -> Path:
    """Resuelve el clip LOCAL. ``video`` explícito manda; si no, se busca junto al JSON
    (la ruta ``video`` del JSON apunta al pod ``/workspace/...``, no sirve en local)."""
    if video is not None:
        p = Path(video)
        if not p.exists():
            raise FileNotFoundError(f"no existe el video pasado: {p}")
        return p
    name = Path(data.get("video", "")).name
    candidate = tracks_json.parent / name
    if not candidate.exists():
        raise FileNotFoundError(
            f"no se encontró el clip local '{name}' junto al JSON ({tracks_json.parent}); "
            f"pásalo con video=<ruta>"
        )
    return candidate


def _solve_homographies(
    clip: Path, anchors: dict[int, dict], smooth_beta: float, homography: str
) -> tuple[dict[int, tuple], object, int]:
    """H por frame leyendo los píxeles del clip + alfombra/centroides del JSON. CPU local.

    ``homography`` ∈ {``"lines"`` (por líneas, default), ``"masks"`` (legacy por máscaras)}.
    En el camino por líneas el frame se redimensiona a la resolución de la máscara de alfombra
    antes de estimar la H. Returns ``(h_by_frame, vh, n_frames)`` con
    ``h_by_frame[idx] = (H | None, status)``.
    """
    import cv2

    vh: object = (
        VideoHomographyLines(smooth_beta=smooth_beta)
        if homography == "lines"
        else VideoHomography(smooth_beta=smooth_beta)
    )
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el clip: {clip}")
    h_by_frame: dict[int, tuple] = {}
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            a = anchors.get(idx)
            if homography == "lines":
                if a is not None and a["carpet_rle"] is not None:
                    carpet = _largest_component(decode_rle(a["carpet_rle"]))
                    hm, wm = carpet.shape[:2]
                    same = frame.shape[:2] == (hm, wm)
                    fr = frame if same else cv2.resize(frame, (wm, hm))
                    H, status, _ov = vh.update(fr, carpet, a["yc"], a["bc"])
                else:
                    # sin alfombra -> conserva la H previa (o None si aún no hay).
                    H, status = vh.H, ("kept" if vh.H is not None else "none")
                h_by_frame[idx] = (H, status)
            else:
                if a is not None and a["carpet_rle"] is not None:
                    carpet = _largest_component(decode_rle(a["carpet_rle"]))
                    yc, bc = a["yc"], a["bc"]
                else:
                    # sin green_floor -> máscara vacía: solve falla y VideoHomography propaga la
                    # H previa (status "propagated").
                    carpet = np.zeros(frame.shape[:2], dtype=np.uint8)
                    yc = bc = None
                h_by_frame[idx] = vh.update_masks(frame, carpet, yc, bc)
            idx += 1
    finally:
        cap.release()
    return h_by_frame, vh, idx


def compute_metric_positions(
    tracks_json: str | Path,
    video: str | Path | None = None,
    *,
    smooth_beta: float = 0.4,
    homography: str = "lines",
) -> MetricResult:
    """Proyecta robots/balón a cm. ``video`` por defecto = clip junto al JSON.

    ``homography`` ∈ {``"lines"`` (default, por líneas) | ``"masks"`` (legacy por máscaras)}.
    """
    if homography not in ("lines", "masks"):
        raise ValueError(f"homography inválida: {homography!r} (usa 'lines' o 'masks')")

    tracks_json = Path(tracks_json)
    data = json.loads(tracks_json.read_text(encoding="utf-8"))
    fps = data.get("fps")

    frame_to_objs, _max_index = _load_tracks_from_json(tracks_json)  # obj_id, cls, foot_xy
    anchors = _load_field_anchors(data)
    clip = _resolve_clip(tracks_json, data, video)

    h_by_frame, vh, n_frames = _solve_homographies(
        clip, anchors, smooth_beta, homography
    )

    posiciones: list[MetricPosition] = []
    for idx in sorted(frame_to_objs):
        H, status = h_by_frame.get(idx, (None, "missing"))
        for obj_id, cls, foot in frame_to_objs[idx]:
            if H is None:
                posiciones.append(MetricPosition(obj_id, cls, idx, None, status))
                continue
            xy = project_points(np.array([foot], dtype=np.float32), H)[0]
            posiciones.append(
                MetricPosition(obj_id, cls, idx, (float(xy[0]), float(xy[1])), status)
            )

    n_con_cm = sum(1 for p in posiciones if p.xy_cm is not None)
    frames_con_H = sum(1 for H, _ in h_by_frame.values() if H is not None)
    resumen = {
        "video": data.get("video"),
        "clip_local": str(clip),
        "fps": fps,
        "n_frames": n_frames,
        "homography": homography,
        "pct_H_valida": round(100.0 * frames_con_H / n_frames, 1) if n_frames else 0.0,
        "n_posiciones": len(posiciones),
        "n_con_cm": n_con_cm,
    }
    if homography == "lines":
        resumen["lines_stats"] = vh.stats()
    else:
        resumen["n_estimated"] = vh.n_estimated
        resumen["n_propagated"] = vh.n_propagated
        resumen["n_rejected"] = vh.n_rejected
        resumen["init_max_err_cm"] = vh.init_max_err_cm

    H_por_frame = {idx: H for idx, (H, _status) in h_by_frame.items()}
    return MetricResult(posiciones=posiciones, resumen=resumen, H_por_frame=H_por_frame)


def write_metric_positions_json(result: MetricResult, path: str | Path) -> Path:
    """Escribe el resultado a JSON (resumen + posiciones)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "posiciones": [
            {
                "obj_id": p.obj_id,
                "class": p.cls,
                "frame_index": p.frame_index,
                "xy_cm": list(p.xy_cm) if p.xy_cm is not None else None,
                "status_H": p.status_H,
            }
            for p in result.posiciones
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
