# -*- coding: utf-8 -*-
"""Fase 6 — posiciones en cm usando la homografía POR LÍNEAS de Rodrigo (nb07,
``VideoHomographyLines``), NO el ``auto_homography`` viejo que truena.

Arregla el size-mismatch de T3: la carpet-RLE del tracking JSON está a resolución de
inferencia y el .mp4 del demo a otra → aquí **redimensiono el frame a la resolución de la
máscara** antes de ``VideoHomographyLines.update`` (que detecta líneas blancas). Los foot
points del tracking JSON ya están en esa misma resolución → la H (px→cm) los proyecta bien.
CPU local (lee el .mp4 + cv2; sin SAM3/YOLO: la carpet y centroides salen del JSON).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.homography import project_points
from src.core.homography_multifeature import VideoHomographyLines
from src.core.inference_schema import decode_rle
from src.core.metric_positions import (
    MetricPosition,
    MetricResult,
    _load_field_anchors,
    _resolve_clip,
)
from src.core.minimap_pipeline import _largest_component, _load_tracks_from_json


def compute_cm_positions_lines(
    tracks_json: str | Path,
    video: str | Path | None = None,
    *,
    smooth_beta: float = 0.4,
    min_overlap: float = 0.40,
) -> MetricResult:
    """Proyecta foot points a cm con la homografía por líneas. Devuelve MetricResult (cm)."""
    import cv2

    tracks_json = Path(tracks_json)
    data = json.loads(tracks_json.read_text(encoding="utf-8"))
    fps = data.get("fps")
    frame_to_objs, _max = _load_tracks_from_json(tracks_json)
    anchors = _load_field_anchors(data)
    clip = _resolve_clip(tracks_json, data, video)

    vh = VideoHomographyLines(min_overlap=min_overlap, smooth_beta=smooth_beta)
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el clip: {clip}")
    h_by: dict[int, tuple] = {}
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            a = anchors.get(idx)
            if a is not None and a["carpet_rle"] is not None:
                carpet = _largest_component(decode_rle(a["carpet_rle"]))
                hm, wm = carpet.shape[:2]
                fr = frame if frame.shape[:2] == (hm, wm) else cv2.resize(frame, (wm, hm))
                H, status, _ov = vh.update(fr, carpet, a["yc"], a["bc"])
            else:
                H, status = (vh.H, "kept" if vh.H is not None else "none")
            h_by[idx] = (H, status)
            idx += 1
    finally:
        cap.release()

    pos: list[MetricPosition] = []
    for fidx in sorted(frame_to_objs):
        H, status = h_by.get(fidx, (None, "none"))
        for obj_id, cls, foot in frame_to_objs[fidx]:
            if H is None:
                pos.append(MetricPosition(obj_id, cls, fidx, None, status))
                continue
            xy = project_points(np.array([foot], dtype=np.float32), H)[0]
            pos.append(MetricPosition(obj_id, cls, fidx, (float(xy[0]), float(xy[1])), status))

    n_cm = sum(1 for p in pos if p.xy_cm is not None)
    resumen = {
        "fps": fps, "units": "cm", "n_frames": idx, "n_con_cm": n_cm,
        "homography": "VideoHomographyLines (nb07)", "lines_stats": vh.stats(),
    }
    return MetricResult(posiciones=pos, resumen=resumen)
