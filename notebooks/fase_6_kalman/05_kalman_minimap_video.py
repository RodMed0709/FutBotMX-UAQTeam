# -*- coding: utf-8 -*-
"""Fase 6 — VIDEO del minimap Kalman: trayectoria en cm con estela + elipse de incertidumbre
que CRECE en las oclusiones (frames predict-only). Visual del aporte del Kalman. CPU local.
Uso (pod):  python 05_kalman_minimap_video.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.core import field_template as ft
from src.core.kalman_kinematics import compute_kalman_states, load_metric_result_from_json
from src.core.video_writer import open_video_writer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cm_positions_lines import compute_cm_positions_lines  # noqa: E402

REPO = Path("/workspace/FutBotMX-UAQTeam")
TRACKS = REPO / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
OUT = REPO / "outputs/inference/fase6_kalman/IMG_9933_5m30/IMG_9933_5m30_kalman_minimap.mp4"
MAX_SECONDS = 20.0
TRAIL = 45  # frames de estela


def main() -> None:
    import cv2

    cache = TRACKS.with_name(TRACKS.stem + "_cm_lines.json")
    raw = load_metric_result_from_json(cache) if cache.exists() else compute_cm_positions_lines(TRACKS)
    fps = raw.resumen.get("fps") or 30.0
    kres = compute_kalman_states(raw, fps=fps)

    # estado por (obj, frame): (xy, source, sigma) + clase
    by_obj = {}
    for o in kres.por_obj:
        by_obj[o.obj_id] = (o.cls, {s.frame_index: s for s in o.estados})
    all_frames = sorted({s.frame_index for o in kres.por_obj for s in o.estados})
    if not all_frames:
        print("sin estados"); return
    f0, f1 = all_frames[0], min(all_frames[-1], all_frames[0] + int(MAX_SECONDS * fps))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open_video_writer(OUT, fps=fps) as append:
        for f in range(f0, f1 + 1):
            canvas, to_px = ft.render_field(scale=2.6, margin_cm=10.0)
            # porterías
            cv2.rectangle(canvas, to_px((0, 55)), to_px((18, 127)), (255, 230, 0), -1)
            cv2.rectangle(canvas, to_px((225, 55)), to_px((243, 127)), (40, 120, 255), -1)
            for oid, (cls, states) in by_obj.items():
                col = (255, 120, 0) if cls == "orange_ball" else (40, 120, 255)
                # estela
                pts = [to_px(states[k].xy_cm) for k in range(max(f0, f - TRAIL), f + 1) if k in states]
                for j in range(1, len(pts)):
                    cv2.line(canvas, pts[j - 1], pts[j], col, 2, cv2.LINE_AA)
                # actual + elipse de incertidumbre si está ocluido (predicted)
                s = states.get(f)
                if s is not None:
                    p = to_px(s.xy_cm)
                    r = 8 if cls == "orange_ball" else 10
                    cv2.circle(canvas, p, r, col, -1, cv2.LINE_AA)
                    cv2.circle(canvas, p, r, (255, 255, 255), 1, cv2.LINE_AA)
                    if s.source == "predicted":
                        rad = max(4, int(round(s.pos_sigma_cm * 2.6)))
                        cv2.circle(canvas, p, rad, (255, 0, 0), 2, cv2.LINE_AA)  # incertidumbre
                        cv2.putText(canvas, "OCCLUDED (predict)", (p[0] + 8, p[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Kalman cm-space trajectory + occlusion uncertainty",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            append(canvas)  # render_field da RGB
            n += 1
    print(f"[fase6] video minimap Kalman -> {OUT} ({n} frames, {n / fps:.1f}s)")


if __name__ == "__main__":
    main()
