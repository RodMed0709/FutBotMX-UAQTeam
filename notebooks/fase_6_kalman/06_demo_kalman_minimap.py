# -*- coding: utf-8 -*-
"""Fase 6 — demo mosaico con NUESTRO minimap de homografía (nb07) + Kalman como 4º panel,
en vez del heatmap outdated. Layout: Original | Tracking | [Segmentación] | Minimap-homografía
(trayectoria cm + estela + elipse de oclusión). NO toca el código de eventos/posesión del compa
(solo reusa render_obj_id_overlay, que es visualización, + nuestra homografía/Kalman). CPU local.
Uso (pod):  python 06_demo_kalman_minimap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.core import field_template as ft
from src.core.kalman_kinematics import compute_kalman_states, load_metric_result_from_json
from src.core.track_overlay import render_obj_id_overlay
from src.core.video_writer import open_video_writer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cm_positions_lines import compute_cm_positions_lines  # noqa: E402

REPO = Path("/workspace/FutBotMX-UAQTeam")
CLIPDIR = REPO / "outputs/inference/fase5_clips/IMG_9933_5m30"
TRACKS = CLIPDIR / "IMG_9933_5m30.json"
CLIP = CLIPDIR / "IMG_9933_5m30.mp4"
OUT = REPO / "outputs/inference/fase6_kalman/IMG_9933_5m30/IMG_9933_5m30_demo_homografia_kalman.mp4"
H = 540          # alto de cada panel
TRAIL = 45
MAX_SECONDS = 20.0


def _fit(frame, h):
    import cv2
    sh, sw = frame.shape[:2]
    return cv2.resize(frame, (max(1, int(round(h * sw / sh))), h))


def _label(img, text):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _minimap_frame(f, by_obj, f0):
    import cv2
    canvas, to_px = ft.render_field(scale=2.6, margin_cm=10.0)
    cv2.rectangle(canvas, to_px((0, 55)), to_px((18, 127)), (255, 230, 0), -1)
    cv2.rectangle(canvas, to_px((225, 55)), to_px((243, 127)), (40, 120, 255), -1)
    for _oid, (cls, states) in by_obj.items():
        col = (255, 120, 0) if cls == "orange_ball" else (40, 120, 255)
        pts = [to_px(states[k].xy_cm) for k in range(max(f0, f - TRAIL), f + 1) if k in states]
        for j in range(1, len(pts)):
            cv2.line(canvas, pts[j - 1], pts[j], col, 2, cv2.LINE_AA)
        s = states.get(f)
        if s is not None:
            p = to_px(s.xy_cm); r = 8 if cls == "orange_ball" else 10
            cv2.circle(canvas, p, r, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, r, (255, 255, 255), 1, cv2.LINE_AA)
            if s.source == "predicted":
                cv2.circle(canvas, p, max(4, int(round(s.pos_sigma_cm * 2.6))), (255, 0, 0), 2, cv2.LINE_AA)
    return canvas  # RGB


def main() -> None:
    import cv2

    cache = TRACKS.with_name(TRACKS.stem + "_cm_lines.json")
    raw = load_metric_result_from_json(cache) if cache.exists() else compute_cm_positions_lines(TRACKS)
    fps = raw.resumen.get("fps") or 30.0
    kres = compute_kalman_states(raw, fps=fps)
    by_obj = {o.obj_id: (o.cls, {s.frame_index: s for s in o.estados}) for o in kres.por_obj}
    all_f = sorted({s.frame_index for o in kres.por_obj for s in o.estados})
    f0 = all_f[0] if all_f else 0

    # paneles de video
    seg = CLIPDIR / "IMG_9933_5m30_seg.mp4"
    obj_id = CLIPDIR / "IMG_9933_5m30_obj_id.mp4"
    if not obj_id.exists():
        print("[demo] generando overlay de tracking…")
        obj_id = render_obj_id_overlay(TRACKS, video_path=CLIP, output_path=obj_id)
    cap_orig = cv2.VideoCapture(str(CLIP))
    cap_trk = cv2.VideoCapture(str(obj_id))
    cap_seg = cv2.VideoCapture(str(seg)) if seg.exists() else None
    if cap_seg is None:
        print("[demo] aviso: no hay _seg.mp4 (se genera en el pod) -> panel Segmentación omitido")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    max_frames = int(MAX_SECONDS * fps)
    n = 0
    with open_video_writer(OUT, fps=fps) as append:
        f = 0
        while n < max_frames:
            ok1, fr_orig = cap_orig.read()
            ok2, fr_trk = cap_trk.read()
            if not (ok1 and ok2):
                break
            cells = [_label(_fit(cv2.cvtColor(fr_orig, cv2.COLOR_BGR2RGB), H), "Original"),
                     _label(_fit(cv2.cvtColor(fr_trk, cv2.COLOR_BGR2RGB), H), "Tracking")]
            if cap_seg is not None:
                ok3, fr_seg = cap_seg.read()
                if ok3:
                    cells.append(_label(_fit(cv2.cvtColor(fr_seg, cv2.COLOR_BGR2RGB), H), "Segmentacion"))
            cells.append(_label(_fit(_minimap_frame(f, by_obj, f0), H), "Minimap homografia + Kalman"))
            append(cv2.hconcat(cells))
            n += 1; f += 1
    cap_orig.release(); cap_trk.release()
    if cap_seg is not None:
        cap_seg.release()
    print(f"[fase6] demo homografia+Kalman -> {OUT} ({n} frames, {n / fps:.1f}s)")


if __name__ == "__main__":
    main()
