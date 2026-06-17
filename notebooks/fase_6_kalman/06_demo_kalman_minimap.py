# -*- coding: utf-8 -*-
"""Fase 6 — demo mosaico AUTOCONTENIDO (no toca código de eventos/posesión/demo del compa).
Dibuja TODO a la resolución del JSON (redimensiona el frame del clip) para que cajas/máscaras
alineen. Paneles: Original | Segmentación | Tracking (cajas+id+estela, SIN máscara) |
Minimap homografía+Kalman | Heatmap (homografía, nuestro, acumulado). CPU local.
Uso (pod):  python 06_demo_kalman_minimap.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from src.core import field_template as ft
from src.core.inference_schema import decode_rle
from src.core.kalman_kinematics import compute_kalman_states, load_metric_result_from_json
from src.core.video_writer import open_video_writer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cm_positions_lines import compute_cm_positions_lines  # noqa: E402

REPO = Path("/workspace/FutBotMX-UAQTeam")
CD = REPO / "outputs/inference/fase5_clips/IMG_9933_5m30"
TRACKS = CD / "IMG_9933_5m30.json"
CLIP = CD / "IMG_9933_5m30.mp4"
OUT = REPO / "outputs/inference/fase6_kalman/IMG_9933_5m30/IMG_9933_5m30_demo_homografia_kalman.mp4"
H = 420
TRAIL = 45
MAX_SECONDS = 20.0
HEAT_BIN = 6.0  # cm por celda
COLORS = {  # RGB
    "robot": (255, 60, 60), "robot_a": (219, 0, 255), "robot_b": (230, 30, 200),
    "orange_ball": (255, 140, 0), "yellow_zone": (255, 230, 0),
    "blue_zone": (40, 120, 255), "green_floor": (50, 220, 70),
}
MOVING = {"orange_ball", "robot", "robot_a", "robot_b"}


def _fit(frame, h):
    import cv2
    sh, sw = frame.shape[:2]
    return cv2.resize(frame, (max(1, int(round(h * sw / sh))), h))


def _label(img, text):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    import cv2

    data = json.loads(TRACKS.read_text(encoding="utf-8"))
    res = data.get("resolution", {})
    Wt, Ht = int(res.get("width", 0)), int(res.get("height", 0))
    frames = {int(fr["frame_index"]): fr.get("detections", {}) for fr in data.get("frames", [])}

    cache = TRACKS.with_name(TRACKS.stem + "_cm_lines.json")
    raw = load_metric_result_from_json(cache) if cache.exists() else compute_cm_positions_lines(TRACKS)
    fps = raw.resumen.get("fps") or 30.0
    kres = compute_kalman_states(raw, fps=fps)
    by_obj = {o.obj_id: (o.cls, {s.frame_index: s for s in o.estados}) for o in kres.por_obj}
    all_f = sorted({s.frame_index for o in kres.por_obj for s in o.estados})
    f0 = all_f[0] if all_f else 0

    # rejilla del heatmap (cm) + estado acumulado
    cols = int(np.ceil(ft.LENGTH_CM / HEAT_BIN)); rowsg = int(np.ceil(ft.WIDTH_CM / HEAT_BIN))
    grid = np.zeros((rowsg, cols), float)
    trails = defaultdict(lambda: deque(maxlen=TRAIL))  # tracking px trails

    cap = cv2.VideoCapture(str(CLIP))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    max_frames = int(MAX_SECONDS * fps)
    n = 0
    base_field, to_px = ft.render_field(scale=2.6, margin_cm=10.0)
    with open_video_writer(OUT, fps=fps) as append:
        f = 0
        while n < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if Wt and (bgr.shape[1] != Wt or bgr.shape[0] != Ht):
                bgr = cv2.resize(bgr, (Wt, Ht))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = frames.get(f, {})

            # --- Segmentación: máscaras coloreadas (rle) ---
            seg = rgb.copy()
            for cls, lst in dets.items():
                col = np.array(COLORS.get(cls, (200, 200, 200)))
                for d in lst:
                    if d.get("rle"):
                        m = decode_rle(d["rle"]).astype(bool)
                        if m.shape != seg.shape[:2]:
                            m = cv2.resize(m.astype(np.uint8), (seg.shape[1], seg.shape[0])) > 0
                        seg[m] = (0.5 * seg[m] + 0.5 * col).astype(np.uint8)

            # --- Tracking: cajas + id + estela, SIN máscara ---
            trk = rgb.copy()
            for cls, lst in dets.items():
                if cls not in MOVING:
                    continue
                col = COLORS.get(cls, (200, 200, 200))
                for d in lst:
                    oid = d.get("obj_id")
                    if d.get("bbox"):
                        x, y, w, hh = [int(v) for v in d["bbox"]]
                        cv2.rectangle(trk, (x, y), (x + w, y + hh), col, 2)
                        cv2.putText(trk, f"{cls[:5]} #{oid}", (x, max(0, y - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                    if d.get("centroid") and oid is not None:
                        trails[oid].append((int(d["centroid"][0]), int(d["centroid"][1])))
                for oid, pts in trails.items():
                    for j in range(1, len(pts)):
                        cv2.line(trk, pts[j - 1], pts[j], (255, 255, 0), 1, cv2.LINE_AA)

            # --- Minimap homografía + Kalman ---
            mm = base_field.copy()
            cv2.rectangle(mm, to_px((0, 55)), to_px((18, 127)), (255, 230, 0), -1)
            cv2.rectangle(mm, to_px((225, 55)), to_px((243, 127)), (40, 120, 255), -1)
            for _oid, (cls, states) in by_obj.items():
                cc = (255, 120, 0) if cls == "orange_ball" else (40, 120, 255)
                pts = [to_px(states[k].xy_cm) for k in range(max(f0, f - TRAIL), f + 1) if k in states]
                for j in range(1, len(pts)):
                    cv2.line(mm, pts[j - 1], pts[j], cc, 2, cv2.LINE_AA)
                s = states.get(f)
                if s is not None:
                    p = to_px(s.xy_cm); r = 8 if cls == "orange_ball" else 10
                    cv2.circle(mm, p, r, cc, -1, cv2.LINE_AA); cv2.circle(mm, p, r, (255, 255, 255), 1)
                    if s.source == "predicted":
                        cv2.circle(mm, p, max(4, int(round(s.pos_sigma_cm * 2.6))), (255, 0, 0), 2)
                    # acumula heatmap
                    cx = min(max(s.xy_cm[0], 0), ft.LENGTH_CM - 1e-6); cy = min(max(s.xy_cm[1], 0), ft.WIDTH_CM - 1e-6)
                    grid[int(cy / HEAT_BIN), int(cx / HEAT_BIN)] += 1.0

            # --- Heatmap homografía (nuestro, acumulado) ---
            hm = base_field.copy()
            if grid.max() > 0:
                g = (255 * grid / grid.max()).astype(np.uint8)
                gcol = cv2.applyColorMap(cv2.resize(g, (hm.shape[1], hm.shape[0])), cv2.COLORMAP_JET)
                gcol = cv2.cvtColor(gcol, cv2.COLOR_BGR2RGB)
                hm = (0.55 * hm + 0.45 * gcol).astype(np.uint8)

            cells = [_label(_fit(rgb, H), "Original"),
                     _label(_fit(seg, H), "Segmentacion"),
                     _label(_fit(trk, H), "Tracking"),
                     _label(_fit(mm, H), "Minimap homografia+Kalman"),
                     _label(_fit(hm, H), "Heatmap (homografia)")]
            append(cv2.hconcat(cells))
            n += 1; f += 1
    cap.release()
    print(f"[fase6] demo 5 paneles -> {OUT} ({n} frames, {n / fps:.1f}s)")


if __name__ == "__main__":
    main()
