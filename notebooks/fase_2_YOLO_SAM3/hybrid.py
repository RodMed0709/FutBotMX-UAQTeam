"""HÍBRIDO (fase_2) — todo el pipeline hasta ahora, en uno.

YOLO (cada frame) -> CAJAS -> SAM3 box-prompt -> MÁSCARAS tight
   -> tracker IoU (IDs estables, color por track, coasting en oclusiones)
   -> green_floor por SAM3 text-prompt (cada N frames).

Combina: re-detección por frame (YOLO) + máscaras finas (SAM3, el centro) +
consistencia temporal e IDs (tracker). green_floor segmentado por SAM3.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from PIL import Image

import pipeline_yolo_sam3 as pl

decord.bridge.set_bridge("native")

# colores distintos por track (para ver IDs)
PALETTE = [
    (60, 130, 255), (255, 90, 90), (90, 255, 130), (255, 200, 0), (200, 100, 255),
    (0, 220, 220), (255, 140, 0), (140, 255, 0), (255, 0, 180), (120, 180, 255),
]
GREEN_COLOR = (40, 200, 70)


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def render_hybrid(video_path, out_path, models, device="cuda",
                  conf=0.4, green_every=10, iou_thr=0.3, max_age=8, alpha=0.5):
    yolo, tproc, tmodel, vproc, vmodel = models
    vr = decord.VideoReader(str(video_path))
    fps = float(vr.get_avg_fps()) or 30.0
    H, W = vr[0].shape[:2]
    n = len(vr)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracks = {}  # tid -> {cls,color,box,mask,age}
    next_tid = 0
    green = np.zeros((H, W), bool)

    for i in range(n):
        frame = vr[i].asnumpy()
        img = Image.fromarray(frame)
        res = yolo.predict(img, imgsz=960, conf=conf, device=device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        masks = pl.boxes_to_masks(tproc, tmodel, img, boxes.tolist(), device)

        assigned = set()
        for di in range(len(boxes)):
            box, c = boxes[di], int(cls[di])
            best_tid, best = None, iou_thr
            for tid, t in tracks.items():
                if t["cls"] != c or tid in assigned:
                    continue
                v = box_iou(box, t["box"])
                if v > best:
                    best, best_tid = v, tid
            if best_tid is None:
                best_tid = next_tid
                next_tid += 1
                tracks[best_tid] = {"cls": c, "color": PALETTE[best_tid % len(PALETTE)]}
            assigned.add(best_tid)
            tracks[best_tid].update(
                box=box, mask=(masks[di] if di < len(masks) else None), age=0)

        # coasting: tracks no vistos envejecen, se borran tras max_age
        for tid in list(tracks):
            if tid not in assigned:
                tracks[tid]["age"] = tracks[tid].get("age", 0) + 1
                if tracks[tid]["age"] > max_age:
                    del tracks[tid]

        if i % green_every == 0:
            green = pl.text_mask(vproc, vmodel, img, pl.GREEN_PROMPT, device)

        ov = frame.copy()
        if green.any():
            ov[green] = ((1 - 0.4) * ov[green] + 0.4 * np.array(GREEN_COLOR)).astype("uint8")
        for tid, t in tracks.items():
            m = t.get("mask")
            if m is not None and m.any():
                col = np.array(t["color"])
                ov[m] = ((1 - alpha) * ov[m] + alpha * col).astype("uint8")
                x0, y0, x1, y1 = [int(v) for v in t["box"]]
                cv2.rectangle(ov, (x0, y0), (x1, y1), t["color"], 2)
                cv2.putText(ov, f"{pl.CLASS_NAMES[t['cls']]}#{tid}", (x0, max(0, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, t["color"], 2)
        vw.write(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        if i % 25 == 0:
            torch.cuda.empty_cache()
            print(f"  frame {i}/{n} tracks={len(tracks)}", flush=True)
    vw.release()
    return str(out_path)


if __name__ == "__main__":
    REPO = Path("/workspace/FutBotMX-UAQTeam")
    F2 = REPO / "notebooks" / "fase_2_YOLO_SAM3"
    models = pl.load_models(REPO / "assets" / "sam3", F2 / "best.pt")
    vids = [
        ("IMG_9871", REPO / "data/raw/17Abril/Cámaras/IMG_9871.MOV"),
        ("video-836", REPO / "data/raw/17Abril/video-836_singular_display.mov"),
    ]
    for name, path in vids:
        print(f"=== híbrido {name} ===", flush=True)
        render_hybrid(path, F2 / f"demo_hybrid_{name}.mp4", models)
    print("HYBRID_DONE", flush=True)
