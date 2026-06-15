"""Minimap fase_4 SOBRE SAM3 (corre en el POD GPU).

Une lo que el equipo quiere para categoria Profesional:
- **YOLO (best.pt)** detecta cajas: robot, orange_ball, yellow_zone, blue_zone.
- **SAM3 text-prompt** segmenta `green_floor` (la alfombra) -> mascara de alfombra.
- **Homografia** (`auto_homography.solve_masks`): white-lines DENTRO de la alfombra
  SAM3 -> 4 esquinas del rectangulo interior; orientacion por centroides de las
  cajas yellow/blue de YOLO. => homografia construida explicitamente sobre SAM3.
- **Tracking** greedy NN sobre los foot-points de las cajas robot/balon (IDs).
- **Minimap** canonico con trails, compuesto sobre el video.

Uso (en el pod):
    cd /workspace/FutBotMX-UAQTeam/notebooks/fase_4_homografia
    python pod_minimap_sam3.py <video> <out.mp4> [n_src_frames] [every]
"""
import os
import sys

import numpy as np
import cv2
import torch
from PIL import Image
import decord

REPO = "/workspace/FutBotMX-UAQTeam"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # auto_homography, minimap_auto
sys.path.insert(0, os.path.join(REPO, "src", "core"))                   # field_template (flat import)
sys.path.insert(0, os.path.join(REPO, "notebooks", "fase_2_YOLO_SAM3"))  # pipeline_yolo_sam3

import imageio.v2 as imageio
import field_template as ft
import auto_homography as ah
from minimap_auto import Minimap, GreedyTracker, draw_field_overlay
import pipeline_yolo_sam3 as p2

SAM3_PATH = os.path.join(REPO, "assets", "sam3")
YOLO_PT = os.path.join(REPO, "notebooks", "fase_2_YOLO_SAM3", "best.pt")
DEVICE = "cuda"
ROBOT, BALL, YELLOW, BLUE = 0, 1, 2, 3


def box_centroid(boxes, cls, target):
    """Centroide (cx,cy) de la primera caja de clase ``target`` (o None)."""
    for b, c in zip(boxes, cls):
        if int(c) == target:
            x1, y1, x2, y2 = b
            return (float((x1 + x2) / 2), float((y1 + y2) / 2))
    return None


def largest_component(mask):
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m * 255
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8) * 255


def run(video, out_path, models, start=0, n_src=300, every=2, conf=0.25):
    yolo, tproc, tmodel, vproc, vmodel = models
    vr = decord.VideoReader(video)
    fps = vr.get_avg_fps() or 30.0
    end = min(start + n_src, len(vr))
    writer = imageio.get_writer(out_path, fps=max(1.0, fps / every), macro_block_size=1, codec="libx264")

    vh = ah.VideoHomography(smooth_beta=0.4)
    tracker, mini = GreedyTracker(), Minimap()
    n_obj = 0
    for i in range(start, end, every):
        rgb = vr[i].asnumpy()
        pil = Image.fromarray(rgb)

        res = yolo.predict(pil, imgsz=960, conf=conf, device=DEVICE, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()

        green = p2.text_mask(vproc, vmodel, pil, p2.GREEN_PROMPT, DEVICE)
        carpet = largest_component(np.asarray(green))

        yc = box_centroid(boxes, cls, YELLOW)
        bc = box_centroid(boxes, cls, BLUE)
        # white es invariante al orden de canal -> pasar RGB esta bien.
        H, src = vh.update_masks(rgb, carpet, yc, bc)

        dets = []
        for b, c in zip(boxes, cls):
            x1, y1, x2, y2 = b
            ci = int(c)
            if ci == ROBOT:
                dets.append(("robot", (float((x1 + x2) / 2), float(y2))))
            elif ci == BALL:
                dets.append(("ball", (float((x1 + x2) / 2), float((y1 + y2) / 2))))
        objs = tracker.update(dets)
        n_obj += len(objs)

        projected = []
        if H is not None and objs:
            feet = np.array([p for _, _, p in objs], np.float32).reshape(-1, 1, 2)
            cm = cv2.perspectiveTransform(feet, H).reshape(-1, 2)
            projected = [(oid, cl, float(x), float(y)) for (oid, cl, _), (x, y) in zip(objs, cm)]
        mini.update(projected)

        frame = rgb.copy()
        draw_field_overlay(frame, vh.prev_H, vh.last_corners if src == "anchors" else None)
        writer.append_data(mini.composite(frame))

    writer.close()
    return {"out": out_path, "frames": len(range(start, end, every)),
            "estimated": vh.n_estimated, "propagated": vh.n_propagated,
            "rejected": vh.n_rejected, "objs": n_obj}


# Segmentos para generar varios videos (clip estable de cada uno).
CAM = "/workspace/FutBotMX-UAQTeam/data/raw/18abril/Camara_superior"
JOBS = [
    (f"{CAM}/IMG_9933.MOV", "IMG_9933_a", 1800, 300),
    (f"{CAM}/IMG_9933.MOV", "IMG_9933_b", 9000, 300),
    (f"{CAM}/IMG_9933.MOV", "IMG_9933_c", 15000, 300),
    (f"{CAM}/IMG_9938.MOV", "IMG_9938_a", 1800, 300),
    (f"{CAM}/IMG_9938.MOV", "IMG_9938_b", 9000, 300),
]


def main():
    out_dir = "/workspace/FutBotMX-UAQTeam/notebooks/fase_4_homografia/outputs"
    os.makedirs(out_dir, exist_ok=True)
    models = p2.load_models(SAM3_PATH, YOLO_PT, DEVICE)
    for video, name, start, n in JOBS:
        out = os.path.join(out_dir, f"{name}_minimap.mp4")
        r = run(video, out, models, start=start, n_src=n, every=2)
        print(name, r)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        main()
    else:
        m = p2.load_models(SAM3_PATH, YOLO_PT, DEVICE)
        video = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else "minimap_sam3.mp4"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        print(run(video, out, m, start=start, n_src=n, every=2))
