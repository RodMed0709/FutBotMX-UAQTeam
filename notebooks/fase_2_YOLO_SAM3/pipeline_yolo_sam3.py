"""Pipeline SAM3-céntrico (fase_2): YOLO localiza (cajas) -> SAM3 box-prompt hace
las MÁSCARAS; green_floor por SAM3 text-prompt. SAM3 = centro, YOLO solo acelera.
Corre en GPU del pod. Genera videos de overlay.

Componentes:
- YOLO (best.pt)            -> cajas rápidas (robot, orange_ball, yellow_zone)
- Sam3TrackerModel          -> box-prompt: caja -> máscara fina (el corazón)
- Sam3VideoModel (text)     -> green_floor (text-prompt, región estática)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, Sam3TrackerModel, Sam3TrackerProcessor
from ultralytics import YOLO

decord.bridge.set_bridge("native")

CLASS_NAMES = {0: "robot", 1: "orange_ball", 2: "yellow_zone", 3: "blue_zone"}
COLORS = {
    "robot": (60, 130, 255),
    "orange_ball": (255, 140, 0),
    "yellow_zone": (255, 230, 0),
    "blue_zone": (30, 90, 220),
    "green_floor": (50, 220, 70),
}
GREEN_PROMPT = "green playing surface with lines"


def load_models(sam3_path, yolo_pt, device="cuda"):
    yolo = YOLO(str(yolo_pt))
    tproc = Sam3TrackerProcessor.from_pretrained(str(sam3_path))
    tmodel = Sam3TrackerModel.from_pretrained(
        str(sam3_path), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    vproc = AutoProcessor.from_pretrained(str(sam3_path))
    vmodel = AutoModel.from_pretrained(
        str(sam3_path), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    return yolo, tproc, tmodel, vproc, vmodel


@torch.no_grad()
def boxes_to_masks(tproc, tmodel, img, boxes, device):
    """YOLO cajas xyxy -> SAM3 box-prompt -> máscaras bool (N,H,W)."""
    H, W = img.size[1], img.size[0]
    if not boxes:
        return np.zeros((0, H, W), bool)
    inp = tproc(images=[img], input_boxes=[boxes], return_tensors="pt").to(device)
    inp2 = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in inp.items()}
    out = tmodel(**inp2, multimask_output=False)
    masks = tproc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])
    m = np.array(masks[0])
    if m.ndim == 4:
        m = m[:, 0]
    return m.astype(bool)


@torch.no_grad()
def text_mask(vproc, vmodel, img, text, device):
    """SAM3 text-prompt -> máscara bool (H,W) (logits -> bilinear -> threshold)."""
    sess = vproc.init_video_session(video=[img], inference_device=device, dtype=torch.bfloat16)
    sess = vproc.add_text_prompt(sess, text=text)
    o = vmodel(inference_session=sess, frame_idx=0)
    W, H = img.size
    full = np.zeros((H, W), bool)
    for oid in o.object_ids:
        lg = o.obj_id_to_mask[oid].detach().cpu().float().numpy()
        if lg.ndim == 4:
            lg = lg[0, 0]
        elif lg.ndim == 3:
            lg = lg[0]
        lo = cv2.resize(lg.astype("float32"), (W, H), interpolation=cv2.INTER_LINEAR)
        full |= lo > 0
    del sess
    return full


def _overlay(frame, items, alpha=0.5):
    ov = frame.copy()
    for mask, color in items:
        if mask.any():
            ov[mask] = ((1 - alpha) * ov[mask] + alpha * np.array(color)).astype("uint8")
    return ov


def render(video_path, out_path, models, device="cuda", mode="yolo_sam3",
           conf=0.4, green_every=5, max_frames=None):
    """mode='yolo' (solo cajas) o 'yolo_sam3' (cajas->máscaras + green_floor)."""
    yolo, tproc, tmodel, vproc, vmodel = models
    vr = decord.VideoReader(str(video_path))
    fps = float(vr.get_avg_fps()) or 30.0
    H, W = vr[0].shape[:2]
    n = len(vr) if not max_frames else min(len(vr), max_frames)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    green = np.zeros((H, W), bool)
    for i in range(n):
        frame = vr[i].asnumpy()
        img = Image.fromarray(frame)
        res = yolo.predict(img, imgsz=960, conf=conf, device=device, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        if mode == "yolo":
            ov = frame.copy()
            for (x0, y0, x1, y1), c in zip(xyxy.astype(int), cls):
                col = COLORS[CLASS_NAMES[c]]
                cv2.rectangle(ov, (x0, y0), (x1, y1), col, 3)
                cv2.putText(ov, CLASS_NAMES[c], (x0, max(0, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        else:
            masks = boxes_to_masks(tproc, tmodel, img, xyxy.tolist(), device)
            if i % green_every == 0:
                green = text_mask(vproc, vmodel, img, GREEN_PROMPT, device)
            items = [(green, COLORS["green_floor"])]
            for j, c in enumerate(cls):
                if j < len(masks):
                    items.append((masks[j], COLORS[CLASS_NAMES[c]]))
            ov = _overlay(frame, items)
            for (x0, y0, x1, y1), c in zip(xyxy.astype(int), cls):
                cv2.rectangle(ov, (x0, y0), (x1, y1), COLORS[CLASS_NAMES[c]], 2)
        vw.write(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        if i % 25 == 0:
            torch.cuda.empty_cache()
    vw.release()
    return str(out_path)
