"""Propagación de máscaras SAM3 en video (fase_2) — consistencia temporal.

Mecanismo del notebook 03: init_video_session(todos los frames) -> add_text_prompt
por clase en frame 0 -> model(session, frame_idx=i) propaga manteniendo obj_ids.
Resultado: máscaras temporalmente consistentes, color estable por obj_id (no parpadeo).
SAM3 100% al centro. kernels (0.12.x) se usa automático en la inferencia.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

decord.bridge.set_bridge("native")

# (prompt, color RGB) — orden = asignación de obj_ids
DEFAULT_PROMPTS = [
    ("robot", (60, 130, 255)),
    ("orange ball", (255, 140, 0)),
    ("green playing surface with lines", (50, 220, 70)),
    ("yellow zone", (255, 230, 0)),
]


def load_video_sam3(sam3_path, device="cuda"):
    proc = AutoProcessor.from_pretrained(str(sam3_path))
    model = AutoModel.from_pretrained(
        str(sam3_path), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    return proc, model


@torch.no_grad()
def render_propagation(video_path, out_path, proc, model, device="cuda",
                       prompts=None, max_frames=None, alpha=0.5):
    prompts = prompts or DEFAULT_PROMPTS
    vr = decord.VideoReader(str(video_path))
    fps = float(vr.get_avg_fps()) or 30.0
    H, W = vr[0].shape[:2]
    n = len(vr) if not max_frames else min(len(vr), max_frames)
    frames = [Image.fromarray(vr[i].asnumpy()) for i in range(n)]

    session = proc.init_video_session(video=frames, inference_device=device, dtype=torch.bfloat16)
    # rangos de obj_id por prompt (para color por clase)
    bounds = []
    for text, color in prompts:
        before = session.get_obj_num()
        session = proc.add_text_prompt(session, text=text)
        after = session.get_obj_num()
        bounds.append((before, after, color))
        print(f"  prompt '{text}': obj_ids [{before},{after})", flush=True)

    def color_for(oid):
        for b, a, col in bounds:
            if b <= oid < a:
                return col
        return (200, 200, 200)

    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        out = model(inference_session=session, frame_idx=i)
        ov = np.array(frames[i]).copy()
        for oid in out.object_ids:
            lg = out.obj_id_to_mask[oid].detach().cpu().float().numpy()
            if lg.ndim == 4:
                lg = lg[0, 0]
            elif lg.ndim == 3:
                lg = lg[0]
            mask = cv2.resize(lg.astype("float32"), (W, H), interpolation=cv2.INTER_LINEAR) > 0
            col = np.array(color_for(int(oid)))
            ov[mask] = ((1 - alpha) * ov[mask] + alpha * col).astype("uint8")
        vw.write(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        if i % 25 == 0:
            torch.cuda.empty_cache()
            print(f"  frame {i}/{n}", flush=True)
    vw.release()
    return str(out_path)


if __name__ == "__main__":
    REPO = Path("/workspace/FutBotMX-UAQTeam")
    F2 = REPO / "notebooks" / "fase_2_YOLO_SAM3"
    p, m = load_video_sam3(REPO / "assets" / "sam3")
    print("render propagación...", flush=True)
    render_propagation(REPO / "data/raw/17Abril/Cámaras/IMG_9871.MOV",
                       F2 / "demo_sam3_propagation.mp4", p, m)
    print("PROP_DONE", flush=True)
