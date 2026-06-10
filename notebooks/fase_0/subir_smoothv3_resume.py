"""Resume del smoothv3: sube SOLO los frames faltantes al project 378046 existente.

Detecta que imagenes ya estan en el dataset y procesa unicamente las que faltan
(SAM3 + TTA x3 + merge por instancia, idem al script original). Util cuando el run
completo se atoro por un 503 de Supervisely.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervisely as sly
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

REPO = Path("/workspace/FutBotMX-UAQTeam")
FRAMES_CSV = REPO / "assets" / "testing_frames.csv"
SAM3_PATH = REPO / "assets" / "sam3"
ENV_FILE = Path("/workspace/.env")

PROJECT_ID = 378046
MIN_AREA = 100
IOU_MATCH = 0.3
MIN_VOTES = 2
GAMMA = 0.8

CLASSES = [
    {"name": "robot", "prompt": "robot", "color": [60, 130, 255]},
    {"name": "orange_ball", "prompt": "orange ball", "color": [255, 100, 0]},
    {"name": "green_floor", "prompt": "green playing surface with lines", "color": [50, 220, 70]},
    {"name": "yellow_zone", "prompt": "yellow zone", "color": [255, 230, 0]},
]


def load_env():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Detection:
    logits: np.ndarray
    score: float


@torch.no_grad()
def segment_with_text(model, processor, image, text, device):
    session = processor.init_video_session(video=[image], inference_device=device, dtype=torch.bfloat16)
    session = processor.add_text_prompt(session, text=text)
    out = model(inference_session=session, frame_idx=0)
    dets = []
    for oid in out.object_ids:
        m = out.obj_id_to_mask[oid].detach().cpu().float().numpy()
        if m.ndim == 4:
            m = m[0, 0]
        elif m.ndim == 3:
            m = m[0]
        dets.append(Detection(logits=m, score=float(out.obj_id_to_score.get(oid, 0.0))))
    del session
    return dets


def logits_to_full(logits, w0, h0):
    lo = logits.astype(np.float32)
    if lo.shape != (h0, w0):
        lo = cv2.resize(lo, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return lo


def make_views(img):
    arr = np.asarray(img).astype(np.float32)
    gamma = np.clip(((arr / 255.0) ** GAMMA) * 255.0, 0, 255).astype(np.uint8)
    return [("id", img), ("hflip", img.transpose(Image.FLIP_LEFT_RIGHT)), ("gamma", Image.fromarray(gamma))]


def tta_class_masks(model, processor, img, prompt, device):
    w, h = img.size
    res = []
    for _ri, (name, aug) in enumerate(make_views(img)):
        for det in segment_with_text(model, processor, aug, prompt, device):
            full = logits_to_full(det.logits, w, h)
            if name == "hflip":
                full = full[:, ::-1].copy()
            res.append((_ri, full, det.score))
    return res


def merge_tta(res):
    items = [(ri, full > 0.0, full, sc) for ri, full, sc in res]
    used = [False] * len(items)
    finals = []
    for i in range(len(items)):
        if used[i]:
            continue
        ri_i, bin_i, _, _ = items[i]
        runs = {ri_i}
        group = [i]
        union = bin_i.copy()
        used[i] = True
        for j in range(len(items)):
            if used[j]:
                continue
            rj, bin_j, _, _ = items[j]
            if rj in runs:
                continue
            inter = np.logical_and(union, bin_j).sum()
            uni = np.logical_or(union, bin_j).sum()
            if uni > 0 and inter / uni > IOU_MATCH:
                group.append(j)
                runs.add(rj)
                used[j] = True
                union = np.logical_or(union, bin_j)
        if len(runs) >= MIN_VOTES:
            avg = np.mean([items[k][2] for k in group], axis=0)
            finals.append(avg > 0.0)
    return finals


def bitmap_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) < MIN_AREA:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    crop = mask[y0 : y1 + 1, x0 : x1 + 1].astype(bool)
    return sly.Bitmap(crop, origin=sly.PointLocation(row=y0, col=x0))


def main():
    load_env()
    api = sly.Api(os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com"), os.environ["SUPERVISELY_API_TOKEN"])
    meta = sly.ProjectMeta.from_json(api.project.get_meta(PROJECT_ID))
    obj_classes = {c["name"]: meta.obj_classes.get(c["name"]) for c in CLASSES}
    ds = api.dataset.get_list(PROJECT_ID)[0]
    existing = {im.name for im in api.image.get_list(ds.id)}
    print(f"dataset '{ds.name}' ya tiene {len(existing)} imgs")

    df = pd.read_csv(FRAMES_CSV).sort_values(["video_id", "frame_index"])
    missing = [r for r in df.to_dict("records") if Path(r["imagen"]).name not in existing]
    print(f"faltantes: {len(missing)}")
    if not missing:
        print("nada que subir.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(SAM3_PATH))
    model = AutoModel.from_pretrained(str(SAM3_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    print(f"SAM3 load {time.time() - t0:.1f}s")

    ok, failed = 0, []
    for row in missing:
        png = REPO / row["imagen"]
        try:
            img = Image.open(png).convert("RGB")
            w, h = img.size
            labels = []
            for cls in CLASSES:
                for mask in merge_tta(tta_class_masks(model, processor, img, cls["prompt"], device)):
                    bmp = bitmap_from_mask(mask)
                    if bmp is not None:
                        labels.append(sly.Label(bmp, obj_classes[cls["name"]]))
            info = api.image.upload_path(ds.id, name=png.name, path=str(png))
            api.annotation.upload_ann(info.id, sly.Annotation(img_size=(h, w), labels=labels))
            ok += 1
            print(f"  ok {png.name} ({ok}/{len(missing)})")
        except Exception as e:  # noqa: BLE001
            failed.append((png.name, str(e)[:120]))
    print(f"\nResume listo: {ok}/{len(missing)} subidas, {len(failed)} fallaron. Total dataset ahora: {len(existing) + ok}")
    for n, e in failed[:5]:
        print(f"  FALLO {n}: {e}")


if __name__ == "__main__":
    main()
