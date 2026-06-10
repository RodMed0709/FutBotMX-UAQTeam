"""smoothv3 — SAM3 + TTA x3 (promediado a nivel instancia) -> Supervisely, 600 frames.

Por cada frame y clase se corre SAM3 sobre 3 vistas TTA (identidad, flip-horizontal,
gamma). Cada deteccion da logits ~288x288 -> upscale BILINEAR a full-res -> se
des-aumenta la geometria (un-flip). Luego se EMPAREJAN las instancias entre las 3
vistas por IoU, se PROMEDIAN los logits de cada instancia emparejada y se queda la
instancia solo si aparece en >=2 de 3 vistas (votacion -> mata falsos positivos).
La mascara final (logit_promedio > 0) se sube como sly.Bitmap (1 instancia c/u).

Modos:
  python subir_smoothv3_tta.py --smoke 6      # 6 frames, NO sube, imprime conteos
  python subir_smoothv3_tta.py                # 600 frames, sube a Supervisely
"""
from __future__ import annotations

import argparse
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

PROJECT_NAME = "FutBot_Testing_600_smoothv3_tta"
DATASET_NAME = "testing_600"
MIN_AREA = 100
IOU_MATCH = 0.3   # umbral para considerar misma instancia entre vistas TTA
MIN_VOTES = 2     # instancia valida si aparece en >=2 de 3 vistas
GAMMA = 0.8

CLASSES = [
    {"name": "robot", "prompt": "robot", "color": [60, 130, 255]},
    {"name": "orange_ball", "prompt": "orange ball", "color": [255, 100, 0]},
    {"name": "green_floor", "prompt": "green playing surface with lines", "color": [50, 220, 70]},
    {"name": "yellow_zone", "prompt": "yellow zone", "color": [255, 230, 0]},
]


def load_env() -> None:
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


def make_views(img: Image.Image):
    """3 vistas TTA: (nombre, imagen_aug). geometria se revierte luego."""
    arr = np.asarray(img).astype(np.float32)
    gamma = np.clip(((arr / 255.0) ** GAMMA) * 255.0, 0, 255).astype(np.uint8)
    return [
        ("id", img),
        ("hflip", img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("gamma", Image.fromarray(gamma)),
    ]


def tta_class_masks(model, processor, img, prompt, device):
    """Corre las 3 vistas; devuelve lista (run_idx, logit_full HxW, score) geom-corregida."""
    w, h = img.size
    res = []
    for ri, (name, aug) in enumerate(make_views(img)):
        for det in segment_with_text(model, processor, aug, prompt, device):
            full = logits_to_full(det.logits, w, h)
            if name == "hflip":
                full = full[:, ::-1].copy()  # des-flip
            res.append((ri, full, det.score))
    return res


def merge_tta(res):
    """Empareja instancias entre vistas por IoU, promedia logits, vota >=MIN_VOTES."""
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
            score = float(np.mean([items[k][3] for k in group]))
            finals.append((avg > 0.0, score))
    return finals


def bitmap_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) < MIN_AREA:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    crop = mask[y0 : y1 + 1, x0 : x1 + 1].astype(bool)
    return sly.Bitmap(crop, origin=sly.PointLocation(row=y0, col=x0))


def all_frames():
    df = pd.read_csv(FRAMES_CSV)
    return df.sort_values(["video_id", "frame_index"]).to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="N frames, no sube, imprime conteos")
    args = ap.parse_args()
    load_env()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(SAM3_PATH))
    model = AutoModel.from_pretrained(str(SAM3_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    print(f"SAM3 load {time.time() - t0:.1f}s device={device}")

    frames = all_frames()
    if args.smoke:
        frames = frames[: args.smoke]

    api = dataset = obj_classes = None
    if not args.smoke:
        server = os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com")
        api = sly.Api(server, os.environ["SUPERVISELY_API_TOKEN"])
        team = api.team.get_list()[0]
        wss = api.workspace.get_list(team.id)
        ws = wss[0] if wss else api.workspace.create(team.id, "futbot", "")
        obj_classes = {c["name"]: sly.ObjClass(c["name"], sly.Bitmap, color=c["color"]) for c in CLASSES}
        meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(list(obj_classes.values())))
        project = api.project.create(ws.id, PROJECT_NAME, type=sly.ProjectType.IMAGES, change_name_if_conflict=True)
        api.project.update_meta(project.id, meta.to_json())
        dataset = api.dataset.create(project.id, DATASET_NAME, change_name_if_conflict=True)
        print(f"project id={project.id} name={project.name!r}")

    ok, failed = 0, []
    per_class = {c["name"]: 0 for c in CLASSES}
    t_start = time.time()
    for row in frames:
        png = REPO / row["imagen"]
        try:
            img = Image.open(png).convert("RGB")
            w, h = img.size
            labels = []
            counts = {}
            for cls in CLASSES:
                res = tta_class_masks(model, processor, img, cls["prompt"], device)
                finals = merge_tta(res)
                n = 0
                for mask, _score in finals:
                    bmp = bitmap_from_mask(mask)
                    if bmp is None:
                        continue
                    per_class[cls["name"]] += 1
                    n += 1
                    if not args.smoke:
                        labels.append(sly.Label(bmp, obj_classes[cls["name"]]))
                counts[cls["name"]] = n
            if args.smoke:
                print(f"  {png.name}: {counts}")
            else:
                info = api.image.upload_path(dataset.id, name=png.name, path=str(png))
                api.annotation.upload_ann(info.id, sly.Annotation(img_size=(h, w), labels=labels))
            ok += 1
            if not args.smoke and ok % 25 == 0:
                torch.cuda.empty_cache()
                print(f"  {ok}/{len(frames)}  ({time.time() - t_start:.0f}s)")
        except Exception as e:  # noqa: BLE001
            failed.append((png.name, str(e)[:120]))

    print(f"\nListo: {ok}/{len(frames)} procesadas, {len(failed)} fallaron.")
    print("instancias por clase:", per_class)
    for n, e in failed[:5]:
        print(f"  FALLO {n}: {e}")
    if not args.smoke:
        print(f"Project '{PROJECT_NAME}' en Supervisely")


if __name__ == "__main__":
    main()
