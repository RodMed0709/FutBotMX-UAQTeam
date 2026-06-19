"""SAM3 -> mascaras BITMAP suaves -> Supervisely. Project NUEVO 'smoothv2' con las
4 clases (robot, orange_ball, green_floor, yellow_zone). 100 frames (5/video x 20).

Mascara: logits ~288x288 -> upscale BILINEAR -> threshold (igual 02/03). Cada
object_id = una instancia (sly.Label + sly.Bitmap). Corre en el pod (GPU).
"""
from __future__ import annotations

import os
import random
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

N_PER_VIDEO = 5
SEED = 42
REPO = Path("/workspace/FutBotMX-UAQTeam")
FRAMES_CSV = REPO / "assets" / "testing_frames.csv"
SAM3_PATH = REPO / "assets" / "sam3"
ENV_FILE = Path("/workspace/.env")

PROJECT_NAME = "FutBot_Testing_100_smoothv2"
DATASET_NAME = "testing_100"
MIN_AREA = 100

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


def mask_from_logits(logits, w0, h0):
    lo = logits.astype(np.float32)
    if lo.shape != (h0, w0):
        lo = cv2.resize(lo, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return lo > 0.0


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


def bitmap_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) < MIN_AREA:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    crop = mask[y0 : y1 + 1, x0 : x1 + 1].astype(bool)
    return sly.Bitmap(crop, origin=sly.PointLocation(row=y0, col=x0))


def sample_frames():
    df = pd.read_csv(FRAMES_CSV)
    rng = random.Random(SEED)
    picked = []
    for _, grp in df.groupby("video_id"):
        rows = grp.to_dict("records")
        rng.shuffle(rows)
        picked.extend(rows[:N_PER_VIDEO])
    picked.sort(key=lambda r: (r["video_id"], r["frame_index"]))
    return picked


def main() -> None:
    load_env()
    server = os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com")
    api = sly.Api(server, os.environ["SUPERVISELY_API_TOKEN"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(SAM3_PATH))
    model = AutoModel.from_pretrained(str(SAM3_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    print(f"SAM3 load {time.time() - t0:.1f}s device={device}")

    team = api.team.get_list()[0]
    wss = api.workspace.get_list(team.id)
    ws = wss[0] if wss else api.workspace.create(team.id, "futbot", "")

    obj_classes = {c["name"]: sly.ObjClass(c["name"], sly.Bitmap, color=c["color"]) for c in CLASSES}
    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(list(obj_classes.values())))
    project = api.project.create(ws.id, PROJECT_NAME, type=sly.ProjectType.IMAGES, change_name_if_conflict=True)
    api.project.update_meta(project.id, meta.to_json())
    dataset = api.dataset.create(project.id, DATASET_NAME, change_name_if_conflict=True)
    print(f"project id={project.id} name={project.name!r}")

    samples = sample_frames()
    ok, failed = 0, []
    per_class = {c["name"]: 0 for c in CLASSES}
    t_start = time.time()
    for row in samples:
        png = REPO / row["imagen"]
        try:
            img = Image.open(png).convert("RGB")
            w, h = img.size
            labels = []
            for cls in CLASSES:
                for det in segment_with_text(model, processor, img, cls["prompt"], device):
                    mask = mask_from_logits(det.logits, w, h)
                    bmp = bitmap_from_mask(mask)
                    if bmp is None:
                        continue
                    labels.append(sly.Label(bmp, obj_classes[cls["name"]]))
                    per_class[cls["name"]] += 1
            info = api.image.upload_path(dataset.id, name=png.name, path=str(png))
            api.annotation.upload_ann(info.id, sly.Annotation(img_size=(h, w), labels=labels))
            ok += 1
            if ok % 20 == 0:
                torch.cuda.empty_cache()
                print(f"  {ok}/100  ({time.time() - t_start:.0f}s)")
        except Exception as e:  # noqa: BLE001
            failed.append((png.name, str(e)[:120]))

    print(f"\nListo: {ok}/100 subidas, {len(failed)} fallaron.")
    print("labels por clase:", per_class)
    for n, e in failed[:5]:
        print(f"  FALLO {n}: {e}")
    print(f"Project '{project.name}' (id={project.id}) en {server}")


if __name__ == "__main__":
    main()
