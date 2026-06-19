"""Agrega la clase 'yellow_zone' (rectangulo amarillo = zona/porteria donde llega
la pelota) al project _smooth EXISTENTE, sin re-subir imagenes.

Por cada imagen del dataset: corre SAM3 con prompt 'yellow zone', genera mascaras
bitmap suaves (bilinear logits, igual que 02/03) y las ANEXA a la anotacion
existente (no borra robot/ball/floor). Idempotente por imagen (sobrescribe ann con
la union). Corre en el pod (GPU).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervisely as sly
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

REPO = Path("/workspace/FutBotMX-UAQTeam")
FRAMES_DIR = REPO / "data" / "testing_frames"
SAM3_PATH = REPO / "assets" / "sam3"
ENV_FILE = Path("/workspace/.env")

PROJECT_ID = 378011  # FutBot_Testing_100_smooth
NEW_CLASS = "yellow_zone"
NEW_PROMPT = "yellow zone"
NEW_COLOR = [255, 230, 0]
MIN_AREA = 100


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


def main() -> None:
    load_env()
    server = os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com")
    api = sly.Api(server, os.environ["SUPERVISELY_API_TOKEN"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(SAM3_PATH))
    model = AutoModel.from_pretrained(str(SAM3_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    print(f"SAM3 load {time.time() - t0:.1f}s device={device}")

    # Agregar la clase nueva al meta (preserva las existentes).
    meta = sly.ProjectMeta.from_json(api.project.get_meta(PROJECT_ID))
    if meta.obj_classes.get(NEW_CLASS) is None:
        new_oc = sly.ObjClass(NEW_CLASS, sly.Bitmap, color=NEW_COLOR)
        meta = meta.add_obj_class(new_oc)
        api.project.update_meta(PROJECT_ID, meta.to_json())
        print(f"clase '{NEW_CLASS}' agregada al meta")
    yellow_oc = meta.obj_classes.get(NEW_CLASS)

    ds = api.dataset.get_list(PROJECT_ID)[0]
    imgs = api.image.get_list(ds.id)
    print(f"dataset '{ds.name}' imgs={len(imgs)}")

    ok, added, failed = 0, 0, []
    t_start = time.time()
    for im in imgs:
        png = FRAMES_DIR / im.name
        try:
            img = Image.open(png).convert("RGB")
            w, h = img.size
            ylabels = []
            for det in segment_with_text(model, processor, img, NEW_PROMPT, device):
                mask = mask_from_logits(det.logits, w, h)
                bmp = bitmap_from_mask(mask)
                if bmp is None:
                    continue
                ylabels.append(sly.Label(bmp, yellow_oc))

            ann_info = api.annotation.download(im.id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)
            ann = ann.add_labels(ylabels)
            api.annotation.upload_ann(im.id, ann)
            ok += 1
            added += len(ylabels)
            if ok % 20 == 0:
                torch.cuda.empty_cache()
                print(f"  {ok}/{len(imgs)}  yellow_labels+={added}  ({time.time() - t_start:.0f}s)")
        except Exception as e:  # noqa: BLE001
            failed.append((im.name, str(e)[:120]))

    print(f"\nListo: {ok}/{len(imgs)} imgs procesadas, {added} labels '{NEW_CLASS}' agregados, {len(failed)} fallaron.")
    for n, e in failed[:5]:
        print(f"  FALLO {n}: {e}")


if __name__ == "__main__":
    main()
