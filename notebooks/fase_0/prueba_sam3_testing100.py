"""prueba_sam3_testing100.py — SAM3 zero-shot instance segmentation sobre 100
frames del set de testing (muestreados de varios videos) -> COCO -> (opcional)
Roboflow. Reemplaza al notebook 04 para ESTE experimento puntual (fuera del
workflow). Corre en el pod con GPU.

Uso:
  python prueba_sam3_testing100.py            # inferencia + COCO + scores
  python prueba_sam3_testing100.py --upload   # ademas sube a Roboflow

Frames: se REUSAN los PNG ya extraidos en data/testing_frames/ (color correcto,
RGB, generados por src.data.eval_frames). Se muestrea N_PER_VIDEO por cada video
de testing (aleatorio, semilla fija) -> 100 frames de distintos videos.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# ---------------- config ----------------
N_PER_VIDEO = 5  # 5 x 20 videos = 100
SEED = 42
REPO = Path("/workspace/FutBotMX-UAQTeam")
FRAMES_CSV = REPO / "assets" / "testing_frames.csv"
FRAMES_DIR = REPO / "data" / "testing_frames"
SAM3_PATH = REPO / "assets" / "sam3"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "testing_100"
ENV_FILE = Path("/workspace/.env")

ROBOFLOW_WORKSPACE = "futbot2026segmentation"
ROBOFLOW_PROJECT = "segmentacion_futbot-kwqqr"
ROBOFLOW_BATCH = "sam3_testing100_eval"

# Prompts ganadores confirmados (probe 2026-05-29).
CLASSES = [
    {"coco_id": 1, "name": "robot", "sam3_prompt": "robot"},
    {"coco_id": 2, "name": "orange_ball", "sam3_prompt": "orange ball"},
    {"coco_id": 3, "name": "green_floor", "sam3_prompt": "green playing surface with lines"},
]


def load_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Detection:
    mask: np.ndarray
    score: float


@torch.no_grad()
def segment_with_text(model, processor, image, text, device):
    session = processor.init_video_session(
        video=[image], inference_device=device, dtype=torch.bfloat16
    )
    session = processor.add_text_prompt(session, text=text)
    out = model(inference_session=session, frame_idx=0)
    dets = []
    for oid in out.object_ids:
        m = out.obj_id_to_mask[oid].detach().cpu().float().numpy()
        if m.ndim == 4:
            m = m[0, 0]
        elif m.ndim == 3:
            m = m[0]
        dets.append(
            Detection(mask=(m > 0.0), score=float(out.obj_id_to_score.get(oid, 0.0)))
        )
    del session
    return dets


def refine_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    h, w = m.shape
    k = max(3, int(min(h, w) * 0.005))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return m.astype(bool)


def mask_to_polygons(mask: np.ndarray, min_area: int = 100):
    mask = refine_mask(mask)
    polys = []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = 0.0015 * cv2.arcLength(c, True)
        cs = cv2.approxPolyDP(c, eps, True)
        if cs.shape[0] < 3:
            continue
        polys.append(cs.flatten().astype(float).tolist())
    return polys


def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min()),
        float(ys.max() - ys.min()),
    )


def sample_frames():
    """5 frames aleatorios por video (semilla fija) -> 100 de distintos videos."""
    df = pd.read_csv(FRAMES_CSV)
    rng = random.Random(SEED)
    picked = []
    for _, grp in df.groupby("video_id"):
        rows = grp.to_dict("records")
        rng.shuffle(rows)
        picked.extend(rows[:N_PER_VIDEO])
    picked.sort(key=lambda r: (r["video_id"], r["frame_index"]))
    return picked


def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(SAM3_PATH))
    model = AutoModel.from_pretrained(
        str(SAM3_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print(f"SAM3 load {time.time() - t0:.1f}s")

    samples = sample_frames()
    nvideos = len({s["video_id"] for s in samples})
    print(f"frames muestreados: {len(samples)} de {nvideos} videos")

    coco = {
        "info": {
            "description": "FutBotMX testing100 SAM3 zero-shot",
            "tool": "SAM3 (facebook/sam3) text-prompt zero-shot",
        },
        "categories": [
            {"id": c["coco_id"], "name": c["name"], "supercategory": c["name"]}
            for c in CLASSES
        ],
        "images": [],
        "annotations": [],
    }
    scores = defaultdict(list)
    imgs_with = defaultdict(set)
    next_img, next_ann = 1, 1
    t_start = time.time()

    for i, row in enumerate(samples):
        png = REPO / row["imagen"]
        img = Image.open(png).convert("RGB")
        w, h = img.size
        coco["images"].append(
            {
                "id": next_img,
                "file_name": png.name,
                "width": w,
                "height": h,
                "video_source": row["video_ruta"],
                "video_frame_idx": int(row["frame_original"]),
                "grupo": row["grupo"],
            }
        )
        for cls in CLASSES:
            for det in segment_with_text(model, processor, img, cls["sam3_prompt"], device):
                mask = det.mask
                if mask.shape != (h, w):
                    mask = (
                        np.array(
                            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                                (w, h), Image.NEAREST
                            )
                        )
                        > 0
                    )
                polys = mask_to_polygons(mask)
                if not polys:
                    continue
                x, y, bw, bh = mask_to_bbox(mask)
                coco["annotations"].append(
                    {
                        "id": next_ann,
                        "image_id": next_img,
                        "category_id": cls["coco_id"],
                        "segmentation": polys,
                        "bbox": [x, y, bw, bh],
                        "area": float(mask.sum()),
                        "iscrowd": 0,
                        "score": det.score,
                    }
                )
                next_ann += 1
                scores[cls["name"]].append(det.score)
                imgs_with[cls["name"]].add(next_img)
        next_img += 1
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"  {i + 1}/{len(samples)}  ({time.time() - t_start:.0f}s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann_file = OUT_DIR / "annotations.json"
    ann_file.write_text(json.dumps(coco, indent=2))
    dt = time.time() - t_start
    print(
        f"\nInferencia: {dt:.0f}s ({dt / 60:.1f} min) | "
        f"imgs={len(coco['images'])} | ann={len(coco['annotations'])}"
    )
    print(f"COCO -> {ann_file}")
    report_scores(coco, scores, imgs_with)
    return coco


def report_scores(coco, scores, imgs_with):
    print("\n===== SCORE SAM3 por clase (text-prompt confidence) =====")
    n_imgs = len(coco["images"])
    print(
        f"{'clase':13s} {'prompt':38s} {'n_inst':>7s} {'imgs':>7s} "
        f"{'score_avg':>10s} {'min':>6s} {'max':>6s}"
    )
    print("-" * 95)
    for cls in CLASSES:
        nm = cls["name"]
        s = scores[nm]
        imgs = f"{len(imgs_with[nm])}/{n_imgs}"
        if s:
            print(
                f"{nm:13s} {cls['sam3_prompt']:38s} {len(s):>7d} {imgs:>7s} "
                f"{np.mean(s):>10.3f} {min(s):>6.3f} {max(s):>6.3f}"
            )
        else:
            print(f"{nm:13s} {cls['sam3_prompt']:38s} {0:>7d} {imgs:>7s} {'--':>10s}")


def upload_roboflow(coco):
    from roboflow import Roboflow
    from tqdm import tqdm

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    print(f"\nUpload -> {project.name} (batch {ROBOFLOW_BATCH})")
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    ok, failed = 0, []
    tmp = OUT_DIR / "per_image_tmp.json"
    for img in tqdm(coco["images"]):
        anns = by_img[img["id"]]
        if not anns:
            continue
        tmp.write_text(
            json.dumps(
                {
                    "info": coco["info"],
                    "categories": coco["categories"],
                    "images": [img],
                    "annotations": anns,
                }
            )
        )
        try:
            project.upload(
                image_path=str(FRAMES_DIR / img["file_name"]),
                annotation_path=str(tmp),
                batch_name=ROBOFLOW_BATCH,
                split="train",
                num_retry_uploads=2,
            )
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed.append((img["file_name"], str(e)))
    tmp.unlink(missing_ok=True)
    print(f"subidas {ok}/{len(coco['images'])}, fallaron {len(failed)}")
    for n, e in failed[:3]:
        print(f"  {n}: {e[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="subir resultado a Roboflow")
    args = ap.parse_args()
    load_env()
    coco = run_inference()
    if args.upload:
        upload_roboflow(coco)


if __name__ == "__main__":
    main()
