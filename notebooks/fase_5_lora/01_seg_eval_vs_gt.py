# -*- coding: utf-8 -*-
"""fase_5_lora / 01 — Benchmark de SEGMENTACION: SAM3 (zero-shot) vs GT manual (600 frames).

Mide cuantitativamente que tan bien segmenta SAM3 contra las anotaciones HUMANAS de los
2 anotadores (dataset/testing_600, formato Supervisely bitmap). Por clase semantica:
IoU, Dice, Boundary-IoU. Los robots se evaluan UNIDOS (robot_a ∪ robot_b) porque SAM3 no
distingue equipo por texto (el equipo es un paso manual posterior, no capacidad de seg).

Esto llena la Tabla 2 del paper MICAI (eval vs GT) — keystone para oral, sin necesitar LoRA.

Salida: outputs/seg_eval/seg_eval_vs_gt[_tta].csv + .json (por clase + media).

Uso (en el pod, GPU):
  python 01_seg_eval_vs_gt.py --limit 10     # smoke (10 frames)
  python 01_seg_eval_vs_gt.py                # 600 frames, zero-shot
  python 01_seg_eval_vs_gt.py --tta          # 600 frames, SAM3 + 3x TTA (consenso >=2/3)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import supervisely as sly
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

REPO = Path("/workspace/FutBotMX-UAQTeam")
DATA = REPO / "notebooks" / "fase_5_lora" / "dataset" / "testing_600"
ANN_DIR = DATA / "ann"
IMG_DIR = DATA / "img"
META_PATH = DATA.parent / "meta.json"
SAM3_PATH = REPO / "assets" / "sam3"
ENV_FILE = Path("/workspace/.env")
OUT_DIR = REPO / "outputs" / "seg_eval"

MIN_AREA = 100
IOU_MATCH = 0.3
MIN_VOTES = 2
GAMMA = 0.8
BOUNDARY_PX = 3  # ancho de la banda de frontera para Boundary-IoU

# Clases SEMANTICAS evaluadas: nombre -> (prompt SAM3, [clases GT a unir]).
EVAL_CLASSES = [
    {"name": "robot", "prompt": "robot", "gt": ["robot_a", "robot_b"]},
    {"name": "orange_ball", "prompt": "orange ball", "gt": ["orange_ball"]},
    {"name": "green_floor", "prompt": "green playing surface with lines", "gt": ["green_floor"]},
    {"name": "yellow_zone", "prompt": "yellow zone", "gt": ["yellow_zone"]},
    {"name": "blue_zone", "prompt": "blue board", "gt": ["blue_zone"]},
]


def load_env() -> None:
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ---------------- GT (Supervisely bitmap) ----------------
def gt_masks_by_class(ann_path: Path, meta: sly.ProjectMeta) -> tuple[dict, int, int]:
    """Mascara binaria GT por clase GT cruda (union de instancias). Devuelve (dict, H, W)."""
    ann = sly.Annotation.load_json_file(str(ann_path), meta)
    h, w = ann.img_size
    out: dict[str, np.ndarray] = {}
    for label in ann.labels:
        name = label.obj_class.name
        geom = label.geometry  # sly.Bitmap
        data = np.asarray(geom.data, dtype=bool)
        r0, c0 = geom.origin.row, geom.origin.col
        canvas = out.setdefault(name, np.zeros((h, w), dtype=bool))
        canvas[r0:r0 + data.shape[0], c0:c0 + data.shape[1]] |= data
    return out, h, w


# ---------------- SAM3 prediccion ----------------
@torch.no_grad()
def _segment(model, processor, image, text, device):
    session = processor.init_video_session(video=[image], inference_device=device, dtype=torch.bfloat16)
    session = processor.add_text_prompt(session, text=text)
    out = model(inference_session=session, frame_idx=0)
    logits = []
    for oid in out.object_ids:
        m = out.obj_id_to_mask[oid].detach().cpu().float().numpy()
        m = m[0, 0] if m.ndim == 4 else (m[0] if m.ndim == 3 else m)
        logits.append(m)
    del session
    return logits


def _to_full(logit, w, h):
    lo = logit.astype(np.float32)
    if lo.shape != (h, w):
        lo = cv2.resize(lo, (w, h), interpolation=cv2.INTER_LINEAR)
    return lo


def pred_mask_zeroshot(model, processor, img, prompt, device, h, w) -> np.ndarray:
    """Mascara semantica (union de instancias, logit>0) de un prompt, single-view."""
    mask = np.zeros((h, w), dtype=bool)
    for lg in _segment(model, processor, img, prompt, device):
        full = _to_full(lg, w, h) > 0.0
        if full.sum() >= MIN_AREA:
            mask |= full
    return mask


def pred_mask_tta(model, processor, pil, prompt, device, h, w) -> np.ndarray:
    """Mascara semantica con 3x TTA (id/hflip/gamma) + voto >=2/3 a nivel instancia."""
    arr = np.asarray(pil).astype(np.float32)
    gamma = np.clip(((arr / 255.0) ** GAMMA) * 255.0, 0, 255).astype(np.uint8)
    views = [("id", pil), ("hflip", pil.transpose(Image.FLIP_LEFT_RIGHT)),
             ("gamma", Image.fromarray(gamma))]
    items = []  # (run_idx, bin_full)
    for ri, (name, aug) in enumerate(views):
        for lg in _segment(model, processor, aug, prompt, device):
            full = _to_full(lg, w, h)
            if name == "hflip":
                full = full[:, ::-1].copy()
            items.append((ri, full > 0.0))
    used = [False] * len(items)
    mask = np.zeros((h, w), dtype=bool)
    for i in range(len(items)):
        if used[i]:
            continue
        ri_i, bin_i = items[i]
        runs, union = {ri_i}, bin_i.copy()
        used[i] = True
        for j in range(len(items)):
            if used[j]:
                continue
            rj, bin_j = items[j]
            if rj in runs:
                continue
            inter = np.logical_and(union, bin_j).sum()
            uni = np.logical_or(union, bin_j).sum()
            if uni > 0 and inter / uni > IOU_MATCH:
                runs.add(rj); used[j] = True
                union = np.logical_or(union, bin_j)
        if len(runs) >= MIN_VOTES and union.sum() >= MIN_AREA:
            mask |= union
    return mask


# ---------------- metricas ----------------
def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else None


def _dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum()) / float(s) if s else None


def _boundary(mask, d):
    """Banda de frontera (mask - erosion) como mascara booleana."""
    if not mask.any():
        return mask
    er = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=d).astype(bool)
    return mask & ~er


def _biou(a, b, d):
    return _iou(_boundary(a, d), _boundary(b, d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", action="store_true", help="usar SAM3 + 3x TTA (default: zero-shot)")
    ap.add_argument("--limit", type=int, default=0, help="N frames (smoke); 0 = todos (600)")
    args = ap.parse_args()
    load_env()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(SAM3_PATH))
    model = AutoModel.from_pretrained(str(SAM3_PATH), dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True).to(device).eval()
    meta = sly.ProjectMeta.from_json(json.loads(META_PATH.read_text(encoding="utf-8")))
    print(f"SAM3 load {time.time() - t0:.1f}s device={device} mode={'TTA' if args.tta else 'zero-shot'}")

    anns = sorted(ANN_DIR.glob("*.json"))
    if args.limit:
        anns = anns[: args.limit]

    # acumuladores por clase semantica
    acc = {c["name"]: {"iou": [], "dice": [], "biou": [], "n": 0, "fp": 0} for c in EVAL_CLASSES}
    failed = []
    t_start = time.time()
    for k, ann_path in enumerate(anns):
        img_path = IMG_DIR / ann_path.stem  # <img>.png.json -> <img>.png
        if not img_path.exists():
            failed.append((ann_path.name, "img faltante")); continue
        try:
            gt_raw, h, w = gt_masks_by_class(ann_path, meta)
            pil = Image.open(img_path).convert("RGB")
            for c in EVAL_CLASSES:
                gt = np.zeros((h, w), dtype=bool)
                for g in c["gt"]:
                    if g in gt_raw:
                        gt |= gt_raw[g]
                if args.tta:
                    pred = pred_mask_tta(model, processor, pil, c["prompt"], device, h, w)
                else:
                    pred = pred_mask_zeroshot(model, processor, pil, c["prompt"], device, h, w)
                if gt.any():
                    acc[c["name"]]["iou"].append(_iou(gt, pred) or 0.0)
                    acc[c["name"]]["dice"].append(_dice(gt, pred) or 0.0)
                    acc[c["name"]]["biou"].append(_biou(gt, pred, BOUNDARY_PX) or 0.0)
                    acc[c["name"]]["n"] += 1
                elif pred.any():
                    acc[c["name"]]["fp"] += 1  # GT vacio pero SAM3 detecta algo
        except Exception as e:  # noqa: BLE001
            failed.append((ann_path.name, str(e)[:100]))
        if (k + 1) % 25 == 0:
            torch.cuda.empty_cache()
            print(f"  {k + 1}/{len(anns)}  ({time.time() - t_start:.0f}s)")

    # tabla
    rows = []
    for c in EVAL_CLASSES:
        a = acc[c["name"]]
        rows.append({
            "class": c["name"],
            "n_frames": a["n"],
            "fp_frames": a["fp"],
            "mIoU": round(float(np.mean(a["iou"])), 4) if a["iou"] else None,
            "Dice": round(float(np.mean(a["dice"])), 4) if a["dice"] else None,
            "BoundaryIoU": round(float(np.mean(a["biou"])), 4) if a["biou"] else None,
        })
    mean_row = {
        "class": "mean",
        "n_frames": sum(r["n_frames"] for r in rows),
        "fp_frames": sum(r["fp_frames"] for r in rows),
        "mIoU": round(float(np.mean([r["mIoU"] for r in rows if r["mIoU"] is not None])), 4),
        "Dice": round(float(np.mean([r["Dice"] for r in rows if r["Dice"] is not None])), 4),
        "BoundaryIoU": round(float(np.mean([r["BoundaryIoU"] for r in rows if r["BoundaryIoU"] is not None])), 4),
    }
    rows.append(mean_row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "tta" if args.tta else "zeroshot"
    payload = {
        "mode": "SAM3+TTA" if args.tta else "SAM3 zero-shot",
        "n_frames_eval": len(anns) - len(failed),
        "boundary_px": BOUNDARY_PX,
        "note": "robots evaluados unidos (robot_a∪robot_b); SAM3 no distingue equipo",
        "rows": rows,
    }
    (OUT_DIR / f"seg_eval_vs_gt_{tag}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    import csv
    with (OUT_DIR / f"seg_eval_vs_gt_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["class", "n_frames", "fp_frames", "mIoU", "Dice", "BoundaryIoU"])
        wcsv.writeheader(); wcsv.writerows(rows)

    print(f"\n=== SEGMENTACION SAM3 ({payload['mode']}) vs GT manual — {payload['n_frames_eval']} frames ===")
    print(f"{'class':14s}{'n':>5s}{'fp':>5s}{'mIoU':>9s}{'Dice':>9s}{'B-IoU':>9s}")
    for r in rows:
        print(f"{r['class']:14s}{r['n_frames']:>5d}{r['fp_frames']:>5d}"
              f"{(r['mIoU'] if r['mIoU'] is not None else -1):>9.4f}"
              f"{(r['Dice'] if r['Dice'] is not None else -1):>9.4f}"
              f"{(r['BoundaryIoU'] if r['BoundaryIoU'] is not None else -1):>9.4f}")
    if failed:
        print(f"\n{len(failed)} fallaron; ej: {failed[:3]}")
    print(f"\n-> {OUT_DIR / f'seg_eval_vs_gt_{tag}.csv'}")


if __name__ == "__main__":
    main()
