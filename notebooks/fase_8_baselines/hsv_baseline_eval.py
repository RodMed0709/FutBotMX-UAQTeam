# -*- coding: utf-8 -*-
"""Fase 8 / E1 — Baseline HSV (umbrales de color) vs GT manual (600 frames).

Detector clasico por color para las clases de color solido; robots NO son color unico (se espera
que falle, ese es el punto). Misma metrica y mismo GT que el eval de SAM3 (01_seg_eval_vs_gt.py):
IoU/Dice/Boundary-IoU por clase semantica, robots unidos. Salida comparable directamente.

Uso (pod, CPU):  python hsv_baseline_eval.py [--limit N]
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
import supervisely as sly

REPO = Path("/workspace/FutBotMX-UAQTeam")
DATA = REPO / "notebooks" / "fase_5_lora" / "dataset" / "testing_600"
ANN_DIR = DATA / "ann"
IMG_DIR = DATA / "img"
META_PATH = DATA.parent / "meta.json"
OUT_DIR = REPO / "outputs" / "seg_eval"
MIN_AREA = 100
BOUNDARY_PX = 3

# Rangos HSV (OpenCV H:0-179) hechos a mano por clase de color. Robot: sin color unico -> None.
HSV_RANGES = {
    "orange_ball": [((5, 120, 120), (20, 255, 255))],
    "yellow_zone": [((20, 80, 120), (35, 255, 255))],
    "green_floor": [((35, 40, 40), (85, 255, 255))],
    "blue_zone":   [((100, 80, 60), (130, 255, 255))],
    "robot":       None,  # multi-color: HSV no puede (limitacion esperada)
}
EVAL_CLASSES = [
    {"name": "robot",       "gt": ["robot_a", "robot_b"]},
    {"name": "orange_ball", "gt": ["orange_ball"]},
    {"name": "green_floor", "gt": ["green_floor"]},
    {"name": "yellow_zone", "gt": ["yellow_zone"]},
    {"name": "blue_zone",   "gt": ["blue_zone"]},
]


def gt_masks_by_class(ann_path, meta):
    ann = sly.Annotation.load_json_file(str(ann_path), meta)
    h, w = ann.img_size
    out = {}
    for label in ann.labels:
        name = label.obj_class.name
        geom = label.geometry
        data = np.asarray(geom.data, dtype=bool)
        r0, c0 = geom.origin.row, geom.origin.col
        canvas = out.setdefault(name, np.zeros((h, w), dtype=bool))
        canvas[r0:r0 + data.shape[0], c0:c0 + data.shape[1]] |= data
    return out, h, w


def hsv_mask(bgr, ranges):
    if ranges is None:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(u) if u else None


def _dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum()) / float(s) if s else None


def _boundary(mask, d):
    if not mask.any():
        return mask
    er = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=d).astype(bool)
    return mask & ~er


def _biou(a, b, d):
    return _iou(_boundary(a, d), _boundary(b, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    meta = sly.ProjectMeta.from_json(json.loads(META_PATH.read_text(encoding="utf-8")))
    anns = sorted(ANN_DIR.glob("*.json"))
    if args.limit:
        anns = anns[: args.limit]
    acc = {c["name"]: {"iou": [], "dice": [], "biou": [], "n": 0, "fp": 0} for c in EVAL_CLASSES}
    failed = []
    for k, ann_path in enumerate(anns):
        img_path = IMG_DIR / ann_path.stem
        if not img_path.exists():
            failed.append(ann_path.name); continue
        try:
            gt_raw, h, w = gt_masks_by_class(ann_path, meta)
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                failed.append(ann_path.name); continue
            for c in EVAL_CLASSES:
                gt = np.zeros((h, w), dtype=bool)
                for g in c["gt"]:
                    if g in gt_raw:
                        gt |= gt_raw[g]
                pred = hsv_mask(bgr, HSV_RANGES[c["name"]])
                if pred is None:
                    pred = np.zeros((h, w), dtype=bool)  # robots: HSV no aplica
                if pred.shape != (h, w):
                    pred = cv2.resize(pred.astype(np.uint8), (w, h), cv2.INTER_NEAREST).astype(bool)
                if pred.sum() < MIN_AREA:
                    pred = np.zeros((h, w), dtype=bool)
                if gt.any():
                    acc[c["name"]]["iou"].append(_iou(gt, pred) or 0.0)
                    acc[c["name"]]["dice"].append(_dice(gt, pred) or 0.0)
                    acc[c["name"]]["biou"].append(_biou(gt, pred, BOUNDARY_PX) or 0.0)
                    acc[c["name"]]["n"] += 1
                elif pred.any():
                    acc[c["name"]]["fp"] += 1
        except Exception as e:  # noqa
            failed.append(f"{ann_path.name}:{str(e)[:60]}")
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(anns)}", flush=True)
    rows = []
    for c in EVAL_CLASSES:
        a = acc[c["name"]]
        rows.append({"class": c["name"], "n_frames": a["n"], "fp_frames": a["fp"],
                     "mIoU": round(float(np.mean(a["iou"])), 4) if a["iou"] else None,
                     "Dice": round(float(np.mean(a["dice"])), 4) if a["dice"] else None,
                     "BoundaryIoU": round(float(np.mean(a["biou"])), 4) if a["biou"] else None})
    valid = [r for r in rows if r["mIoU"] is not None]
    rows.append({"class": "mean", "n_frames": sum(r["n_frames"] for r in rows),
                 "fp_frames": sum(r["fp_frames"] for r in rows),
                 "mIoU": round(float(np.mean([r["mIoU"] for r in valid])), 4),
                 "Dice": round(float(np.mean([r["Dice"] for r in valid])), 4),
                 "BoundaryIoU": round(float(np.mean([r["BoundaryIoU"] for r in valid])), 4)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"mode": "HSV color-threshold baseline", "n_frames_eval": len(anns) - len(failed),
               "boundary_px": BOUNDARY_PX, "note": "robots: HSV no aplica (sin color unico)", "rows": rows}
    (OUT_DIR / "seg_eval_hsv.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "seg_eval_hsv.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "n_frames", "fp_frames", "mIoU", "Dice", "BoundaryIoU"])
        w.writeheader(); w.writerows(rows)
    print("\n=== HSV baseline vs GT ===")
    for r in rows:
        print(f"{r['class']:14s} n={r['n_frames']:>4} mIoU={r['mIoU']} Dice={r['Dice']} BIoU={r['BoundaryIoU']}")
    print(f"-> {OUT_DIR/'seg_eval_hsv.csv'}  (failed {len(failed)})")


if __name__ == "__main__":
    main()
