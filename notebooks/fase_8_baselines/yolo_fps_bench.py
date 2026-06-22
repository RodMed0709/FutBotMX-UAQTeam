# -*- coding: utf-8 -*-
"""Fase 8 / E2 — FPS de YOLO11 standalone (modo box-only, sin SAM3).

Cronometra inferencia de assets/yolo/best.pt sobre frames de test (warm-up descartado),
reporta FPS mean/std + ms/frame en GPU. Respalda el claim "modo box-only en tiempo real".

Uso (pod, GPU):  python yolo_fps_bench.py [--n 400]
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

REPO = Path("/workspace/FutBotMX-UAQTeam")
WEIGHTS = REPO / "assets" / "yolo" / "best.pt"
IMG_DIR = REPO / "notebooks" / "fase_5_lora" / "dataset" / "testing_600" / "img"
OUT = REPO / "outputs" / "benchmark" / "yolo_fps.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS))
    imgs = sorted(IMG_DIR.glob("*"))[: args.n + args.warmup]
    imgs = [str(p) for p in imgs]
    if not imgs:
        print("NO images"); return
    # warm-up
    for p in imgs[: args.warmup]:
        model.predict(p, imgsz=args.imgsz, device=device, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    per = []
    for p in imgs[args.warmup:]:
        t0 = time.perf_counter()
        model.predict(p, imgsz=args.imgsz, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per.append(time.perf_counter() - t0)
    per = np.array(per)
    fps = 1.0 / per
    payload = {
        "model": "YOLO11s (auto-labeled), standalone (no SAM3)",
        "weights": str(WEIGHTS), "gpu": gpu, "imgsz": args.imgsz,
        "n_frames": int(len(per)),
        "ms_per_frame_mean": round(float(per.mean() * 1000), 2),
        "ms_per_frame_std": round(float(per.std() * 1000), 2),
        "fps_mean": round(float(fps.mean()), 2),
        "fps_std": round(float(fps.std()), 2),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
