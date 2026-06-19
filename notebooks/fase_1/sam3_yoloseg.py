"""SAM3 -> YOLO **segmentation** auto-labels (fase_1). Las 4 clases con MASCARA.

Reusa la lib de deteccion (sam3_yolo) para SAM3. Convierte cada mascara de instancia
a un POLIGONO YOLO-seg normalizado: 'class x1 y1 x2 y2 ...' (1 linea por instancia).
Incluye green_floor (su forma real -> homografia). Split por video, resumable,
escritura atomica, score>=0.40.
"""
from __future__ import annotations

import gc
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from PIL import Image

import sam3_yolo as base  # load_sam3, segment_with_text, mask_from_logits, videos_by_split, frame_indices

decord.bridge.set_bridge("native")

SEG_CLASSES = [
    {"id": 0, "name": "robot", "prompt": "robot"},
    {"id": 1, "name": "orange_ball", "prompt": "orange ball"},
    {"id": 2, "name": "green_floor", "prompt": "green playing surface with lines"},
    {"id": 3, "name": "yellow_zone", "prompt": "yellow zone"},
]
MIN_BOX_AREA_PX = 100
MIN_SIDE_PX = 2
SCORE_THRESH = 0.40
EPS_FRAC = 0.004  # suavizado del poligono (fraccion del perimetro)

load_sam3 = base.load_sam3
segment_with_text = base.segment_with_text
mask_from_logits = base.mask_from_logits
videos_by_split = base.videos_by_split
frame_indices = base.frame_indices


def mask_to_yolo_polygons(mask, w, h):
    """Mascara bool (H,W) -> lista de poligonos normalizados (cada uno [x1,y1,...])."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    polys = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < MIN_SIDE_PX or bh < MIN_SIDE_PX or bw * bh < MIN_BOX_AREA_PX:
            continue
        ap = cv2.approxPolyDP(c, EPS_FRAC * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        flat = []
        for px, py in ap:
            flat.append(min(max(px / w, 0.0), 1.0))
            flat.append(min(max(py / h, 0.0), 1.0))
        polys.append(flat)
    return polys


def _atomic_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_img(img, dest):
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    img.save(tmp, "JPEG", quality=95, subsampling=0)
    os.replace(tmp, dest)


def _count(out, split):
    c = Counter()
    for tf in (out / "labels" / split).glob("*.txt"):
        for ln in tf.read_text().splitlines():
            if ln.strip():
                c[int(ln.split()[0])] += 1
    return c


def autolabel_seg(videos, repo_root, out_dir, model, proc, device,
                  frames_per_video=30, val_frac=0.2, seed=42):
    import csv
    import random

    repo_root, out = Path(repo_root), Path(out_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    vids = list(videos)
    rng.shuffle(vids)
    n_val = max(1, int(round(len(vids) * val_frac)))
    val_ids = {vid for vid, _ in vids[:n_val]}

    n_img = {"train": 0, "val": 0}
    skipped = 0
    manifest = []
    plan, total = [], 0
    for vid, ruta in videos:
        try:
            vr = decord.VideoReader(str(repo_root / ruta))
            idxs = frame_indices(len(vr), frames_per_video)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {ruta}: {e}", flush=True)
            continue
        plan.append((vid, ruta, idxs))
        total += len(idxs)
    print(f"videos OK={len(plan)} frames={total} val_videos={len(val_ids)}", flush=True)

    t0 = time.time()
    done = 0
    for vid, ruta, idxs in plan:
        split = "val" if vid in val_ids else "train"
        stem = Path(ruta).stem.replace(" ", "_")
        try:
            vr = decord.VideoReader(str(repo_root / ruta))
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {ruta}: {e}", flush=True)
            continue
        for fi in idxs:
            name = f"{stem}_f{fi:04d}"
            lbl = out / "labels" / split / f"{name}.txt"
            imgp = out / "images" / split / f"{name}.jpg"
            manifest.append([vid, ruta, fi, split, f"images/{split}/{name}.jpg"])
            if lbl.exists() and imgp.exists():
                skipped += 1
                done += 1
                continue
            try:
                img = Image.fromarray(vr[fi].asnumpy())
                w, h = img.size
                lines = []
                for cls in SEG_CLASSES:
                    for det in segment_with_text(model, proc, img, cls["prompt"], device):
                        if det.score < SCORE_THRESH:
                            continue
                        for poly in mask_to_yolo_polygons(mask_from_logits(det.logits, w, h), w, h):
                            lines.append(f"{cls['id']} " + " ".join(f"{v:.6f}" for v in poly))
                _atomic_img(img, imgp)
                _atomic_text(lbl, "\n".join(lines))
                n_img[split] += 1
            except Exception as e:  # noqa: BLE001
                print(f"  frame {ruta}@{fi} fail: {e}", flush=True)
                continue
            finally:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                eta = el / max(1, done - skipped) * (total - done)
                print(f"  {done}/{total} ({el:.0f}s, ETA {eta/60:.0f}min)", flush=True)
        print(f"  done {ruta} [{split}]", flush=True)

    items = sorted(SEG_CLASSES, key=lambda c: c["id"])
    names_block = "\n".join(f"  {c['id']}: {c['name']}" for c in items)
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(items)}\nnames:\n{names_block}\n"
    )
    with (out / "manifest.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video_id", "video_ruta", "frame_idx", "split", "image"])
        wr.writerows(manifest)
    bt, bv = _count(out, "train"), _count(out, "val")
    nm = {c["id"]: c["name"] for c in SEG_CLASSES}
    print(f"\nDataset YOLO-seg -> {out}", flush=True)
    print(f"imgs train={n_img['train']} val={n_img['val']} saltadas={skipped}", flush=True)
    print(f"polys TRAIN: {{ {', '.join(f'{nm[k]}={v}' for k,v in sorted(bt.items()))} }}", flush=True)
    print(f"polys VAL:   {{ {', '.join(f'{nm[k]}={v}' for k,v in sorted(bv.items()))} }}", flush=True)
    if sum(bt.values()) == 0 or sum(bv.values()) == 0:
        print("!!! ALERTA train/val sin poligonos", flush=True)
    return {"out": str(out), "n_img": n_img, "boxes_train": dict(bt), "boxes_val": dict(bv), "val_ids": sorted(val_ids)}
