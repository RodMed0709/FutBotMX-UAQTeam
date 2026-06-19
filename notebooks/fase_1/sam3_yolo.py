"""SAM3 -> YOLO detection auto-labels (fase_1). Robusto para 103 videos desatendido.

Destilacion: SAM3 (maestro) auto-etiqueta frames -> dataset YOLO (alumno).
logits 288 -> upscale BILINEAR -> mask -> caja YOLO normalizada. Split POR VIDEO
(anti-leak). RESUMABLE. Anti-OOM. Escritura atomica. Filtro por score.
"""
from __future__ import annotations

import gc
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

decord.bridge.set_bridge("native")

DEFAULT_CLASSES = [
    {"id": 0, "name": "robot", "prompt": "robot"},
    {"id": 1, "name": "orange_ball", "prompt": "orange ball"},
    {"id": 2, "name": "yellow_zone", "prompt": "yellow zone"},
]
MIN_BOX_AREA_PX = 100   # area de la CAJA (no de la mascara)
MIN_SIDE_PX = 2         # descarta lineas/puntos
SCORE_THRESH = 0.40     # no meter detecciones espurias como GT


@dataclass
class Detection:
    logits: np.ndarray
    score: float


def load_sam3(sam3_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(sam3_path))
    model = AutoModel.from_pretrained(
        str(sam3_path), dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    print(f"SAM3 cargado en {time.time() - t0:.1f}s (device={device})", flush=True)
    return model, processor, device


@torch.no_grad()
def segment_with_text(model, processor, image, text, device):
    session = None
    try:
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
        return dets
    finally:
        del session


def mask_from_logits(logits, w0, h0):
    lo = logits.astype(np.float32)
    if lo.shape != (h0, w0):
        lo = cv2.resize(lo, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return lo > 0.0


def mask_to_yolo_box(mask, w, h):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw_px, bh_px = x1 - x0, y1 - y0
    if bw_px < MIN_SIDE_PX or bh_px < MIN_SIDE_PX:
        return None
    if bw_px * bh_px < MIN_BOX_AREA_PX:
        return None
    xc = ((x0 + x1) / 2) / w
    yc = ((y0 + y1) / 2) / h
    bw = bw_px / w
    bh = bh_px / h
    clip = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return clip(xc), clip(yc), clip(bw), clip(bh)


def videos_by_split(metadata_csv, splits):
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    sel = df[df["split"].isin(splits)]
    return [(int(r["id"]), r["ruta"]) for r in sel.to_dict("records")]


def frame_indices(total, quota):
    if total <= quota:
        return list(range(total))
    return sorted(set(np.linspace(0, total - 1, quota).round().astype(int).tolist()))


def _atomic_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_img(img, dest, ext):
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if ext.lower() in (".jpg", ".jpeg"):
        img.save(tmp, "JPEG", quality=95, subsampling=0)
    else:
        img.save(tmp)
    os.replace(tmp, dest)


def _count_boxes(out, split):
    c = Counter()
    for tf in (out / "labels" / split).glob("*.txt"):
        for ln in tf.read_text().splitlines():
            if ln.strip():
                c[int(ln.split()[0])] += 1
    return c


def autolabel(
    videos, repo_root, out_dir, sam3_model, sam3_proc, device,
    classes=None, frames_per_video=30, val_frac=0.2, img_ext=".jpg", seed=42,
):
    import csv
    import random

    classes = classes or DEFAULT_CLASSES
    repo_root = Path(repo_root)
    out = Path(out_dir)
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

    plan, total_est = [], 0
    for vid, ruta in videos:
        try:
            vr = decord.VideoReader(str(repo_root / ruta))
            idxs = frame_indices(len(vr), frames_per_video)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP video {ruta}: {e}", flush=True)
            continue
        plan.append((vid, ruta, idxs))
        total_est += len(idxs)
    print(f"videos OK={len(plan)} frames_estimados={total_est} val_videos={len(val_ids)}", flush=True)

    t0 = time.time()
    done = 0
    for vid, ruta, idxs in plan:
        split = "val" if vid in val_ids else "train"
        stem = Path(ruta).stem.replace(" ", "_")
        try:
            vr = decord.VideoReader(str(repo_root / ruta))
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP video {ruta}: {e}", flush=True)
            continue
        for fi in idxs:
            name = f"{stem}_f{fi:04d}"
            lbl_path = out / "labels" / split / f"{name}.txt"
            img_path = out / "images" / split / f"{name}{img_ext}"
            manifest.append([vid, ruta, fi, split, f"images/{split}/{name}{img_ext}"])
            if lbl_path.exists() and img_path.exists():
                skipped += 1
                done += 1
                continue
            try:
                frame = vr[fi].asnumpy()
                img = Image.fromarray(frame)
                w, h = img.size
                lines = []
                for cls in classes:
                    for det in segment_with_text(sam3_model, sam3_proc, img, cls["prompt"], device):
                        if det.score < SCORE_THRESH:
                            continue
                        box = mask_to_yolo_box(mask_from_logits(det.logits, w, h), w, h)
                        if box is None:
                            continue
                        xc, yc, bw, bh = box
                        lines.append(f"{cls['id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                _atomic_img(img, img_path, img_ext)       # imagen primero
                _atomic_text(lbl_path, "\n".join(lines))  # .txt = marcador 'done'
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
                eta = el / max(1, (done - skipped)) * (total_est - done)
                print(f"  {done}/{total_est} (nuevas {done - skipped}, {el:.0f}s, ETA {eta / 60:.0f}min)", flush=True)
        print(f"  done {ruta} [{split}]", flush=True)

    write_data_yaml(out, classes)
    with (out / "manifest.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video_id", "video_ruta", "frame_idx", "split", "image"])
        wr.writerows(manifest)

    bt, bv = _count_boxes(out, "train"), _count_boxes(out, "val")
    names = {c["id"]: c["name"] for c in classes}
    print(f"\nDataset YOLO -> {out}", flush=True)
    print(f"imgs nuevas train={n_img['train']} val={n_img['val']} | saltadas(resume)={skipped}", flush=True)
    print(f"cajas TRAIN: {{ {', '.join(f'{names[k]}={v}' for k,v in sorted(bt.items()))} }}  total={sum(bt.values())}", flush=True)
    print(f"cajas VAL:   {{ {', '.join(f'{names[k]}={v}' for k,v in sorted(bv.items()))} }}  total={sum(bv.values())}", flush=True)
    if sum(bt.values()) == 0 or sum(bv.values()) == 0:
        print("!!! ALERTA: train o val sin cajas -> YOLO no entrenara bien. Revisar split.", flush=True)
    return {"out": str(out), "n_img": n_img, "boxes_train": dict(bt), "boxes_val": dict(bv),
            "val_ids": sorted(val_ids), "skipped": skipped}


def write_data_yaml(out_dir, classes):
    out = Path(out_dir)
    items = sorted(classes, key=lambda c: c["id"])
    names_block = "\n".join(f"  {c['id']}: {c['name']}" for c in items)
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(items)}\n"
        f"names:\n{names_block}\n"
    )
    return out / "data.yaml"
