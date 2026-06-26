# -*- coding: utf-8 -*-
"""fase_5_lora / 03 — Diagnostico de factores que sesgan la eval de C3 (600 GT).

Complementa a 02_yolo_eval_vs_gt.py. El 02 da los NUMEROS (mAP, mIoU); este script
da las COMPROBACIONES de por que esos numeros pueden estar sesgados. Una sola pasada
por los 600 frames cubre 4 factores; los otros se comprueban con --conf (en el 02) o
en local. Mapa:

  F1  Nombres de clase     -> inventario de classTitle vs lo esperado (fuga silenciosa)
  F4  IoU semantica        -> instancias GT/frame vs nº detecciones (enmascaramiento)
  F5  FP no penalizados    -> predicciones huerfanas (pred sin GT) que inflan el seg
  F6  Resize de mascaras   -> cuantas mascaras pred no llegan a tamaño nativo
  F3  Cajas desde mascaras -> overlays a disco (caja GT + mascara GT + mascara pred)

  [F2 se comprueba con 02 --conf | F7 es disciplina al escribir | F8 es hash local]

Por defecto corre TODO (necesita GPU + SAM3 + YOLO, igual que la pasada 3 del 02).
Para solo el inventario de clases (F1, sin GPU): --gt-only.

Corre en el POD. Pegar en /workspace y:
    python 03_diagnostico_factores.py                 # 600 frames, F1+F3+F4+F5+F6
    python 03_diagnostico_factores.py --limit 20      # smoke
    python 03_diagnostico_factores.py --gt-only       # solo F1 (sin SAM3/GPU)
    python 03_diagnostico_factores.py --overlays 16   # nº de frames volcados (F3)

Salida: outputs/yolo_eval_vs_gt/diagnostico/{factor1_clases,factor4_instancias,
        factor5_huerfanas,factor6_resolucion}.csv + overlays/*.png + resumen.json
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path("/workspace/FutBotMX-UAQTeam")
DATA = REPO / "notebooks" / "fase_5_lora" / "dataset" / "testing_600"
ANN_DIR = DATA / "ann"
IMG_DIR = DATA / "img"
ENV_FILE = REPO / ".env"
OUT_DIR = REPO / "outputs" / "yolo_eval_vs_gt" / "diagnostico"
OVERLAY_DIR = OUT_DIR / "overlays"

MIN_AREA = 100

# Clases crudas esperadas (las confirmadas por los conteos GT). Cualquier otra string
# en el GT es una FUGA: el 02 la descarta en silencio.
EXPECTED_GT = {"robot_a", "robot_b", "orange_ball", "green_floor", "yellow_zone", "blue_zone"}
# Mapeo del 02: que clases crudas entran al detector y bajo que nombre semantico.
GT_TO_YOLO_ID = {"robot_a": 0, "robot_b": 0, "orange_ball": 1, "yellow_zone": 2, "blue_zone": 3}
EVAL_CLASSES = [
    {"name": "robot", "gt": ["robot_a", "robot_b"]},
    {"name": "orange_ball", "gt": ["orange_ball"]},
    {"name": "yellow_zone", "gt": ["yellow_zone"]},
    {"name": "blue_zone", "gt": ["blue_zone"]},
]
# color BGR por clase semantica (para los overlays).
CLS_COLOR = {"robot": (0, 0, 255), "orange_ball": (0, 165, 255),
             "yellow_zone": (0, 255, 255), "blue_zone": (255, 0, 0)}


def load_env() -> None:
    for ef in (ENV_FILE, Path("/workspace/.env")):
        if ef.exists():
            for line in ef.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


# ---------------- GT (Supervisely bitmap, decodificado sin la lib; igual que 02) ----------------
def _decode_bitmap(s: str) -> np.ndarray:
    raw = base64.b64decode(s)
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    n = np.frombuffer(raw, np.uint8)
    im = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        return im[:, :, 3].astype(bool)
    if im.ndim == 3 and im.shape[2] == 1:
        return im[:, :, 0].astype(bool)
    if im.ndim == 2:
        return im.astype(bool)
    raise RuntimeError(f"formato de bitmap inesperado: shape={im.shape}")


def _iter_gt_objects(ann_path: Path):
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    h, w = ann["size"]["height"], ann["size"]["width"]
    for obj in ann.get("objects", []):
        bm = obj.get("bitmap")
        if not bm:
            continue
        mask = _decode_bitmap(bm["data"])
        col0, row0 = int(bm["origin"][0]), int(bm["origin"][1])
        yield obj["classTitle"], mask, col0, row0, h, w


def gt_instances(ann_path: Path):
    """(lista de (classTitle, mask_full bool, bbox), H, W) — una entrada por instancia."""
    insts = []
    h = w = 0
    for name, mask, c0, r0, h, w in _iter_gt_objects(ann_path):
        full = np.zeros((h, w), dtype=bool)
        full[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] |= mask
        ys, xs = np.where(full)
        if ys.size == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        insts.append((name, full, bbox))
    return insts, h, w


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(u) if u else None


def _write_csv(rows, fieldnames, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT_DIR / (tag + '.csv')}")


# ============================================================ #
# F1 — inventario de classTitle (fuga silenciosa de clases)    #
# ============================================================ #
def factor1(anns) -> dict:
    print("\n===== F1: inventario de classTitle =====")
    c = Counter()
    for ann_path in anns:
        for name, *_ in _iter_gt_objects(ann_path):
            c[name] += 1
    rows, leak = [], []
    for name, n in sorted(c.items(), key=lambda kv: -kv[1]):
        recognized = name in EXPECTED_GT
        if not recognized:
            leak.append(name)
        rows.append({
            "classTitle": name, "count": n,
            "esperada": recognized,
            "mapea_a": ("green_floor->EXCLUIDA" if name == "green_floor"
                        else GT_TO_YOLO_ID.get(name, "DESCARTADA(no mapea)")),
        })
    _write_csv(rows, ["classTitle", "count", "esperada", "mapea_a"], "factor1_clases")
    for r in rows:
        print(f"  {r['classTitle']:14s} {r['count']:5d}  esperada={r['esperada']!s:5s}  -> {r['mapea_a']}")
    if leak:
        print(f"  !!! FUGA: classTitles no esperados (el 02 los descarta en silencio): {leak}")
    else:
        print("  OK: todos los classTitle son los esperados; sin fuga silenciosa.")
    return {"leak": leak, "counts": dict(c)}


# ============================================================ #
# F4/F5/F6 — requieren detecciones (YOLO->SAM3)               #
# ============================================================ #
def factors_detect(anns, n_overlays: int) -> dict:
    import torch

    from src.core.detectors import yolo_sam3
    from src.core.sam3_loader import load_sam3
    from src.core.segmentation import _load_classes

    print("\n===== F4/F5/F6: instancias, huerfanas, resize (YOLO->SAM3) =====")
    bundle = load_sam3()
    all_classes = _load_classes()
    yolo_classes = [c for c in all_classes if c["name"] in {e["name"] for e in EVAL_CLASSES}]
    print("device:", bundle.device, "| overlays:", n_overlays)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # acumuladores
    f4 = {e["name"]: {"gt_inst": [], "n_pred": [], "multi_gt": 0} for e in EVAL_CLASSES}
    f5 = {e["name"]: {"gt_frames": 0, "pred_frames": 0, "orphan": 0, "orphan_px": []} for e in EVAL_CLASSES}
    f6 = {"total_masks": 0, "resized": 0}
    dumped = 0
    t0 = time.time()

    for k, ann_path in enumerate(anns):
        img_path = IMG_DIR / ann_path.stem
        if not img_path.exists():
            continue
        insts, h, w = gt_instances(ann_path)
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        res = yolo_sam3.detect(rgb, classes=yolo_classes, bundle=bundle)

        # GT por clase semantica: nº instancias + mascara union
        gt_inst_count = defaultdict(int)
        gt_union = {e["name"]: np.zeros((h, w), dtype=bool) for e in EVAL_CLASSES}
        raw_to_sem = {g: e["name"] for e in EVAL_CLASSES for g in e["gt"]}
        for name, full, _bbox in insts:
            sem = raw_to_sem.get(name)
            if sem is None:
                continue
            gt_inst_count[sem] += 1
            gt_union[sem] |= full

        for e in EVAL_CLASSES:
            name = e["name"]
            dets = [d for d in res.get(name, []) if d.mask.sum() >= MIN_AREA]
            n_gt = gt_inst_count[name]
            # F4
            f4[name]["gt_inst"].append(n_gt)
            f4[name]["n_pred"].append(len(dets))
            if n_gt > 1:
                f4[name]["multi_gt"] += 1
            # F6: resize de mascaras
            for d in dets:
                f6["total_masks"] += 1
                if d.mask.shape != (h, w):
                    f6["resized"] += 1
            # F5: huerfanas
            has_gt = gt_union[name].any()
            pred_union = np.zeros((h, w), dtype=bool)
            for d in dets:
                m = d.mask
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                pred_union |= m
            has_pred = pred_union.any()
            if has_gt:
                f5[name]["gt_frames"] += 1
            if has_pred:
                f5[name]["pred_frames"] += 1
            if has_pred and not has_gt:
                f5[name]["orphan"] += 1
                f5[name]["orphan_px"].append(int(pred_union.sum()))

        # F3/F6: volcar overlays de los primeros n_overlays frames
        if dumped < n_overlays:
            _dump_overlay(rgb, insts, res, h, w, OVERLAY_DIR / f"{ann_path.stem}.png")
            dumped += 1

        if (k + 1) % 25 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  {k + 1}/{len(anns)}  ({time.time() - t0:.0f}s)")

    # ---- F4 ----
    rows4 = []
    for e in EVAL_CLASSES:
        a = f4[e["name"]]
        gt = a["gt_inst"] or [0]
        rows4.append({
            "class": e["name"],
            "mean_gt_inst": round(float(np.mean(gt)), 3),
            "max_gt_inst": int(np.max(gt)),
            "frames_multi_gt": a["multi_gt"],  # frames con >1 instancia GT (riesgo enmascaramiento)
            "mean_n_pred": round(float(np.mean(a["n_pred"] or [0])), 3),
        })
    _write_csv(rows4, ["class", "mean_gt_inst", "max_gt_inst", "frames_multi_gt", "mean_n_pred"],
               "factor4_instancias")
    # ---- F5 ----
    rows5 = []
    for e in EVAL_CLASSES:
        a = f5[e["name"]]
        rows5.append({
            "class": e["name"],
            "gt_frames": a["gt_frames"],
            "pred_frames": a["pred_frames"],
            "huerfanas": a["orphan"],  # pred sin GT: NO penalizan el seg del 02 (lo inflan)
            "huerfana_px_medio": (round(float(np.mean(a["orphan_px"])), 1) if a["orphan_px"] else 0),
        })
    _write_csv(rows5, ["class", "gt_frames", "pred_frames", "huerfanas", "huerfana_px_medio"],
               "factor5_huerfanas")
    # ---- F6 ----
    pct = (100.0 * f6["resized"] / f6["total_masks"]) if f6["total_masks"] else 0.0
    rows6 = [{"total_masks": f6["total_masks"], "resized": f6["resized"], "pct_resized": round(pct, 2)}]
    _write_csv(rows6, ["total_masks", "resized", "pct_resized"], "factor6_resolucion")

    print("\n  F4 instancias (frames_multi_gt = riesgo de IoU semantica inflada):")
    for r in rows4:
        print(f"    {r['class']:14s} gt_inst~{r['mean_gt_inst']} (max {r['max_gt_inst']})"
              f"  multi_gt={r['frames_multi_gt']}  pred~{r['mean_n_pred']}")
    print("  F5 huerfanas (pred sin GT, inflan el seg):")
    for r in rows5:
        print(f"    {r['class']:14s} gt={r['gt_frames']} pred={r['pred_frames']}"
              f"  huerfanas={r['huerfanas']} (~{r['huerfana_px_medio']} px)")
    print(f"  F6 resize: {f6['resized']}/{f6['total_masks']} mascaras reescaladas ({pct:.1f}%)")
    print(f"  F3 overlays: {dumped} en {OVERLAY_DIR}")
    return {"f4": rows4, "f5": rows5, "f6": rows6[0], "overlays": dumped}


def _dump_overlay(rgb, insts, res, h, w, out_path: Path) -> None:
    """F3: caja GT (verde) + mascara GT (tenue) + mascara pred (color de clase)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    raw_to_sem = {g: e["name"] for e in EVAL_CLASSES for g in e["gt"]}
    # mascara GT union (tenue, blanco) + caja GT (verde)
    for name, full, bbox in insts:
        if raw_to_sem.get(name) is None:
            continue
        x1, y1, x2, y2 = bbox
        bgr[full] = (0.6 * bgr[full] + 0.4 * np.array([220, 220, 220])).astype(np.uint8)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # mascara pred (contorno del color de clase)
    for e in EVAL_CLASSES:
        col = CLS_COLOR[e["name"]]
        for d in res.get(e["name"], []):
            m = d.mask
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bgr, cnts, -1, col, 2)
    cv2.imwrite(str(out_path), bgr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="N frames (smoke); 0 = 600")
    ap.add_argument("--gt-only", action="store_true", help="solo F1 (sin SAM3/GPU)")
    ap.add_argument("--overlays", type=int, default=12, help="nº de overlays a volcar (F3)")
    args = ap.parse_args()
    load_env()
    os.environ.setdefault("CONFIG_FILENAME", "01_yolo_sam3_config.json")

    anns = sorted(ANN_DIR.glob("*.json"))
    if args.limit:
        anns = anns[: args.limit]
    print(f"frames: {len(anns)} | OUT: {OUT_DIR}")

    summary = {"n_frames": len(anns), "f1": factor1(anns)}
    if not args.gt_only:
        summary.update(factors_detect(anns, args.overlays))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> resumen: {OUT_DIR / 'resumen.json'}")


if __name__ == "__main__":
    main()
