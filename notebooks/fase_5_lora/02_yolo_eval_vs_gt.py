# -*- coding: utf-8 -*-
"""fase_5_lora / 02 — Eval del pipeline vs GT humano (600 frames). DOS pasadas.

Cierra el hueco honesto de C3: hoy el mAP del draft (0.895) se midio contra las
pseudo-etiquetas de SAM3 (fidelidad al maestro), NO contra GT humano. Aqui se mide
contra los 600 GT humanos (testing_600), y ademas se evalua a SAM3 en el pipeline
DESPLEGADO (YOLO->SAM3 box-prompt), no solo el detector suelto.

  Pasada (2)  DETECCION: YOLO11 (best.pt) cajas  vs  cajas GT humano
              -> mAP50 / mAP50-95 por clase (via ultralytics val). El numero
              HONESTO del alumno (vs humano), no vs SAM3.

  Pasada (3)  SEGMENTACION END-TO-END: YOLO->SAM3 box-prompt (mascaras)  vs
              mascaras GT humano -> mIoU / Dice / Boundary-IoU por clase,
              COMPARABLE con la Tabla 7 (SAM3-text). Responde: ¿alimentar SAM3 con
              cajas de YOLO conserva la calidad de mascara de SAM3-text?

Criterios (acordados): robot = robot_a ∪ robot_b (YOLO no distingue equipo);
green_floor EXCLUIDO (YOLO no tiene esa clase / la caja seria casi todo el frame);
area minima 100 px; imgsz/conf de la config (yolo: imgsz 960, conf 0.4).

Corre en el POD (GPU + assets/sam3 + assets/yolo/best.pt + dataset testing_600).
Pegar en /workspace y:
    python 02_yolo_eval_vs_gt.py                 # 600 frames, ambas pasadas
    python 02_yolo_eval_vs_gt.py --limit 10      # smoke (10 frames)
    python 02_yolo_eval_vs_gt.py --pass2-only    # solo deteccion (rapido)
    python 02_yolo_eval_vs_gt.py --pass3-only    # solo YOLO->SAM3 seg
    python 02_yolo_eval_vs_gt.py --pass3-only --conf 0.1   # diagnostico Factor 2

--conf solo afecta la PASADA 3 (gatea la deteccion YOLO antes del box-prompt). La
pasada 2 (mAP) es independiente: ultralytics barre umbrales internamente. Bajar
--conf responde si el colapso de blue_zone es recall de deteccion (sube) o limite de
SAM3 (no se mueve). La salida se nombra con sufijo (_confNNN) para NO pisar el
baseline conf=config.

Salida: outputs/yolo_eval_vs_gt/{det_vs_gt,seg_yolo_sam3_vs_gt,comparison_seg}.csv (+ .json)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path("/workspace/FutBotMX-UAQTeam")
DATA = REPO / "notebooks" / "fase_5_lora" / "dataset" / "testing_600"
ANN_DIR = DATA / "ann"
IMG_DIR = DATA / "img"
ENV_FILE = REPO / ".env"           # o /workspace/.env; se intenta ambos
OUT_DIR = REPO / "outputs" / "yolo_eval_vs_gt"
SAM3_SEG_CSV = REPO / "outputs" / "seg_eval" / "seg_eval_vs_gt_zeroshot.csv"  # Tabla 7

MIN_AREA = 100
BOUNDARY_PX = 3

# GT crudas -> id YOLO (robots unidos; green_floor fuera).
GT_TO_YOLO_ID = {"robot_a": 0, "robot_b": 0, "orange_ball": 1, "yellow_zone": 2, "blue_zone": 3}
YOLO_NAMES = {0: "robot", 1: "orange_ball", 2: "yellow_zone", 3: "blue_zone"}

# Clases SEMANTICAS evaluadas en seg (nombre detector -> clases GT a unir).
EVAL_CLASSES = [
    {"name": "robot", "gt": ["robot_a", "robot_b"]},
    {"name": "orange_ball", "gt": ["orange_ball"]},
    {"name": "yellow_zone", "gt": ["yellow_zone"]},
    {"name": "blue_zone", "gt": ["blue_zone"]},
]


def load_env() -> None:
    for ef in (ENV_FILE, Path("/workspace/.env")):
        if ef.exists():
            for line in ef.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


# ---------------- GT (Supervisely bitmap, decodificado sin la lib) ----------------
def _decode_bitmap(s: str) -> np.ndarray:
    """base64(+zlib) -> mascara booleana local (replica sly.Bitmap.base64_2_data)."""
    raw = base64.b64decode(s)
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass  # formato viejo: base64 directo del PNG
    n = np.frombuffer(raw, np.uint8)
    im = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        return im[:, :, 3].astype(bool)        # canal alfa
    if im.ndim == 3 and im.shape[2] == 1:
        return im[:, :, 0].astype(bool)
    if im.ndim == 2:
        return im.astype(bool)
    raise RuntimeError(f"formato de bitmap inesperado: shape={im.shape}")


def _iter_gt_objects(ann_path: Path):
    """Itera (classTitle, mask_local, col0, row0, H, W) de cada objeto del JSON."""
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    h, w = ann["size"]["height"], ann["size"]["width"]
    for obj in ann.get("objects", []):
        bm = obj.get("bitmap")
        if not bm:  # solo geometria bitmap (la usada por el GT)
            continue
        mask = _decode_bitmap(bm["data"])
        col0, row0 = int(bm["origin"][0]), int(bm["origin"][1])  # origin = [x, y]
        yield obj["classTitle"], mask, col0, row0, h, w


def gt_masks_by_class(ann_path: Path):
    """Mascara binaria GT por clase cruda (union de instancias) + (H, W)."""
    out: dict[str, np.ndarray] = {}
    h = w = 0
    for name, mask, c0, r0, h, w in _iter_gt_objects(ann_path):
        canvas = out.setdefault(name, np.zeros((h, w), dtype=bool))
        canvas[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] |= mask
    return out, h, w


def gt_boxes_yolo(ann_path: Path):
    """Lista de lineas YOLO (cls xc yc bw bh, normalizadas) desde las cajas GT."""
    lines = []
    h = w = 0
    for name, mask, c0, r0, h, w in _iter_gt_objects(ann_path):
        cid = GT_TO_YOLO_ID.get(name)
        if cid is None:  # green_floor u otra -> fuera
            continue
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        x1, y1 = c0 + int(xs.min()), r0 + int(ys.min())
        x2, y2 = c0 + int(xs.max()) + 1, r0 + int(ys.max()) + 1
        if (x2 - x1) * (y2 - y1) < MIN_AREA:
            continue
        xc, yc = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines, h, w


# ---------------- metricas seg ----------------
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


def _write(rows, fieldnames, tag, payload_extra=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    payload = {"rows": rows, **(payload_extra or {})}
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT_DIR / (tag + '.csv')}")


# ============================================================ #
# PASADA 2 — DETECCION YOLO vs cajas GT (mAP via ultralytics)  #
# ============================================================ #
def run_pass2(anns) -> None:
    from ultralytics import YOLO

    print("\n===== PASADA 2: deteccion YOLO vs GT humano (mAP) =====")
    ds = OUT_DIR / "yolo_gt_ds"
    img_out, lbl_out = ds / "images" / "val", ds / "labels" / "val"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_lab = 0
    for ann_path in anns:
        stem_img = ann_path.stem  # "<img>.png"
        src_img = IMG_DIR / stem_img
        if not src_img.exists():
            continue
        # symlink imagen (sin copiar)
        dst_img = img_out / stem_img
        if not dst_img.exists():
            try:
                dst_img.symlink_to(src_img)
            except OSError:
                import shutil
                shutil.copy(src_img, dst_img)
        lines, _, _ = gt_boxes_yolo(ann_path)
        (lbl_out / (Path(stem_img).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        n_lab += 1

    names_block = "\n".join(f"  {i}: {n}" for i, n in YOLO_NAMES.items())
    # ultralytics exige train Y val aunque solo validemos -> ambas al mismo split.
    (ds / "data.yaml").write_text(
        f"path: {ds}\ntrain: images/val\nval: images/val\nnames:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"dataset YOLO-GT: {n_lab} imgs etiquetadas -> {ds}")

    weights = str(_yolo_weights())
    imgsz = _yolo_imgsz()
    print(f"weights: {weights} | imgsz: {imgsz}")
    model = YOLO(weights)
    r = model.val(data=str(ds / "data.yaml"), imgsz=imgsz, verbose=False)
    rows = [{
        "class": "all",
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
    }]
    for i, name in model.names.items():
        try:
            rows.append({
                "class": name,
                "mAP50": round(float(r.box.ap50[i]), 4),
                "mAP50_95": round(float(r.box.ap[i]), 4),
            })
        except Exception:
            rows.append({"class": name, "mAP50": None, "mAP50_95": None})
    print(f"{'class':14s}{'mAP50':>9s}{'mAP50-95':>10s}")
    for r_ in rows:
        print(f"{r_['class']:14s}{(r_['mAP50'] or -1):>9.4f}{(r_['mAP50_95'] or -1):>10.4f}")
    _write(rows, ["class", "mAP50", "mAP50_95"], "det_vs_gt",
           {"note": "YOLO best.pt vs cajas GT humano (robots unidos, green_floor fuera)"})


def _yolo_weights() -> Path:
    """Ruta de best.pt desde la config (working_dirs.yolo_weights)."""
    from src.utils import get_abs_path
    cfg = json.loads(get_abs_path(f"configs/{os.environ['CONFIG_FILENAME']}").read_text("utf-8"))
    return get_abs_path(cfg["working_dirs"]["yolo_weights"])


def _yolo_imgsz() -> int:
    from src.utils import get_abs_path
    cfg = json.loads(get_abs_path(f"configs/{os.environ['CONFIG_FILENAME']}").read_text("utf-8"))
    return int(cfg.get("yolo", {}).get("imgsz", 960))


# ============================================================ #
# PASADA 3 — YOLO->SAM3 box-prompt (mascaras) vs GT            #
# ============================================================ #
def run_pass3(anns, conf: float | None = None) -> None:
    import torch

    from src.core.detectors import yolo_sam3
    from src.core.sam3_loader import load_sam3
    from src.core.segmentation import _load_classes

    suffix = "" if conf is None else f"_conf{int(round(conf * 100)):03d}"
    print("\n===== PASADA 3: YOLO->SAM3 (mascaras) vs GT humano =====")
    print(f"conf YOLO: {'config (default)' if conf is None else conf}"
          f"{' | salida: ' + 'seg_yolo_sam3_vs_gt' + suffix if suffix else ''}")
    bundle = load_sam3()
    all_classes = _load_classes()
    yolo_classes = [c for c in all_classes if c["name"] in {e["name"] for e in EVAL_CLASSES}]
    print("clases YOLO->SAM3:", [c["name"] for c in yolo_classes],
          "| device:", bundle.device)

    acc = {e["name"]: {"iou": [], "dice": [], "biou": [], "n": 0} for e in EVAL_CLASSES}
    failed = []
    t0 = time.time()
    for k, ann_path in enumerate(anns):
        img_path = IMG_DIR / ann_path.stem
        if not img_path.exists():
            failed.append((ann_path.name, "img faltante")); continue
        try:
            gt_raw, h, w = gt_masks_by_class(ann_path)
            rgb = np.asarray(Image.open(img_path).convert("RGB"))
            res = yolo_sam3.detect(rgb, classes=yolo_classes, bundle=bundle, conf=conf)
            for e in EVAL_CLASSES:
                gt = np.zeros((h, w), dtype=bool)
                for g in e["gt"]:
                    if g in gt_raw:
                        gt |= gt_raw[g]
                pred = np.zeros((h, w), dtype=bool)
                for det in res.get(e["name"], []):
                    m = det.mask
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if m.sum() >= MIN_AREA:
                        pred |= m
                if gt.any():
                    acc[e["name"]]["iou"].append(_iou(gt, pred) or 0.0)
                    acc[e["name"]]["dice"].append(_dice(gt, pred) or 0.0)
                    acc[e["name"]]["biou"].append(_biou(gt, pred, BOUNDARY_PX) or 0.0)
                    acc[e["name"]]["n"] += 1
        except Exception as ex:  # noqa: BLE001
            failed.append((ann_path.name, str(ex)[:120]))
        if (k + 1) % 25 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  {k + 1}/{len(anns)}  ({time.time() - t0:.0f}s)")

    rows = []
    for e in EVAL_CLASSES:
        a = acc[e["name"]]
        rows.append({
            "class": e["name"], "n_frames": a["n"],
            "mIoU": round(float(np.mean(a["iou"])), 4) if a["iou"] else None,
            "Dice": round(float(np.mean(a["dice"])), 4) if a["dice"] else None,
            "BoundaryIoU": round(float(np.mean(a["biou"])), 4) if a["biou"] else None,
        })
    valid = [r for r in rows if r["mIoU"] is not None]
    rows.append({
        "class": "mean", "n_frames": sum(r["n_frames"] for r in rows),
        "mIoU": round(float(np.mean([r["mIoU"] for r in valid])), 4),
        "Dice": round(float(np.mean([r["Dice"] for r in valid])), 4),
        "BoundaryIoU": round(float(np.mean([r["BoundaryIoU"] for r in valid])), 4),
    })
    print(f"{'class':14s}{'n':>5s}{'mIoU':>9s}{'Dice':>9s}{'B-IoU':>9s}")
    for r in rows:
        print(f"{r['class']:14s}{r['n_frames']:>5d}"
              f"{(r['mIoU'] if r['mIoU'] is not None else -1):>9.4f}"
              f"{(r['Dice'] if r['Dice'] is not None else -1):>9.4f}"
              f"{(r['BoundaryIoU'] if r['BoundaryIoU'] is not None else -1):>9.4f}")
    if failed:
        print(f"\n{len(failed)} fallaron; ej: {failed[:3]}")
    _write(rows, ["class", "n_frames", "mIoU", "Dice", "BoundaryIoU"], f"seg_yolo_sam3_vs_gt{suffix}",
           {"note": "YOLO->SAM3 box-prompt vs GT humano; robots unidos; green_floor fuera",
            "conf": ("config" if conf is None else conf)})

    # comparacion con SAM3-text (Tabla 7) si existe el CSV
    if SAM3_SEG_CSV.exists():
        sam3 = {row["class"]: row for row in csv.DictReader(SAM3_SEG_CSV.open(encoding="utf-8"))}
        comp = []
        for r in rows:
            s = sam3.get(r["class"])
            comp.append({
                "class": r["class"],
                "SAM3text_mIoU": (float(s["mIoU"]) if s else None),
                "YOLO_SAM3_mIoU": r["mIoU"],
                "delta": (round(r["mIoU"] - float(s["mIoU"]), 4) if (s and r["mIoU"] is not None) else None),
            })
        print("\n=== comparacion seg vs Tabla 7 (SAM3-text) ===")
        print(f"{'class':14s}{'SAM3text':>10s}{'YOLO_SAM3':>11s}{'delta':>9s}")
        for c in comp:
            print(f"{c['class']:14s}{(c['SAM3text_mIoU'] or -1):>10.4f}"
                  f"{(c['YOLO_SAM3_mIoU'] or -1):>11.4f}{(c['delta'] if c['delta'] is not None else -9):>9.4f}")
        _write(comp, ["class", "SAM3text_mIoU", "YOLO_SAM3_mIoU", "delta"], f"comparison_seg{suffix}")
    else:
        print(f"\n(aviso) no esta {SAM3_SEG_CSV} -> sin comparacion automatica con Tabla 7")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="N frames (smoke); 0 = 600")
    ap.add_argument("--pass2-only", action="store_true")
    ap.add_argument("--pass3-only", action="store_true")
    ap.add_argument("--conf", type=float, default=None,
                    help="umbral conf YOLO en la pasada 3 (None=config). Sufija la salida")
    args = ap.parse_args()
    load_env()
    os.environ.setdefault("CONFIG_FILENAME", "01_yolo_sam3_config.json")

    anns = sorted(ANN_DIR.glob("*.json"))
    if args.limit:
        anns = anns[: args.limit]
    print(f"frames: {len(anns)} | OUT: {OUT_DIR}")

    if not args.pass3_only:
        run_pass2(anns)
    if not args.pass2_only:
        run_pass3(anns, conf=args.conf)


if __name__ == "__main__":
    main()
