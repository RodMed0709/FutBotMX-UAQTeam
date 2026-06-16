"""extra300 — 300 frames NUEVOS (extra_videos_lora.csv) -> proyecto Supervisely SEPARADO.

Misma receta que `subir_smoothv3_tta.py` (SAM3 + TTA x3, votacion >=2/3, Bitmap por
instancia) PERO:
  - lee `extra_videos_lora.csv` (10 videos, frame_indices random) y EXTRAE los PNG el
    mismo script (los 600 ya venian exportados; estos no).
  - 5 clases: agrega `blue_zone` (zona azul) a las 4 de los 600.
  - crea un PROYECTO NUEVO (no toca el de los 600) + 1 Labeling Job llamado "ARIEL".

Corre en el POD (SAM3 + GPU + token Supervisely en /workspace/.env).

Modos:
  python subir_extra300_blue.py --smoke 6        # 6 frames: extrae, corre SAM3, imprime conteos, NO sube
  python subir_extra300_blue.py                  # full: 300 frames -> proyecto + sube + job ARIEL (placeholder)
  python subir_extra300_blue.py --email a@x.com  # full, asigna el job ARIEL a ese miembro del team
"""
from __future__ import annotations

import argparse
import os
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

REPO = Path("/workspace/FutBotMX-UAQTeam")
EXTRA_CSV = REPO / "extra_videos_lora.csv"
FRAMES_OUT = REPO / "data" / "extra300_frames"   # PNG extraidos (git-ignored)
SAM3_PATH = REPO / "assets" / "sam3"
ENV_FILE = Path("/workspace/.env")

PROJECT_NAME = "FutBot_Extra_300_blue_smoothv3_tta"
DATASET_NAME = "extra_300"
JOB_NAME = "ARIEL"
MIN_AREA = 100
IOU_MATCH = 0.3
MIN_VOTES = 2
GAMMA = 0.8

# 5 clases = las 4 de los 600 + blue_zone (prompt/color de configs/01_yolo_sam3_config.json)
CLASSES = [
    {"name": "robot", "prompt": "robot", "color": [60, 130, 255]},
    {"name": "orange_ball", "prompt": "orange ball", "color": [255, 100, 0]},
    {"name": "green_floor", "prompt": "green playing surface with lines", "color": [50, 220, 70]},
    {"name": "yellow_zone", "prompt": "yellow zone", "color": [255, 230, 0]},
    {"name": "blue_zone", "prompt": "dark blue rectangle", "color": [30, 90, 220]},
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


def logits_to_full(logits, w0, h0):
    lo = logits.astype(np.float32)
    if lo.shape != (h0, w0):
        lo = cv2.resize(lo, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return lo


def make_views(img: Image.Image):
    arr = np.asarray(img).astype(np.float32)
    gamma = np.clip(((arr / 255.0) ** GAMMA) * 255.0, 0, 255).astype(np.uint8)
    return [
        ("id", img),
        ("hflip", img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("gamma", Image.fromarray(gamma)),
    ]


def tta_class_masks(model, processor, img, prompt, device):
    w, h = img.size
    res = []
    for _ri, (name, aug) in enumerate(make_views(img)):
        for det in segment_with_text(model, processor, aug, prompt, device):
            full = logits_to_full(det.logits, w, h)
            if name == "hflip":
                full = full[:, ::-1].copy()
            res.append((_ri, full, det.score))
    return res


def merge_tta(res):
    items = [(ri, full > 0.0, full, sc) for ri, full, sc in res]
    used = [False] * len(items)
    finals = []
    for i in range(len(items)):
        if used[i]:
            continue
        ri_i, bin_i, _, _ = items[i]
        runs = {ri_i}
        group = [i]
        union = bin_i.copy()
        used[i] = True
        for j in range(len(items)):
            if used[j]:
                continue
            rj, bin_j, _, _ = items[j]
            if rj in runs:
                continue
            inter = np.logical_and(union, bin_j).sum()
            uni = np.logical_or(union, bin_j).sum()
            if uni > 0 and inter / uni > IOU_MATCH:
                group.append(j)
                runs.add(rj)
                used[j] = True
                union = np.logical_or(union, bin_j)
        if len(runs) >= MIN_VOTES:
            avg = np.mean([items[k][2] for k in group], axis=0)
            score = float(np.mean([items[k][3] for k in group]))
            finals.append((avg > 0.0, score))
    return finals


def bitmap_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) < MIN_AREA:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    crop = mask[y0 : y1 + 1, x0 : x1 + 1].astype(bool)
    return sly.Bitmap(crop, origin=sly.PointLocation(row=y0, col=x0))


def extract_frames() -> list[Path]:
    """Lee extra_videos_lora.csv, extrae los frame_indices de cada video a PNG.
    Devuelve la lista de PNG (saltando los ya extraidos)."""
    FRAMES_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EXTRA_CSV)
    pngs: list[Path] = []
    for row in df.to_dict("records"):
        video = REPO / row["video_ruta"]
        stem = Path(row["nombre"]).stem
        idxs = [int(x) for x in str(row["frame_indices"]).split()]
        if not video.exists():
            print(f"  [!] FALTA video en el pod: {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        for idx in idxs:
            out_png = FRAMES_OUT / f"{stem}_f{idx:06d}.png"
            if not out_png.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    print(f"  [!] no se pudo leer {stem} frame {idx}")
                    continue
                cv2.imwrite(str(out_png), frame)  # BGR; PIL lo reabre como RGB
            pngs.append(out_png)
        cap.release()
        print(f"  {stem}: {len(idxs)} frames")
    print(f"PNG listos: {len(pngs)}")
    return pngs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="N frames, no sube, imprime conteos")
    ap.add_argument("--email", default=None, help="miembro del team a quien asignar el job ARIEL")
    args = ap.parse_args()
    load_env()

    print("== Extraccion de frames ==")
    pngs = extract_frames()
    if args.smoke:
        pngs = pngs[: args.smoke]
    if not pngs:
        print("Sin frames -> aborto (revisa que los videos esten en data/raw del pod).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(SAM3_PATH))
    model = AutoModel.from_pretrained(str(SAM3_PATH), dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    print(f"SAM3 load {time.time() - t0:.1f}s device={device}")

    api = dataset = obj_classes = None
    if not args.smoke:
        server = os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com")
        api = sly.Api(server, os.environ["SUPERVISELY_API_TOKEN"])
        team = api.team.get_list()[0]
        wss = api.workspace.get_list(team.id)
        ws = wss[0] if wss else api.workspace.create(team.id, "futbot", "")
        obj_classes = {c["name"]: sly.ObjClass(c["name"], sly.Bitmap, color=c["color"]) for c in CLASSES}
        meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(list(obj_classes.values())))
        project = api.project.create(ws.id, PROJECT_NAME, type=sly.ProjectType.IMAGES, change_name_if_conflict=True)
        api.project.update_meta(project.id, meta.to_json())
        dataset = api.dataset.create(project.id, DATASET_NAME, change_name_if_conflict=True)
        print(f"project id={project.id} name={project.name!r}")

    ok, failed = 0, []
    per_class = {c["name"]: 0 for c in CLASSES}
    uploaded_ids: list[int] = []
    t_start = time.time()
    for png in pngs:
        try:
            img = Image.open(png).convert("RGB")
            w, h = img.size
            labels = []
            counts = {}
            for cls in CLASSES:
                res = tta_class_masks(model, processor, img, cls["prompt"], device)
                finals = merge_tta(res)
                n = 0
                for mask, _score in finals:
                    bmp = bitmap_from_mask(mask)
                    if bmp is None:
                        continue
                    per_class[cls["name"]] += 1
                    n += 1
                    if not args.smoke:
                        labels.append(sly.Label(bmp, obj_classes[cls["name"]]))
                counts[cls["name"]] = n
            if args.smoke:
                print(f"  {png.name}: {counts}")
            else:
                info = api.image.upload_path(dataset.id, name=png.name, path=str(png))
                api.annotation.upload_ann(info.id, sly.Annotation(img_size=(h, w), labels=labels))
                uploaded_ids.append(info.id)
            ok += 1
            if not args.smoke and ok % 25 == 0:
                torch.cuda.empty_cache()
                print(f"  {ok}/{len(pngs)}  ({time.time() - t_start:.0f}s)")
        except Exception as e:  # noqa: BLE001
            failed.append((png.name, str(e)[:120]))

    print(f"\nSAM3 listo: {ok}/{len(pngs)} procesadas, {len(failed)} fallaron.")
    print("instancias por clase:", per_class)
    for n, e in failed[:5]:
        print(f"  FALLO {n}: {e}")

    if args.smoke:
        return

    # --- Labeling Job "ARIEL" con las 300 imagenes ---
    members = {u.login: u.id for u in api.user.get_team_members(team.id)}
    assignee = members.get(args.email) if args.email else None
    who = args.email if assignee else f"PLACEHOLDER({list(members)[0]}) -> reasignar"
    uid = assignee or list(members.values())[0]
    job = api.labeling_job.create(
        JOB_NAME, dataset.id, user_ids=[uid], images_ids=uploaded_ids,
        classes_to_label=[c["name"] for c in CLASSES],
        description="300 frames extra (5 clases incl. blue_zone) para segmentar",
    )
    print(f"Job '{JOB_NAME}' creado: {[j.id for j in job]} -> {who}, {len(uploaded_ids)} imgs")
    print(f"\nLISTO. Proyecto '{PROJECT_NAME}' + job '{JOB_NAME}' visibles en Supervisely.")


if __name__ == "__main__":
    main()
