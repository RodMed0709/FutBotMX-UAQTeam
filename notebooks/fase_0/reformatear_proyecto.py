"""Reformatea project 378046 (robusto a estado mixto tras el fallo de colision):

1) RENAME en 2 PASADAS (sin colisiones): cada imagen -> '__tmp_<id>.png' -> nombre
   real '<stem>_f<frame_original:04d>.png'. Decodifica el estado actual (mezcla de
   nombres por-indice y ya-reales) de forma inequivoca antes de tocar nada.
2) CLASES: 'robot' -> 'robot_a' (reescribe labels) + crea 'robot_b' vacia.
   Meta final: robot_a, robot_b, orange_ball, green_floor, yellow_zone.
Regenera assets/gt_sly_name_map.csv.
"""
import os
from pathlib import Path

import pandas as pd
import supervisely as sly

for l in Path("/workspace/.env").read_text().splitlines():
    if "=" in l and not l.strip().startswith("#"):
        k, v = l.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
api = sly.Api(os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com"), os.environ["SUPERVISELY_API_TOKEN"])

PID = 378046
CSV = Path("/workspace/FutBotMX-UAQTeam/assets/testing_frames.csv")
MAP_OUT = Path("/workspace/FutBotMX-UAQTeam/assets/gt_sly_name_map.csv")
COLORS = {
    "robot_a": [60, 130, 255], "robot_b": [230, 30, 200], "orange_ball": [255, 100, 0],
    "green_floor": [50, 220, 70], "yellow_zone": [255, 230, 0],
}

df = pd.read_csv(CSV)
df["stem"] = df["video_ruta"].apply(lambda p: Path(p).stem.replace(" ", "_"))
stem2vid = dict(zip(df["stem"], df["video_id"]))
idx2real = {(int(r["video_id"]), int(r["frame_index"])): int(r["frame_original"]) for r in df.to_dict("records")}
real_by_vid = {}
for r in df.to_dict("records"):
    real_by_vid.setdefault(int(r["video_id"]), set()).add(int(r["frame_original"]))


def parse(name):
    stem, num = name[:-4].rsplit("_f", 1)  # quita .png
    return stem, int(num)


ds = api.dataset.get_list(PID)[0]
imgs = api.image.get_list(ds.id)
print(f"imgs={len(imgs)}")

# Decodifica nombre REAL objetivo para CADA imagen segun estado actual (inequivoco):
#  - NNNN>29  -> ya es frame_original (real) -> se queda
#  - NNNN<=29 -> es indice de cuota -> real = frame_original del indice
id2real = {}
for im in imgs:
    stem, nnnn = parse(im.name)
    vid = stem2vid[stem]
    if nnnn > 29:
        real = nnnn
    else:
        real = idx2real[(vid, nnnn)]
    id2real[im.id] = f"{stem}_f{real:04d}.png"

# unicidad de destinos
assert len(set(id2real.values())) == len(id2real), "COLISION en nombres destino"

# PASADA 1: todo a temporal unico
for im in imgs:
    api.image.rename(im.id, f"__tmp_{im.id}.png")
print("pasada 1 (tmp) ok")
# PASADA 2: tmp -> real
for im in imgs:
    api.image.rename(im.id, id2real[im.id])
print("pasada 2 (real) ok")

# mapa de traza
rows = []
for r in df.to_dict("records"):
    vid, fi = int(r["video_id"]), int(r["frame_index"])
    rows.append({
        "video_id": vid, "frame_index": fi, "frame_original": int(r["frame_original"]),
        "grupo": r["grupo"], "video_ruta": r["video_ruta"],
        "orig_upload": Path(r["imagen"]).name,
        "sly_name": f"{r['stem']}_f{int(r['frame_original']):04d}.png",
    })
pd.DataFrame(rows).to_csv(MAP_OUT, index=False)
print(f"mapa -> {MAP_OUT}")

# ---------- CLASES ----------
meta = sly.ProjectMeta.from_json(api.project.get_meta(PID))
add = [sly.ObjClass(n, sly.Bitmap, color=COLORS[n]) for n in ("robot_a", "robot_b") if meta.obj_classes.get(n) is None]
if add:
    meta = meta.add_obj_classes(add)
    api.project.update_meta(PID, meta.to_json())
robot_a = meta.obj_classes.get("robot_a")

imgs = api.image.get_list(ds.id)
ids = [im.id for im in imgs]
moved = 0
for info in api.annotation.download_batch(ds.id, ids):
    ann = sly.Annotation.from_json(info.annotation, meta)
    changed = False
    new_labels = []
    for lb in ann.labels:
        if lb.obj_class.name == "robot":
            new_labels.append(lb.clone(obj_class=robot_a)); changed = True; moved += 1
        else:
            new_labels.append(lb)
    if changed:
        api.annotation.upload_ann(info.image_id, ann.clone(labels=new_labels))
print(f"labels robot -> robot_a: {moved}")

final_names = ["robot_a", "robot_b", "orange_ball", "green_floor", "yellow_zone"]
final = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(
    [meta.obj_classes.get(n) or sly.ObjClass(n, sly.Bitmap, color=COLORS[n]) for n in final_names]))
api.project.update_meta(PID, final.to_json())
print("meta final:", [o.name for o in sly.ProjectMeta.from_json(api.project.get_meta(PID)).obj_classes])
print("LISTO")
