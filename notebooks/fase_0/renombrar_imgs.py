"""Renombra las 600 imagenes del project 378046 a nombres legibles
'<video_stem>_f<frame_index:04d>.png' (ej IMG_9779_f0000.png) SIN re-subir ni
tocar las anotaciones. Escribe assets/gt_sly_name_map.csv para preservar la traza
con testing_frames.csv. Idempotente.
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

df = pd.read_csv(CSV)
df["old"] = df["imagen"].apply(lambda p: Path(p).name)
df["stem"] = df["video_ruta"].apply(lambda p: Path(p).stem.replace(" ", "_"))
df["sly_name"] = df.apply(lambda r: f"{r['stem']}_f{int(r['frame_index']):04d}.png", axis=1)

assert df["sly_name"].is_unique, "COLISION en sly_name -> abortar"
old2new = dict(zip(df["old"], df["sly_name"]))
done = set(df["sly_name"])

ds = api.dataset.get_list(PID)[0]
imgs = api.image.get_list(ds.id)
print(f"imgs={len(imgs)}")

renamed, already, fail = 0, 0, []
for im in imgs:
    if im.name in done:
        already += 1
        continue
    new = old2new.get(im.name)
    if not new:
        fail.append((im.name, "no en CSV"))
        continue
    try:
        api.image.rename(im.id, new)
        renamed += 1
        if renamed % 100 == 0:
            print(f"  renombradas {renamed}")
    except Exception as e:  # noqa: BLE001
        fail.append((im.name, str(e)[:100]))

# Guardar mapa de traza (sin tocar testing_frames.csv).
df[["video_id", "frame_index", "frame_original", "grupo", "video_ruta", "old", "sly_name"]].rename(
    columns={"old": "old_name"}
).to_csv(MAP_OUT, index=False)

print(f"\nrenombradas={renamed} ya_estaban={already} fail={len(fail)}")
for n, e in fail[:5]:
    print(f"  FAIL {n}: {e}")
print(f"mapa -> {MAP_OUT}")
# muestra
for r in df.head(3).to_dict("records"):
    print(f"  {r['old']} -> {r['sly_name']}  (video_id={r['video_id']}, frame_real={r['frame_original']})")
