"""Arregla los Labeling Jobs: estaban creados con la clase vieja 'robot', por eso
no muestran los robots tras renombrar a 'robot_a'. Archiva los viejos y recrea con
las 5 clases nuevas. Split por video 10/10 igual que antes.
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
MAP = Path("/workspace/FutBotMX-UAQTeam/assets/gt_sly_name_map.csv")
CLASSES = ["robot_a", "robot_b", "orange_ball", "green_floor", "yellow_zone"]
GROUP_A = [0, 2, 3, 20, 24, 25, 37, 45, 46, 56]
GROUP_B = [69, 71, 78, 82, 84, 88, 89, 90, 94, 95]
OLD_JOBS = [185731, 185732]

# Diagnostico: que clases tenian los jobs viejos
for jid in OLD_JOBS:
    try:
        info = api.labeling_job.get_info_by_id(jid)
        print(f"job {jid}: name={info.name!r} classes={getattr(info,'classes_to_label',None)}")
    except Exception as e:  # noqa: BLE001
        print(f"job {jid} info fail: {str(e)[:80]}")

# Archivar viejos
for jid in OLD_JOBS:
    try:
        api.labeling_job.archive(jid)
        print(f"archivado {jid}")
    except Exception as e:  # noqa: BLE001
        print(f"archive fail {jid}: {str(e)[:80]}")

team = api.team.get_list()[0]
members = {u.login: u.id for u in api.user.get_team_members(team.id)}
rodrigo = members.get("rodrigomed07@gmail.com", list(members.values())[0])

m = pd.read_csv(MAP)
name2vid = dict(zip(m["sly_name"], m["video_id"]))
ds = api.dataset.get_list(PID)[0]
imgs = api.image.get_list(ds.id)
idsA = [im.id for im in imgs if name2vid.get(im.name) in GROUP_A]
idsB = [im.id for im in imgs if name2vid.get(im.name) in GROUP_B]
print(f"grupo A={len(idsA)} grupo B={len(idsB)}")

jobA = api.labeling_job.create("GT_grupoA_rodrigo", ds.id, user_ids=[rodrigo], images_ids=idsA,
                               classes_to_label=CLASSES, description="GT - 10 videos grupo A")
print(f"Job A nuevo: {[j.id for j in jobA]} (300 imgs, rodrigo)")
jobB = api.labeling_job.create("GT_grupoB_PLACEHOLDER_reasignar", ds.id, user_ids=[rodrigo], images_ids=idsB,
                               classes_to_label=CLASSES, description="GT - 10 videos grupo B")
print(f"Job B nuevo: {[j.id for j in jobB]} (300 imgs, placeholder)")
print("LISTO - clases en jobs:", CLASSES)
