"""Crea 2 Labeling Jobs en el project 378046, split por video (10/10).

Job A -> grupo A (videos primeros 10) asignado a rodrigo.
Job B -> grupo B (videos ultimos 10). Si se pasa --email de un miembro del team,
se asigna a esa persona; si no, se crea placeholder en rodrigo para reasignar luego.

Uso:
  python crear_jobs.py                      # A=rodrigo, B=placeholder(rodrigo)
  python crear_jobs.py --email otra@x.com   # A=rodrigo, B=esa persona (debe ser miembro)
"""
import argparse
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
CLASSES = ["robot", "orange_ball", "green_floor", "yellow_zone"]
GROUP_A = [0, 2, 3, 20, 24, 25, 37, 45, 46, 56]
GROUP_B = [69, 71, 78, 82, 84, 88, 89, 90, 94, 95]

ap = argparse.ArgumentParser()
ap.add_argument("--email", default=None)
args = ap.parse_args()

team = api.team.get_list()[0]
members = {u.login: u.id for u in api.user.get_team_members(team.id)}
rodrigo = members.get("rodrigomed07@gmail.com", list(members.values())[0])

m = pd.read_csv(MAP)
name2vid = dict(zip(m["sly_name"], m["video_id"]))

ds = api.dataset.get_list(PID)[0]
imgs = api.image.get_list(ds.id)
idsA = [im.id for im in imgs if name2vid.get(im.name) in GROUP_A]
idsB = [im.id for im in imgs if name2vid.get(im.name) in GROUP_B]
print(f"imgs grupo A={len(idsA)}  grupo B={len(idsB)}  (sin asignar: {len(imgs) - len(idsA) - len(idsB)})")

jobA = api.labeling_job.create(
    "GT_grupoA_rodrigo", ds.id, user_ids=[rodrigo], images_ids=idsA,
    classes_to_label=CLASSES, description="GT testing - 10 videos grupo A",
)
print(f"Job A creado: {[j.id for j in jobA]} -> rodrigo, {len(idsA)} imgs")

uidB = members.get(args.email) if args.email else None
if uidB:
    nameB, assignB = "GT_grupoB_persona2", uidB
else:
    nameB, assignB = "GT_grupoB_PLACEHOLDER_reasignar", rodrigo
jobB = api.labeling_job.create(
    nameB, ds.id, user_ids=[assignB], images_ids=idsB,
    classes_to_label=CLASSES, description="GT testing - 10 videos grupo B",
)
who = args.email if uidB else "PLACEHOLDER(rodrigo) -> reasignar a persona2"
print(f"Job B creado: {[j.id for j in jobB]} -> {who}, {len(idsB)} imgs")
print("\nLISTO. Jobs visibles en Supervisely -> Labeling Jobs.")
