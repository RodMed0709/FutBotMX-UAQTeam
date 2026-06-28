# -*- coding: utf-8 -*-
"""TRACKING_EXPERIMENT -> Supervisely. 3 clips frame-a-frame con pre-anotaciones
(Bitmap por deteccion + tag obj_id). Project IMAGES + 1 labeling job por dataset.
Token por env SUPERVISELY_API_TOKEN. Corre en el pod.

FIX vs v1: (1) videos en /workspace/Meta_Glasses (NO REPO/); copia ASCII para evitar
acento 'Camaras' que rompia cv2. (2) job por dataset con dataset.id (no project.id).
"""
import os, json, shutil
from pathlib import Path
import cv2, numpy as np, supervisely as sly
from pycocotools import mask as maskutil

REPO = Path("/workspace/FutBotMX-UAQTeam")
PROJECT_NAME = "TRACKING_EXPERIMENT"
SRC = {  # rutas REALES (con acento) -> se copian a ASCII
    "IMG_9830": "/workspace/Meta_Glasses/17Abril/Cámaras/IMG_9830.MOV",
    "IMG_9812": "/workspace/Meta_Glasses/17Abril/Cámaras/IMG_9812.MOV",
    "IMG_9856": "/workspace/Meta_Glasses/17Abril/Cámaras/IMG_9856.MOV",
}
ASCII_DIR = REPO / "outputs/_te_src"; ASCII_DIR.mkdir(parents=True, exist_ok=True)
JSON = lambda s: REPO / f"outputs/inference/tracking_exp/{s}/{s}.json"
COLORS = {"robot": [255, 60, 60], "orange_ball": [255, 140, 0],
          "yellow_zone": [230, 230, 0], "blue_zone": [40, 90, 220]}
TMP = REPO / "outputs/_tracking_exp_frames"; TMP.mkdir(parents=True, exist_ok=True)

def rle_to_mask(rle):
    r = {"size": rle["size"], "counts": rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]}
    return maskutil.decode(r).astype(bool)

def main():
    api = sly.Api(os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com"),
                  os.environ["SUPERVISELY_API_TOKEN"])
    team = api.team.get_list()[0]
    wss = api.workspace.get_list(team.id); ws = wss[0] if wss else api.workspace.create(team.id, "futbot", "")
    objc = {n: sly.ObjClass(n, sly.Bitmap, color=c) for n, c in COLORS.items()}
    tag_obj = sly.TagMeta("obj_id", sly.TagValueType.ANY_NUMBER)
    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(list(objc.values())),
                           tag_metas=sly.TagMetaCollection([tag_obj]))
    project = api.project.create(ws.id, PROJECT_NAME, type=sly.ProjectType.IMAGES, change_name_if_conflict=True)
    api.project.update_meta(project.id, meta.to_json())
    print(f"project id={project.id} name={project.name!r}", flush=True)
    uid = api.user.get_team_members(team.id)[0].id
    for stem, src in SRC.items():
        ascii_vid = ASCII_DIR / f"{stem}.MOV"
        if not ascii_vid.exists(): shutil.copy(src, ascii_vid)   # ASCII -> cv2 abre OK
        cap = cv2.VideoCapture(str(ascii_vid))
        if not cap.isOpened():
            print(f"FAIL open {stem}", flush=True); continue
        d = json.load(open(JSON(stem)))
        byframe = {f["frame_index"]: f for f in d.get("frames", [])}
        ds = api.dataset.create(project.id, stem, change_name_if_conflict=True)
        ids = []; idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok: break
            f = byframe.get(idx)
            if f is not None:
                h, w = bgr.shape[:2]; labels = []
                for cls, dets in f.get("detections", {}).items():
                    if cls not in objc: continue
                    for det in dets:
                        rle = det.get("rle")
                        if not rle: continue
                        m = rle_to_mask(rle)
                        if m.shape != (h, w) or not m.any(): continue
                        tag = sly.Tag(tag_obj, value=int(det.get("obj_id", -1)))
                        labels.append(sly.Label(sly.Bitmap(m), objc[cls], tags=sly.TagCollection([tag])))
                png = TMP / f"{stem}_{idx:05d}.png"; cv2.imwrite(str(png), bgr)
                info = api.image.upload_path(ds.id, name=png.name, path=str(png))
                api.annotation.upload_ann(info.id, sly.Annotation(img_size=(h, w), labels=labels))
                ids.append(info.id)
                if len(ids) % 50 == 0: print(f"  {stem}: {len(ids)}", flush=True)
            idx += 1
        cap.release()
        api.labeling_job.create(f"TRACKING_EXPERIMENT_{stem}", ds.id, user_ids=[uid],
                                images_ids=ids, classes_to_label=list(objc.keys()))
        print(f"DONE {stem}: {len(ids)} frames + job", flush=True)
    print(f"ALL DONE. project id={project.id}", flush=True)

if __name__ == "__main__":
    main()
