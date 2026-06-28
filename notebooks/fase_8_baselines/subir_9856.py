# -*- coding: utf-8 -*-
"""Sube SOLO IMG_9856 al project EXISTENTE TRACKING_EXPERIMENT (378729) + su labeling job.
4 clases (robot/orange_ball/yellow_zone/blue_zone). Token por env. Corre en pod."""
import os, json, shutil
from pathlib import Path
import cv2, supervisely as sly
from pycocotools import mask as maskutil

REPO = Path("/workspace/FutBotMX-UAQTeam")
PID = 378729
STEM = "IMG_9856"
SRC = "/workspace/Meta_Glasses/17Abril/Cámaras/IMG_9856.MOV"
ASCII_VID = REPO / "outputs/_te_src/IMG_9856.MOV"
JSONP = REPO / f"outputs/inference/tracking_exp/{STEM}/{STEM}.json"
COLORS = {"robot": [255, 60, 60], "orange_ball": [255, 140, 0],
          "yellow_zone": [230, 230, 0], "blue_zone": [40, 90, 220]}
TMP = REPO / "outputs/_tracking_exp_frames"; TMP.mkdir(parents=True, exist_ok=True)

def rle_to_mask(rle):
    r = {"size": rle["size"], "counts": rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]}
    return maskutil.decode(r).astype(bool)

def main():
    api = sly.Api("https://app.supervisely.com", os.environ["SUPERVISELY_API_TOKEN"])
    team = api.team.get_list()[0]
    objc = {n: sly.ObjClass(n, sly.Bitmap, color=c) for n, c in COLORS.items()}
    tag_obj = sly.TagMeta("obj_id", sly.TagValueType.ANY_NUMBER)
    # quitar dataset IMG_9856 parcial si quedo
    for ds0 in api.dataset.get_list(PID):
        if ds0.name.startswith(STEM):
            api.dataset.remove(ds0.id); print(f"removido parcial {ds0.name}", flush=True)
    if not ASCII_VID.exists(): shutil.copy(SRC, ASCII_VID)
    cap = cv2.VideoCapture(str(ASCII_VID))
    assert cap.isOpened(), "no abre video"
    byframe = {f["frame_index"]: f for f in json.load(open(JSONP))["frames"]}
    ds = api.dataset.create(PID, STEM, change_name_if_conflict=True)
    uid = api.user.get_team_members(team.id)[0].id
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
            png = TMP / f"{STEM}_{idx:05d}.png"; cv2.imwrite(str(png), bgr)
            info = api.image.upload_path(ds.id, name=png.name, path=str(png))
            api.annotation.upload_ann(info.id, sly.Annotation(img_size=(h, w), labels=labels))
            ids.append(info.id)
            if len(ids) % 50 == 0: print(f"  {len(ids)}", flush=True)
        idx += 1
    cap.release()
    api.labeling_job.create(f"TRACKING_EXPERIMENT_{STEM}", ds.id, user_ids=[uid],
                            images_ids=ids, classes_to_label=list(objc.keys()))
    print(f"DONE {STEM}: {len(ids)} frames + job", flush=True)

if __name__ == "__main__":
    main()
