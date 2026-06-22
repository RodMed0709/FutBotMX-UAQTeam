# -*- coding: utf-8 -*-
"""Fase 8 / E3 — YOLO11 val por clase (mAP50 + mAP50:95) vs labels del val split (teacher).

Desglose por clase pedido por el review: muestra si blue_zone arrastra (fidelidad al teacher,
los labels del val son pseudo-labels de SAM3). Salida: outputs/benchmark/yolo_val_perclass.json

Uso (pod, GPU):  python yolo_val_perclass.py
"""
import json
from ultralytics import YOLO

REPO = "/workspace/FutBotMX-UAQTeam"
m = YOLO(REPO + "/assets/yolo/best.pt")
r = m.val(data=REPO + "/notebooks/fase_1/yolo_dataset/data.yaml", split="val", imgsz=960, verbose=True)
names = m.names
out = {"mAP50_all": round(float(r.box.map50), 4), "mAP50_95_all": round(float(r.box.map), 4),
       "note": "val labels = SAM3 pseudo-labels (teacher fidelity); green_floor no es clase YOLO",
       "per_class": {}}
for i, ci in enumerate(r.box.ap_class_index):
    out["per_class"][names[int(ci)]] = {"mAP50": round(float(r.box.ap50[i]), 4),
                                         "mAP50_95": round(float(r.box.ap[i]), 4)}
open(REPO + "/outputs/benchmark/yolo_val_perclass.json", "w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
