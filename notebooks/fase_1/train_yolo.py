"""Entrena YOLO11 sobre el dataset auto-etiquetado (fase_1). Desatendido.

Modelo: yolo11s (el mas nuevo de Ultralytics, balance velocidad/precision).
Salida en notebooks/fase_1/runs/yolo11s_futbot. Al final reporta mAP.
"""
from pathlib import Path

from ultralytics import YOLO

FASE1 = Path("/workspace/FutBotMX-UAQTeam/notebooks/fase_1")
DATA = FASE1 / "yolo_dataset" / "data.yaml"

model = YOLO("yolo11s.pt")  # pretrained (auto-descarga)
model.train(
    data=str(DATA),
    epochs=100,
    imgsz=960,
    batch=16,
    patience=25,
    device=0,
    project=str(FASE1 / "runs"),
    name="yolo11s_futbot",
    exist_ok=True,
    verbose=True,
)
metrics = model.val()
print("VAL mAP50-95:", float(metrics.box.map), flush=True)
print("VAL mAP50   :", float(metrics.box.map50), flush=True)
print("pesos en:", FASE1 / "runs" / "yolo11s_futbot" / "weights" / "best.pt", flush=True)
print("TRAIN_DONE", flush=True)
