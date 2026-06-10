"""Genera los 2 videos demo sobre un video de TESTING (no visto): uno solo-YOLO
(cajas) y uno YOLO+SAM3 (cajas->máscaras + green_floor). Corre en GPU del pod.
"""
from pathlib import Path

import pipeline_yolo_sam3 as pl

REPO = Path("/workspace/FutBotMX-UAQTeam")
F2 = REPO / "notebooks" / "fase_2_YOLO_SAM3"
SAM = REPO / "assets" / "sam3"
YOLO_PT = F2 / "best.pt"
VIDEO = REPO / "data/raw/17Abril/Cámaras/IMG_9871.MOV"  # testing, jamás visto

print("cargando modelos...", flush=True)
models = pl.load_models(SAM, YOLO_PT)

print("render YOLO-only...", flush=True)
pl.render(VIDEO, F2 / "demo_yolo_only.mp4", models, mode="yolo")

print("render YOLO+SAM3...", flush=True)
pl.render(VIDEO, F2 / "demo_yolo_sam3.mp4", models, mode="yolo_sam3", green_every=5)

print("DEMOS_DONE", flush=True)
