"""Demos fase_2 sobre cámara superior (campo completo → portería azul visible).

Recorta un clip activo de IMG_9933 y renderiza yolo-only + yolo_sam3 con el
best.pt de 4 clases (incluye blue_zone).
"""
from pathlib import Path
import cv2, decord
import pipeline_yolo_sam3 as pl

decord.bridge.set_bridge("native")
REPO = Path("/workspace/FutBotMX-UAQTeam")
F2 = REPO / "notebooks" / "fase_2_YOLO_SAM3"
SAM = REPO / "assets" / "sam3"
SRC = REPO / "data/raw/18abril/Camara_superior/IMG_9933.MOV"
CLIP = F2 / "IMG_9933_superior_clip.mp4"
START, NF = 11000, 200

vr = decord.VideoReader(str(SRC))
fps = float(vr.get_avg_fps())
H, W = vr[0].shape[:2]
vw = cv2.VideoWriter(str(CLIP), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
for i in range(START, START + NF):
    vw.write(cv2.cvtColor(vr[i].asnumpy(), cv2.COLOR_RGB2BGR))
vw.release()
print(f"clip listo: {CLIP} ({NF}f @ {fps:.0f}fps)", flush=True)

models = pl.load_models(SAM, F2 / "best.pt")
print("render yolo-only superior...", flush=True)
pl.render(CLIP, F2 / "demo_yolo_only_superior.mp4", models, mode="yolo")
print("render yolo+sam3 superior...", flush=True)
pl.render(CLIP, F2 / "demo_yolo_sam3_superior.mp4", models, mode="yolo_sam3", green_every=5)
print("DEMOS_SUPERIOR_DONE", flush=True)
