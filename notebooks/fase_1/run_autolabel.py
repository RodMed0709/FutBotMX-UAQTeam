"""Runner: auto-labela los 103 videos NO-testing (splits 0=reserva + 1=finetuning)
con SAM3 -> dataset YOLO en notebooks/fase_1/yolo_dataset. Desatendido + resumable.
"""
from pathlib import Path

import sam3_yolo as sy

REPO = Path("/workspace/FutBotMX-UAQTeam")
OUT = REPO / "notebooks" / "fase_1" / "yolo_dataset"

model, proc, device = sy.load_sam3(REPO / "assets" / "sam3")
videos = sy.videos_by_split(REPO / "assets" / "db_metadata.csv", [0, 1])  # no-testing
print(f"AUTOLABEL {len(videos)} videos -> {OUT}", flush=True)
res = sy.autolabel(
    videos, REPO, OUT, model, proc, device,
    frames_per_video=30, val_frac=0.2, img_ext=".jpg",
)
print("DONE", res, flush=True)
