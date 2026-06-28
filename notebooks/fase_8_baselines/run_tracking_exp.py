# -*- coding: utf-8 -*-
"""Fase 8 / TRACKING_EXPERIMENT — pre-anotaciones (YOLO+SAM3+ByteTrack, con mascaras).

Corre el pipeline completo en 3 clips frescos (reserve, no usados en 600/300) para generar
pre-anotaciones (tracks obj_id estables + mascaras por frame) que se suben a Supervisely;
el humano solo corrige (reframing). ByteTrack bridgea oclusiones cortas (lost_track_buffer).
Sin homografia.

Uso (pod, GPU):  CONFIG_FILENAME=01_yolo_sam3_config.json python run_tracking_exp.py
"""
import os, time, sys
os.environ.setdefault("CONFIG_FILENAME", "01_yolo_sam3_config.json")
sys.path.insert(0, "/workspace/FutBotMX-UAQTeam")
from src.core.tracking import track_video

CLIPS = [
    "/workspace/Meta_Glasses/17Abril/Cámaras/IMG_9830.MOV",
    "/workspace/Meta_Glasses/17Abril/Cámaras/IMG_9812.MOV",
    "/workspace/Meta_Glasses/17Abril/Cámaras/IMG_9856.MOV",
]
t0 = time.time()
for c in CLIPS:
    ts = time.time()
    print(f"[{time.time()-t0:6.1f}s] START {c}", flush=True)
    try:
        r = track_video(c, include_masks=True, render_video=True, run_label="tracking_exp")
        nt = len(r.get("tracks", [])) if isinstance(r, dict) else "?"
        print(f"[{time.time()-t0:6.1f}s] DONE {c} tracks={nt} ({time.time()-ts:.0f}s)", flush=True)
    except Exception as e:
        print(f"[{time.time()-t0:6.1f}s] FAIL {c}: {e}", flush=True)
print(f"[{time.time()-t0:6.1f}s] ALL DONE -> outputs/inference/.../tracking_exp", flush=True)
