import time
from pathlib import Path
from src.utils import PROJECT_ROOT
from src.core.track_overlay import render_obj_id_overlay
from src.core.inference import run_inference
ROOT = PROJECT_ROOT
OUT = ROOT/'notebooks/fase_7_visuales/outputs'
raw = Path('/workspace/Meta_Glasses/17Abril/video-722_singular_display.mov')
stem = 'video-722_singular_display'
t0=time.time(); print('[722] inferencia COMPLETA ...', flush=True)
res = run_inference(str(raw), mode='tracking', detector='yolo_sam3', tracker='bytetrack',
      include_masks=True, render_video=False, max_frames=None,
      run_label='visuales_fresh', progress=False)
out = render_obj_id_overlay(res['json'], video_path=str(raw),
      output_path=OUT/f'overlay_{stem}.mp4', draw_masks=True, trajectory_window=30)
print(f'OK -> {out}  ({time.time()-t0:.0f}s)', flush=True)
