import shutil, time
from pathlib import Path
from src.utils import PROJECT_ROOT
from src.core.event_broadcast_overlay import render_broadcast_overlay
ROOT = PROJECT_ROOT
OUT = ROOT/'notebooks/fase_7_visuales/outputs'
stem = 'IMG_9933_5m30'
j = ROOT/f'outputs/inference/fase5_clips/{stem}/{stem}.json'
t0=time.time(); print('[v2] broadcast+boxes', stem, flush=True)
res = render_broadcast_overlay(str(j), layout=2, goal_source='strict',
      draw_field_on_video=True, draw_boxes=True, max_frames=None, progress=False)
dst = OUT/f'broadcast_{stem}_v2.mp4'; shutil.copy(res.video, dst)
print(f'OK -> {dst}  ({time.time()-t0:.0f}s)', flush=True)
