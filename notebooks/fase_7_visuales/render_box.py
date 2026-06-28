import shutil, time
from pathlib import Path
from src.utils import PROJECT_ROOT
from src.core.event_broadcast_overlay import render_broadcast_overlay

ROOT = PROJECT_ROOT
OUT = ROOT / 'notebooks/fase_7_visuales/outputs'
STEMS = ['IMG_9933_5m30', 'IMG_9938_min1']   # solo 2

for stem in STEMS:
    j = ROOT / f'outputs/inference/fase5_clips/{stem}/{stem}.json'
    dst = OUT / f'broadcast_box_{stem}.mp4'
    if dst.exists():
        print('skip exists', dst.name, flush=True); continue
    t0 = time.time(); print('[box] render', stem, flush=True)
    res = render_broadcast_overlay(str(j), layout=2, goal_source='strict',
          draw_field_on_video=True, draw_boxes=True, box_labels=False,
          max_frames=2300, progress=False)   # ~38s, suficiente para ventana de 35s
    shutil.copy(res.video, dst)
    print(f'  OK -> {dst.name}  ({time.time()-t0:.0f}s)', flush=True)
print('ALL DONE', flush=True)
