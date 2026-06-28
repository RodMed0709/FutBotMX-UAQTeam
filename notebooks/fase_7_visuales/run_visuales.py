# -*- coding: utf-8 -*-
"""Genera overlays completos para ~12 videos diversos -> outputs/ junto al notebook.
Grupos: A cenital broadcast (reuse JSON) | B reuse overlay (mask+id+estela) | C fresh inference."""
import time, shutil
from pathlib import Path
from src.utils import PROJECT_ROOT
from src.core.track_overlay import render_obj_id_overlay
from src.core.event_broadcast_overlay import render_broadcast_overlay
from src.core.inference import run_inference

ROOT = PROJECT_ROOT
OUT = ROOT / 'notebooks/fase_7_visuales/outputs'
OUT.mkdir(parents=True, exist_ok=True)
MG = Path('/workspace/Meta_Glasses/17Abril')

def log(*a): print(*a, flush=True)

def overlay(stem, jpath, raw):
    j = Path(jpath) if str(jpath).startswith('/') else ROOT/jpath
    if not j.exists(): log('   [skip]', stem, 'no json', j); return
    if not Path(raw).exists(): log('   [skip]', stem, 'no raw', raw); return
    t0 = time.time(); log(f'[overlay] {stem} ...')
    try:
        out = render_obj_id_overlay(str(j), video_path=str(raw),
              output_path=OUT/f'overlay_{stem}.mp4', draw_masks=True, trajectory_window=30)
        log(f'   OK -> {out}  ({time.time()-t0:.0f}s)')
    except Exception as e:
        import traceback; traceback.print_exc(); log('   FAIL', stem, e)

# ---------------- GROUP A: cenital broadcast ----------------
CENITAL = ['IMG_9933_5m30','IMG_9933_8m00','IMG_9933_min1','IMG_9938_5m00','IMG_9938_min1']
for stem in CENITAL:
    j = ROOT/f'outputs/inference/fase5_clips/{stem}/{stem}.json'
    if not j.exists(): log('[A skip]', stem, 'no json'); continue
    t0 = time.time(); log(f'[A] broadcast {stem} ...')
    try:
        res = render_broadcast_overlay(str(j), layout=2, goal_source='strict',
              draw_field_on_video=True, max_frames=None, progress=False)
        dst = OUT/f'broadcast_{stem}.mp4'; shutil.copy(res.video, dst)
        log(f'   OK -> {dst}  ({time.time()-t0:.0f}s)')
    except Exception as e:
        import traceback; traceback.print_exc(); log('   FAIL', stem, e)

# ---------------- GROUP B: reuse overlay ----------------
TRK = 'outputs/inference/trackers/yolo_sam3+bytetrack'
REUSE = [
 ('video-597_singular_display', f'{TRK}/video-597_singular_display/video-597_singular_display.json', MG/'video-597_singular_display.mov'),
 ('video-714_singular_display', f'{TRK}/video-714_singular_display/video-714_singular_display.json', MG/'video-714_singular_display.mov'),
 ('video-836_singular_display', f'{TRK}/video-836_singular_display/video-836_singular_display.json', MG/'video-836_singular_display.mov'),
 ('IMG_9780', 'outputs/inference/fase3_eventos/IMG_9780/IMG_9780.json', MG/'Cámaras/IMG_9780.MOV'),
]
for stem, jp, raw in REUSE:
    overlay(stem, jp, raw)

# ---------------- GROUP C: fresh inference + overlay ----------------
FRESH = [
 ('video-848_singular_display', MG/'video-848_singular_display.mov', None),
 ('IMG_9871', MG/'Cámaras/IMG_9871.MOV', None),
 ('video-722_singular_display', MG/'video-722_singular_display.mov', 300),
]
for stem, raw, maxf in FRESH:
    if not Path(raw).exists(): log('[C skip]', stem, 'no raw', raw); continue
    t0 = time.time(); log(f'[C] inferencia {stem} (max_frames={maxf}) ...')
    try:
        res = run_inference(str(raw), mode='tracking', detector='yolo_sam3', tracker='bytetrack',
              include_masks=True, render_video=False, max_frames=maxf,
              run_label='visuales_fresh', progress=False)
        overlay(stem, res['json'], raw)
    except Exception as e:
        import traceback; traceback.print_exc(); log('   FAIL', stem, e)

log('=== DONE ===')
print('\n=== outputs ===')
for p in sorted(OUT.glob('*.mp4')):
    print('  ', p.name, f'{p.stat().st_size/1e6:.1f}MB')
