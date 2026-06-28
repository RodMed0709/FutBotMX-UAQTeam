import time
from pathlib import Path
from src.utils import PROJECT_ROOT
from src.core.track_overlay import render_obj_id_overlay
from src.core.inference import run_inference
ROOT = PROJECT_ROOT
OUT = ROOT/'notebooks/fase_7_visuales/outputs'
MG = Path('/workspace/Meta_Glasses/17Abril')
def log(*a): print(*a, flush=True)
FRESH = [
 ('video-848_singular_display', MG/'video-848_singular_display.mov', None),
 ('IMG_9871', MG/'Cámaras/IMG_9871.MOV', None),
 ('video-722_singular_display', MG/'video-722_singular_display.mov', 300),
]
for stem, raw, maxf in FRESH:
    if not Path(raw).exists(): log('[skip]', stem, 'no raw'); continue
    t0=time.time(); log(f'[C] inferencia {stem} (max_frames={maxf}) ...')
    try:
        res = run_inference(str(raw), mode='tracking', detector='yolo_sam3', tracker='bytetrack',
              include_masks=True, render_video=False, max_frames=maxf,
              run_label='visuales_fresh', progress=False)
        out = render_obj_id_overlay(res['json'], video_path=str(raw),
              output_path=OUT/f'overlay_{stem}.mp4', draw_masks=True, trajectory_window=30)
        log(f'   OK -> {out}  ({time.time()-t0:.0f}s)')
    except Exception as e:
        import traceback; traceback.print_exc(); log('   FAIL', stem, e)
log('=== FRESH DONE ===')
