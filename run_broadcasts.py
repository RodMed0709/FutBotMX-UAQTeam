import time
from pathlib import Path
from src.core.event_broadcast_overlay import render_broadcast_overlay

CLIPS = [
    'outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json',
    'outputs/inference/fase5_clips/IMG_9933_8m00/IMG_9933_8m00.json',
    'outputs/inference/fase5_clips/IMG_9933_min1/IMG_9933_min1.json',
    'outputs/inference/fase5_clips/IMG_9938_5m00/IMG_9938_5m00.json',
    'outputs/inference/fase5_clips/IMG_9938_min1/IMG_9938_min1.json',
]
for i, j in enumerate(CLIPS, 1):
    t0 = time.time()
    print(f'[{i}/{len(CLIPS)}] >>> {j}', flush=True)
    try:
        res = render_broadcast_overlay(j, layout=2, goal_source='strict', draw_field_on_video=True, max_frames=None, progress=False)
        dt = time.time()-t0
        print(f'    OK -> {res.video}  resumen={res.resumen}  ({dt:.0f}s)', flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'    FAIL {j}: {e}', flush=True)
print('=== DONE ===', flush=True)
