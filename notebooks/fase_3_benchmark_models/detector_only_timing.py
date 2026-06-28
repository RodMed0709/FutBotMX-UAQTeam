"""Timing de YOLO **solo cajas** (box-only, sin SAM3) — cierra la fila ausente de la Tabla
de detectores y reemplaza el "11.3 FPS" sin fuente.

Measurement-only: NO toca ``src/``. Reusa los building blocks del benchmark para que el número
sea **directamente comparable** a ``outputs/benchmark/detectors.csv`` (mismas 5 testing videos,
misma definición FPS = frames/wall-time, mismo idiom de VRAM pico que ``run_batch``):

- ``benchmark_videos(5, 42)``  -> los MISMOS 5 clips de testing del benchmark.
- ``iter_frames`` / ``get_frame_count`` -> streaming + denominador de FPS.
- ``detect_boxes`` (``src/core/detectors/yolo_boxes``) -> YOLO **puro**, sin SAM3 (es la primera
  mitad de ``yolo_sam3.detect``; aísla el costo del detector).
- ``torch.cuda.reset_peak_memory_stats`` + ``perf_counter`` + ``max_memory_allocated`` -> igual
  que ``src/core/batch.py``.

Solo cronometra la llamada a ``detect_boxes`` por frame (throughput puro del detector); excluye
warm-up (carga de modelo + init CUDA). Corre en el **pod** (necesita ``best.pt`` + GPU).

Uso (pod)
---------
    python notebooks/fase_3_benchmark_models/detector_only_timing.py \
        --n 5 --warmup 10 --out outputs/benchmark/detector_only.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.utils import PROJECT_ROOT


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def _reset_peak_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _read_peak_vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="nº de videos de testing (mismos que el benchmark)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=10, help="frames de warm-up excluidos del cronómetro")
    ap.add_argument("--max-frames", type=int, default=800,
                    help="tope de frames por video (FPS estable sin procesar videos largos completos; None=todos)")
    ap.add_argument("--conf", type=float, default=None, help="conf YOLO (None ⇒ el de la config)")
    ap.add_argument("--out", default="outputs/benchmark/detector_only.csv")
    args = ap.parse_args()

    from src.core.detectors.yolo_boxes import detect_boxes
    from src.core.frame_extraction import get_frame_count, iter_frames
    from src.core.segmentation import _load_classes
    from src.eval.benchmark import benchmark_videos

    classes = _load_classes()
    yolo_classes = [c for c in classes if "yolo_id" in c]
    names = [c["name"] for c in yolo_classes]
    print(f"clases YOLO (con yolo_id): {names}")

    videos = benchmark_videos(n=args.n, seed=args.seed)
    print(f"videos ({len(videos)}): {[Path(v).stem for v in videos]}\n")

    rows: list[dict] = []
    tot_frames = 0
    tot_time = 0.0
    peak_vram_all = 0.0

    for vi, vid in enumerate(videos):
        vpath = _abs(vid)
        nframes = get_frame_count(vpath)
        _reset_peak_vram()
        f_timed = 0
        t_acc = 0.0
        for fi, (_src_idx, frame) in enumerate(iter_frames(vpath, max_frames=args.max_frames)):
            t0 = time.perf_counter()
            detect_boxes(frame, classes=yolo_classes, conf=args.conf)
            dt = time.perf_counter() - t0
            if fi >= args.warmup:  # excluye warm-up
                t_acc += dt
                f_timed += 1
        vram = _read_peak_vram_mb()
        fps = f_timed / t_acc if t_acc > 0 else 0.0
        peak_vram_all = max(peak_vram_all, vram or 0.0)
        tot_frames += f_timed
        tot_time += t_acc
        rows.append({"clip": vpath.stem, "frames_timed": f_timed,
                     "elapsed_s": round(t_acc, 3), "fps": round(fps, 2),
                     "peak_vram_mb": round(vram, 1) if vram else None})
        print(f"  {vpath.stem:34} {f_timed:4} frames  {t_acc:7.2f}s  {fps:7.1f} FPS  "
              f"VRAM {vram and round(vram,1)} MB")

    agg_fps = tot_frames / tot_time if tot_time > 0 else 0.0

    out = _abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "frames_timed", "elapsed_s", "fps", "peak_vram_mb"])
        w.writeheader()
        w.writerows(rows)
        w.writerow({"clip": "ALL", "frames_timed": tot_frames, "elapsed_s": round(tot_time, 3),
                    "fps": round(agg_fps, 2), "peak_vram_mb": round(peak_vram_all, 1)})

    print(f"\n=== YOLO box-only (sin SAM3) — AGREGADO ===")
    print(f"  {tot_frames} frames cronometrados  |  {agg_fps:.1f} FPS  |  VRAM pico {peak_vram_all:.1f} MB")
    print(f"  vs yolo_sam3 pipeline (detectors.csv 1.71 FPS) → {agg_fps/1.71:.1f}× más rápido")
    print(f"  guardado: {out}")


if __name__ == "__main__":
    main()
