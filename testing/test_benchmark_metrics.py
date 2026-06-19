"""Prueba manual de las métricas del benchmark (tarea benchmark_metrics).

LOCAL (sin GPU): ejercita ``video_metrics`` (trayectoria + máscara), ``aggregate_config``
y ``comparison_table``/``write_table`` con **JSON sintéticos** (RLE real fabricado con
``encode_rle``). No requiere SAM3 ni el manifiesto.

Uso:
    python testing/test_benchmark_metrics.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.inference_schema import encode_rle  # noqa: E402
from src.eval.benchmark import (  # noqa: E402
    aggregate_config,
    comparison_table,
    video_metrics,
    write_table,
)


def _block_mask(x: int, y: int, size: int = 10, hw: int = 100) -> np.ndarray:
    """Máscara booleana (hw, hw) con un bloque size×size en la esquina (x, y)."""
    m = np.zeros((hw, hw), dtype=bool)
    m[y : y + size, x : x + size] = True
    return m


def _tracking_doc() -> dict:
    """JSON tracking sintético: 2 tracks (misma clase) + frames con rle de obj_id 1."""
    rle = encode_rle(
        _block_mask(10, 10)
    )  # misma máscara en 3 frames -> IoU 1, jitter 0

    def det(obj_id):
        return {
            "obj_id": obj_id,
            "bbox": [10, 10, 10, 10],
            "centroid": [15.0, 15.0],
            "score": 0.9,
            "rle": rle,
        }

    frames = [{"frame_index": i, "detections": {"robot": [det(1)]}} for i in (0, 1, 2)]
    tracks = [
        {  # track recto (accel 0 -> smoothness 0); termina en frame 2
            "obj_id": 1,
            "class": "robot",
            "observations": [
                {
                    "frame_index": 0,
                    "bbox": [0, 0, 0, 0],
                    "centroid": [10.0, 10.0],
                    "score": 0.9,
                },
                {
                    "frame_index": 1,
                    "bbox": [0, 0, 0, 0],
                    "centroid": [20.0, 10.0],
                    "score": 0.9,
                },
                {
                    "frame_index": 2,
                    "bbox": [0, 0, 0, 0],
                    "centroid": [30.0, 10.0],
                    "score": 0.9,
                },
            ],
        },
        {  # arranca en frame 3 cerca del fin del track 1 -> fragmento
            "obj_id": 2,
            "class": "robot",
            "observations": [
                {
                    "frame_index": 3,
                    "bbox": [0, 0, 0, 0],
                    "centroid": [31.0, 10.0],
                    "score": 0.9,
                },
                {
                    "frame_index": 4,
                    "bbox": [0, 0, 0, 0],
                    "centroid": [40.0, 10.0],
                    "score": 0.9,
                },
            ],
        },
    ]
    return {
        "resolution": {"height": 100, "width": 100},
        "frames": frames,
        "tracks": tracks,
    }


def _segmentation_doc() -> dict:
    """JSON segmentación sintético: sin ``tracks`` y sin ``rle``."""
    frames = [
        {
            "frame_index": 0,
            "detections": {
                "robot": [
                    {
                        "obj_id": 0,
                        "bbox": [1, 1, 5, 5],
                        "centroid": [3.0, 3.0],
                        "score": 0.8,
                    }
                ]
            },
        }
    ]
    return {"resolution": {"height": 100, "width": 100}, "frames": frames}


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    # 1) Trayectoria + máscara sobre el doc tracking.
    m = video_metrics(_tracking_doc())
    assert m["tracklet_len"] == 2.5, f"tracklet_len {m['tracklet_len']}"  # (3+2)/2
    assert m["frag_rate"] == 0.5, f"frag_rate {m['frag_rate']}"  # 1 de 2 tracks
    assert abs(m["smoothness"]) < 1e-9, f"smoothness {m['smoothness']}"  # recto
    assert m["mask_iou"] == 1.0, f"mask_iou {m['mask_iou']}"  # máscara idéntica
    assert abs(m["com_jitter"]) < 1e-9, f"com_jitter {m['com_jitter']}"
    print("  [ok] trayectoria + máscara (doc tracking)")

    # 2) Segmentación: trayectoria y máscara en None.
    ms = video_metrics(_segmentation_doc())
    assert all(ms[k] is None for k in ("tracklet_len", "frag_rate", "smoothness"))
    assert ms["mask_iou"] is None and ms["com_jitter"] is None
    print("  [ok] segmentación -> trayectoria/máscara N/A")

    # 3) aggregate_config: 2 done + 1 skipped (ignorado), funde timing.
    with tempfile.TemporaryDirectory() as d:
        jp1 = Path(d) / "a.json"
        jp2 = Path(d) / "b.json"
        jp1.write_text(json.dumps(_tracking_doc()), encoding="utf-8")
        jp2.write_text(json.dumps(_tracking_doc()), encoding="utf-8")
        entries = [
            {"status": "done", "json": str(jp1), "fps": 10.0, "peak_vram_mb": 2000.0},
            {"status": "done", "json": str(jp2), "fps": 20.0, "peak_vram_mb": 2200.0},
            {"status": "skipped", "json": None, "fps": None, "peak_vram_mb": None},
        ]
        row = aggregate_config("yolo_sam3+botsort", entries)
        assert row["config"] == "yolo_sam3+botsort"
        assert row["fps"] == 15.0 and row["peak_vram_mb"] == 2100.0, "timing fundido"
        assert row["tracklet_len"] == 2.5 and row["mask_iou"] == 1.0
        print("  [ok] aggregate_config (done promediado, skipped ignorado)")

        # 4) Config sin tracking: trayectoria/máscara None en la fila.
        jp3 = Path(d) / "c.json"
        jp3.write_text(json.dumps(_segmentation_doc()), encoding="utf-8")
        seg_row = aggregate_config(
            "sam3_text+none",
            [{"status": "done", "json": str(jp3), "fps": 5.0, "peak_vram_mb": 1800.0}],
        )
        assert seg_row["fps"] == 5.0
        assert all(
            seg_row[k] is None for k in ("tracklet_len", "frag_rate", "mask_iou")
        )
        print("  [ok] config sin tracking -> calidad N/A en la fila")

        # 5) Tabla + CSV.
        df = comparison_table([row, seg_row])
        assert list(df.columns) == [
            "config",
            "fps",
            "peak_vram_mb",
            "tracklet_len",
            "frag_rate",
            "smoothness",
            "mask_iou",
            "com_jitter",
        ]
        out = write_table(df, Path(d) / "comparison.csv")
        assert out.exists()
        import pandas as pd

        reloaded = pd.read_csv(out)
        assert len(reloaded) == 2 and "config" in reloaded.columns
        print("  [ok] comparison_table + write_table (CSV releído)")

    print("\n== Resultado ==")
    print("  OK: las pruebas de benchmark_metrics pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
