"""Harness de T5 `metric_heatmap` (fase_5 · Capa B). Corre en CPU local (sin GPU).

Genera el heatmap de ocupación (balón y robots) en cm sobre la cancha y valida invariantes +
casos borde + escribe los PNG.

    python testing/test_metric_heatmap.py [ruta/al/tracks.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

from src.core import field_template as ft
from src.core.metric_heatmap import compute_heatmaps, write_heatmap_png
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT
    / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _edge_cases() -> None:
    """Categoría sin muestras → grid de ceros; render no rompe."""
    from src.core.metric_heatmap import render_heatmap

    empty = np.zeros((36, 49), dtype=float)
    img = render_heatmap(empty, bin_cm=5.0)
    assert img.ndim == 3 and img.shape[2] == 3
    print("casos borde OK")


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    result = compute_heatmaps(tracks)
    print("resumen:\n" + json.dumps(result.resumen, indent=2, ensure_ascii=False))

    # --- invariantes ---
    for cat, g in result.grids.items():
        assert (g >= 0).all(), f"grid con valores negativos en {cat}"
    ball_peak = result.resumen["por_categoria"]["ball"]["celda_pico_cm"]
    if result.resumen["por_categoria"]["ball"]["n_muestras"] > 0:
        # el balón se concentró hacia la portería azul (x alto) en este clip de gol
        assert ball_peak[0] > ft.LENGTH_CM / 2.0, \
            f"el pico del balón no cae en la mitad azul: {ball_peak}"
    print("invariantes OK")

    _edge_cases()

    stem = tracks.stem
    for cat, g in result.grids.items():
        out = write_heatmap_png(
            g, result.bin_cm, PROJECT_ROOT / "outputs" / f"heatmap_{cat}_{stem}.png"
        )
        print(f"heatmap {cat}:", out)
    print("OK")


if __name__ == "__main__":
    main()
