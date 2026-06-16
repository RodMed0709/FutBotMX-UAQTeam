"""T5 (fase_5 · Capa B) — mapa de calor de ocupación en cm.

Acumula las **posiciones en cm** de T3 (`metric_positions`) en una rejilla de la cancha
canónica (`field_template`, 243×182 cm) y produce un **heatmap** del balón y de los robots
(agregado) superpuesto sobre el campo. Refuerza la convocatoria 3.5.2 (visualización de
ocupación / "flujo de juego"). Solo cámara superior (donde T3 es fiable); en píxeles no es útil
porque las cámaras se mueven. Corre en **CPU local** (numpy + cv2; sin matplotlib, sin GPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core import field_template as ft
from src.core.events_core import BALL_CLASSES
from src.core.metric_positions import MetricResult, compute_metric_positions

DEFAULT_BIN_CM = 5.0
DEFAULT_SIGMA_CM = 10.0
CATS = ("ball", "robot")


@dataclass
class HeatmapResult:
    grids: dict[str, np.ndarray]  # cat -> rejilla de conteos (rows=y, cols=x)
    bin_cm: float
    resumen: dict


def _category(cls: str) -> str | None:
    if cls in BALL_CLASSES:
        return "ball"
    if cls == "robot":
        return "robot"
    return None


def _histogram(result: MetricResult, bin_cm: float) -> dict[str, np.ndarray]:
    """Histograma 2D de posiciones (clip a la cancha) por categoría."""
    cols = int(np.ceil(ft.LENGTH_CM / bin_cm))
    rows = int(np.ceil(ft.WIDTH_CM / bin_cm))
    grids = {c: np.zeros((rows, cols), dtype=float) for c in CATS}
    for p in result.posiciones:
        if p.xy_cm is None:
            continue
        cat = _category(p.cls)
        if cat is None:
            continue
        x = min(max(p.xy_cm[0], 0.0), ft.LENGTH_CM - 1e-6)
        y = min(max(p.xy_cm[1], 0.0), ft.WIDTH_CM - 1e-6)
        grids[cat][int(y / bin_cm), int(x / bin_cm)] += 1.0
    return grids


def _smooth_normalize(grid: np.ndarray, sigma_cm: float, bin_cm: float) -> np.ndarray:
    """Suaviza (gaussiano, cv2) y normaliza a 0..1 por el máximo."""
    import cv2

    if grid.max() <= 0:
        return grid
    sigma_cells = max(sigma_cm / bin_cm, 1e-6)
    k = int(2 * np.ceil(3 * sigma_cells) + 1)  # kernel impar
    sm = cv2.GaussianBlur(grid, (k, k), sigma_cells)
    return sm / sm.max() if sm.max() > 0 else sm


def _peak_cm(grid: np.ndarray, bin_cm: float) -> tuple[float, float] | None:
    """Centro (cm) de la celda más visitada, o ``None`` si está vacía."""
    if grid.max() <= 0:
        return None
    r, c = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return ((c + 0.5) * bin_cm, (r + 0.5) * bin_cm)


def compute_heatmaps(
    source: str | Path | MetricResult,
    *,
    bin_cm: float = DEFAULT_BIN_CM,
    sigma_cm: float = DEFAULT_SIGMA_CM,
) -> HeatmapResult:
    """Heatmaps (conteos) de balón y robots. ``source`` = ruta tracks_json (llama a T3) o
    un ``MetricResult``."""
    result = source if isinstance(source, MetricResult) else compute_metric_positions(Path(source))
    grids = _histogram(result, bin_cm)
    total_celdas = grids["ball"].size
    por_cat = {}
    for cat, g in grids.items():
        por_cat[cat] = {
            "n_muestras": int(g.sum()),
            "celda_pico_cm": _peak_cm(g, bin_cm),
            "pct_ocupacion": round(100.0 * int((g > 0).sum()) / total_celdas, 1),
        }
    resumen = {
        "fps": result.resumen.get("fps"),
        "bin_cm": bin_cm,
        "sigma_cm": sigma_cm,
        "rejilla": list(grids["ball"].shape),  # [rows, cols]
        "por_categoria": por_cat,
        "nota": "ocupación en cm (cámara superior); mapa indicativo (tracking/H)",
    }
    return HeatmapResult(grids=grids, bin_cm=bin_cm, resumen=resumen)


def render_heatmap(
    grid: np.ndarray,
    bin_cm: float,
    *,
    sigma_cm: float = DEFAULT_SIGMA_CM,
    scale: float = 2.6,
    margin_cm: float = 10.0,
    max_alpha: float = 0.8,
) -> np.ndarray:
    """Heatmap coloreado sobre la cancha canónica (imagen BGR)."""
    import cv2

    canvas, to_px = ft.render_field(scale=scale, margin_cm=margin_cm)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    norm = _smooth_normalize(grid, sigma_cm, bin_cm)

    x0, y0 = to_px((0.0, 0.0))
    x1, y1 = to_px((ft.LENGTH_CM, ft.WIDTH_CM))
    fw, fh = x1 - x0, y1 - y0
    dens = cv2.resize(norm, (fw, fh), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap((dens * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(dens, 0.0, max_alpha)[..., None]

    roi = canvas[y0:y1, x0:x1].astype(float)
    canvas[y0:y1, x0:x1] = (roi * (1 - alpha) + colored.astype(float) * alpha).astype(np.uint8)
    return canvas


def write_heatmap_png(
    grid: np.ndarray, bin_cm: float, path: str | Path, *, sigma_cm: float = DEFAULT_SIGMA_CM
) -> Path:
    """Renderiza y escribe el heatmap a PNG."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), render_heatmap(grid, bin_cm, sigma_cm=sigma_cm))
    return path
