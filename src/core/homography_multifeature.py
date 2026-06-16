"""Solver de homografia multi-feature + detectores de landmarks (fase_4 v2).

Reutiliza las primitivas de ``auto_homography`` (mascara alfombra/blanco, esquinas
del rectangulo interior, ajuste robusto de rectas) y agrega:

- ``detect_center_circle``: centro del circulo central (cm template = (121.5, 91))
  por componente conexa "anillada" (no borde, no linea, no diminuta) + ``fitEllipse``.
  Sirve como **landmark held-out** para medir reproj/jitter sin circularidad, y como
  correspondencia extra para el solver.

El solver completo (RANSAC sobre N correspondencias) se construye encima en nb02/03.
"""

from __future__ import annotations

import numpy as np

from src.core.auto_homography import carpet_and_white


def detect_center_circle(
    img_bgr: np.ndarray,
    white: np.ndarray | None = None,
    min_axis_ratio: float = 0.22,
    min_area_frac: float = 0.0004,
    max_area_frac: float = 0.25,
) -> tuple[float, float] | None:
    """Detecta el centro del circulo central (px imagen) o ``None``.

    No usa la homografia: aisla la componente blanca mas "anillada" dentro de la
    alfombra. Rechaza:
    - componentes que tocan el borde del frame (rectangulo exterior, recortes),
    - lineas (elipse super-alargada: ``min(axes)/max(axes) < min_axis_ratio``),
    - blobs diminutos o gigantes (fuera de ``[min_area_frac, max_area_frac]`` del frame).

    Entre las candidatas elige la de mejor puntaje = redondez * calidad de anillo
    (hueca, no solida). Devuelve el centro de la elipse ajustada.

    Args:
        img_bgr: frame BGR.
        white: mascara blanca dentro de alfombra; si ``None``, se calcula con
            ``carpet_and_white``.
        min_axis_ratio: razon minima de ejes de la elipse (filtra lineas rectas).
        min_area_frac / max_area_frac: rango de area de la componente (frac del frame).

    Returns:
        ``(cx, cy)`` en pixeles, o ``None`` si no hay candidata plausible.
    """
    import cv2

    if white is None:
        out = carpet_and_white(img_bgr)
        white = out[0] if out else None
    if white is None:
        return None

    H, W = white.shape
    area_img = float(H * W)
    n, lab, stats, _cent = cv2.connectedComponentsWithStats((white > 0).astype(np.uint8), 8)

    best = None
    best_score = -1.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area_frac * area_img or area > max_area_frac * area_img:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:  # toca borde
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 15:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        try:
            (ex, ey), (axa, axb), _ang = cv2.fitEllipse(pts)
        except cv2.error:
            continue
        major, minor = max(axa, axb), min(axa, axb)
        if major < 1:
            continue
        ratio = minor / major
        if ratio < min_axis_ratio:  # linea recta
            continue
        # "anillo": hueco -> fill (area/bbox) bajo-medio; solido -> alto.
        fill = area / (w * h + 1e-6)
        ring = 1.0 - min(1.0, abs(fill - 0.30) / 0.30)  # 1 en fill~0.30, cae lejos
        score = ratio * (0.4 + 0.6 * ring)
        if score > best_score:
            best_score = score
            best = (float(ex), float(ey))
    return best
