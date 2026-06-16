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


def _remove_long_lines(white: np.ndarray, min_len_frac: float = 0.16, thick: int = 9) -> np.ndarray:
    """Quita los segmentos rectos LARGOS del blob blanco, deja las curvas (circulo).

    Las marcas del campo (rectangulo, linea central, areas, lineas de gol) son rectas
    largas; el circulo central solo aporta cuerdas cortas. Detecta segmentos con
    ``HoughLinesP`` y los borra (dibujados gruesos) -> el residual es ~el circulo.
    """
    import cv2

    H, W = white.shape
    min_len = int(min(H, W) * min_len_frac)
    lines = cv2.HoughLinesP(white, 1, np.pi / 180.0, threshold=60,
                            minLineLength=min_len, maxLineGap=12)
    residual = white.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(residual, (int(x1), int(y1)), (int(x2), int(y2)), 0, thick)
    return residual


def detect_center_circle(
    img_bgr: np.ndarray,
    white: np.ndarray | None = None,
    min_axis_ratio: float = 0.20,
    min_major_frac: float = 0.04,
    max_major_frac: float = 0.6,
) -> tuple[float, float] | None:
    """Detecta el centro del circulo central (px imagen) o ``None``.

    Estrategia (no usa homografia): las lineas blancas del campo forman UN blob
    conexo, asi que aislar por componente falla. En cambio se **quitan las rectas
    largas** (``_remove_long_lines``) y sobre el residual (mayormente el circulo) se
    ajusta una elipse por componente, eligiendo la mas redonda y centrada.

    Args:
        img_bgr: frame BGR.
        white: mascara blanca dentro de alfombra; si ``None``, se calcula.
        min_axis_ratio: razon minima de ejes (filtra restos de recta).
        min_major_frac / max_major_frac: eje mayor de la elipse como fraccion de
            ``min(H, W)`` (tamano plausible del circulo proyectado).

    Returns:
        ``(cx, cy)`` en pixeles, o ``None``.
    """
    import cv2

    if white is None:
        out = carpet_and_white(img_bgr)
        white = out[0] if out else None
    if white is None:
        return None

    H, W = white.shape
    wy, wx = np.where(white > 0)
    if len(wx) < 50:
        return None
    white_centroid = np.array([wx.mean(), wy.mean()])
    short = min(H, W)
    diag = float(np.hypot(H, W))

    residual = _remove_long_lines(white)
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _cent = cv2.connectedComponentsWithStats((residual > 0).astype(np.uint8), 8)

    best = None
    best_score = -1.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 30:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 12:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        try:
            (ex, ey), (axa, axb), _ang = cv2.fitEllipse(pts)
        except cv2.error:
            continue
        major, minor = max(axa, axb), min(axa, axb)
        if major < min_major_frac * short or major > max_major_frac * short:
            continue
        ratio = minor / major
        if ratio < min_axis_ratio:
            continue
        d = float(np.linalg.norm(np.array([ex, ey]) - white_centroid)) / diag
        near = float(np.exp(-(d / 0.20) ** 2))
        score = (0.4 + 0.6 * ratio) * near
        if score > best_score:
            best_score = score
            best = (float(ex), float(ey))
    return best
