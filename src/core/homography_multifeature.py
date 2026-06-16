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

from src.core.auto_homography import _fit_line_robust, carpet_and_white


def inner_corners_extrapolated(
    white: np.ndarray,
    frac: float = 0.80,
    min_side_px: int = 15,
    max_oob_frac: float = 1.5,
):
    """4 esquinas del rectangulo interior, tolerando esquinas OCLUIDAS/cortadas.

    Igual que ``auto_homography.inner_corners`` (clasifica blancos en 4 lados con los
    ejes del ``minAreaRect``, ajusta una recta robusta por lado e intersecta lados
    adyacentes), PERO **no rechaza** una esquina por caer fuera del frame: si la
    esquina superior-derecha esta comida/recortada, las rectas del lado superior y
    derecho se ajustan con sus pixeles visibles y se **extrapolan** hasta cruzarse,
    recuperando la esquina "virtual". El mapa no se rompe porque la linea blanca
    sigue su recta aunque el vertice no se vea.

    Args:
        white: mascara binaria de lineas blancas (dentro de la alfombra).
        frac: umbral (fraccion del semieje) para clasificar un pixel como de borde.
        min_side_px: minimo de pixeles por lado para ajustar su recta.
        max_oob_frac: cuanto puede salirse una esquina del frame, como fraccion del
            tamano del frame (1.5 = hasta 1.5x fuera). Filtra intersecciones absurdas
            (lados casi paralelos) sin exigir que la esquina sea visible.

    Returns:
        ``np.ndarray (4,2) float32`` con winding consistente, o ``None`` si no se
        pueden ajustar los 4 lados.
    """
    import cv2

    ys, xs = np.where(white)
    if len(xs) < 4 * min_side_px:
        return None
    H, W = white.shape
    pts = np.column_stack([xs, ys]).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    center = box.mean(0)
    e0, e1 = box[1] - box[0], box[2] - box[1]
    n0, n1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    if n0 < 1 or n1 < 1:
        return None
    if n0 >= n1:
        u, half_u, v, half_v = e0 / n0, n0 / 2, e1 / n1, n1 / 2
    else:
        u, half_u, v, half_v = e1 / n1, n1 / 2, e0 / n0, n0 / 2
    du = (pts - center) @ u
    dv = (pts - center) @ v
    qu, qv = half_u * frac, half_v * frac
    sides = {"u-": pts[du < -qu], "u+": pts[du > qu],
             "v-": pts[dv < -qv], "v+": pts[dv > qv]}
    lines = {}
    for k, p in sides.items():
        if len(p) < min_side_px:
            return None
        lines[k] = _fit_line_robust(p)

    def inter(a, b):
        (p, d), (q, e) = lines[a], lines[b]
        A = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(np.linalg.det(A)) < 0.2:  # lados casi paralelos -> degenerado
            return None
        t = np.linalg.solve(A, q - p)
        return p + t[0] * d

    cs = [inter("u-", "v-"), inter("u+", "v-"), inter("u+", "v+"), inter("u-", "v+")]
    if any(c is None for c in cs):
        return None
    q = np.array(cs, np.float32)
    # Permitir esquinas FUERA del frame (ocluidas/cortadas), pero no absurdas.
    lo = np.array([-max_oob_frac * W, -max_oob_frac * H])
    hi = np.array([(1 + max_oob_frac) * W, (1 + max_oob_frac) * H])
    if np.any(q < lo) or np.any(q > hi):
        return None
    c = q.mean(0)
    a = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    return q[np.argsort(a)]


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
