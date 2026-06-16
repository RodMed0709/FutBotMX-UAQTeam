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


def field_white_lines(img_bgr: np.ndarray, carpet_mask: np.ndarray,
                      close_ksize: int = 25, white_v_min: int = 140,
                      white_s_max: int = 90) -> np.ndarray:
    """Líneas blancas del campo a partir de la máscara verde de SAM3 (idea Rodrigo).

    SAM3 segmenta MUY bien la alfombra ``green_floor`` y **excluye** las líneas
    blancas: cada línea es un *hueco* (no-verde) **rodeado de verde** (la alfombra se
    extiende a ambos lados de la línea: borde a 12 cm de la pared, centro/áreas dentro).
    Entonces:

    1. ``filled`` = cierre morfológico de la máscara verde → tapa los huecos de línea.
    2. ``gaps`` = ``filled - verde`` → las líneas blancas (+ robots/objetos internos).
    3. se queda solo con lo **blanquecino** (alto valor, baja saturación) → descarta
       robots oscuros, deja las líneas.

    Devuelve máscara binaria ``uint8`` de líneas blancas, limpia y lista para ajustar
    rectas / medir overlap.
    """
    import cv2

    # SAM3 incluye las líneas DENTRO del green_floor (el piso es un objeto con sus
    # marcas), así que las líneas NO son huecos: se extraen por color (blanquecino)
    # dentro de la región del campo. Se cierra la máscara verde para abarcar la franja
    # de líneas y el borde, y se intersecta con píxeles de alta luminancia/baja sat.
    g = (carpet_mask > 0).astype(np.uint8) * 255
    field = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))
    field = cv2.dilate(field, np.ones((7, 7), np.uint8))   # alcanza la línea de borde
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    whiteish = cv2.inRange(hsv, (0, 0, white_v_min), (180, white_s_max, 255))
    white = cv2.bitwise_and(whiteish, field)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return white


def _template_perimeter_cm(step: float = 4.0) -> np.ndarray:
    """Puntos (cm) densos del rectángulo interior + línea central, para medir overlap."""
    from src.core import field_landmarks as fl
    tl = fl.LANDMARK_POINTS["inner_tl"]; tr = fl.LANDMARK_POINTS["inner_tr"]
    br = fl.LANDMARK_POINTS["inner_br"]; bl = fl.LANDMARK_POINTS["inner_bl"]
    ct = fl.LANDMARK_POINTS["center_top"]; cb = fl.LANDMARK_POINTS["center_bot"]

    def seg(a, b):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step))
        t = np.linspace(0, 1, n)[:, None]
        return np.array(a) * (1 - t) + np.array(b) * t

    return np.vstack([seg(tl, tr), seg(tr, br), seg(br, bl), seg(bl, tl), seg(ct, cb)])


_PERIM_CM = None


def registration_overlap(white: np.ndarray, H: np.ndarray, band: int = 7) -> float:
    """Overlap global del template proyectado (rectángulo+central) sobre la blanca."""
    import cv2
    global _PERIM_CM
    if _PERIM_CM is None:
        _PERIM_CM = _template_perimeter_cm()
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0.0
    pts = cv2.perspectiveTransform(_PERIM_CM.reshape(-1, 1, 2).astype(np.float64), Hinv).reshape(-1, 2)
    return line_overlap_score(white, pts, band=band)


def field_quad_from_white(white: np.ndarray):
    """4 esquinas del campo (perspectiva) por convex-hull del blanco → cuadrilátero.

    El blanco traza el perímetro del campo (las líneas de borde son el contorno
    externo; las áreas/central quedan dentro del hull). El hull aproximado a 4
    vértices da las 4 esquinas en perspectiva — robusto en vista lateral, donde el
    rectángulo es un trapecio (minAreaRect falla).

    Returns:
        ``np.ndarray (4,2) float32`` (winding por ángulo) o ``None``.
    """
    import cv2

    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.int32)
    if len(pts) < 8:
        return None
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    quad = None
    for f in (0.01, 0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, f * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return None
    c = quad.mean(0)
    a = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    return quad[np.argsort(a)]


def solve_lines_masks(img_bgr, carpet_mask, yc=None, bc=None):
    """Homografía imagen->cm por ajuste de líneas blancas, orientada y con overlap.

    1. ``field_white_lines`` (blanco dentro del verde de SAM3).
    2. ``inner_corners_extrapolated`` (4 esquinas, tolera oclusión/fuera de frame).
    3. prueba las 4 rotaciones del etiquetado contra el template, elige la **orientación
       válida** (amarillo a la izquierda x<centro, azul a la derecha) de **mayor overlap**.

    Returns:
        dict ``{H, corners, overlap, white, ok}``. ``ok=False`` si no se pudo ajustar.
    """
    import cv2
    from src.core import field_landmarks as fl

    from src.core import field_template as ft

    white = field_white_lines(img_bgr, carpet_mask)
    res = {"H": None, "corners": None, "overlap": 0.0, "white": white, "ok": False}

    inner = np.array([fl.LANDMARK_POINTS[n] for n in
                      ["inner_tl", "inner_tr", "inner_br", "inner_bl"]], np.float64)
    carpet = np.array(ft.CARPET_CORNERS, np.float64)

    # Tres fuentes de esquinas, cada una con su template destino:
    #  - esquinas interiores extrapoladas (cenital) -> rectángulo INTERIOR
    #  - cuadrilátero del blanco (perspectiva)       -> rectángulo INTERIOR
    #  - cuadrilátero del VERDE de SAM3 (blob limpio) -> esquinas de ALFOMBRA
    cand = []
    ic = inner_corners_extrapolated(white)
    if ic is not None:
        cand.append((ic, inner))
    wq = field_quad_from_white(white)
    if wq is not None:
        cand.append((wq, inner))
    gq = field_quad_from_white((carpet_mask > 0).astype(np.uint8) * 255)
    if gq is not None:
        cand.append((gq, carpet))
    if not cand:
        return res

    cx = fl.CENTER_CIRCLE[0]
    best = None
    for corners, tgt0 in cand:
        for r in range(4):
            tgt = np.roll(tgt0, -r, axis=0)
            H, _ = cv2.findHomography(corners.astype(np.float64), tgt, 0)
            if H is None:
                continue
            ok_orient = True
            if yc is not None:
                yx = cv2.perspectiveTransform(np.array([[yc]], np.float64), H)[0, 0, 0]
                ok_orient = ok_orient and (yx < cx)
            if bc is not None:
                bx = cv2.perspectiveTransform(np.array([[bc]], np.float64), H)[0, 0, 0]
                ok_orient = ok_orient and (bx > cx)
            if not ok_orient:
                continue
            ov = registration_overlap(white, H)
            if best is None or ov > best[2]:
                best = (H, corners, ov)
    if best is None:
        return res
    res.update({"H": best[0], "corners": best[1], "overlap": best[2], "ok": True})
    return res


class VideoHomographyLines:
    """Homografía por video: re-ajuste por líneas cada frame, EMA + gate de overlap.

    Resuelve la queja de Rodrigo (líneas chuecas que se quedan fijas): **cada frame**
    re-ajusta contra la línea blanca real (dinámico, sigue a la cámara), pero solo
    **acepta** el ajuste si su ``overlap`` con la blanca supera ``min_overlap`` (nunca
    fija algo torcido). Suaviza con EMA para que no tiemble. Si el frame ajusta mal,
    **conserva la última H buena** (no congela una mala).

    Args:
        min_overlap: overlap mínimo del template proyectado sobre la blanca para aceptar.
        smooth_beta: peso del histórico en la EMA (0=salta al nuevo, 1=ignora el nuevo).
            Bajo = más dinámico/responsivo a la cámara.
    """

    def __init__(self, min_overlap: float = 0.40, smooth_beta: float = 0.4):
        self.min_overlap = min_overlap
        self.smooth_beta = smooth_beta
        self.H: np.ndarray | None = None
        self.overlap = 0.0
        self.n_fit = 0
        self.n_kept = 0
        self.n_none = 0

    def update(self, img_bgr, carpet_mask, yc=None, bc=None):
        """Procesa un frame -> ``(H, status, overlap)``; status in {fit, kept, none}."""
        r = solve_lines_masks(img_bgr, carpet_mask, yc, bc)
        if r["ok"] and r["overlap"] >= self.min_overlap:
            Hn = r["H"] / r["H"][2, 2]
            if self.H is None:
                self.H = Hn
            else:
                self.H = (1 - self.smooth_beta) * Hn + self.smooth_beta * self.H
            self.overlap = r["overlap"]
            self.n_fit += 1
            return self.H, "fit", r["overlap"]
        if self.H is not None:
            self.n_kept += 1
            return self.H, "kept", r["overlap"]
        self.n_none += 1
        return None, "none", r["overlap"]

    def stats(self) -> dict:
        return {"fit": self.n_fit, "kept": self.n_kept, "none": self.n_none}


def line_overlap_score(white: np.ndarray, projected_pts: np.ndarray, band: int = 6) -> float:
    """Fracción de puntos proyectados (de una línea del template) que caen sobre blanco.

    Mide qué tan bien una línea proyectada **OVERLAPa** la línea blanca real (criterio
    de calidad pedido por Rodrigo). ``projected_pts`` ``(N,2)`` en px imagen; ``band``
    dilata la máscara blanca para tolerar grosor/ruido. 1.0 = encaje perfecto.
    """
    import cv2

    H, W = white.shape
    wd = cv2.dilate(white, np.ones((band, band), np.uint8))
    pts = np.round(projected_pts).astype(int)
    ok = 0
    tot = 0
    for x, y in pts:
        if 0 <= x < W and 0 <= y < H:
            tot += 1
            if wd[y, x] > 0:
                ok += 1
    return (ok / tot) if tot else 0.0


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
