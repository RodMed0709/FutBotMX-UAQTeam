"""Homografia imagen->campo a partir de mascaras SAM3 (fase_4).

Estima, por frame, la homografia que lleva pixeles de la imagen a coordenadas
metricas del campo (``src.core.field_template``), usando como **anclas** las
mascaras SAM3 que el pipeline ya produce:

- ``green_floor`` (alfombra): su **cuadrilatero** (4 esquinas) es el ancla
  primaria, presente casi en todo frame. Se mapea a las esquinas de la alfombra.
- ``yellow_zone`` / ``blue_zone`` (porterias de color): sus **centroides**
  **orientan** el cuadrilatero (que extremo es el amarillo y cual el azul) y
  refinan ``H``. Resolver la orientacion es la parte normalmente dificil del
  registro de cancha; aqui la fija el color.

Orientacion sin ambiguedad: se prueban las 4 rotaciones del mapeo
esquinas-imagen -> esquinas-alfombra y se elige la que **minimiza el error de
reproyeccion** de los centroides de porteria (amarillo cerca de ``x`` chico, azul
cerca de ``x`` grande). Esto resuelve a la vez el giro y el "flip" vertical.

Robustez ante camara movil:
- ``cv2.findHomography(..., RANSAC)`` sobre todas las anclas descarta atipicos.
- **Suavizado temporal** (EMA sobre la matriz) mata el jitter de mascaras.
- **Propagacion**: si faltan anclas o ``H`` no valida, se reusa la ``H`` previa.

``cv2`` se importa de forma perezosa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import field_template as ft


@dataclass
class HomographyState:
    """Estado persistente del estimador a lo largo del video."""

    prev_H: np.ndarray | None = None
    n_estimated: int = 0
    n_propagated: int = 0


@dataclass
class FrameHomography:
    """Resultado de estimar ``H`` en un frame.

    Attributes:
        H: homografia imagen->cm ``(3, 3)`` o ``None`` si nunca hubo una valida.
        source: ``"anchors"`` (resuelta este frame) o ``"propagated"`` (reusada).
        n_anchors: numero de correspondencias usadas (0 si propagada).
    """

    H: np.ndarray | None
    source: str
    n_anchors: int = 0


def _mask_pixels(mask: np.ndarray) -> np.ndarray:
    """Pixeles activos de una mascara booleana como ``(N, 2)`` float32 ``(x, y)``."""
    ys, xs = np.where(mask)
    return np.column_stack([xs, ys]).astype(np.float32)


def mask_centroid(mask: np.ndarray | None) -> tuple[float, float] | None:
    """Centroide ``(cx, cy)`` (pixeles) de una mascara booleana, o ``None`` si vacia."""
    if mask is None:
        return None
    pts = _mask_pixels(mask)
    if len(pts) == 0:
        return None
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def goal_endpoints(mask: np.ndarray, min_pixels: int = 30) -> list[tuple[float, float]]:
    """Dos endpoints del eje mayor de la barra de porteria (orden: ``y`` ascendente).

    Soporte para el camino *solo-porterias* (sin campo). Usa ``minAreaRect`` y
    devuelve los puntos medios de los dos lados cortos (extremos del eje largo).
    """
    import cv2

    pts = _mask_pixels(mask)
    if len(pts) < min_pixels:
        return []
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    d01 = float(np.linalg.norm(box[0] - box[1]))
    d12 = float(np.linalg.norm(box[1] - box[2]))
    # Barra cuadrada (vista de frente): el eje mayor es ambiguo (puede girar 90°).
    # Sin un eje claro, no confiar en los endpoints (los usa solo el respaldo).
    d_long, d_short = max(d01, d12), min(d01, d12)
    if d_long <= 0 or d_short / d_long > 0.6:
        return []
    if d01 <= d12:
        m1, m2 = (box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0
    else:
        m1, m2 = (box[1] + box[2]) / 2.0, (box[3] + box[0]) / 2.0
    endpoints = sorted([m1, m2], key=lambda p: p[1])
    return [(float(p[0]), float(p[1])) for p in endpoints]


def field_quad(mask: np.ndarray, min_area: float = 2000.0) -> np.ndarray | None:
    """Cuadrilatero (4 esquinas) de la alfombra, **forzado a 4 puntos**.

    Toma el contorno externo de mayor area, su envolvente convexa, y reduce a 4
    vertices: prueba ``approxPolyDP`` con epsilon creciente y, si no logra 4, cae a
    ``minAreaRect`` (rectangulo rotado). Devuelve las 4 esquinas en orden
    **horario** (mismo sentido que las esquinas del template en coords imagen),
    o ``None`` si el area es insuficiente.
    """
    import cv2

    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None

    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for frac in (0.02, 0.04, 0.06, 0.08, 0.10, 0.14):
        approx = cv2.approxPolyDP(hull, frac * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)

    return _order_clockwise(quad)


def _order_clockwise(quad: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos en sentido horario (coords imagen, ``y`` hacia abajo)."""
    pts = np.asarray(quad, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def project_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Proyecta puntos ``(N, 2)`` de imagen a cm via ``H`` (perspectiva)."""
    import cv2

    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _goal_error(H: np.ndarray, yc, bc) -> float:
    """Error de reproyeccion (cm) de los centroides de porteria a su objetivo."""
    err, n = 0.0, 0
    for c, target in ((yc, ft.YELLOW_GOAL_CENTER), (bc, ft.BLUE_GOAL_CENTER)):
        if c is None:
            continue
        p = project_points(np.array([c], np.float32), H)[0]
        err += float(np.hypot(p[0] - target[0], p[1] - target[1]))
        n += 1
    return err / n if n else float("inf")


def _oriented_candidates(quad: np.ndarray):
    """Las (hasta) 4 homografias de mapear las esquinas-imagen a la alfombra.

    Returns:
        Lista de ``(H, carpet_assignment)`` para las 4 rotaciones del mapeo.
    """
    import cv2

    # Solo las 4 rotaciones (mismo sentido de giro): el cuadrilatero de imagen y el
    # template se ordenan ambos en sentido horario (``_order_clockwise``), y una
    # camara cenital no espeja la escena, asi que nunca hace falta una reflexion.
    # Anadir candidatos reflejados reintroduciria la ambiguedad de flip vertical.
    carpet = np.array(ft.CARPET_CORNERS, dtype=np.float32)
    out = []
    for r in range(4):
        assigned = np.roll(carpet, -r, axis=0)
        H, _ = cv2.findHomography(quad, assigned, 0)
        if _is_usable(H):
            out.append((H, assigned))
    return out


def _continuity_cm(H: np.ndarray, prev_H: np.ndarray | None, ref_pt: np.ndarray) -> float:
    """Distancia (cm) entre donde ``H`` y ``prev_H`` mapean un punto de referencia.

    Mide cuanto "salta" la proyeccion respecto al frame anterior; sirve para
    desempatar entre orientaciones equivalentes (p. ej. el flip vertical cuando
    solo se ve una porteria) y mantener el minimap estable. ``0`` si no hay previa.
    """
    if prev_H is None:
        return 0.0
    a = project_points(np.array([ref_pt], np.float32), H)[0]
    b = project_points(np.array([ref_pt], np.float32), prev_H)[0]
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _refine(quad, assigned, yc, bc, ransac_thresh):
    """RANSAC final sobre 4 esquinas + centroides de porteria (anclas conocidas)."""
    import cv2

    img = [tuple(p) for p in quad]
    tmpl = [tuple(p) for p in assigned]
    if yc is not None:
        img.append(yc)
        tmpl.append(ft.YELLOW_GOAL_CENTER)
    if bc is not None:
        img.append(bc)
        tmpl.append(ft.BLUE_GOAL_CENTER)
    H, _ = cv2.findHomography(
        np.array(img, np.float32), np.array(tmpl, np.float32), cv2.RANSAC, ransac_thresh
    )
    return (H if _is_usable(H) else None), len(img)


def _solve_goals_only(yellow_mask, blue_mask, ransac_thresh):
    """Camino de respaldo: H desde los 4 endpoints de porteria (sin campo)."""
    import cv2

    ye = goal_endpoints(yellow_mask) if yellow_mask is not None else []
    be = goal_endpoints(blue_mask) if blue_mask is not None else []
    if len(ye) != 2 or len(be) != 2:
        return None, 0
    img = np.array(ye + be, np.float32)
    tmpl = np.array(
        ft.YELLOW_GOAL_ENDPOINTS + ft.BLUE_GOAL_ENDPOINTS, np.float32
    )
    H, _ = cv2.findHomography(img, tmpl, cv2.RANSAC, ransac_thresh)
    return H, 4


def _is_usable(H: np.ndarray | None) -> bool:
    """``H`` no nula, finita y con ``H[2,2]`` no degenerado (normalizable)."""
    return (
        H is not None
        and np.all(np.isfinite(H))
        and abs(float(H[2, 2])) > 1e-8
    )


def _smooth_H(H_new: np.ndarray, H_prev: np.ndarray | None, beta: float) -> np.ndarray:
    """EMA sobre la matriz (normalizada a ``H[2,2]=1``) para reducir jitter.

    Precondicion: ``H_new`` ya validada con :func:`_is_usable`. Si ``H_prev`` no es
    usable, se devuelve solo la normalizacion de ``H_new``.
    """
    a = H_new / H_new[2, 2]
    if not _is_usable(H_prev) or beta <= 0.0:
        return a
    b = H_prev / H_prev[2, 2]
    return (1.0 - beta) * a + beta * b


def estimate_homography(
    field_mask: np.ndarray | None,
    yellow_mask: np.ndarray | None,
    blue_mask: np.ndarray | None,
    state: HomographyState,
    smooth_beta: float = 0.4,
    ransac_thresh: float = 12.0,
) -> FrameHomography:
    """Estima ``H`` (imagen->cm) para un frame y actualiza ``state`` in-place.

    Estrategia:
    1. Si hay cuadrilatero de campo y >=1 porteria: orienta por porterias y refina.
    2. Si no hay campo pero si las dos porterias: usa sus 4 endpoints.
    3. Si nada valida: propaga la ``H`` previa.

    Returns:
        ``FrameHomography`` con la ``H`` final (suavizada) y su procedencia.
    """
    yc = mask_centroid(yellow_mask)
    bc = mask_centroid(blue_mask)
    quad = field_quad(field_mask) if field_mask is not None else None
    cands = _oriented_candidates(quad) if quad is not None else []
    tol = ft.LENGTH_CM * 0.22  # ~53 cm: rechaza encajes flojos sin sobre-propagar

    H, n = None, 0
    # Punto de referencia de continuidad: una esquina del campo (fuera del eje de
    # simetria), para que un cambio de orientacion produzca un salto detectable.
    ref = quad[0] if quad is not None else None

    if cands and (yc is not None or bc is not None):
        # Orientacion: menor error de porteria, desempatado por continuidad temporal.
        scored = sorted(
            (
                (_goal_error(H0, yc, bc) + _continuity_cm(H0, state.prev_H, ref), _goal_error(H0, yc, bc), H0, asg)
                for H0, asg in cands
            ),
            key=lambda t: t[0],
        )
        _, err, H0, asg = scored[0]
        if err < tol:
            Href, n = _refine(quad, asg, yc, bc, ransac_thresh)
            # Quedarse con la mejor entre el refinamiento y el candidato orientado.
            if _is_usable(Href) and _goal_error(Href, yc, bc) <= err:
                H = Href
            else:
                H, n = H0, 4
    if H is None and cands and yc is None and bc is None and _is_usable(state.prev_H):
        # Sin porteria este frame pero con campo: mantener orientacion por continuidad.
        H = min(cands, key=lambda c: _continuity_cm(c[0], state.prev_H, ref))[0]
        n = 4
    if H is None:
        Hg, ng = _solve_goals_only(yellow_mask, blue_mask, ransac_thresh)
        if _is_usable(Hg) and _goal_error(Hg, yc, bc) < tol:
            H, n = Hg, ng

    if _is_usable(H):
        H = _smooth_H(H, state.prev_H, smooth_beta)
        if _is_usable(H):  # el suavizado no debe degradar la matriz
            state.prev_H = H
            state.n_estimated += 1
            return FrameHomography(H=H, source="anchors", n_anchors=int(n))

    state.n_propagated += 1
    return FrameHomography(H=state.prev_H, source="propagated", n_anchors=0)
