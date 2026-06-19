"""Modelo metrico del campo Copa FutBotMX + render del minimap base (fase_4).

Geometria oficial (Reglas Copa FutBotMX 2026, seccion 7 y Figura 1), en
centimetros. Origen en la esquina superior izquierda de la **alfombra verde**
(la region que segmenta SAM3 como ``green_floor``, delimitada por las paredes
negras), con eje ``x`` a lo largo del **largo** (243 cm) y eje ``y`` a lo largo
del **ancho** (182 cm).

Este modulo es la **fuente de verdad** geometrica para la homografia
(``src.core.homography``) y para el dibujo del minimap (``src.core.minimap``):

- ``*_CM`` / ``*_ENDPOINTS`` / ``CARPET_CORNERS``: puntos-ancla del template en
  cm, usados como correspondencias mundo<-imagen.
- ``render_field``: dibuja la cancha 2D vista superior (cenital) sobre un lienzo
  RGB y devuelve la funcion ``world_to_px`` (cm -> pixel del minimap).

``cv2`` se importa de forma perezosa (estilo del repo).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# --- Dimensiones oficiales (cm) -------------------------------------------------
LENGTH_CM: float = 243.0  # largo total de la alfombra (eje x)
WIDTH_CM: float = 182.0   # ancho total de la alfombra (eje y)

LINE_BORDER_CM: float = 12.0  # inset de las lineas blancas respecto a las paredes
INNER_LENGTH_CM: float = LENGTH_CM - 2 * LINE_BORDER_CM  # 219
INNER_WIDTH_CM: float = WIDTH_CM - 2 * LINE_BORDER_CM     # 158

CENTER_CM: tuple[float, float] = (LENGTH_CM / 2.0, WIDTH_CM / 2.0)  # (121.5, 91)
CIRCLE_RADIUS_CM: float = 30.0  # diametro 60

GOAL_MOUTH_CM: float = 60.0  # ancho de la porteria / linea de gol
GOAL_LINE_X_LEFT_CM: float = LINE_BORDER_CM          # 12
GOAL_LINE_X_RIGHT_CM: float = LENGTH_CM - LINE_BORDER_CM  # 231

# Barra de color de cada porteria (entre la pared y la linea de gol).
YELLOW_GOAL_X_CM: float = 6.0
BLUE_GOAL_X_CM: float = LENGTH_CM - 6.0  # 237

# Borde superior/inferior de la boca de porteria (centrada en y=91).
_GOAL_TOP_Y_CM: float = CENTER_CM[1] - GOAL_MOUTH_CM / 2.0     # 61
_GOAL_BOTTOM_Y_CM: float = CENTER_CM[1] + GOAL_MOUTH_CM / 2.0  # 121

PENALTY_DEPTH_CM: float = 25.0
PENALTY_WIDTH_CM: float = 80.0
PENALTY_CORNER_R_CM: float = 14.0  # las esquinas internas del area chica son redondeadas (D)
_PEN_TOP_Y_CM: float = CENTER_CM[1] - PENALTY_WIDTH_CM / 2.0     # 51
_PEN_BOTTOM_Y_CM: float = CENTER_CM[1] + PENALTY_WIDTH_CM / 2.0  # 131


def _penalty_outline_cm(goal_x: float, inner_x: float, r: float = PENALTY_CORNER_R_CM, n: int = 10):
    """Contorno (cm) del area chica con las **dos esquinas internas redondeadas**.

    El area chica real no es un rectangulo de puntas: el borde que mira al centro
    tiene esquinas en arco (forma de D). ``goal_x`` es la linea de gol y ``inner_x``
    el borde interno; el signo decide hacia donde redondear.
    """
    s = 1.0 if inner_x > goal_x else -1.0  # direccion del borde interno
    cx = inner_x - s * r
    pts = [(goal_x, _PEN_TOP_Y_CM), (cx, _PEN_TOP_Y_CM)]
    for t in np.linspace(-np.pi / 2.0, 0.0, n):  # arco superior
        pts.append((cx + s * r * np.cos(t), (_PEN_TOP_Y_CM + r) + r * np.sin(t)))
    for t in np.linspace(0.0, np.pi / 2.0, n):    # arco inferior
        pts.append((cx + s * r * np.cos(t), (_PEN_BOTTOM_Y_CM - r) + r * np.sin(t)))
    pts.append((goal_x, _PEN_BOTTOM_Y_CM))
    return pts

# --- Puntos-ancla para la homografia (cm) --------------------------------------
# Endpoints (extremos) de la barra de color de cada porteria: dos puntos por
# porteria, no colineales con los de la otra -> 4 correspondencias bastan para H.
YELLOW_GOAL_ENDPOINTS: list[tuple[float, float]] = [
    (YELLOW_GOAL_X_CM, _GOAL_TOP_Y_CM),
    (YELLOW_GOAL_X_CM, _GOAL_BOTTOM_Y_CM),
]
BLUE_GOAL_ENDPOINTS: list[tuple[float, float]] = [
    (BLUE_GOAL_X_CM, _GOAL_TOP_Y_CM),
    (BLUE_GOAL_X_CM, _GOAL_BOTTOM_Y_CM),
]
YELLOW_GOAL_CENTER: tuple[float, float] = (YELLOW_GOAL_X_CM, CENTER_CM[1])
BLUE_GOAL_CENTER: tuple[float, float] = (BLUE_GOAL_X_CM, CENTER_CM[1])

# Esquinas de la alfombra (soporte extra para RANSAC si el campo se ve completo).
CARPET_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (LENGTH_CM, 0.0),
    (LENGTH_CM, WIDTH_CM),
    (0.0, WIDTH_CM),
]

# Colores RGB del minimap (consistentes con la cancha real).
_C_CARPET = (34, 139, 34)
_C_LINE = (245, 245, 245)
_C_WALL = (25, 25, 25)
_C_YELLOW = (255, 215, 0)
_C_BLUE = (30, 90, 220)


def render_field(
    scale: float = 2.2,
    margin_cm: float = 10.0,
) -> tuple[np.ndarray, Callable[[tuple[float, float]], tuple[int, int]]]:
    """Dibuja la cancha 2D (vista superior) y devuelve ``(canvas, world_to_px)``.

    Args:
        scale: pixeles por centimetro del minimap.
        margin_cm: margen (cm) alrededor de la alfombra, para que las paredes y
            las porterias no queden pegadas al borde del lienzo.

    Returns:
        Tupla ``(canvas, world_to_px)``:
        - ``canvas``: ``np.ndarray (H, W, 3) uint8`` RGB con la cancha dibujada.
        - ``world_to_px``: funcion ``(x_cm, y_cm) -> (px, py)`` (int) que mapea
          coordenadas del template a pixeles del minimap (incluye el margen).
    """
    import cv2

    def to_px(pt: tuple[float, float]) -> tuple[int, int]:
        x_cm, y_cm = pt
        return (
            int(round((x_cm + margin_cm) * scale)),
            int(round((y_cm + margin_cm) * scale)),
        )

    w_px = int(round((LENGTH_CM + 2 * margin_cm) * scale))
    h_px = int(round((WIDTH_CM + 2 * margin_cm) * scale))
    canvas = np.full((h_px, w_px, 3), _C_WALL, dtype=np.uint8)

    lw = max(1, int(round(2.0 * scale / 2.5)))  # grosor de linea ~ proporcional

    # Alfombra verde (toda la superficie hasta las paredes).
    cv2.rectangle(canvas, to_px((0, 0)), to_px((LENGTH_CM, WIDTH_CM)), _C_CARPET, -1)

    # Rectangulo interior de lineas blancas (219 x 158).
    cv2.rectangle(
        canvas,
        to_px((LINE_BORDER_CM, LINE_BORDER_CM)),
        to_px((LENGTH_CM - LINE_BORDER_CM, WIDTH_CM - LINE_BORDER_CM)),
        _C_LINE,
        lw,
    )

    # Linea central + circulo central.
    cv2.line(
        canvas,
        to_px((CENTER_CM[0], LINE_BORDER_CM)),
        to_px((CENTER_CM[0], WIDTH_CM - LINE_BORDER_CM)),
        _C_LINE,
        lw,
    )
    cv2.circle(canvas, to_px(CENTER_CM), int(round(CIRCLE_RADIUS_CM * scale)), _C_LINE, lw)
    cv2.circle(canvas, to_px(CENTER_CM), max(2, lw + 1), _C_LINE, -1)

    # Areas de penalti (forma de D: esquinas internas redondeadas).
    for goal_x, inner_x in (
        (GOAL_LINE_X_LEFT_CM, GOAL_LINE_X_LEFT_CM + PENALTY_DEPTH_CM),
        (GOAL_LINE_X_RIGHT_CM, GOAL_LINE_X_RIGHT_CM - PENALTY_DEPTH_CM),
    ):
        outline = np.array([to_px(p) for p in _penalty_outline_cm(goal_x, inner_x)], dtype=np.int32)
        cv2.polylines(canvas, [outline], False, _C_LINE, lw, cv2.LINE_AA)

    # Porterias de color (amarilla a la izquierda, azul a la derecha).
    cv2.line(
        canvas,
        to_px((YELLOW_GOAL_X_CM, _GOAL_TOP_Y_CM)),
        to_px((YELLOW_GOAL_X_CM, _GOAL_BOTTOM_Y_CM)),
        _C_YELLOW,
        max(2, lw + 2),
    )
    cv2.line(
        canvas,
        to_px((BLUE_GOAL_X_CM, _GOAL_TOP_Y_CM)),
        to_px((BLUE_GOAL_X_CM, _GOAL_BOTTOM_Y_CM)),
        _C_BLUE,
        max(2, lw + 2),
    )

    return canvas, to_px
