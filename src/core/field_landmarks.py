"""Landmarks nombrados del campo Copa FutBotMX, en cm (fase_4 v2).

Capa sobre ``field_template.py``: expone el conjunto de **puntos y lineas
distinguibles** del campo como correspondencias mundo (cm) etiquetadas, para el
solver de homografia multi-feature y para las metricas de consistencia.

Sistema de coordenadas identico a ``field_template``: origen en la esquina de la
alfombra, ``x`` a lo largo del largo (243), ``y`` a lo largo del ancho (182).

- ``LANDMARK_POINTS``: dict ``nombre -> (x, y)`` de puntos esquina/interseccion
  inequivocos (sirven directo como correspondencias para ``cv2.findHomography``).
- ``LANDMARK_LINES``: dict ``nombre -> ((x1, y1), (x2, y2))`` de segmentos de
  linea blanca (para correspondencias linea-a-linea / muestreo).
- ``CENTER_CIRCLE``: ``(cx, cy, r)`` del circulo central (restriccion conica).
- ``static_world_points``: puntos del campo usados como referencia estatica para
  medir el jitter temporal.
"""

from __future__ import annotations

import numpy as np

from src.core import field_template as ft

# --- atajos a las constantes oficiales -----------------------------------------
_L = ft.LINE_BORDER_CM                     # 12
_RX = ft.LENGTH_CM - ft.LINE_BORDER_CM     # 231  (borde derecho de lineas)
_TY = ft.LINE_BORDER_CM                    # 12   (borde superior de lineas)
_BY = ft.WIDTH_CM - ft.LINE_BORDER_CM      # 170  (borde inferior de lineas)
_CX, _CY = ft.CENTER_CM                    # 121.5, 91
_R = ft.CIRCLE_RADIUS_CM                   # 30
_GT, _GB = ft._GOAL_TOP_Y_CM, ft._GOAL_BOTTOM_Y_CM    # 61, 121 (boca porteria)
_PT, _PB = ft._PEN_TOP_Y_CM, ft._PEN_BOTTOM_Y_CM      # 51, 131 (area chica)
_GLX_L, _GLX_R = ft.GOAL_LINE_X_LEFT_CM, ft.GOAL_LINE_X_RIGHT_CM  # 12, 231

# --- puntos-ancla nombrados (cm) -----------------------------------------------
LANDMARK_POINTS: dict[str, tuple[float, float]] = {
    # esquinas del rectangulo interior de lineas (219 x 158)
    "inner_tl": (_L, _TY),
    "inner_tr": (_RX, _TY),
    "inner_br": (_RX, _BY),
    "inner_bl": (_L, _BY),
    # linea central (interseccion con bordes y centro)
    "center_top": (_CX, _TY),
    "center_bot": (_CX, _BY),
    "center": (_CX, _CY),
    # circulo central (interseccion con linea central y ejes)
    "circle_top": (_CX, _CY - _R),
    "circle_bot": (_CX, _CY + _R),
    "circle_left": (_CX - _R, _CY),
    "circle_right": (_CX + _R, _CY),
    # esquinas del area chica sobre la linea de gol (puntas no redondeadas)
    "penL_top": (_GLX_L, _PT),
    "penL_bot": (_GLX_L, _PB),
    "penR_top": (_GLX_R, _PT),
    "penR_bot": (_GLX_R, _PB),
    # boca de porteria sobre la linea interior (extremos del segmento de 60 cm)
    "goalL_top": (_GLX_L, _GT),
    "goalL_bot": (_GLX_L, _GB),
    "goalR_top": (_GLX_R, _GT),
    "goalR_bot": (_GLX_R, _GB),
    # extremos de la barra de color de cada porteria (amarilla/azul)
    "postY_top": (ft.YELLOW_GOAL_X_CM, _GT),
    "postY_bot": (ft.YELLOW_GOAL_X_CM, _GB),
    "postB_top": (ft.BLUE_GOAL_X_CM, _GT),
    "postB_bot": (ft.BLUE_GOAL_X_CM, _GB),
}

# --- segmentos de linea blanca (cm) --------------------------------------------
LANDMARK_LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "border_top": ((_L, _TY), (_RX, _TY)),
    "border_bot": ((_L, _BY), (_RX, _BY)),
    "border_left": ((_L, _TY), (_L, _BY)),
    "border_right": ((_RX, _TY), (_RX, _BY)),
    "center_line": ((_CX, _TY), (_CX, _BY)),
    "penL_front": ((_GLX_L + ft.PENALTY_DEPTH_CM, _PT), (_GLX_L + ft.PENALTY_DEPTH_CM, _PB)),
    "penR_front": ((_GLX_R - ft.PENALTY_DEPTH_CM, _PT), (_GLX_R - ft.PENALTY_DEPTH_CM, _PB)),
}

# --- circulo central (restriccion conica) --------------------------------------
CENTER_CIRCLE: tuple[float, float, float] = (_CX, _CY, _R)


def points_array(names: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Devuelve ``(nombres, arreglo (N,2) cm)`` de los landmarks pedidos.

    Args:
        names: subconjunto de claves de ``LANDMARK_POINTS``; si ``None``, todos.

    Returns:
        Tupla ``(nombres, pts)`` con ``pts`` ``float32 (N, 2)`` en cm, alineado
        por indice con ``nombres``.
    """
    if names is None:
        names = list(LANDMARK_POINTS.keys())
    pts = np.array([LANDMARK_POINTS[n] for n in names], dtype=np.float32)
    return names, pts


def static_world_points() -> np.ndarray:
    """Puntos estaticos del campo (cm) para medir jitter temporal.

    Usa las 4 esquinas interiores + el centro: cubren el campo y son los mas
    estables. Devuelve ``float32 (5, 2)``.
    """
    names = ["inner_tl", "inner_tr", "inner_br", "inner_bl", "center"]
    return points_array(names)[1]


def draw_landmarks(canvas: np.ndarray, world_to_px, radius: int = 3) -> np.ndarray:
    """Dibuja los puntos-ancla sobre un lienzo (cm->px via ``world_to_px``).

    Util para inspeccion visual del template. Modifica ``canvas`` in-place y lo
    devuelve.
    """
    import cv2

    for name, pt in LANDMARK_POINTS.items():
        px = world_to_px(pt)
        cv2.circle(canvas, px, radius, (255, 0, 255), -1, cv2.LINE_AA)
    return canvas
