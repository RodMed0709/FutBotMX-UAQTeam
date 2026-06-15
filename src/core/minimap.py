"""Minimap 2D con trayectorias proyectadas + composicion sobre el video (fase_4).

Acumula, por ``obj_id``, las posiciones proyectadas al campo (cm) y las dibuja como
**trails** sobre la cancha 2D (vista superior) de ``src.core.field_template``. Luego
**compone** ese minimap en la esquina superior derecha del frame del video.

Flujo de uso (por frame):

    renderer.update([(obj_id, class_name, x_cm, y_cm), ...])
    mini = renderer.render()
    frame_out = renderer.composite(frame, mini)

``cv2`` se importa de forma perezosa.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from src.core import field_template as ft

# Paleta para robots (cicla por obj_id). El balon tiene color propio.
_ROBOT_PALETTE = [
    (60, 130, 255),
    (255, 80, 80),
    (80, 220, 120),
    (200, 120, 255),
    (255, 170, 40),
    (40, 220, 220),
]
_BALL_COLOR = (255, 140, 0)
_BALL_CLASSES = {"orange_ball", "ball"}


def _point_color(obj_id: int, class_name: str) -> tuple[int, int, int]:
    """Color del objeto: balon fijo; robots por paleta segun ``obj_id``."""
    if class_name in _BALL_CLASSES:
        return _BALL_COLOR
    return _ROBOT_PALETTE[obj_id % len(_ROBOT_PALETTE)]


def draw_field_overlay(
    frame: np.ndarray,
    H: np.ndarray | None,
    corners: np.ndarray | None = None,
) -> np.ndarray:
    """Reproyecta la cancha sobre el frame del video para confirmar la homografia.

    Con ``H`` (img->cm) dibuja, vía ``H^-1``, el **rectangulo interior** (verde) y el
    **circulo central** (cian) en coordenadas de imagen; si se pasan ``corners`` (las 4
    esquinas detectadas en la imagen), las marca en azul. Replica ``draw_field_overlay``
    de la demo (camino C). Colores en convencion **RGB** (lo que entrega ``iter_frames``).
    Dibuja in place sobre ``frame`` y lo devuelve. ``cv2`` perezoso.
    """
    import cv2

    out = frame
    b = ft.LINE_BORDER_CM
    if H is not None:
        Hinv = np.linalg.inv(H)

        def c2i(p: tuple[float, float]) -> tuple[int, int]:
            q = cv2.perspectiveTransform(np.array([[p]], np.float32), Hinv).reshape(2)
            return int(q[0]), int(q[1])

        inner = [(b, b), (ft.LENGTH_CM - b, b),
                 (ft.LENGTH_CM - b, ft.WIDTH_CM - b), (b, ft.WIDTH_CM - b)]
        cv2.polylines(out, [np.array([c2i(p) for p in inner], np.int32)], True,
                      (0, 255, 0), 3, cv2.LINE_AA)
        circ = [c2i((ft.CENTER_CM[0] + ft.CIRCLE_RADIUS_CM * np.cos(t),
                     ft.CENTER_CM[1] + ft.CIRCLE_RADIUS_CM * np.sin(t)))
                for t in np.linspace(0, 2 * np.pi, 40)]
        cv2.polylines(out, [np.array(circ, np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
    if corners is not None:
        for p in np.asarray(corners).astype(int):
            cv2.circle(out, (int(p[0]), int(p[1])), 13, (255, 0, 0), -1, cv2.LINE_AA)
    return out


def orientation_k(field_center: tuple[float, float], goal_center: tuple[float, float]) -> int:
    """Cuantas rotaciones de 90deg (CCW, ``np.rot90``) alinear el minimap con la imagen.

    El minimap canonico tiene la porteria amarilla a la **izquierda**. Segun hacia
    donde quede la amarilla respecto al centro del campo en la imagen (arriba/abajo/
    izquierda/derecha), se rota el minimap para que coincida con la orientacion del
    campo en el video (campo vertical -> minimap vertical, etc.).
    """
    dx = goal_center[0] - field_center[0]
    dy = goal_center[1] - field_center[1]  # imagen: y hacia abajo
    ang = float(np.degrees(np.arctan2(dy, dx)))
    # amarilla a la: derecha(0)->k2, abajo(90)->k1, izquierda(180)->k0, arriba(-90)->k3
    targets = {0.0: 2, 90.0: 1, 180.0: 0, -90.0: 3}
    best = min(targets, key=lambda a: abs(((ang - a + 180.0) % 360.0) - 180.0))
    return targets[best]


class MinimapRenderer:
    """Dibuja la cancha 2D con trails acumulados y la compone sobre el video."""

    def __init__(
        self,
        scale: float = 2.6,
        margin_cm: float = 3.0,
        trail_len: int = 64,
        panel_width_frac: float = 0.34,
        trail_persist: int = 24,
    ) -> None:
        """Inicializa el renderer.

        Args:
            scale: pixeles por cm del minimap.
            margin_cm: margen alrededor de la alfombra.
            trail_len: cuantas posiciones recientes conserva cada trail.
            panel_width_frac: ancho del minimap como fraccion del ancho del frame
                al componerlo.
            trail_persist: frames sin actualizarse tras los que un trail se purga
                (evita dots congelados de objetos que ya salieron de cuadro).
        """
        self._base, self._to_px = ft.render_field(scale=scale, margin_cm=margin_cm)
        self._trail_len = trail_len
        self._panel_width_frac = panel_width_frac
        self._trail_persist = trail_persist
        self._trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
        self._class_of: dict[int, str] = {}
        self._last_seen: dict[int, int] = {}
        self._frame: int = 0
        self._rotate_k: int = 0
        self._oriented: bool = False

    def orient_once(
        self,
        field_center: tuple[float, float] | None,
        goal_center: tuple[float, float] | None,
    ) -> None:
        """Fija la orientacion del minimap (una sola vez) a partir de los centroides.

        Usa el centro del campo y el de la porteria amarilla en la **imagen** para
        rotar el minimap y que coincida con la orientacion del campo en el video.
        No-op si ya esta orientado o si falta algun centroide.
        """
        if self._oriented or field_center is None or goal_center is None:
            return
        self._rotate_k = orientation_k(field_center, goal_center)
        self._oriented = True

    @property
    def base(self) -> np.ndarray:
        """Lienzo base (cancha sin trails), copia segura."""
        return self._base.copy()

    def update(self, projected: list[tuple[int, str, float, float]]) -> None:
        """Agrega posiciones proyectadas (cm) de este frame a los trails.

        Descarta puntos que caen absurdamente fuera del campo (proyeccion mala),
        con una tolerancia de un campo extra alrededor.
        """
        self._frame += 1
        x_lim = (-ft.LENGTH_CM, 2 * ft.LENGTH_CM)
        y_lim = (-ft.WIDTH_CM, 2 * ft.WIDTH_CM)
        for obj_id, class_name, x_cm, y_cm in projected:
            if not (x_lim[0] <= x_cm <= x_lim[1] and y_lim[0] <= y_cm <= y_lim[1]):
                continue
            self._trails[obj_id].append((x_cm, y_cm))
            self._class_of[obj_id] = class_name
            self._last_seen[obj_id] = self._frame

        # Purga de trails no actualizados recientemente (objetos fuera de cuadro).
        stale = [
            oid for oid, seen in self._last_seen.items()
            if self._frame - seen > self._trail_persist
        ]
        for oid in stale:
            self._trails.pop(oid, None)
            self._class_of.pop(oid, None)
            self._last_seen.pop(oid, None)

    def render(self) -> np.ndarray:
        """Dibuja el minimap del estado actual (cancha + trails + marcadores).

        Replica el render de la demo (camino C): el **balon** es un circulo naranja;
        los **robots** son un **cuadro gris** (marcador uniforme). El **trail** sigue la
        paleta por ``obj_id`` (robots) / naranja (balon) para distinguir trayectorias.
        """
        import cv2

        canvas = self._base.copy()
        for obj_id, trail in self._trails.items():
            if not trail:
                continue
            class_name = self._class_of.get(obj_id, "robot")
            color = _point_color(obj_id, class_name)  # color del trail
            pts = [self._to_px(p) for p in trail]
            x, y = pts[-1]
            if len(pts) >= 2:
                cv2.polylines(
                    canvas, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA
                )
            if class_name in _BALL_CLASSES:
                cv2.circle(canvas, (x, y), 12, _BALL_COLOR, -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 12, (20, 20, 20), 2, cv2.LINE_AA)
            else:
                s = 14
                cv2.rectangle(canvas, (x - s, y - s), (x + s, y + s), (175, 175, 175), -1)
                cv2.rectangle(canvas, (x - s, y - s), (x + s, y + s), (35, 35, 35), 2)
        if self._rotate_k:
            canvas = np.ascontiguousarray(np.rot90(canvas, self._rotate_k))
        return canvas

    def composite(self, frame: np.ndarray, minimap: np.ndarray | None = None) -> np.ndarray:
        """Pega el minimap en la esquina superior derecha del ``frame`` (copia).

        Args:
            frame: frame RGB ``(H, W, 3) uint8`` del video (o su overlay).
            minimap: minimap a componer; si es ``None`` se renderiza el actual.

        Returns:
            Nuevo frame ``(H, W, 3) uint8`` con el minimap superpuesto y un marco.
        """
        import cv2

        mini = minimap if minimap is not None else self.render()
        out = frame.copy()
        fh, fw = out.shape[:2]

        target_w = max(1, int(round(fw * self._panel_width_frac)))
        mh, mw = mini.shape[:2]
        target_h = max(1, int(round(mh * target_w / mw)))
        mini_rs = cv2.resize(mini, (target_w, target_h), interpolation=cv2.INTER_AREA)

        pad = max(6, int(round(fw * 0.01)))
        x0 = fw - target_w - pad
        y0 = pad
        x1, y1 = x0 + target_w, y0 + target_h
        if x0 < 0 or y1 > fh:  # frame demasiado chico: no compone
            return out

        cv2.rectangle(out, (x0 - 2, y0 - 2), (x1 + 1, y1 + 1), (255, 255, 255), 2)
        out[y0:y1, x0:x1] = mini_rs
        return out
