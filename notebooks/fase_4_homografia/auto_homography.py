"""Homografia AUTOMATICA imagen->campo por canales de color (fase_4, sin SAM3).

Pipeline (validado en cámara superior Copa FutBotMX):

1. ``white_mask``  — aisla verde (alfombra) y, dentro de ella, las lineas blancas.
   El rectangulo interior de juego (219x158 cm, inset 12 cm) es ancla mas robusta
   que el borde de la alfombra, porque el frame suele recortar los margenes pero
   el rectangulo interior queda completo.
2. ``inner_corners`` — 4 esquinas del rectangulo interior por **fit de las 4
   rectas-lado + interseccion** (tolera perspectiva; el ``minAreaRect`` solo da los
   ejes para clasificar pixeles por lado, no las esquinas finales).
3. ``solve`` — homografia imagen->cm. La orientacion (cual lado es el largo y que
   porteria va a cada extremo) se resuelve con el **color de las porterias**:
   reproyecta los centroides amarillo/azul y elige la rotacion del mapeo que los
   manda cerca de su objetivo (amarillo x chico, azul x grande). Las 4 esquinas
   ajustan H exacta para cualquier rotacion, asi que el color es la unica señal
   de orientacion fiable.

Todo opera sobre máscaras de color → no depende de SAM3 ni GPU. ``cv2`` perezoso.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import field_template as ft


# Rangos HSV (OpenCV: H 0-180). Calibrados sobre cámara superior IMG_9933/9938.
GREEN_LO, GREEN_HI = (35, 60, 30), (90, 255, 255)
WHITE_LO, WHITE_HI = (0, 0, 150), (180, 70, 255)
YELLOW_LO, YELLOW_HI = (20, 90, 120), (35, 255, 255)
BLUE_LO, BLUE_HI = (100, 90, 60), (130, 255, 255)

# Error de reproyeccion de porteria sobre el cual el encaje se considera malo.
# ~16% del largo del campo; arriba de esto conviene propagar la H previa.
MAX_GOAL_ERR_CM = 40.0


@dataclass
class AutoH:
    """Resultado de la homografia automatica de un frame."""
    H: np.ndarray | None          # imagen->cm (3,3) o None si fallo
    corners: np.ndarray | None    # 4 esquinas imagen del rect. interior (horario)
    goal_err_cm: float            # error de reproyeccion de centroides de porteria
    ok: bool


def _centroid(mask):
    ys, xs = np.where(mask)
    return None if len(xs) < 80 else (float(xs.mean()), float(ys.mean()))


def carpet_and_white(img):
    """Devuelve ``(white_mask, carpet_contour, masks_dict)`` para inspeccion."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, {}
    carpet = max(cnts, key=cv2.contourArea)
    cm = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(cm, [carpet], -1, 255, -1)
    cm = cv2.erode(cm, np.ones((9, 9), np.uint8))
    white = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    white = cv2.bitwise_and(white, cm)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    masks = {
        "green": green, "carpet": cm, "white": white,
        "yellow": cv2.inRange(hsv, YELLOW_LO, YELLOW_HI),
        "blue": cv2.inRange(hsv, BLUE_LO, BLUE_HI),
    }
    return white, carpet, masks


def _fit_line_robust(p):
    """Recta ``(punto, dir_unit)`` por Huber + 2da pasada que recorta outliers (MAD).

    Huber pondera pero no recorta las colas grandes; las areas-D, la linea central
    o sombras pueden sesgar el lado. La 2da pasada (DIST_L2 sobre inliers a <=3·MAD)
    los elimina. Devuelve ``None`` si quedan pocos puntos.
    """
    import cv2

    vx, vy, x0, y0 = cv2.fitLine(p, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    o, d = np.array([x0, y0]), np.array([vx, vy])
    nrm = np.array([-vy, vx])           # normal unitaria a la recta
    dist = np.abs((p - o) @ nrm)
    mad = float(np.median(dist))
    keep = p[dist <= 3.0 * mad + 1e-3] if mad > 0 else p
    if len(keep) < 10:
        keep = p
    vx, vy, x0, y0 = cv2.fitLine(keep, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def inner_corners(white, frac=0.80, min_side_px=20):
    """4 esquinas del rectangulo interior (horario) o ``None``.

    Clasifica los pixeles blancos en 4 lados usando los ejes del ``minAreaRect``
    (derivados de ``boxPoints``, no del angulo —cuya convencion cambia entre
    versiones de OpenCV—), ajusta una recta robusta a cada lado e intersecta
    lados adyacentes. ``frac`` = umbral (fraccion del semieje) de "pixel de borde".
    """
    import cv2

    ys, xs = np.where(white)
    if len(xs) < 4 * min_side_px:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))     # 4 esquinas en orden conocido
    center = box.mean(0)
    e0, e1 = box[1] - box[0], box[2] - box[1]
    n0, n1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    if n0 < 1 or n1 < 1:
        return None
    # u = lado LARGO, v = lado corto (independiente de la convencion del angulo).
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
        # det = sin(angulo entre lados); lados adyacentes deben ser ~perpendiculares.
        # <0.2 (~11.5 grados) = ajuste degenerado -> rechazar.
        if abs(np.linalg.det(A)) < 0.2:
            return None
        t = np.linalg.solve(A, q - p)
        return p + t[0] * d

    cs = [inter("u-", "v-"), inter("u+", "v-"), inter("u+", "v+"), inter("u-", "v+")]
    if any(c is None for c in cs):
        return None
    q = np.array(cs, np.float32)
    # Cuadrilatero convexo con area razonable (descarta esquinas colineales/locas).
    if cv2.contourArea(cv2.convexHull(q)) < 0.02 * white.shape[0] * white.shape[1]:
        return None
    c = q.mean(0)
    a = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    return q[np.argsort(a)]  # winding consistente (no se prueban candidatos espejo)


def _white_in_carpet(img, carpet_mask):
    """Lineas blancas dentro de una mascara de alfombra (interna HSV o externa SAM3)."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cm = (carpet_mask > 0).astype(np.uint8) * 255
    # CRITICO: las lineas blancas son huecos en la mascara de alfombra (no son
    # verdes). Cerrar rellena esos huecos -> la region cubre las lineas; si no, al
    # intersectar con 'white' se borran las lineas (white ~ vacio). El HSV verde
    # local ya cerraba 25x25; el green_floor de SAM3 viene crudo, hay que cerrarlo.
    cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cm = cv2.erode(cm, np.ones((9, 9), np.uint8))
    white = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    white = cv2.bitwise_and(white, cm)
    return cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _orient_and_build(img, corners, yc, bc) -> AutoH:
    """Dadas las 4 esquinas y los centroides de porteria (imagen), resuelve la
    orientacion y devuelve la H. Compartido por ``solve`` (HSV) y ``solve_masks``
    (anclas externas SAM3/YOLO)."""
    import cv2

    b = ft.LINE_BORDER_CM
    inner_cm = np.array([[b, b], [ft.LENGTH_CM - b, b],
                         [ft.LENGTH_CM - b, ft.WIDTH_CM - b], [b, ft.WIDTH_CM - b]], np.float32)

    def goal_err(H):
        e, n = 0.0, 0
        for cc, tgt in ((yc, ft.YELLOW_GOAL_CENTER), (bc, ft.BLUE_GOAL_CENTER)):
            if cc is None:
                continue
            p = cv2.perspectiveTransform(np.array([[cc]], np.float32), H).reshape(2)
            e += float(np.hypot(p[0] - tgt[0], p[1] - tgt[1])); n += 1
        return e / n if n else float("inf")

    # Gate geometrico: una esquina muy fuera de la imagen = interseccion degenerada
    # (un lado del rectangulo quedo cortado por el borde del frame). Rechazar.
    h_img, w_img = img.shape[:2]
    in_bounds = np.all((corners[:, 0] > -0.5 * w_img) & (corners[:, 0] < 1.5 * w_img) &
                       (corners[:, 1] > -0.5 * h_img) & (corners[:, 1] < 1.5 * h_img))
    if not in_bounds:
        return AutoH(None, corners, float("inf"), False)

    def proj(c, H):
        return cv2.perspectiveTransform(np.array([[c]], np.float32), H).reshape(2)

    have_y, have_b = yc is not None, bc is not None
    L2 = ft.LENGTH_CM / 2.0
    best, scored = None, []
    for r in range(4):
        H, _ = cv2.findHomography(corners, np.roll(inner_cm, -r, axis=0), 0)
        if H is None:
            continue
        # Orientacion dura POR PORTERIA VISIBLE: la amarilla debe caer en x<L/2 y/o
        # la azul en x>L/2. Como las esquinas tienen winding consistente (no hay
        # candidato espejo), una sola porteria basta para fijar la orientacion: mata
        # el flip 180 (manda la porteria al lado equivocado) y el giro 90/270.
        if have_y and proj(yc, H)[0] >= L2:
            continue
        if have_b and proj(bc, H)[0] <= L2:
            continue
        err = goal_err(H)
        scored.append(err)
        if best is None or err < best[0]:
            best = (err, H)
    if best is None:
        return AutoH(None, corners, float("inf"), False)
    err, H = best
    # Ambiguo solo si quedan 2 candidatos casi empatados tras el check duro.
    scored.sort()
    ambig = len(scored) >= 2 and (scored[1] - scored[0]) < 10.0
    ok = ((have_y or have_b) and np.isfinite(err) and err <= MAX_GOAL_ERR_CM and not ambig)
    return AutoH(H=H if ok else None, corners=corners, goal_err_cm=err, ok=ok)


def solve(img) -> AutoH:
    """H automatica del frame con anclas por HSV interno (alfombra verde + porterias)."""
    white, _, masks = carpet_and_white(img)
    if white is None:
        return AutoH(None, None, float("inf"), False)
    corners = inner_corners(white)
    if corners is None:
        return AutoH(None, None, float("inf"), False)
    return _orient_and_build(img, corners, _centroid(masks["yellow"]), _centroid(masks["blue"]))


def solve_masks(img, carpet_mask, yellow_centroid=None, blue_centroid=None) -> AutoH:
    """H construida sobre anclas EXTERNAS: alfombra de SAM3 (``green_floor``) y
    centroides de porteria de cajas YOLO. Decopla del HSV interno -> homografia
    explicitamente sobre la segmentacion SAM3 (innovacion 3.7.3)."""
    white = _white_in_carpet(img, carpet_mask)
    corners = inner_corners(white)
    if corners is None:
        return AutoH(None, None, float("inf"), False)
    return _orient_and_build(img, corners, yellow_centroid, blue_centroid)


# Esquinas del rectangulo interior en cm (para medir el "salto" de la H).
_REF_CM = np.array([[ft.LINE_BORDER_CM, ft.LINE_BORDER_CM],
                    [ft.LENGTH_CM - ft.LINE_BORDER_CM, ft.LINE_BORDER_CM],
                    [ft.LENGTH_CM - ft.LINE_BORDER_CM, ft.WIDTH_CM - ft.LINE_BORDER_CM],
                    [ft.LINE_BORDER_CM, ft.WIDTH_CM - ft.LINE_BORDER_CM]], np.float32)


class VideoHomography:
    """Estima H por frame con: lock del arranque + GATE DE CONSISTENCIA temporal +
    EMA + propagacion.

    Premisa: la camara cenital casi no se mueve, asi que entre frames consecutivos
    el campo se proyecta casi igual. Una H que "salta" mucho de un frame al
    siguiente es un **falso positivo** (esquinas/orientacion mal), no movimiento
    real -> se RECHAZA y se mantiene la H previa buena. La primera H (el ancla)
    debe ser solida (error bajo) antes de fijarse.
    """

    def __init__(self, smooth_beta: float = 0.4, max_jump_px: float = 70.0,
                 init_max_err_cm: float = 22.0):
        self.beta = smooth_beta
        self.max_jump_px = max_jump_px      # salto maximo permitido de las esquinas (px)
        self.init_max_err_cm = init_max_err_cm
        self.prev_H: np.ndarray | None = None
        self.last_corners: np.ndarray | None = None
        self.n_estimated = 0
        self.n_propagated = 0
        self.n_rejected = 0

    def _jump_px(self, Hn: np.ndarray, Hp: np.ndarray) -> float:
        """Desplazamiento medio (px) de las esquinas del campo entre dos H (cm->img)."""
        import cv2

        try:
            a = cv2.perspectiveTransform(_REF_CM.reshape(-1, 1, 2), np.linalg.inv(Hn)).reshape(-1, 2)
            b = cv2.perspectiveTransform(_REF_CM.reshape(-1, 1, 2), np.linalg.inv(Hp)).reshape(-1, 2)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.linalg.norm(a - b, axis=1).mean())

    def _ingest(self, res) -> tuple[np.ndarray | None, str]:
        if res.ok and res.H is not None:
            Hn = res.H / res.H[2, 2]
            if self.prev_H is None:
                # Lock del arranque: solo fijar el ancla si el error es bajo.
                if res.goal_err_cm <= self.init_max_err_cm:
                    self.prev_H, self.last_corners = Hn, res.corners
                    self.n_estimated += 1
                    return Hn, "anchors"
                self.n_propagated += 1
                return None, "init"        # aun sin ancla fiable
            # Gate de consistencia: si la H salta demasiado = falso positivo.
            if self._jump_px(Hn, self.prev_H) > self.max_jump_px:
                self.n_rejected += 1
                return self.prev_H, "rejected"
            H = (1.0 - self.beta) * Hn + self.beta * self.prev_H if self.beta > 0 else Hn
            self.prev_H, self.last_corners = H, res.corners
            self.n_estimated += 1
            return H, "anchors"
        self.n_propagated += 1
        return self.prev_H, "propagated"

    def update(self, img) -> tuple[np.ndarray | None, str]:
        """Camino HSV interno (sin SAM3)."""
        return self._ingest(solve(img))

    def update_masks(self, img, carpet_mask, yc=None, bc=None) -> tuple[np.ndarray | None, str]:
        """Camino SAM3: alfombra ``green_floor`` + centroides de porteria YOLO."""
        return self._ingest(solve_masks(img, carpet_mask, yc, bc))
