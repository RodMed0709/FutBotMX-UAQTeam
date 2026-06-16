"""Homografia temporal robusta por seguimiento (fase_4 v2).

Resuelve el problema del jitter ("las lineas se mueven cada frame") y la oclusion
("casi nunca ves 4 puntos") **no re-detectando** las lineas cada frame. En su lugar:

1. **Ancla** (SAM3): cuando hay una deteccion confiable de campo (via
   ``auto_homography.solve_masks`` con la alfombra ``green_floor`` y los centroides
   amarillo/azul), fija ``H`` y **siembra puntos rastreables** sobre la alfombra,
   guardando sus coordenadas de cancha (cm) bajo esa ``H``.
2. **Propaga** (flujo optico): en los frames siguientes sigue esos puntos con
   Lucas-Kanade y recalcula ``H`` por RANSAC entre los puntos seguidos (imagen) y
   sus coordenadas de cancha **fijas**. Asi ``H`` se mueve solo con la camara, no por
   re-deteccion -> sin jitter, y basta con unos pocos puntos (tolera oclusion).
3. **Re-siembra** puntos cuando se agotan (oclusion) usando la ``H`` actual.
4. **Corrige** la deriva lenta con una correccion suave (EMA) hacia un ancla fresca,
   solo si difiere de forma consistente (la camara de verdad se movio).

Basado en el seguimiento de keypoints + flujo optico para campos deportivos
(Real-Time Camera Pose Estimation for Sports Fields, arXiv:2003.14109) + filtrado
temporal de la homografia.
"""

from __future__ import annotations

import numpy as np

from src.core.auto_homography import solve_masks


def _project(H, pts):
    import cv2
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64)).reshape(-1, 2)


class VideoHomographyTracked:
    """Homografia imagen->cm estable por anclaje SAM3 + propagacion por flujo optico.

    Args:
        min_track: minimo de puntos seguidos para confiar en la propagacion.
        target_pts: cuantos puntos sembrar al anclar/re-sembrar.
        correct_every: cada cuantos frames intentar una correccion de deriva con un
            ancla fresca (EMA suave); 0 desactiva la correccion.
        correct_beta: ganancia de la correccion (0=ignora ancla, 1=salta al ancla).
            Bajo (p. ej. 0.15) = corrige despacio, sin jitter.
        max_correct_px: si el ancla fresca difiere de la H seguida por mas de esto
            (en px de esquinas reproyectadas), se considera salto real y se re-ancla
            duro en vez de corregir suave.
    """

    def __init__(self, min_track: int = 12, target_pts: int = 120,
                 correct_every: int = 20, correct_beta: float = 0.15,
                 max_correct_px: float = 60.0):
        self.min_track = min_track
        self.target_pts = target_pts
        self.correct_every = correct_every
        self.correct_beta = correct_beta
        self.max_correct_px = max_correct_px
        self.H: np.ndarray | None = None
        self.prev_gray = None
        self.pts_img: np.ndarray | None = None   # (N,2) puntos en imagen
        self.pts_cm: np.ndarray | None = None    # (N,2) sus coords de cancha fijas
        self.since_anchor = 10 ** 9
        self.n_anchored = 0
        self.n_tracked = 0
        self.n_corrected = 0
        self.n_lost = 0

    # -- siembra de puntos rastreables sobre la alfombra, coords cm bajo H ----------
    def _seed(self, gray, carpet_mask, H) -> bool:
        import cv2
        mask = (carpet_mask > 0).astype(np.uint8)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.target_pts,
                                      qualityLevel=0.01, minDistance=10, mask=mask)
        if pts is None or len(pts) < self.min_track:
            return False
        pts = pts.reshape(-1, 2).astype(np.float32)
        self.pts_img = pts
        self.pts_cm = _project(H, pts)
        self.H = np.asarray(H, np.float64)
        return True

    def _anchor(self, bgr, gray, carpet_mask, yc, bc) -> bool:
        res = solve_masks(bgr, carpet_mask, yc, bc)
        if not (res.ok and res.H is not None):
            return False
        if self._seed(gray, carpet_mask, res.H):
            self.since_anchor = 0
            self.n_anchored += 1
            return True
        return False

    def _corner_diff_px(self, Ha, Hb) -> float:
        # diferencia media de las 4 esquinas de cancha reproyectadas a imagen
        import cv2
        from src.core import field_template as ft
        corners_cm = np.array([(ft.LINE_BORDER_CM, ft.LINE_BORDER_CM),
                               (ft.LENGTH_CM - ft.LINE_BORDER_CM, ft.LINE_BORDER_CM),
                               (ft.LENGTH_CM - ft.LINE_BORDER_CM, ft.WIDTH_CM - ft.LINE_BORDER_CM),
                               (ft.LINE_BORDER_CM, ft.WIDTH_CM - ft.LINE_BORDER_CM)], np.float64)
        try:
            ia = _project(np.linalg.inv(Ha), corners_cm)
            ib = _project(np.linalg.inv(Hb), corners_cm)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.linalg.norm(ia - ib, axis=1).mean())

    def update(self, bgr, carpet_mask, yc=None, bc=None) -> tuple[np.ndarray | None, str]:
        """Procesa un frame. Devuelve ``(H, status)``.

        ``status`` in {anchored, tracked, corrected, reanchored, lost}.
        """
        import cv2
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1) propagar por flujo optico si hay puntos
        if self.pts_img is not None and self.prev_gray is not None and len(self.pts_img) >= self.min_track:
            nxt, stt, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.pts_img.astype(np.float32), None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            ok = stt.ravel().astype(bool)
            gi, gc = nxt[ok], self.pts_cm[ok]
            if len(gi) >= self.min_track:
                Hn, inl = cv2.findHomography(gi, gc, cv2.RANSAC, 6.0)
                if Hn is not None:
                    inl = inl.ravel().astype(bool) if inl is not None else np.ones(len(gi), bool)
                    self.H = Hn
                    self.pts_img, self.pts_cm = gi[inl], gc[inl]
                    self.prev_gray = gray
                    self.since_anchor += 1
                    status = "tracked"
                    self.n_tracked += 1
                    # re-sembrar si quedan pocos puntos (oclusion): mantiene continuidad
                    if len(self.pts_img) < self.target_pts * 0.4:
                        self._seed(gray, carpet_mask, self.H)
                    # correccion suave de deriva con ancla fresca
                    if self.correct_every and self.since_anchor % self.correct_every == 0:
                        res = solve_masks(bgr, carpet_mask, yc, bc)
                        if res.ok and res.H is not None:
                            d = self._corner_diff_px(res.H, self.H)
                            if d > self.max_correct_px:
                                if self._seed(gray, carpet_mask, res.H):
                                    self.since_anchor = 0; status = "reanchored"
                            else:
                                Hc = (1 - self.correct_beta) * self.H + self.correct_beta * (res.H / res.H[2, 2])
                                self.H = Hc
                                self._seed(gray, carpet_mask, Hc)  # re-fija coords cm bajo H corregida
                                self.n_corrected += 1; status = "corrected"
                    return self.H, status

        # 2) sin tracking fiable -> intentar anclar
        if self._anchor(bgr, gray, carpet_mask, yc, bc):
            self.prev_gray = gray
            return self.H, "anchored"

        # 3) nada: propagar la ultima H (congelada)
        self.prev_gray = gray
        self.n_lost += 1
        return self.H, "lost"

    def stats(self) -> dict:
        return {"anchored": self.n_anchored, "tracked": self.n_tracked,
                "corrected": self.n_corrected, "lost": self.n_lost}
