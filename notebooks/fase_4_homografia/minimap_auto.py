"""Driver AUTOCONTENIDO fase_4: video -> homografia automatica por color ->
minimap de trayectorias. NO usa SAM3 ni el repo (`src.core.*`): solo cv2 +
`field_template` + `auto_homography`. Corre local sin GPU.

- Homografia: ``auto_homography.VideoHomography`` (color, EMA + propagacion).
- Objetos: deteccion por color local (robots = blobs oscuros en la alfombra;
  balon = blob naranja). Punto-pie = centro-inferior (robot) / centroide (balon).
- Minimap: cancha canonica de ``field_template`` (la H normaliza la orientacion,
  asi que NO hace falta rotar el minimap segun la camara).

Uso: ``python minimap_auto.py <video> <out.mp4> [max_frames]``.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_template as ft
import auto_homography as ah

ROBOT_PALETTE = [(60, 130, 255), (255, 80, 80), (80, 220, 120),
                 (200, 120, 255), (255, 170, 40), (40, 220, 220)]
BALL_COLOR = (255, 140, 0)


def detect_objects(bgr, carpet_mask):
    """Detecta robots (blobs oscuros) y balon (naranja) dentro de la alfombra.

    Devuelve lista ``(class, foot_xy)`` con ``class in {'robot','ball'}``.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    inside = carpet_mask > 0
    dets = []

    # Robots: oscuros (V bajo) y poco verdes, dentro de la alfombra.
    dark = ((hsv[:, :, 2] < 110) & inside).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    H_img, W_img = bgr.shape[:2]
    n, _, stats, cents = cv2.connectedComponentsWithStats(dark)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (1500 < area < 40000):           # robot ~ tam medio; descarta sombras/ruido
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        ar = max(w, h) / max(1, min(w, h))
        if ar > 1.8:                             # robot es ~circular; rechaza sombras/celular alargado
            continue
        if x <= 2 or y <= 2 or x + w >= W_img - 2 or y + h >= H_img - 2:
            continue                             # toca el borde = mano/pie entrando, no robot
        dets.append(("robot", (float(x + w / 2.0), float(y + h))))

    # Balon: naranja saturado, pequeno.
    orange = cv2.inRange(hsv, (5, 120, 120), (20, 255, 255))
    orange = cv2.bitwise_and(orange, carpet_mask)
    n, _, stats, cents = cv2.connectedComponentsWithStats(orange)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if 30 < area < 4000:
            dets.append(("ball", (float(cents[i][0]), float(cents[i][1]))))
    return dets


def draw_field_overlay(frame, H, corners=None):
    """Pinta sobre el frame las 4 esquinas detectadas + el rectangulo interior y el
    circulo central reproyectados por H (cm->imagen). Confirma visualmente la
    homografia. Colores en convencion del frame que recibe (driver usa RGB)."""
    out = frame
    b = ft.LINE_BORDER_CM
    if H is not None:
        Hinv = np.linalg.inv(H)

        def c2i(p):
            q = cv2.perspectiveTransform(np.array([[p]], np.float32), Hinv).reshape(2)
            return int(q[0]), int(q[1])

        inner = [(b, b), (ft.LENGTH_CM - b, b), (ft.LENGTH_CM - b, ft.WIDTH_CM - b), (b, ft.WIDTH_CM - b)]
        cv2.polylines(out, [np.array([c2i(p) for p in inner], np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
        circ = [c2i((ft.CENTER_CM[0] + ft.CIRCLE_RADIUS_CM * np.cos(t),
                     ft.CENTER_CM[1] + ft.CIRCLE_RADIUS_CM * np.sin(t))) for t in np.linspace(0, 2 * np.pi, 40)]
        cv2.polylines(out, [np.array(circ, np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
    if corners is not None:
        for p in np.asarray(corners).astype(int):
            cv2.circle(out, (int(p[0]), int(p[1])), 13, (255, 0, 0), -1, cv2.LINE_AA)
    return out


class GreedyTracker:
    """Tracker minimo por vecino mas cercano (por clase) para ids estables."""

    def __init__(self, gate_px=120.0, max_age=10):
        self.gate, self.max_age = gate_px, max_age
        self.tracks: dict[int, dict] = {}
        self.next = 0

    def update(self, dets):
        for t in self.tracks.values():
            t["age"] += 1
        out, used = [], set()
        for cls, pt in dets:
            best, bd = None, self.gate
            for tid, t in self.tracks.items():
                if tid in used or t["class"] != cls:
                    continue
                d = float(np.hypot(t["pt"][0] - pt[0], t["pt"][1] - pt[1]))
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best, self.next = self.next, self.next + 1
            used.add(best)
            self.tracks[best] = {"class": cls, "pt": pt, "age": 0}
            out.append((best, cls, pt))
        for tid in [k for k, t in self.tracks.items() if t["age"] > self.max_age]:
            del self.tracks[tid]
        return out


class Minimap:
    """Cancha canonica + trails acumulados por obj_id, compuesta sobre el frame."""

    def __init__(self, scale=2.6, trail_len=64, persist=24, panel_frac=0.34):
        # margen chico: los robots viven en el verde, no hace falta marco negro grueso.
        self.base, self.to_px = ft.render_field(scale=scale, margin_cm=3.0)
        self.trail_len, self.persist, self.panel_frac = trail_len, persist, panel_frac
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
        self.cls: dict[int, str] = {}
        self.seen: dict[int, int] = {}
        self.f = 0

    def update(self, projected):
        self.f += 1
        for oid, cls, x, y in projected:
            if -ft.LENGTH_CM <= x <= 2 * ft.LENGTH_CM and -ft.WIDTH_CM <= y <= 2 * ft.WIDTH_CM:
                self.trails[oid].append((x, y))
                self.cls[oid] = cls
                self.seen[oid] = self.f
        for oid in [o for o, s in self.seen.items() if self.f - s > self.persist]:
            self.trails.pop(oid, None); self.cls.pop(oid, None); self.seen.pop(oid, None)

    def render(self):
        c = self.base.copy()
        for oid, tr in self.trails.items():
            if not tr:
                continue
            cls = self.cls.get(oid, "robot")
            pts = [self.to_px(p) for p in tr]
            x, y = pts[-1]
            if cls == "ball":
                # balon: circulo NARANJA grande y visible.
                cv2.circle(c, (x, y), 12, BALL_COLOR, -1, cv2.LINE_AA)
                cv2.circle(c, (x, y), 12, (20, 20, 20), 2, cv2.LINE_AA)
                if len(pts) >= 2:
                    cv2.polylines(c, [np.array(pts, np.int32)], False, BALL_COLOR, 2, cv2.LINE_AA)
            else:
                # robot: CUADRO GRIS grande (no puntito); trail tenue por obj_id.
                trail_col = ROBOT_PALETTE[oid % len(ROBOT_PALETTE)]
                if len(pts) >= 2:
                    cv2.polylines(c, [np.array(pts, np.int32)], False, trail_col, 2, cv2.LINE_AA)
                s = 14
                cv2.rectangle(c, (x - s, y - s), (x + s, y + s), (175, 175, 175), -1)
                cv2.rectangle(c, (x - s, y - s), (x + s, y + s), (35, 35, 35), 2)
        return c

    def composite(self, frame):
        mini = self.render()
        out = frame.copy(); fh, fw = out.shape[:2]
        tw = max(1, int(fw * self.panel_frac)); mh, mw = mini.shape[:2]
        th = max(1, int(mh * tw / mw))
        mini = cv2.resize(mini, (tw, th), interpolation=cv2.INTER_AREA)
        pad = max(6, int(fw * 0.01)); x0, y0 = fw - tw - pad, pad
        if x0 < 0 or y0 + th > fh:
            return out
        cv2.rectangle(out, (x0 - 2, y0 - 2), (x0 + tw + 1, y0 + th + 1), (255, 255, 255), 2)
        out[y0:y0 + th, x0:x0 + tw] = mini
        return out


def render_minimap_auto(video, out_path, max_frames=None, every=1):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"no pude leer {video}")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / every, (w, h))
    vh = ah.VideoHomography(smooth_beta=0.4)
    tracker, mini = GreedyTracker(), Minimap()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        if i % every == 0:
            white, carpet, masks = ah.carpet_and_white(frame)
            H, _ = vh.update(frame)
            objs = tracker.update(detect_objects(frame, masks["carpet"]) if masks else [])
            projected = []
            if H is not None and objs:
                feet = np.array([p for _, _, p in objs], np.float32).reshape(-1, 1, 2)
                cm = cv2.perspectiveTransform(feet, H).reshape(-1, 2)
                projected = [(oid, cls, float(x), float(y)) for (oid, cls, _), (x, y) in zip(objs, cm)]
            mini.update(projected)
            writer.write(mini.composite(frame))
            n += 1
        i += 1
    cap.release(); writer.release()
    return {"out": out_path, "n_frames": n,
            "estimated": vh.n_estimated, "propagated": vh.n_propagated}


if __name__ == "__main__":
    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "minimap_auto.mp4"
    mf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(render_minimap_auto(video, out, max_frames=mf))
