"""Figura de validacion held-out: clic humano vs homografia (1 panel, para el paper).

Dibuja sobre UN frame representativo:
  - la plantilla del campo (rectangulo interior, linea central, circulo, areas)
    reproyectada cm->px por la H desplegada (calidad global del registro);
  - cada landmark HELD-OUT como: clic humano (verde) vs su posicion segun la H
    (rojo, = cm conocido reproyectado), unidos por el vector de error;
  - el error mediano del frame en el caption.

El frame se elige automaticamente como el de error mediano del frame mas cercano a la
mediana GLOBAL (representativo, ni el mejor ni el peor). Solo lee clicks + clip + JSON;
no toca pipeline ni entregables. LOCAL (sin GPU).

Uso
---
    python notebooks/fase_4_homografia/held_out_figure.py \
        --json outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json \
        --clip outputs/fase5_clips/IMG_9933_5m30.mp4 \
        --clicks outputs/homografia_heldout/clicks_IMG_9933_5m30.csv \
        --out .specs/drafts/paper_md_draft/proy_latex/fig_homography_heldout.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.core import field_landmarks as fl
from src.core import field_template as ft
from src.utils import PROJECT_ROOT

# reutiliza la asignacion robusta top/bot del script de error
import importlib.util as _u
_spec = _u.spec_from_file_location(
    "_hoe", Path(__file__).with_name("held_out_error.py"))
_hoe = _u.module_from_spec(_spec)
_spec.loader.exec_module(_hoe)


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def _template_polylines_cm() -> dict[str, np.ndarray]:
    """Polilineas del campo en cm para dibujar la plantilla reproyectada."""
    L = fl.LANDMARK_POINTS
    inner = np.array([L["inner_tl"], L["inner_tr"], L["inner_br"], L["inner_bl"], L["inner_tl"]])
    center = np.array([L["center_top"], L["center_bot"]])
    cx, cy, r = fl.CENTER_CIRCLE
    th = np.linspace(0, 2 * np.pi, 72)
    circle = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])
    depth = ft.PENALTY_DEPTH_CM
    glL = ft.GOAL_LINE_X_LEFT_CM
    glR = ft.GOAL_LINE_X_RIGHT_CM
    yT, yB = ft._PEN_TOP_Y_CM, ft._PEN_BOTTOM_Y_CM
    penL = np.array([[glL, yT], [glL + depth, yT], [glL + depth, yB], [glL, yB]])
    penR = np.array([[glR, yT], [glR - depth, yT], [glR - depth, yB], [glR, yB]])
    return {"inner": inner, "center": center, "circle": circle, "penL": penL, "penR": penR}


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--clicks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=None, help="forzar frame (si no, auto)")
    args = ap.parse_args()

    from src.core.metric_positions import compute_metric_positions

    res = compute_metric_positions(_abs(args.json), video=_abs(args.clip))
    H = res.H_por_frame

    # mediciones por frame (con asignacion robusta top/bot)
    clicks = list(csv.DictReader(open(_abs(args.clicks))))
    per_frame: dict[int, list] = {}
    all_err: list[float] = []
    for c in clicks:
        idx = int(c["frame_index"])
        fam = _hoe._family_of(c["landmark"])
        if fam not in _hoe.FAMILIES or H.get(idx) is None:
            continue
        h = np.asarray(H[idx], float)
        px, py = float(c["px_x"]), float(c["px_y"])
        cm = _hoe._project_px_to_cm(h, px, py)
        name = _hoe._assign_landmark(fam, cm)
        gx, gy = fl.LANDMARK_POINTS[name]
        err = float(np.hypot(cm[0] - gx, cm[1] - gy))
        per_frame.setdefault(idx, []).append((px, py, name, err))
        all_err.append(err)

    global_med = float(np.median(all_err))

    # frame representativo: >=6 clics, error medio del frame mas cercano a la mediana global
    if args.frame is not None:
        sel = args.frame
    else:
        cand = {i: np.mean([e for *_, e in v]) for i, v in per_frame.items() if len(v) >= 6}
        sel = min(cand, key=lambda i: abs(cand[i] - global_med))
    pts = per_frame[sel]
    frame_med = float(np.median([e for *_, e in pts]))
    print(f"frame elegido: {sel}  |  {len(pts)} held-out  |  mediana frame {frame_med:.1f} cm "
          f"(global {global_med:.1f} cm)")

    # leer el frame
    cap = cv2.VideoCapture(str(_abs(args.clip)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sel)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no se pudo leer el frame {sel}")
    # alinear al dominio de la H (igual que el pipeline): clicks estan en esa resolucion
    from src.core.metric_positions import _largest_green_rle
    from src.core.inference_schema import decode_rle
    import json
    data = json.load(open(_abs(args.json)))
    rle = _largest_green_rle(next(f for f in data["frames"] if f["frame_index"] == sel)
                             ["detections"].get("green_floor", []))
    if rle is not None:
        ch, cw = decode_rle(rle).shape[:2]
        if img.shape[:2] != (ch, cw):
            img = cv2.resize(img, (cw, ch))

    Hsel = np.asarray(H[sel], float)
    Hinv = np.linalg.inv(Hsel)

    # 1) contexto minimo: solo rectangulo interior + circulo, finos (la plantilla
    #    completa confunde porque la distorsion de barril curva las lineas reales).
    tpl = _template_polylines_cm()
    for name in ("inner", "circle"):
        px = cv2.perspectiveTransform(tpl[name].reshape(-1, 1, 2).astype(np.float64), Hinv).reshape(-1, 2)
        cv2.polylines(img, [np.round(px).astype(np.int32)], True, (230, 200, 80), 2, cv2.LINE_AA)

    # 2) held-out (protagonista): clic humano (verde) vs landmark segun H (rojo) + error cm.
    #    Marcas GRANDES para que se lean en el paper a tamano de columna sin hacer zoom.
    R = 22                  # radio de los marcadores (px)
    FS, FT, FTo = 1.6, 4, 11  # escala, grosor, grosor de contorno del texto
    for px, py, name, err in pts:
        gx, gy = fl.LANDMARK_POINTS[name]
        hp = cv2.perspectiveTransform(np.array([[[gx, gy]]], np.float64), Hinv)[0, 0]
        a = (int(round(px)), int(round(py)))
        b = (int(round(hp[0])), int(round(hp[1])))
        cv2.line(img, a, b, (0, 0, 0), 9, cv2.LINE_AA)             # contorno del vector error
        cv2.line(img, a, b, (0, 255, 255), 5, cv2.LINE_AA)        # error (amarillo)
        cv2.circle(img, b, R, (0, 0, 0), 6, cv2.LINE_AA)          # segun H (rojo, con halo)
        cv2.circle(img, b, R, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.circle(img, a, R, (0, 230, 0), -1, cv2.LINE_AA)       # clic humano (verde)
        cv2.circle(img, a, R, (0, 0, 0), 3, cv2.LINE_AA)
        lbl = f"{err:.0f} cm"
        lp = (a[0] + R + 6, a[1] - R)
        cv2.putText(img, lbl, lp, cv2.FONT_HERSHEY_SIMPLEX, FS, (0, 0, 0), FTo, cv2.LINE_AA)
        cv2.putText(img, lbl, lp, cv2.FONT_HERSHEY_SIMPLEX, FS, (0, 255, 255), FT, cv2.LINE_AA)

    # 3) recortar a la cancha (quita barras negras) usando la bbox de la alfombra
    if rle is not None:
        carpet = decode_rle(rle)
        if carpet.shape[:2] != img.shape[:2]:
            carpet = cv2.resize(carpet, (img.shape[1], img.shape[0]))
        ys, xs = np.where(carpet > 0)
        m = 40
        y1, y2 = max(0, ys.min() - m), min(img.shape[0], ys.max() + m)
        x1, x2 = max(0, xs.min() - m), min(img.shape[1], xs.max() + m)
        img = img[y1:y2, x1:x2].copy()

    # 4) la leyenda ya NO se quema en la imagen: vive en el caption de LaTeX (puede
    #    saltar de linea y usar el ancho de columna). Asi la figura queda limpia y las
    #    marcas grandes se leen sin zoom.

    out = _abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"guardado: {out}  ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
