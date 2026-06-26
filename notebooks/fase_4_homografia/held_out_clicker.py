"""Clicker de landmarks HELD-OUT para medir el error de la homografia por lineas.

Objetivo
--------
La homografia desplegada (``VideoHomographyLines``, la que corre en ``main.py`` ->
``compute_metric_positions``) ajusta ``H`` SOLO con las 4 esquinas del rectangulo
interior (``inner_*``). Para medir su error en cm de forma **no circular** se usan
landmarks que **NUNCA entran al ajuste** (held-out): linea central, circulo central
y esquinas de las areas. Sus coordenadas en cm ya estan en ``field_landmarks.py``.

Este script muestra unos pocos frames del clip crudo y deja que un humano **haga
clic** en esos puntos held-out (los que se vean claros). Guarda un CSV
``frame_index, landmark, px_x, px_y`` que ``held_out_error.py`` convierte en error cm.

NO modifica nada del pipeline ni de los entregables: solo lee el clip + el JSON de
tracking (para la resolucion de la mascara) y escribe un CSV de medicion.

Corre LOCAL (sin GPU/SAM3). Necesita ventana grafica (cv2.imshow).

Uso
---
    python notebooks/fase_4_homografia/held_out_clicker.py \
        --json outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json \
        --clip outputs/fase5_clips/IMG_9933_5m30.mp4 \
        --n 10 \
        --out outputs/homografia_heldout/clicks_IMG_9933_5m30.csv

Controles (por landmark):
    - clic izquierdo  -> coloca el punto (se puede reposicionar reclicando)
    - n / ENTER / ESPACIO -> acepta y pasa al siguiente landmark
    - s              -> SALTA este landmark (no se ve en el frame)
    - b              -> regresa al landmark anterior
    - q              -> guarda lo hecho y termina
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.core import field_landmarks as fl
from src.utils import PROJECT_ROOT

# --- landmarks held-out a clicar (NO inner_*), con pista humana --------------------
# El solver fuerza amarillo a la IZQUIERDA en cm (x<121.5) y azul a la DERECHA, asi
# que penL = lado de la porteria AMARILLA, penR = lado de la porteria AZUL.
HELD_OUT: list[tuple[str, str]] = [
    ("center_top", "LINEA CENTRAL toca el borde (un extremo de la linea central)"),
    ("center_bot", "LINEA CENTRAL toca el borde (el otro extremo)"),
    ("circle_top", "La LINEA CENTRAL entra al CIRCULO: cruce de un lado"),
    ("circle_bot", "La LINEA CENTRAL entra al CIRCULO: cruce del otro lado"),
    ("penY_top", "AREA del lado AMARILLO: una esquina sobre la linea de gol"),
    ("penY_bot", "AREA del lado AMARILLO: la otra esquina sobre la linea de gol"),
    ("penB_top", "AREA del lado AZUL: una esquina sobre la linea de gol"),
    ("penB_bot", "AREA del lado AZUL: la otra esquina sobre la linea de gol"),
]

# Mapa de nombre humano -> nombre real en field_landmarks (penY/penB = penL/penR).
NAME_MAP = {
    "center_top": "center_top", "center_bot": "center_bot",
    "circle_top": "circle_top", "circle_bot": "circle_bot",
    "penY_top": "penL_top", "penY_bot": "penL_bot",
    "penB_top": "penR_top", "penB_bot": "penR_bot",
}


def _carpet_shape_by_frame(data: dict) -> dict[int, tuple[int, int]]:
    """Por frame con green_floor: (H, W) de la mascara de alfombra (dominio de la H)."""
    from src.core.metric_positions import _largest_green_rle
    from src.core.inference_schema import decode_rle

    out: dict[int, tuple[int, int]] = {}
    for fr in data.get("frames", []):
        rle = _largest_green_rle(fr.get("detections", {}).get("green_floor", []))
        if rle is not None:
            out[int(fr["frame_index"])] = decode_rle(rle).shape[:2]
    return out


def _pick_frames(frames_with_carpet: list[int], n: int) -> list[int]:
    """n indices repartidos uniformemente entre los frames que tienen alfombra."""
    if n >= len(frames_with_carpet):
        return frames_with_carpet
    step = len(frames_with_carpet) / n
    return [frames_with_carpet[int(i * step)] for i in range(n)]


def _click_one_frame(frame_bgr, frame_idx: int) -> list[tuple[str, float, float]]:
    """Devuelve [(landmark_real, px_x, px_y)] para un frame. UI cv2."""
    import cv2

    H, W = frame_bgr.shape[:2]
    scale = min(1.0, 1200.0 / max(H, W))  # encoge para que quepa en pantalla
    disp0 = cv2.resize(frame_bgr, (int(W * scale), int(H * scale))) if scale < 1 else frame_bgr.copy()

    win = f"held-out clicker  |  frame {frame_idx}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {"pt": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)

    cv2.setMouseCallback(win, on_mouse)

    results: list[tuple[str, float, float]] = []
    i = 0
    while i < len(HELD_OUT):
        human_name, hint = HELD_OUT[i]
        state["pt"] = None
        while True:
            img = disp0.copy()
            # ya colocados
            for _, px, py in results:
                cv2.circle(img, (int(px * scale), int(py * scale)), 5, (0, 200, 0), -1)
            # punto actual tentativo
            if state["pt"] is not None:
                cv2.circle(img, state["pt"], 6, (0, 0, 255), 2)
            bar = img.copy()
            cv2.rectangle(bar, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
            cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)
            cv2.putText(img, f"[{i + 1}/{len(HELD_OUT)}] {human_name}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, hint, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, "clic=poner  n=ok  s=saltar  b=atras  q=guardar/salir",
                        (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.imshow(win, img)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("n"), 13, 32):  # n / ENTER / ESPACIO
                if state["pt"] is not None:
                    px, py = state["pt"][0] / scale, state["pt"][1] / scale
                    results.append((NAME_MAP[human_name], px, py))
                    i += 1
                    break
            elif k == ord("s"):
                i += 1
                break
            elif k == ord("b"):
                i = max(0, i - 1)
                if results and i < len(results):
                    results.pop()
                break
            elif k == ord("q"):
                cv2.destroyWindow(win)
                return results
    cv2.destroyWindow(win)
    return results


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="tracking JSON (include_masks=True)")
    ap.add_argument("--clip", required=True, help="clip CRUDO (no el segmentado)")
    ap.add_argument("--n", type=int, default=10, help="numero de frames a clicar")
    ap.add_argument("--frames", default=None, help="indices explicitos coma-separados")
    ap.add_argument("--out", required=True, help="CSV de salida de clics")
    args = ap.parse_args()

    import json

    def _abs(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else PROJECT_ROOT / q

    data = json.load(open(_abs(args.json)))
    carpet_shapes = _carpet_shape_by_frame(data)
    frames_ok = sorted(carpet_shapes)
    if not frames_ok:
        raise SystemExit("el JSON no tiene green_floor con rle (include_masks?)")

    if args.frames:
        sel = [int(x) for x in args.frames.split(",")]
    else:
        sel = _pick_frames(frames_ok, args.n)
    print(f"frames a clicar ({len(sel)}): {sel}")

    cap = cv2.VideoCapture(str(_abs(args.clip)))
    if not cap.isOpened():
        raise SystemExit(f"no se pudo abrir el clip: {args.clip}")

    rows: list[tuple[int, str, float, float]] = []
    for idx in sel:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {idx}: no se pudo leer, se omite")
            continue
        hm, wm = carpet_shapes[idx]
        if frame.shape[:2] != (hm, wm):  # alinear al dominio de la H (igual que el pipeline)
            frame = cv2.resize(frame, (wm, hm))
        clicks = _click_one_frame(frame, idx)
        for name, px, py in clicks:
            rows.append((idx, name, round(px, 2), round(py, 2)))
        print(f"  frame {idx}: {len(clicks)} clics")
    cap.release()

    out = _abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "landmark", "px_x", "px_y"])
        w.writerows(rows)
    print(f"\nguardado: {out}  ({len(rows)} clics held-out)")
    # sanity: los landmarks usados existen en field_landmarks
    unknown = {r[1] for r in rows} - set(fl.LANDMARK_POINTS)
    if unknown:
        print(f"AVISO: landmarks no reconocidos: {unknown}")


if __name__ == "__main__":
    main()
