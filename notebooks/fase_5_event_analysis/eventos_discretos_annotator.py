"""Anotador ligero de EVENTOS DISCRETOS (GT manual para las Tablas de eventos).

Reproduce un clip y deja al humano marcar cada **evento discreto real** con su frame y
calificador. Cubre el set completo que el pipeline contempla:

    gol        portería  (yellow | blue)        -> event_shot_goal
    tiro       portería  (yellow | blue)        -> event_shot_goal
    fuera      causa     (salida | area)        -> event_field_violations
    lack       (sin calificador)                -> event_field_violations (lack_of_progress)
    pushing    portería  (yellow | blue | -)    -> event_field_violations

Salida: un CSV de GT ligero ``clip,frame,tipo,calificador,note`` que consume
``eventos_discretos_eval.py`` para sacar Precision/Recall por tipo de evento.

IMPORTANTE (honestidad del alcance): los detectores **desplegados** de gol/tiro/fuera son de
**capa métrica (cm)** y solo aplican a video de **cámara superior** compatible con homografía.
Anota sobre esos clips (los que tienen tracking JSON cenital). Es **solo anotación**: no toca
el pipeline ni ``src/``, no corre modelos. LOCAL (solo el clip .mp4).

Controles (ventana de OpenCV)
-----------------------------
    espacio   play / pausa
    d / →  +1     e  +10     f  +30        (avanzar)
    a / ←  -1     w  -10     r  -30        (retroceder)
    g        ir a un frame (se escribe en la terminal)
    u        deshacer la última marca      s / ESC  guardar y salir

    Marcar evento = 1 tecla de TIPO y luego (si aplica) 1 tecla de CALIFICADOR:
      1 gol     -> y/b (portería)
      2 tiro    -> y/b (portería)
      3 fuera   -> s (salida de campo) / a (entra al área chica)
      4 lack    (se marca directo, sin calificador)
      5 pushing -> y/b (portería) o ENTER para sin zona

Uso
---
    python notebooks/fase_5_event_analysis/tabla9_goal_annotator.py \
        --clip outputs/fase5_clips/IMG_9933_5m30.mp4 \
        --out outputs/eventos_gt/events_gt.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils import PROJECT_ROOT

# tipo -> (etiqueta, conjunto de calificadores válidos, obligatorio?)
TYPES = {
    ord("1"): ("gol", {"y": "yellow", "b": "blue"}, True),
    ord("2"): ("tiro", {"y": "yellow", "b": "blue"}, True),
    ord("3"): ("fuera", {"s": "salida_campo", "a": "area_chica"}, True),
    ord("4"): ("lack_of_progress", {}, False),
    ord("5"): ("pushing", {"y": "yellow", "b": "blue"}, False),
}
_COL = {"gol": (0, 230, 255), "tiro": (0, 200, 120), "fuera": (60, 60, 255),
        "lack_of_progress": (255, 200, 0), "pushing": (255, 120, 0)}


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def _load_existing(out: Path, stem: str) -> list[dict]:
    """Marcas previas de OTROS clips (se conservan); las de este clip se reescriben."""
    if not out.exists():
        return []
    return [r for r in csv.DictReader(open(out)) if r["clip"] != stem]


def _hud(img, cur, last, playing, marks, stem, pending):
    import cv2

    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 96), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)
    mine = [m for m in marks if m["clip"] == stem]
    cv2.putText(img, f"{stem}  frame {cur}/{last}  {'PLAY' if playing else 'PAUSA'}  "
                f"eventos: {len(mine)}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "1 gol  2 tiro  3 fuera  4 lack  5 pushing   u undo   s guardar",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 255, 180), 1, cv2.LINE_AA)
    if pending:
        tipo, quals, _ = pending
        opts = " / ".join(f"{k}={v}" for k, v in quals.items()) or "ENTER"
        cv2.putText(img, f">> {tipo}: elige calificador [{opts}]  (ESC cancela)",
                    (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    for m in mine:
        if abs(int(m["frame"]) - cur) <= 1:
            col = _COL.get(m["tipo"], (255, 255, 255))
            lbl = f"{m['tipo']}" + (f" [{m['calificador']}]" if m["calificador"] else "")
            cv2.putText(img, lbl, (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, col, 4, cv2.LINE_AA)


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", default=None, help="identidad del clip (default: nombre del archivo)")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    clip = _abs(args.clip)
    stem = args.stem or clip.stem
    out = _abs(args.out)

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"no se pudo abrir el clip: {clip}")
    last = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"clip {stem}: {last + 1} frames @ {fps:.1f} fps")

    marks = _load_existing(out, stem)
    cur, playing, pending = 0, False, None
    win = f"GT eventos — {stem}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def read(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        return fr if ok else None

    def commit(tipo, cal):
        marks.append({"clip": stem, "frame": cur, "tipo": tipo, "calificador": cal, "note": ""})
        print(f"  + {tipo}" + (f" [{cal}]" if cal else "") + f" @ frame {cur}")

    while True:
        frame = read(cur)
        if frame is None:
            cur = max(0, cur - 1)
            playing = False
            continue
        disp = frame.copy()
        _hud(disp, cur, last, playing, marks, stem, pending)
        if args.scale != 1.0:
            disp = cv2.resize(disp, None, fx=args.scale, fy=args.scale)
        cv2.imshow(win, disp)

        key = cv2.waitKey(max(1, int(1000 / fps)) if (playing and not pending) else 30) & 0xFF
        if playing and not pending and key == 255:
            cur = min(last, cur + 1)
            if cur == last:
                playing = False
            continue

        # --- modo calificador: esperando la 2a tecla ---
        if pending is not None:
            tipo, quals, mandatory = pending
            if key == 27:  # ESC cancela
                pending = None
            elif key == 13 and not mandatory:  # ENTER = sin calificador
                commit(tipo, "")
                pending = None
            elif chr(key) in quals if key < 256 else False:
                commit(tipo, quals[chr(key)])
                pending = None
            continue

        # --- navegación / acciones ---
        if key in (ord("s"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key in (ord("d"), 83):
            cur = min(last, cur + 1)
        elif key in (ord("a"), 81):
            cur = max(0, cur - 1)
        elif key == ord("e"):
            cur = min(last, cur + 10)
        elif key == ord("w"):
            cur = max(0, cur - 10)
        elif key == ord("f"):
            cur = min(last, cur + 30)
        elif key == ord("r"):
            cur = max(0, cur - 30)
        elif key == ord("u"):
            mine = [m for m in marks if m["clip"] == stem]
            if mine:
                marks.remove(mine[-1])
                print("  - undo")
        elif key == ord("g"):
            try:
                cur = max(0, min(last, int(input("ir a frame: "))))
            except (ValueError, EOFError):
                pass
        elif key in TYPES:
            tipo, quals, mandatory = TYPES[key]
            if not quals:  # lack: marca directo
                commit(tipo, "")
            else:
                pending = (tipo, quals, mandatory)
                playing = False

    cap.release()
    cv2.destroyAllWindows()

    out.parent.mkdir(parents=True, exist_ok=True)
    marks.sort(key=lambda m: (m["clip"], int(m["frame"])))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "frame", "tipo", "calificador", "note"])
        w.writeheader()
        w.writerows(marks)
    mine = [m for m in marks if m["clip"] == stem]
    print(f"\nguardado: {out}  ({len(marks)} marcas totales, {len(mine)} de {stem})")


if __name__ == "__main__":
    main()
