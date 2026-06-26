"""Error en cm de la homografia POR LINEAS sobre landmarks held-out (clicks humanos).

Mide el error de reproyeccion del metodo **desplegado** (``VideoHomographyLines`` via
``compute_metric_positions``, el mismo que usa ``main.py`` para el video de espectador)
de forma **no circular**: usa los clics humanos en landmarks que NO entran al ajuste
(``held_out_clicker.py``) y los compara contra la geometria fisica conocida del campo
(``field_landmarks.py``, en cm).

Procedimiento (faithful al pipeline):
1. ``compute_metric_positions(json, video=clip)`` -> ``H_por_frame`` (la H real, con
   EMA + propagacion, identica a la del entregable). NO se re-ajusta nada aparte.
2. Por cada clic ``(frame, landmark, px)`` -> proyecta px->cm con ``H_por_frame[frame]``.
3. error = || proj_cm - cm_conocido(landmark) ||. Reporta mediana / p90 / max,
   global y por landmark, + cobertura de H.

NO toca el pipeline ni los entregables: solo lee el JSON + el clip + el CSV de clics
y escribe un CSV de errores. Corre LOCAL (sin GPU/SAM3).

Uso
---
    python notebooks/fase_4_homografia/held_out_error.py \
        --json outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json \
        --clip outputs/fase5_clips/IMG_9933_5m30.mp4 \
        --clicks outputs/homografia_heldout/clicks_IMG_9933_5m30.csv \
        --out outputs/homografia_heldout/error_IMG_9933_5m30.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.core import field_landmarks as fl
from src.utils import PROJECT_ROOT


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


# Familias de landmarks held-out: el clic identifica de forma fiable la FAMILIA
# (centro vs circulo vs area amarilla vs area azul: el humano distingue el lado por
# color de porteria y centro/circulo a ojo), pero el sufijo top/bot puede salir
# invertido respecto al eje-y del template (convencion imagen-y vs template-y). Por eso
# la asignacion top/bot NO se confia del CSV: dentro de cada familia se elige el
# candidato cuya cm proyectada esta mas cerca. Es una correccion deterministica y NO
# circular (los 2 candidatos de una familia distan 60-158 cm, muy por encima del error).
FAMILIES: dict[str, list[str]] = {
    "center": ["center_top", "center_bot"],
    "circle": ["circle_top", "circle_bot"],
    "penL": ["penL_top", "penL_bot"],
    "penR": ["penR_top", "penR_bot"],
}


def _family_of(label: str) -> str:
    """Familia a partir de la etiqueta del clic (quita el sufijo _top/_bot)."""
    return label.rsplit("_", 1)[0]


def _project_px_to_cm(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Proyecta un punto imagen (px) a cm con la homografia px->cm."""
    import cv2

    pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
    cm = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(cm[0]), float(cm[1])


def _assign_landmark(family: str, proj_cm: tuple[float, float]) -> str:
    """Dentro de la familia, el landmark canonico cuya cm esta mas cerca de la proyeccion."""
    cands = FAMILIES[family]
    px, py = proj_cm
    return min(cands, key=lambda n: np.hypot(px - fl.LANDMARK_POINTS[n][0],
                                             py - fl.LANDMARK_POINTS[n][1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--clicks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.core.metric_positions import compute_metric_positions

    # 1) La H real del metodo desplegado (default homography="lines").
    print("calculando H_por_frame (homografia por lineas, igual que el entregable)...")
    res = compute_metric_positions(_abs(args.json), video=_abs(args.clip))
    H_by_frame = res.H_por_frame
    n_valid = sum(1 for h in H_by_frame.values() if h is not None)
    cov = 100.0 * n_valid / max(1, len(H_by_frame))
    print(f"  H valida en {n_valid}/{len(H_by_frame)} frames ({cov:.1f}% cobertura)")

    # 2) clics held-out
    clicks = list(csv.DictReader(open(_abs(args.clicks))))
    rows: list[dict] = []
    skipped_noH = 0
    for c in clicks:
        idx = int(c["frame_index"])
        family = _family_of(c["landmark"])
        if family not in FAMILIES:
            print(f"  AVISO: familia desconocida '{c['landmark']}', se omite")
            continue
        H = H_by_frame.get(idx)
        if H is None:
            skipped_noH += 1
            continue
        px_cm, py_cm = _project_px_to_cm(np.asarray(H, float), float(c["px_x"]), float(c["px_y"]))
        name = _assign_landmark(family, (px_cm, py_cm))  # top/bot por cercania (no del CSV)
        gx, gy = fl.LANDMARK_POINTS[name]  # cm conocido (held-out)
        err = float(np.hypot(px_cm - gx, py_cm - gy))
        rows.append({
            "frame_index": idx, "landmark": name, "clicked_as": c["landmark"],
            "px_x": float(c["px_x"]), "px_y": float(c["px_y"]),
            "proj_x_cm": round(px_cm, 2), "proj_y_cm": round(py_cm, 2),
            "gt_x_cm": gx, "gt_y_cm": gy, "err_cm": round(err, 2),
        })

    if not rows:
        raise SystemExit("no hay clics con H valida; revisa el CSV de clics / frames")

    errs = np.array([r["err_cm"] for r in rows])

    def stats(a):
        return float(np.median(a)), float(np.percentile(a, 90)), float(a.max()), float(a.mean())

    med, p90, mx, mean = stats(errs)

    # 3) salida CSV detalle
    out = _abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 4) resumen por landmark
    print(f"\n=== Error held-out (homografia por lineas) — {len(rows)} mediciones ===")
    print(f"cobertura H: {cov:.1f}%   |   clics sin H (omitidos): {skipped_noH}")
    print(f"GLOBAL  mediana={med:.2f}  p90={p90:.2f}  max={mx:.2f}  media={mean:.2f}  (cm)")
    print("\npor landmark:")
    print(f"  {'landmark':<12} {'n':>3}  {'mediana':>8} {'p90':>7} {'max':>7}")
    for name in sorted({r["landmark"] for r in rows}):
        a = np.array([r["err_cm"] for r in rows if r["landmark"] == name])
        m, p, x, _ = stats(a)
        print(f"  {name:<12} {len(a):>3}  {m:>8.2f} {p:>7.2f} {x:>7.2f}")

    # patron espacial: centro del campo (centro/circulo) vs bordes (areas, distorsion)
    center_e = np.array([r["err_cm"] for r in rows
                         if _family_of(r["landmark"]) in ("center", "circle")])
    edge_e = np.array([r["err_cm"] for r in rows
                       if _family_of(r["landmark"]) in ("penL", "penR")])
    print("\npor zona del campo:")
    if len(center_e):
        m, p, x, _ = stats(center_e)
        print(f"  centro/circulo  n={len(center_e):>2}  mediana={m:5.2f} p90={p:6.2f} max={x:6.2f}")
    if len(edge_e):
        m, p, x, _ = stats(edge_e)
        print(f"  esquinas area   n={len(edge_e):>2}  mediana={m:5.2f} p90={p:6.2f} max={x:6.2f}")

    # misclicks groseros (>40 cm: area medio fuera de cuadro / clic equivocado)
    GROSS = 40.0
    clean = errs[errs <= GROSS]
    n_gross = int((errs > GROSS).sum())
    medc, p90c, mxc, _ = stats(clean) if len(clean) else (med, p90, mx, mean)

    print(f"\nguardado detalle: {out}")
    print("\n>>> para el paper (Tabla 6):")
    print(f"    Held-out landmark error (line homography): "
          f"median {med:.1f} / p90 {p90:.1f} / max {mx:.1f} cm  (n={len(rows)})")
    print(f"    excl. {n_gross} gross misclicks (>{GROSS:.0f} cm): "
          f"median {medc:.1f} / p90 {p90c:.1f} / max {mxc:.1f} cm")
    print(f"    Valid-H coverage: {cov:.1f}% of frames")


if __name__ == "__main__":
    main()
