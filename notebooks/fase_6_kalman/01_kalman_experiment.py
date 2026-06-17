# -*- coding: utf-8 -*-
"""Fase 6 — experimentos del KF en cm (CPU local, desde JSON). Genera las tablas del paper:
 T6.1 recuperación de oclusión (error cm vs largo de hueco; KF vs hold vs lineal),
 T6.2 suavidad de velocidad (Var de aceleración: finite-diff vs KF),
 T6.3 continuidad/cobertura (huecos rellenados),
 T6.4 headline (v_max balón: T4 vs KF),
 T6.5 consistencia (NIS medio).

Insumo: clips cenitales (tracking JSON + .mp4) -> T3 metric_positions (cachea JSON) -> KF.
Uso (pod):  python 01_kalman_experiment.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.core.kalman_kinematics import CLASS_PARAMS, compute_kalman_states
from src.core.kalman_state import KalmanCV
from src.core.metric_kinematics import compute_kinematics
from src.core.metric_positions import MetricPosition, MetricResult
from src.core.minimap_pipeline import _load_tracks_from_json

# UNIDADES = PIXELES (espacio de imagen). El KF corre sobre los centroides del tracking JSON
# directamente (sin homografia/video). La conversion a cm es via la homografia (T3, Ec. en el
# paper); T3 esta roto para los clips re-encodeados del demo (RLE size != frame size), asi que
# el experimento se hace en px: el aporte del KF (oclusion/suavizado/NIS) es unidad-agnostico.
REPO = Path("/workspace/FutBotMX-UAQTeam")
CLIPS = [
    REPO / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json",
    REPO / "outputs/inference/fase5_clips/IMG_9938_min1/IMG_9938_min1.json",
    REPO / "outputs/inference/fase5_clips/IMG_9933_min1/IMG_9933_min1.json",
]
OUT_TABLES = REPO / "assets" / "fase6" / "tables"
OUT_T3 = REPO / "outputs" / "inference" / "fase6_kalman"
GAPS = (1, 2, 3, 5, 8, 12)
STRIDE = 8
WARMUP = 30
MOVING = set(CLASS_PARAMS) - {"robot"}  # clases que filtramos


def get_metric_result(tracks_json: Path) -> MetricResult:
    """Posiciones en PIXELES por (obj_id, frame) desde los centroides/foot del tracking JSON
    (sin homografia/video). Espeja MetricResult para reusar el driver del KF."""
    frame_to_objs, _max = _load_tracks_from_json(tracks_json)
    fps = json.loads(tracks_json.read_text(encoding="utf-8")).get("fps") or 30.0
    pos = []
    for idx in sorted(frame_to_objs):
        for obj_id, cls, foot in frame_to_objs[idx]:
            pos.append(MetricPosition(obj_id, cls, int(idx),
                                      (float(foot[0]), float(foot[1])), "estimated"))
    return MetricResult(posiciones=pos, resumen={"fps": fps, "units": "px"})


def measured_series(result: MetricResult) -> dict[int, tuple[str, list]]:
    """Por obj_id móvil: lista ordenada de (frame, (x,y)) con detección real."""
    by: dict[int, tuple[str, list]] = {}
    for p in result.posiciones:
        if p.cls not in CLASS_PARAMS or p.xy_cm is None:
            continue
        cls, lst = by.setdefault(p.obj_id, (p.cls, []))
        lst.append((p.frame_index, p.xy_cm, p.status_H))
    for _o, (_c, lst) in by.items():
        lst.sort(key=lambda r: r[0])
    return by


# ---------------- 4a: recuperación de oclusión (synthetic hold-out) ----------------
def occlusion_recovery(result: MetricResult, fps: float) -> dict:
    """Para cada clase y largo de hueco g: error de posición (cm) de hold/lineal/kalman."""
    errs: dict[tuple[str, int, str], list] = {}
    for _oid, (cls, lst) in measured_series(result).items():
        params = CLASS_PARAMS[cls]
        n = len(lst)
        if n < WARMUP + max(GAPS) + 2:
            continue
        frames = [r[0] for r in lst]
        xy = {r[0]: np.array(r[1], dtype=float) for r in lst}
        present = set(frames)
        for g in GAPS:
            t = frames[WARMUP]
            while t + g + 1 <= frames[-1]:
                hidden = [t + k for k in range(1, g + 1)]
                if not all(h in present for h in hidden) or (t not in present) or (t - 1 not in present):
                    t += STRIDE
                    continue
                # hold: última posición vista
                last = xy[t]
                # lineal: velocidad por diferencia finita previa
                v_lin = (xy[t] - xy[t - 1]) * fps
                # kalman: warmup desde t-WARMUP, luego predict-only g pasos
                kf = None
                for f in range(t - WARMUP, t + 1):
                    if f not in present:
                        continue
                    if kf is None:
                        kf = KalmanCV(tuple(xy[f]), params)
                    else:
                        kf.predict(1.0 / fps)
                        kf.update(tuple(xy[f]), present and "estimated")
                for k, h in enumerate(hidden, start=1):
                    true = xy[h]
                    errs.setdefault((cls, g, "hold"), []).append(float(np.linalg.norm(last - true)))
                    pred_lin = xy[t] + v_lin * (k / fps)
                    errs.setdefault((cls, g, "linear"), []).append(float(np.linalg.norm(pred_lin - true)))
                    kf.predict(1.0 / fps)
                    errs.setdefault((cls, g, "kalman"), []).append(float(np.linalg.norm(np.array(kf.pos) - true)))
                t += STRIDE
    rows = []
    for (cls, g, method), vals in sorted(errs.items()):
        rows.append({"class": cls, "gap": g, "method": method,
                     "n": len(vals), "mean_err_cm": round(float(np.mean(vals)), 2),
                     "median_err_cm": round(float(np.median(vals)), 2)})
    return {"rows": rows}


# ---------------- 4b: suavidad de velocidad ----------------
def velocity_smoothness(result: MetricResult, kres, fps: float) -> dict:
    rows = []
    # finite-diff (crudo) por obj
    fd_by_cls: dict[str, list] = {}
    for _oid, (cls, lst) in measured_series(result).items():
        if len(lst) < 5:
            continue
        sp = []
        for a, b in zip(lst, lst[1:]):
            dt = (b[0] - a[0]) / fps
            if dt > 0:
                sp.append(float(np.hypot(b[1][0] - a[1][0], b[1][1] - a[1][1])) / dt)
        acc = np.diff(sp) * fps
        if acc.size:
            fd_by_cls.setdefault(cls, []).append(float(np.var(acc)))
    # kalman por obj
    kf_by_cls: dict[str, list] = {}
    for o in kres.por_obj:
        sp = [s.speed_cms for s in o.estados]
        acc = np.diff(sp) * fps
        if acc.size:
            kf_by_cls.setdefault(o.cls, []).append(float(np.var(acc)))
    for cls in sorted(set(fd_by_cls) | set(kf_by_cls)):
        fd = float(np.mean(fd_by_cls.get(cls, [0]))) if fd_by_cls.get(cls) else None
        kf = float(np.mean(kf_by_cls.get(cls, [0]))) if kf_by_cls.get(cls) else None
        red = round(100.0 * (1 - kf / fd), 1) if (fd and kf is not None and fd > 0) else None
        rows.append({"class": cls,
                     "var_accel_finitediff": round(fd, 1) if fd is not None else None,
                     "var_accel_kalman": round(kf, 1) if kf is not None else None,
                     "reduction_pct": red})
    return {"rows": rows}


# ---------------- 4c: continuidad / cobertura ----------------
def continuity(kres) -> dict:
    rows = []
    for o in kres.por_obj:
        cov = round(100.0 * o.n_measured / o.n_frames, 1) if o.n_frames else 0.0
        rows.append({"obj_id": o.obj_id, "class": o.cls, "n_frames": o.n_frames,
                     "measured": o.n_measured, "filled_occlusion": o.n_predicted,
                     "gated": o.n_gated, "coverage_pct_after_KF": round(100.0, 1),
                     "measured_pct": cov})
    return {"rows": rows}


# ---------------- 4d: headline v_max (T4 vs KF) ----------------
def headline(result: MetricResult, kres, fps: float) -> dict:
    t4 = compute_kinematics(result, fps=fps)
    t4_ball = max((o.v_max_cms for o in t4.por_obj if o.cls in ("orange_ball", "ball")), default=0.0)
    kf_ball = max((o.v_max_cms for o in kres.por_obj if o.cls == "orange_ball"), default=0.0)
    return {"rows": [
        {"metric": "ball_vmax_cms", "T4_finitediff": round(t4_ball, 1), "kalman": round(kf_ball, 1)},
    ]}


# ---------------- 4e: NIS ----------------
def nis_consistency(kres) -> dict:
    rows = []
    by: dict[str, list] = {}
    for o in kres.por_obj:
        for s in o.estados:
            if s.nis is not None:
                by.setdefault(o.cls, []).append(s.nis)
    for cls, vals in sorted(by.items()):
        rows.append({"class": cls, "n": len(vals), "mean_nis": round(float(np.mean(vals)), 2),
                     "frac_gt_5.99": round(float(np.mean(np.array(vals) > 5.99)), 3)})
    return {"rows": rows}


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUT_TABLES / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    agg = {"occlusion": [], "smoothness": [], "continuity": [], "headline": [], "nis": []}
    for tracks_json in CLIPS:
        if not tracks_json.exists():
            print(f"[skip] no existe {tracks_json}")
            continue
        stem = tracks_json.stem
        print(f"\n==== {stem} ====")
        try:
            result = get_metric_result(tracks_json)
        except Exception as e:  # noqa: BLE001  -- T3 puede fallar (homografia/clip); seguir con el resto
            print(f"  [T3 FALLO en {stem}: {str(e)[:160]}] -> se omite este clip")
            continue
        fps = result.resumen.get("fps") or 30.0
        kres = compute_kalman_states(result, fps=fps)
        print(f"  objs móviles: {kres.resumen['n_obj']} | "
              f"oclusión rellenada: {kres.resumen['frames_rellenados_oclusion']} frames | "
              f"gated: {kres.resumen['frames_gated']}")
        # T3 estados KF a JSON
        from src.core.kalman_kinematics import write_kalman_states_json
        write_kalman_states_json(kres, OUT_T3 / stem / f"{stem}_kalman_states.json")
        for r in occlusion_recovery(result, fps)["rows"]:
            r["clip"] = stem; agg["occlusion"].append(r)
        for r in velocity_smoothness(result, kres, fps)["rows"]:
            r["clip"] = stem; agg["smoothness"].append(r)
        for r in continuity(kres)["rows"]:
            r["clip"] = stem; agg["continuity"].append(r)
        for r in headline(result, kres, fps)["rows"]:
            r["clip"] = stem; agg["headline"].append(r)
        for r in nis_consistency(kres)["rows"]:
            r["clip"] = stem; agg["nis"].append(r)

    write_csv("T6_1_occlusion_recovery.csv", agg["occlusion"])
    write_csv("T6_2_velocity_smoothness.csv", agg["smoothness"])
    write_csv("T6_3_continuity.csv", agg["continuity"])
    write_csv("T6_4_headline_vmax.csv", agg["headline"])
    write_csv("T6_5_nis.csv", agg["nis"])

    print("\n===== T6.1 OCCLUSION RECOVERY (mean err cm) =====")
    print(f"{'clip':18s}{'class':12s}{'gap':>4s}{'method':>9s}{'n':>6s}{'mean':>8s}")
    for r in agg["occlusion"]:
        print(f"{r['clip']:18s}{r['class']:12s}{r['gap']:>4d}{r['method']:>9s}{r['n']:>6d}{r['mean_err_cm']:>8.2f}")
    print("\n===== T6.2 VELOCITY SMOOTHNESS (Var accel) =====")
    for r in agg["smoothness"]:
        print(f"  {r['clip']:18s}{r['class']:12s} fd={r['var_accel_finitediff']} kf={r['var_accel_kalman']} red={r['reduction_pct']}%")
    print("\n===== T6.4 HEADLINE v_max balón =====")
    for r in agg["headline"]:
        print(f"  {r['clip']:18s} T4={r['T4_finitediff']} KF={r['kalman']}")
    print("\n===== T6.5 NIS =====")
    for r in agg["nis"]:
        print(f"  {r['clip']:18s}{r['class']:12s} mean_NIS={r['mean_nis']} frac>5.99={r['frac_gt_5.99']}")
    print(f"\n-> tablas en {OUT_TABLES}")


if __name__ == "__main__":
    main()
