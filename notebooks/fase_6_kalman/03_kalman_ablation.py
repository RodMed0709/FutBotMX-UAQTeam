# -*- coding: utf-8 -*-
"""Fase 6 — ablación NIS: barre σ_z (R) y σ_a (Q) por clase y elige el que da NIS medio ≈ 2
(filtro consistente con el error real de medición). Clip fiable IMG_9933_5m30.
Uso (pod):  python 03_kalman_ablation.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.core.kalman_kinematics import compute_kalman_states, load_metric_result_from_json
from src.core.kalman_state import KFParams

REPO = Path("/workspace/FutBotMX-UAQTeam")
CACHE = REPO / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30_cm_lines.json"
OUT = REPO / "assets" / "fase6" / "tables"
SIGMA_Z = [2.0, 3.0, 5.0, 8.0, 15.0]
SIGMA_A = {"orange_ball": [400.0, 800.0, 1600.0], "robot": [125.0, 250.0, 500.0]}


def mean_nis_by_class(raw, fps, sz, sa_ball, sa_robot):
    cp = {
        "orange_ball": KFParams(sigma_a=sa_ball, sigma_z=sz, max_gap_frames=15),
        "robot": KFParams(sigma_a=sa_robot, sigma_z=sz, max_gap_frames=30),
        "robot_a": KFParams(sigma_a=sa_robot, sigma_z=sz, max_gap_frames=30),
        "robot_b": KFParams(sigma_a=sa_robot, sigma_z=sz, max_gap_frames=30),
    }
    kres = compute_kalman_states(raw, fps=fps, class_params=cp)
    by = {}
    for o in kres.por_obj:
        for s in o.estados:
            if s.nis is not None:
                by.setdefault(o.cls, []).append(s.nis)
    return {c: float(np.mean(v)) for c, v in by.items() if v}


def main() -> None:
    raw = load_metric_result_from_json(CACHE)
    fps = raw.resumen.get("fps") or 30.0
    rows = []
    for sz in SIGMA_Z:
        for sab in SIGMA_A["orange_ball"]:
            for sar in SIGMA_A["robot"]:
                nis = mean_nis_by_class(raw, fps, sz, sab, sar)
                rows.append({"sigma_z": sz, "sigma_a_ball": sab, "sigma_a_robot": sar,
                             "NIS_ball": round(nis.get("orange_ball", -1), 2),
                             "NIS_robot": round(nis.get("robot", nis.get("robot_a", -1)), 2)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "T6_ablation_nis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # mejor por clase: NIS más cercano a 2 (dof=2)
    def best(key):
        return min((r for r in rows if r[key] > 0), key=lambda r: abs(r[key] - 2.0))
    bb, br = best("NIS_ball"), best("NIS_robot")
    print("=== Ablación NIS (objetivo ~2) — IMG_9933_5m30 ===")
    print(f"{'sz':>5}{'sa_ball':>9}{'sa_rob':>8}{'NIS_ball':>10}{'NIS_robot':>11}")
    for r in rows:
        print(f"{r['sigma_z']:>5}{r['sigma_a_ball']:>9}{r['sigma_a_robot']:>8}"
              f"{r['NIS_ball']:>10}{r['NIS_robot']:>11}")
    print(f"\nMEJOR balón: sigma_z={bb['sigma_z']} sigma_a={bb['sigma_a_ball']} -> NIS={bb['NIS_ball']}")
    print(f"MEJOR robot: sigma_z={br['sigma_z']} sigma_a={br['sigma_a_robot']} -> NIS={br['NIS_robot']}")
    print(f"\n-> {OUT/'T6_ablation_nis.csv'}")


if __name__ == "__main__":
    main()
