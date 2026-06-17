# -*- coding: utf-8 -*-
"""Fase 6 / SDD-B (T6) — Ablación de eventos: goles y v_max del balón CON vs SIN Kalman.

Sobre el clip fiable ``IMG_9933_5m30`` (cámara superior), corre la capa de eventos con las
posiciones **crudas** (T3) y con las **refinadas por Kalman** (``apply_kalman_to_metric``), y
compara: (i) goles/tiros detectados (``event_shot_goal``, ruta cm, estricto), (ii) ``v_max`` del
balón (diferencias finitas T4 vs Kalman), (iii) frames de oclusión rellenados. CPU local.

Uso (local o pod):  python notebooks/fase_6_kalman/07_ablation_events_kalman.py
"""
from __future__ import annotations

import json

from src.core.event_shot_goal import compute_shot_vs_goal
from src.core.kalman_kinematics import apply_kalman_to_metric, compute_kalman_states
from src.core.metric_kinematics import compute_kinematics
from src.core.metric_positions import compute_metric_positions
from src.utils import PROJECT_ROOT

TRACKS = PROJECT_ROOT / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
OUT = PROJECT_ROOT / "assets/fase6/ablation_events_kalman.json"
BALL = {"orange_ball", "ball"}


def _resumen_eventos(eventos) -> dict:
    return {
        "goles": sum(1 for e in eventos if e.tipo == "gol"),
        "tiros": sum(1 for e in eventos if e.tipo == "tiro"),
        "detalle": [(e.tipo, e.zona, e.frame_inicio, e.frame_fin) for e in eventos],
    }


def main() -> None:
    if not TRACKS.exists():
        raise FileNotFoundError(f"no está el JSON de tracking: {TRACKS}")
    fps = json.loads(TRACKS.read_text(encoding="utf-8")).get("fps")

    print(f"[T6] homografía + posiciones cm (líneas) sobre {TRACKS.name}…")
    metric = compute_metric_positions(TRACKS)
    fps = fps or metric.resumen.get("fps")

    # --- SIN Kalman (crudo / actual) ---
    eventos_raw = compute_shot_vs_goal(metric, route="cm").eventos
    kin = compute_kinematics(metric, fps=fps)
    vmax_finite = max((o.v_max_cms for o in kin.por_obj if o.cls in BALL), default=0.0)

    # --- CON Kalman ---
    kres = compute_kalman_states(metric, fps=fps)
    metric_k = apply_kalman_to_metric(metric, kres)
    eventos_kal = compute_shot_vs_goal(metric_k, route="cm").eventos
    vmax_kalman = max((o.v_max_cms for o in kres.por_obj if o.cls in BALL), default=0.0)
    n_predicted = kres.resumen.get("frames_rellenados_oclusion", 0)

    tabla = {
        "clip": TRACKS.stem,
        "fps": fps,
        "sin_kalman": {**_resumen_eventos(eventos_raw), "v_max_balon_cms": round(vmax_finite, 1)},
        "con_kalman": {**_resumen_eventos(eventos_kal), "v_max_balon_cms": round(vmax_kalman, 1)},
        "frames_oclusion_rellenados": n_predicted,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tabla, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ABLACIÓN EVENTOS (IMG_9933_5m30) ===")
    print(f"{'':18}{'SIN Kalman':>14}{'CON Kalman':>14}")
    print(f"{'goles':18}{tabla['sin_kalman']['goles']:>14}{tabla['con_kalman']['goles']:>14}")
    print(f"{'tiros':18}{tabla['sin_kalman']['tiros']:>14}{tabla['con_kalman']['tiros']:>14}")
    print(f"{'v_max balon cm/s':18}{vmax_finite:>14.1f}{vmax_kalman:>14.1f}")
    print(f"{'oclusion rellena':18}{'-':>14}{n_predicted:>14}")
    print(f"\n[T6] tabla escrita -> {OUT}")


if __name__ == "__main__":
    main()
