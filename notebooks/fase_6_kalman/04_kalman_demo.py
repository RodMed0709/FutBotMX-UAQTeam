# -*- coding: utf-8 -*-
"""Fase 6 — demo T7 alimentado por Kalman. Inyecta el MetricResult de estados Kalman
(cm, suavizado, oclusiones rellenadas) a compose_demo, así el panel de velocidad/heatmap usa
el estado Kalman en vez de las diferencias finitas de T4. CPU local.
Uso (pod):  python 04_kalman_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.core.demo_overlay import compose_demo
from src.core.kalman_kinematics import compute_kalman_states, load_metric_result_from_json
from src.core.metric_positions import MetricPosition, MetricResult

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cm_positions_lines import compute_cm_positions_lines  # noqa: E402

REPO = Path("/workspace/FutBotMX-UAQTeam")
TRACKS = REPO / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
OUT = REPO / "outputs/inference/fase6_kalman/IMG_9933_5m30/IMG_9933_5m30_demo_kalman.mp4"


def main() -> None:
    cache = TRACKS.with_name(TRACKS.stem + "_cm_lines.json")
    raw = load_metric_result_from_json(cache) if cache.exists() else compute_cm_positions_lines(TRACKS)
    fps = raw.resumen.get("fps") or 30.0
    kres = compute_kalman_states(raw, fps=fps)
    # MetricResult a partir de los estados Kalman (suavizado, oclusiones rellenadas)
    pos = [MetricPosition(o.obj_id, o.cls, s.frame_index, s.xy_cm, "estimated")
           for o in kres.por_obj for s in o.estados]
    kmet = MetricResult(posiciones=pos, resumen=raw.resumen)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = compose_demo(TRACKS, output_path=OUT, max_seconds=20.0, metric=kmet)
    print(f"[fase6] demo Kalman -> {out}")


if __name__ == "__main__":
    main()
