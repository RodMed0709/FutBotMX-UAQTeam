# -*- coding: utf-8 -*-
"""Fase 6 — INTEGRACION: el Kalman ALIMENTA las métricas de fase_5 + minimap con incertidumbre.

Construye un `MetricResult` a partir de los ESTADOS Kalman (cm, suavizados, oclusiones
rellenadas) y se lo pasa a las funciones de fase_5 (distancia/velocidad T4, zonas T6, gol
geométrico) SIN modificar su código — solo cambia la fuente de posiciones (Kalman vs T3 crudo).
Además dibuja un minimap con la **trayectoria Kalman + elipse de incertidumbre** en las
oclusiones. CPU local (reusa el cache cm de cm_positions_lines).

Uso (pod):  python 02_kalman_integration.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from src.core.event_goal_geometric import compute_geometric_goals
from src.core.kalman_kinematics import compute_kalman_states, load_metric_result_from_json
from src.core.metric_field_zones import compute_field_zones
from src.core.metric_kinematics import compute_kinematics
from src.core.metric_positions import MetricPosition, MetricResult, write_metric_positions_json
from src.core import field_template as ft

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cm_positions_lines import compute_cm_positions_lines  # noqa: E402

REPO = Path("/workspace/FutBotMX-UAQTeam")
CLIPS = [
    REPO / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json",
]
OUT_FIG = REPO / "assets" / "fase6" / "figures"
OUT_TAB = REPO / "assets" / "fase6" / "tables"


def get_cm(tracks_json: Path) -> MetricResult:
    cache = tracks_json.with_name(tracks_json.stem + "_cm_lines.json")
    if cache.exists():
        return load_metric_result_from_json(cache)
    res = compute_cm_positions_lines(tracks_json)
    write_metric_positions_json(res, cache)
    return res


def kalman_metric(kres) -> MetricResult:
    """MetricResult a partir de los estados Kalman (cm, suavizados, oclusiones rellenadas)."""
    pos = []
    for o in kres.por_obj:
        for s in o.estados:
            pos.append(MetricPosition(o.obj_id, o.cls, s.frame_index, s.xy_cm, "estimated"))
    return MetricResult(posiciones=pos, resumen=kres.resumen)


def ball_vmax(kin) -> float:
    return max((o.v_max_cms for o in kin.por_obj if o.cls in ("orange_ball", "ball")), default=0.0)


def render_minimap(kres, raw: MetricResult, out_png: Path) -> None:
    """Minimap: trayectoria Kalman (gruesa, rellena oclusión) + elipse de incertidumbre +
    posiciones crudas T3 (puntos tenues) para contraste."""
    import cv2

    canvas, to_px = ft.render_field(scale=2.2, margin_cm=10.0)
    # crudas T3 (tenue)
    raw_by = {}
    for p in raw.posiciones:
        if p.xy_cm is not None and p.cls in ("orange_ball", "robot", "robot_a", "robot_b"):
            raw_by.setdefault(p.obj_id, []).append(p.xy_cm)
    for _o, pts in raw_by.items():
        for xy in pts:
            cv2.circle(canvas, to_px(xy), 2, (180, 180, 180), -1, cv2.LINE_AA)
    # Kalman (color por clase) + elipses de incertidumbre en frames 'predicted'
    for o in kres.por_obj:
        col = (255, 80, 0) if o.cls == "orange_ball" else (0, 90, 230)  # RGB
        ps = [to_px(s.xy_cm) for s in o.estados]
        for j in range(1, len(ps)):
            cv2.line(canvas, ps[j - 1], ps[j], col, 2, cv2.LINE_AA)
        for s, p in zip(o.estados, ps):
            if s.source == "predicted":
                r = max(2, int(round(s.pos_sigma_cm * 2.2)))
                cv2.circle(canvas, p, r, (255, 0, 0), 1, cv2.LINE_AA)  # elipse incertidumbre
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    rows = []
    for tracks_json in CLIPS:
        if not tracks_json.exists():
            print(f"[skip] {tracks_json}"); continue
        stem = tracks_json.stem
        print(f"\n==== {stem} ====")
        raw = get_cm(tracks_json)
        fps = raw.resumen.get("fps") or 30.0
        kres = compute_kalman_states(raw, fps=fps)
        kmet = kalman_metric(kres)

        # --- métricas: crudo (T3/T4) vs Kalman-fed ---
        kin_raw, kin_kf = compute_kinematics(raw, fps=fps), compute_kinematics(kmet, fps=fps)
        goals_raw = compute_geometric_goals(raw, fps=fps)
        goals_kf = compute_geometric_goals(kmet, fps=fps)
        zon_raw = compute_field_zones(tracks_json, metric=raw, fps=fps)
        zon_kf = compute_field_zones(tracks_json, metric=kmet, fps=fps)

        def half(z):
            return z.por_esquema["mitades"]["presencia"]["ball"]
        row = {
            "clip": stem,
            "ball_vmax_raw": ball_vmax(kin_raw), "ball_vmax_kalman": ball_vmax(kin_kf),
            "goals_raw": goals_raw.resumen.get("total_eventos"),
            "goals_kalman": goals_kf.resumen.get("total_eventos"),
            "ball_azul_raw": round(half(zon_raw).get("azul", 0.0), 1),
            "ball_azul_kalman": round(half(zon_kf).get("azul", 0.0), 1),
            "occlusion_filled": kres.resumen["frames_rellenados_oclusion"],
        }
        rows.append(row)
        render_minimap(kres, raw, OUT_FIG / f"{stem}_kalman_minimap.png")
        print(f"  v_max balón: raw {row['ball_vmax_raw']:.1f} -> KF {row['ball_vmax_kalman']:.1f} cm/s")
        print(f"  goles: raw {row['goals_raw']} -> KF {row['goals_kalman']}")
        print(f"  balón mitad azul: raw {row['ball_azul_raw']}% -> KF {row['ball_azul_kalman']}%")
        print(f"  oclusión rellenada: {row['occlusion_filled']} frames | minimap -> {stem}_kalman_minimap.png")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUT_TAB / "T6_6_kalman_fed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n-> tabla {OUT_TAB/'T6_6_kalman_fed_metrics.csv'} + figuras en {OUT_FIG}")


if __name__ == "__main__":
    main()
