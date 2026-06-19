# -*- coding: utf-8 -*-
"""Test manual de SDD-B (Kalman como fuente de posiciones). CPU, sin GPU.

(1) ``refine_with_kalman`` conserva H_por_frame/resumen y solo toca el balón;
(2) la oclusión del balón queda rellenada con ``source="predicted"``;
(3) regresión: ``event_shot_goal`` sobre el metric **crudo** es idéntico al de hoy;
(4) guarda conservadora: un cruce de gol solo-predicho NO declara gol.

Uso: python testing/test_kalman_position_source.py
"""
import numpy as np

from src.core.event_shot_goal import compute_shot_vs_goal
from src.core.kalman_kinematics import refine_with_kalman
from src.core.metric_positions import MetricPosition, MetricResult


def _metric_ball_robot():
    """Balón obj 1 (frames 0..10, ocluido 4/5/6) + robot obj 2 (frames 0..5). Sin source (crudo)."""
    pos = []
    for f in range(11):
        xy = None if f in (4, 5, 6) else (10.0 + 5 * f, 20.0 + 2 * f)
        pos.append(MetricPosition(1, "orange_ball", f, xy, "estimated"))
    for f in range(6):
        pos.append(MetricPosition(2, "robot", f, (50.0 + 3 * f, 40.0 - f), "estimated"))
    H = {f: np.eye(3) for f in range(11)}
    return MetricResult(posiciones=pos, resumen={"fps": 30}, H_por_frame=H)


def test_refine_preserves_and_ball_only():
    metric = _metric_ball_robot()
    ref = refine_with_kalman(metric)
    assert ref.H_por_frame is metric.H_por_frame, "H_por_frame debe conservarse"
    assert ref.resumen is metric.resumen, "resumen debe conservarse"
    rob_in = {p.frame_index: p.xy_cm for p in metric.posiciones if p.cls == "robot"}
    rob_out = {p.frame_index: (p.xy_cm, p.source) for p in ref.posiciones if p.cls == "robot"}
    assert all(rob_out[f][0] == rob_in[f] and rob_out[f][1] is None for f in rob_in), \
        "los robots NO deben cambiar (ball_only)"
    print("[1] H/resumen preservados; robots intactos; solo el balón se refina")


def test_occlusion_filled_predicted():
    ref = refine_with_kalman(_metric_ball_robot())
    ball = {p.frame_index: p for p in ref.posiciones if p.cls == "orange_ball"}
    for f in (4, 5, 6):
        assert ball[f].xy_cm is not None, f"oclusión {f} debe quedar rellenada"
        assert ball[f].source == "predicted", f"oclusión {f} debe ser predicted ({ball[f].source})"
    meas = [ball[f].source for f in range(11) if f not in (4, 5, 6)]
    assert all(s in ("measured", "gated") for s in meas), f"medidos: {meas}"
    print(f"[2] oclusión 4/5/6 rellenada como predicted; sources={[ball[f].source for f in range(11)]}")


def _metric_goal(cross_source):
    """Balón cruzando la portería azul (x>=231, boca y~90); 3,4 cruzan. ``cross_source`` en 3,4."""
    xs = {0: 218, 1: 222, 2: 226, 3: 235, 4: 238, 5: 210, 6: 200, 7: 190}
    pos = []
    for f, x in xs.items():
        src = cross_source if f in (3, 4) else ("measured" if cross_source else None)
        pos.append(MetricPosition(1, "orange_ball", f, (float(x), 90.0), "estimated", source=src))
    return MetricResult(posiciones=pos, resumen={"fps": 30})


def _tipos(metric):
    return [(e.tipo, e.zona) for e in compute_shot_vs_goal(metric, route="cm").eventos]


def test_goal_regression_and_guard():
    raw = _tipos(_metric_goal(None))          # crudo (regresión)
    meas = _tipos(_metric_goal("measured"))   # cruce medido
    pred = _tipos(_metric_goal("predicted"))  # cruce solo-predicho
    assert any(t == "gol" for t, _ in raw), "regresión: crudo debe dar GOL (comportamiento actual)"
    assert raw == meas, "crudo y medido deben ser idénticos"
    assert all(t != "gol" for t, _ in pred), "cruce solo-predicho NO debe dar gol"
    assert any(t == "tiro" for t, _ in pred), "cruce solo-predicho debe degradar a tiro"
    print(f"[3] regresión crudo={raw} == medido={meas}")
    print(f"[4] guarda: cruce solo-predicho={pred} (gol suprimido)")


if __name__ == "__main__":
    test_refine_preserves_and_ball_only()
    test_occlusion_filled_predicted()
    test_goal_regression_and_guard()
    print("\nOK: 4/4 tests de kalman_position_source pasaron.")
