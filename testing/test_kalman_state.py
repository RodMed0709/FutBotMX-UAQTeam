# -*- coding: utf-8 -*-
"""Test manual del KF (CPU, sin GPU). Trayectoria CV sintética + ruido:
(1) velocidad recuperada ≈ verdadera; (2) hueco -> error acotado; (3) NIS ≈ 2.
Uso: python testing/test_kalman_state.py
"""
import numpy as np

from src.core.kalman_state import KFParams, KalmanCV, run_kalman_on_track

FPS = 30.0
rng = np.random.default_rng(42)


def _truth(n=120, vx=120.0, vy=-40.0):
    """Posiciones CV verdaderas (cm) a FPS, vel constante (cm/s)."""
    t = np.arange(n) / FPS
    return np.stack([50 + vx * t, 90 + vy * t], axis=1), (vx, vy)


def test_velocity_recovery():
    truth, (vx, vy) = _truth()
    meas = truth + rng.normal(0, 15.0, truth.shape)  # ruido ~ sigma_z
    params = KFParams(sigma_a=250.0, sigma_z=15.0)
    samples = [(i, (float(meas[i, 0]), float(meas[i, 1])), "estimated") for i in range(len(meas))]
    states = run_kalman_on_track(samples, "robot_a", 1, FPS, params)
    vfx, vfy = states[-1].vxy_cms
    err = np.hypot(vfx - vx, vfy - vy)
    print(f"[1] vel recuperada=({vfx:.1f},{vfy:.1f}) vs verdadera=({vx},{vy})  err={err:.1f} cm/s")
    assert err < 40.0, f"velocidad mal recuperada (err {err:.1f})"


def test_occlusion_bounded():
    truth, _ = _truth()
    params = KFParams(sigma_a=250.0, sigma_z=15.0, max_gap_frames=20)
    # ocultar frames 60..70
    samples = []
    for i in range(len(truth)):
        xy = None if 60 <= i <= 70 else (float(truth[i, 0]), float(truth[i, 1]))
        samples.append((i, xy, "estimated"))
    states = run_kalman_on_track(samples, "robot_a", 1, FPS, params)
    by_f = {s.frame_index: s for s in states}
    s65 = by_f[65]
    err = np.hypot(s65.xy_cm[0] - truth[65, 0], s65.xy_cm[1] - truth[65, 1])
    print(f"[2] frame 65 ocluido: pred err={err:.1f} cm, sigma={s65.pos_sigma_cm:.1f}, source={s65.source}")
    assert s65.source == "predicted"
    assert err < 40.0, f"predicción de oclusión mala (err {err:.1f})"
    assert s65.pos_sigma_cm > by_f[59].pos_sigma_cm, "la incertidumbre debe crecer en el hueco"


def test_nis_consistency():
    truth, _ = _truth(n=200)
    meas = truth + rng.normal(0, 15.0, truth.shape)
    params = KFParams(sigma_a=250.0, sigma_z=15.0)
    samples = [(i, (float(meas[i, 0]), float(meas[i, 1])), "estimated") for i in range(len(meas))]
    states = run_kalman_on_track(samples, "robot_a", 1, FPS, params)
    nis = [s.nis for s in states if s.nis is not None]
    m = float(np.mean(nis[5:]))  # descartar transitorio
    print(f"[3] NIS medio={m:.2f} (esperado ~2)")
    assert 0.8 < m < 5.0, f"NIS fuera de rango ({m:.2f}) -> R/Q mal calibrados"


if __name__ == "__main__":
    test_velocity_recovery()
    test_occlusion_bounded()
    test_nis_consistency()
    print("\nOK: 3/3 tests del KF pasaron.")
