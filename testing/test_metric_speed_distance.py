"""Harness de T4 `metric_speed_distance` (fase_5 · Capa B). Corre en CPU local (sin GPU).

Calcula velocidad (cm/s) y distancia (cm) por `obj_id` desde la salida métrica de T3, y valida
invariantes + casos borde + una viz (curva de velocidad + barras de distancia).

    python testing/test_metric_speed_distance.py [ruta/al/tracks.json]

Por defecto usa el clip del gol de cámara superior (`IMG_9933_5m30`).
"""

import json
import statistics
import sys
from pathlib import Path

from src.core.metric_kinematics import (
    KinematicsResult,
    compute_kinematics,
    write_kinematics_json,
)
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT
    / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)
BALL_CLASSES = {"orange_ball", "ball"}


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _print_table(result: KinematicsResult) -> None:
    print(f"{'obj':>5} {'clase':<13} {'n':>5} {'dur_s':>6} {'dist_cm':>9} "
          f"{'v_med':>7} {'v_max':>7}")
    for o in result.por_obj:
        print(f"{o.obj_id:>5} {o.cls:<13} {o.n_muestras:>5} {o.dur_s:>6.1f} "
              f"{o.dist_cm:>9.1f} {o.v_media_cms:>7.1f} {o.v_max_cms:>7.1f}")


def _plot(result: KinematicsResult, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_v, ax_d) = plt.subplots(2, 1, figsize=(12, 7))
    for o in result.por_obj:
        if o.serie:
            xs = [f for f, _ in o.serie]
            ys = [v for _, v in o.serie]
            ax_v.plot(xs, ys, lw=0.8, label=f"{o.cls} #{o.obj_id}")
    ax_v.set_xlabel("frame")
    ax_v.set_ylabel("velocidad (cm/s)")
    ax_v.set_title("Velocidad por obj_id (suavizada)")
    ax_v.legend(fontsize=6, ncol=4)

    objs = result.por_obj
    ax_d.bar([f"{o.cls[:4]}#{o.obj_id}" for o in objs], [o.dist_cm for o in objs])
    ax_d.set_ylabel("distancia (cm)")
    ax_d.set_title("Distancia recorrida por obj_id")
    ax_d.tick_params(axis="x", rotation=90, labelsize=6)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print("viz:", png_path)


def _edge_cases() -> None:
    """Objeto con <2 muestras y objeto con todos los segmentos outliers (no rompen)."""
    from src.core.metric_kinematics import _kinematics

    una, _ = _kinematics("robot", 1, [(0, (10.0, 10.0))], fps=30.0, max_speed_cms=300.0,
                         win=5, with_series=False)
    assert una.dist_cm == 0.0 and una.v_max_cms == 0.0
    # dos muestras muy lejanas en 1 frame -> velocidad enorme -> outlier
    salto, n_out = _kinematics("robot", 2, [(0, (0.0, 0.0)), (1, (200.0, 0.0))], fps=30.0,
                               max_speed_cms=300.0, win=5, with_series=False)
    assert n_out == 1 and salto.dist_cm == 0.0
    print("casos borde OK")


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    result = compute_kinematics(tracks, with_series=True)  # serie para la viz
    print("resumen:\n" + json.dumps(result.resumen, indent=2, ensure_ascii=False))
    _print_table(result)

    # --- invariantes ---
    for o in result.por_obj:
        assert o.dist_cm >= 0.0, f"distancia negativa en #{o.obj_id}"
        assert o.v_max_cms >= o.v_media_cms - 1e-6, f"v_max<v_media en #{o.obj_id}"
    assert result.resumen["segmentos_outlier_descartados"] >= 0
    # el balón debe tener picos ≥ que la mediana de robots (coherencia cualitativa)
    ball_vmax = [o.v_max_cms for o in result.por_obj if o.cls in BALL_CLASSES]
    robot_vmax = [o.v_max_cms for o in result.por_obj if o.cls == "robot"]
    if ball_vmax and robot_vmax:
        assert max(ball_vmax) >= statistics.median(robot_vmax), \
            "el balón no alcanza la velocidad pico esperada"
    print("invariantes OK")

    _edge_cases()

    stem = tracks.stem
    _plot(result, PROJECT_ROOT / "outputs" / f"metric_speed_distance_{stem}.png")
    out = write_kinematics_json(
        result, PROJECT_ROOT / "outputs" / f"metric_speed_distance_{stem}.json"
    )
    print("escrito:", out)
    print("OK")


if __name__ == "__main__":
    main()
