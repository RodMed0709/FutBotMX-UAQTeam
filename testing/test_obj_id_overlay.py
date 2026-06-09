"""Prueba manual del overlay por obj_id (tarea obj_id_overlay).

El post-pase **no requiere SAM3**, así que casi todo se valida **localmente sin GPU**:

  A) LOCAL (sintético): firma; helpers (`_label`, `_text_color`); validación de
     `mode="tracking"`; dibujo por frame (`_compose_frame`) inspeccionando píxeles
     exactos (sin pasar por la codificación mp4): cajas color por clase, filtro de
     clases, warm-up, y `draw_masks` sin `rle` (aviso + degrade). Más un end-to-end que
     escribe un mp4 sobre un video sintético.

  B) JSON REAL (opcional): correr `render_obj_id_overlay` sobre un JSON de tracking
     real (p. ej. de batch_inference); también local, no requiere GPU.

Uso:
    python testing/test_obj_id_overlay.py
    python testing/test_obj_id_overlay.py /ruta/a/un/tracking.json   # Parte B
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.track_overlay import (  # noqa: E402
    _compose_frame,
    _label,
    _text_color,
    render_obj_id_overlay,
)
from src.core.video_writer import open_video_writer  # noqa: E402

H, W, N = 120, 160, 6
ROBOT = (60, 130, 255)
BALL = (10, 35, 80)
FLOOR = (230, 0, 120)
_CLASSES = [
    {"name": "robot", "color": list(ROBOT)},
    {"name": "orange_ball", "color": list(BALL)},
    {"name": "green_floor", "color": list(FLOOR)},
]


def _synthetic_payload(video: str) -> dict:
    """JSON de tracking sintético: robot (id 0), balón (id 1, warm-up en f0), piso (id 2)."""
    frames = []
    tracks: dict[int, dict] = {
        0: {"obj_id": 0, "class": "robot", "observations": []},
        1: {"obj_id": 1, "class": "orange_ball", "observations": []},
        2: {"obj_id": 2, "class": "green_floor", "observations": []},
    }
    for f in range(N):
        robot_box = [10 + f, 10, 30, 30]
        ball_box = [80, 60 + f, 12, 12]
        floor_box = [5, 90, 50, 20]
        # balón en warm-up en el primer frame (obj_id = -1).
        ball_id = -1 if f == 0 else 1
        frames.append(
            {
                "frame_index": f,
                "detections": {
                    "robot": [
                        {
                            "obj_id": 0,
                            "bbox": robot_box,
                            "centroid": [25 + f, 25],
                            "score": 0.9,
                        }
                    ],
                    "orange_ball": [
                        {
                            "obj_id": ball_id,
                            "bbox": ball_box,
                            "centroid": [86, 66 + f],
                            "score": 0.8,
                        }
                    ],
                    "green_floor": [
                        {
                            "obj_id": 2,
                            "bbox": floor_box,
                            "centroid": [30, 100],
                            "score": 0.7,
                        }
                    ],
                },
            }
        )
        tracks[0]["observations"].append(
            {
                "frame_index": f,
                "bbox": robot_box,
                "centroid": [25 + f, 25],
                "score": 0.9,
            }
        )
        if ball_id == 1:
            tracks[1]["observations"].append(
                {
                    "frame_index": f,
                    "bbox": ball_box,
                    "centroid": [86, 66 + f],
                    "score": 0.8,
                }
            )
        tracks[2]["observations"].append(
            {"frame_index": f, "bbox": floor_box, "centroid": [30, 100], "score": 0.7}
        )
    return {
        "schema_version": "1.0",
        "video": video,
        "mode": "tracking",
        "fps": 4,
        "resolution": {"height": H, "width": W},
        "num_frames": N,
        "include_masks": False,
        "classes": ["robot", "orange_ball", "green_floor"],
        "config": {"classes": _CLASSES},
        "frames": frames,
        "tracks": [tracks[0], tracks[1], tracks[2]],
    }


def _has_color(img: np.ndarray, color: tuple) -> bool:
    return bool(np.any(np.all(img == np.array(color, dtype=np.uint8), axis=-1)))


def part_a_local() -> None:
    print("== Parte A — local (sintético, sin GPU) ==")

    # Firma.
    params = inspect.signature(render_obj_id_overlay).parameters
    for name, default in {
        "video_path": None,
        "output_path": None,
        "draw_masks": False,
        "trajectory_window": None,
        "excluded_classes": None,
    }.items():
        assert name in params and params[name].default == default, f"firma: {name}"
    print("  [ok] firma de render_obj_id_overlay")

    # Helpers.
    assert _label("robot", -1) == "robot" and _label("robot", 3) == "robot #3"
    assert _text_color((255, 255, 255)) == (0, 0, 0)
    assert _text_color((0, 0, 0)) == (255, 255, 255)
    print("  [ok] _label (warm-up sin #id) y _text_color (luminancia)")

    payload = _synthetic_payload("dummy.MOV")
    color_map = {c["name"]: tuple(c["color"]) for c in _CLASSES}
    frame_by_index = {f["frame_index"]: f for f in payload["frames"]}
    from src.core.track_overlay import _trajectories_by_obj

    traj = _trajectories_by_obj(payload)
    gray = np.full((H, W, 3), 100, dtype=np.uint8)

    def compose(excluded, draw_masks=False):
        return _compose_frame(
            gray,
            2,
            frame_by_index,
            traj,
            color_map,
            excluded=set(excluded),
            window=60,
            alpha=0.55,
            thickness=2,
            font_scale=0.5,
            draw_masks=draw_masks,
        )

    # Filtro de clases: green_floor excluido -> su color NO aparece; robot SÍ.
    out_excl = compose(["green_floor"])
    assert _has_color(out_excl, ROBOT), "robot deberia dibujarse"
    assert _has_color(out_excl, BALL), "balón deberia dibujarse"
    assert not _has_color(out_excl, FLOOR), "green_floor NO deberia dibujarse"
    print("  [ok] filtro de clases (green_floor no se dibuja; robot/balón sí)")

    # Sin exclusión -> green_floor SÍ aparece.
    out_all = compose([])
    assert _has_color(out_all, FLOOR), "sin exclusión green_floor deberia aparecer"
    print("  [ok] sin exclusión, green_floor se dibuja")

    # draw_masks sin rle -> aviso, no crashea, sigue dibujando cajas.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out_m = compose(["green_floor"], draw_masks=True)
        assert any(
            "rle" in str(x.message) for x in w
        ), "deberia avisar por falta de rle"
    assert _has_color(out_m, ROBOT), "con draw_masks sin rle siguen las cajas"
    print("  [ok] draw_masks sin rle -> aviso + degrade a cajas/estela")

    # Validación de modo: JSON de segmentación -> ValueError (sin escribir).
    with tempfile.TemporaryDirectory() as d:
        seg = Path(d) / "seg.json"
        seg.write_text(json.dumps({"mode": "segmentation"}), encoding="utf-8")
        try:
            render_obj_id_overlay(seg, video_path="nope.MOV")
            raise AssertionError("se esperaba ValueError para mode!=tracking")
        except ValueError:
            print("  [ok] mode='segmentation' -> ValueError")

    # End-to-end: video sintético real -> escribe mp4 no vacío.
    with tempfile.TemporaryDirectory() as d:
        dtmp = Path(d)
        vid = dtmp / "clip.mp4"
        with open_video_writer(vid, fps=4) as append:
            for _ in range(N):
                append(np.full((H, W, 3), 100, dtype=np.uint8))
        pl = _synthetic_payload(str(vid))
        jpath = dtmp / "clip.json"
        jpath.write_text(json.dumps(pl), encoding="utf-8")
        out = render_obj_id_overlay(jpath)
        assert out.name == "clip_obj_id.mp4", f"nombre de salida inesperado: {out.name}"
        assert out.exists() and out.stat().st_size > 0, "mp4 vacío o ausente"
        print("  [ok] end-to-end: escribe clip_obj_id.mp4 (no vacío)")
    print()


def part_b_real(json_path: str) -> None:
    print(f"== Parte B — JSON real: {json_path} ==")
    out = render_obj_id_overlay(json_path)
    assert Path(out).exists() and Path(out).stat().st_size > 0
    print(f"  [ok] overlay escrito en {out}\n")


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
    part_a_local()
    if len(sys.argv) > 1:
        part_b_real(sys.argv[1])
    else:
        print("(Parte B omitida: pasa la ruta de un JSON de tracking real)\n")
    print("== Resultado ==")
    print("  OK: las pruebas de obj_id_overlay pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
