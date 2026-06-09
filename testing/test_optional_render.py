"""Prueba manual del render de mp4 opcional (tarea optional_render).

Dos partes:

  A) LOCAL (sin GPU): introspeccion de firma — ``run_pipeline`` y ``track_video``
     exponen el parametro ``render_video`` con default ``True``. No requiere SAM3.

  B) POD (GPU): corre los orquestadores reales con render ON/OFF en ambos modos y
     verifica que (a) el JSON existe siempre, (b) el mp4 existe solo con render ON,
     (c) el retorno trae ``"video"`` = ruta/``None`` segun el flag, y (d) el caso
     ``render_video=False, include_masks=True`` produce JSON con ``rle`` y sin mp4.
     Requiere modelo SAM3 + GPU.

Uso:
    python testing/test_optional_render.py          # solo Parte A (local)
    python testing/test_optional_render.py pod       # Parte A + Parte B (en el pod)
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline import run_pipeline  # noqa: E402
from src.core.tracking import track_video  # noqa: E402


def part_a_local() -> None:
    """Parte A — la firma de ambos orquestadores incluye render_video=True."""
    print("== Parte A — local (sin GPU) ==")
    for fn in (run_pipeline, track_video):
        params = inspect.signature(fn).parameters
        assert "render_video" in params, f"{fn.__name__} no expone render_video"
        default = params["render_video"].default
        assert default is True, f"{fn.__name__}.render_video default != True: {default}"
        print(f"  [ok] {fn.__name__}(render_video=True) por defecto")
    print()


def _pick_non_forced_video() -> str:
    """Ruta (relativa) del video de menor id NO forzado a testing."""
    import pandas as pd

    from src.data.eval_frames import _load_eval_frames_config
    from src.data.metadata import _load_metadata_config
    from src.utils import get_abs_path

    _, metadata_csv, _, _ = _load_metadata_config()
    _, _, _, forced_testing = _load_eval_frames_config()
    df = pd.read_csv(get_abs_path(metadata_csv)).sort_values("id")
    forced = set(forced_testing)
    for ruta in df["ruta"]:
        if ruta not in forced:
            return ruta
    raise RuntimeError("No se encontro ningun video no-forzado en db_metadata.csv.")


def _iter_dets(data: dict):
    """Itera todas las detecciones de la vista frame-indexed de un payload."""
    for f in data["frames"]:
        for dets in f["detections"].values():
            yield from dets


def part_b_pod() -> None:
    """Parte B — orquestadores reales en el pod (GPU), render ON/OFF."""
    print("== Parte B — pod (GPU) ==")
    video = _pick_non_forced_video()
    print(f"  video: {video}")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # 1) Seg-only ON: mp4 existe, JSON existe, retorno trae ruta de video.
        res = run_pipeline(video, output_path=tmp / "seg_on.mp4", render_video=True)
        assert (
            res["video"] is not None and Path(res["video"]).exists()
        ), "seg ON sin mp4"
        assert Path(res["json"]).exists(), "seg ON sin JSON"
        print("  [ok] seg-only render ON -> mp4 + JSON")

        # 2) Seg-only OFF: mp4 NO existe, JSON si, retorno "video" is None.
        res = run_pipeline(video, output_path=tmp / "seg_off.mp4", render_video=False)
        assert res["video"] is None, "seg OFF deberia devolver video=None"
        assert not (tmp / "seg_off.mp4").exists(), "seg OFF no debe escribir mp4"
        assert Path(res["json"]).exists(), "seg OFF sin JSON"
        print("  [ok] seg-only render OFF -> JSON, sin mp4, video=None")

        # 3) Tracking ON: mp4 existe, JSON unificado (frames+tracks), retorno con ruta.
        res = track_video(
            video, output_path=tmp / "trk_on.mp4", max_frames=6, render_video=True
        )
        assert (
            res["video"] is not None and Path(res["video"]).exists()
        ), "trk ON sin mp4"
        data = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
        assert "frames" in data and "tracks" in data, "trk ON sin frames+tracks"
        print("  [ok] tracking render ON -> mp4 + JSON (frames+tracks)")

        # 4) Tracking OFF: mp4 NO existe, JSON unificado si, retorno "video" is None.
        res = track_video(
            video, output_path=tmp / "trk_off.mp4", max_frames=6, render_video=False
        )
        assert res["video"] is None, "trk OFF deberia devolver video=None"
        assert not (tmp / "trk_off.mp4").exists(), "trk OFF no debe escribir mp4"
        data = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
        assert "frames" in data and "tracks" in data, "trk OFF sin frames+tracks"
        print("  [ok] tracking render OFF -> JSON (frames+tracks), sin mp4, video=None")

        # 5) Ortogonalidad: render OFF + include_masks ON -> rle en JSON, sin mp4.
        res = track_video(
            video,
            output_path=tmp / "trk_masks.mp4",
            max_frames=4,
            render_video=False,
            include_masks=True,
        )
        assert res["video"] is None and not (tmp / "trk_masks.mp4").exists()
        data = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
        assert any("rle" in det for det in _iter_dets(data)), "OFF+masks sin rle"
        print("  [ok] render OFF + include_masks ON -> rle sin mp4\n")


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
    part_a_local()

    if len(sys.argv) > 1 and sys.argv[1] == "pod":
        part_b_pod()
    else:
        print("(Parte B omitida: pasa 'pod' como argumento para correrla en GPU)\n")

    print("== Resultado ==")
    print("  OK: las pruebas de optional_render pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
