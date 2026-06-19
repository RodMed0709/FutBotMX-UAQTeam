"""Prueba manual de la fachada única de inferencia (tarea unified_inference).

Dos partes:

  A) LOCAL (sin GPU): introspeccion de firma de ``run_inference`` (y de la firma
     ampliada de ``run_pipeline``: ``classes``/``bundle``) + validacion de ``mode`` y
     ``sampling`` que falla con ``ValueError`` **antes** de cargar SAM3. No requiere
     modelo ni GPU.

  B) POD (GPU): corre la fachada real en ambos modos sobre un clip corto y verifica el
     retorno unificado (``{"json", "video", "index"}``: ``index`` is ``None`` en
     segmentacion, dict en tracking), el muestreo por modo, el caso OFF+masks y el
     reuso de un ``bundle`` precargado en ambos modos. Requiere modelo SAM3 + GPU.

Uso:
    python testing/test_unified_inference.py          # solo Parte A (local)
    python testing/test_unified_inference.py pod       # Parte A + Parte B (en el pod)
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

from src.core.inference import run_inference  # noqa: E402
from src.core.pipeline import run_pipeline  # noqa: E402


def _assert_value_error(call, label: str) -> None:
    """Verifica que ``call()`` levante ValueError (sin tocar el modelo)."""
    try:
        call()
    except ValueError:
        print(f"  [ok] {label} -> ValueError")
        return
    raise AssertionError(f"{label}: se esperaba ValueError y no se levanto")


def part_a_local() -> None:
    """Parte A — firma de la fachada + validacion temprana (sin GPU)."""
    print("== Parte A — local (sin GPU) ==")

    # Firma de la fachada: parametros y defaults clave.
    params = inspect.signature(run_inference).parameters
    expected = {
        "mode": "segmentation",
        "sampling": "auto",
        "max_frames": None,
        "classes": None,
        "bundle": None,
        "include_masks": False,
        "render_video": True,
    }
    for name, default in expected.items():
        assert name in params, f"run_inference no expone {name}"
        assert (
            params[name].default == default
        ), f"run_inference.{name} default {params[name].default!r} != {default!r}"
    print("  [ok] firma de run_inference con defaults esperados")

    # run_pipeline ahora expone classes y bundle (default None).
    rp = inspect.signature(run_pipeline).parameters
    for name in ("classes", "bundle"):
        assert name in rp, f"run_pipeline no expone {name}"
        assert rp[name].default is None, f"run_pipeline.{name} default != None"
    print("  [ok] run_pipeline expone classes/bundle (default None)")

    # Validacion temprana: estos casos fallan ANTES de cargar SAM3.
    bad = "data/does_not_exist.MOV"  # si la validacion no precede, fallaria distinto
    _assert_value_error(lambda: run_inference(bad, mode="bad"), "mode='bad'")
    _assert_value_error(
        lambda: run_inference(bad, mode="tracking", sampling="quota"),
        "sampling='quota' + tracking",
    )
    _assert_value_error(
        lambda: run_inference(bad, mode="segmentation", sampling="contiguous"),
        "sampling='contiguous' + segmentation",
    )
    _assert_value_error(
        lambda: run_inference(bad, mode="segmentation", sampling="rara"),
        "sampling desconocido",
    )
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
    """Parte B — fachada real en el pod (GPU), ambos modos + reuso de bundle."""
    print("== Parte B — pod (GPU) ==")
    from src.core.sam3_loader import load_sam3

    video = _pick_non_forced_video()
    print(f"  video: {video}")

    # Carga unica del modelo: se reusa en ambos modos (verifica la ampliacion).
    bundle = load_sam3()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # 1) segmentation auto: retorno {"json","video","index": None}, mp4 + JSON.
        res = run_inference(
            video, mode="segmentation", output_path=tmp / "seg.mp4", bundle=bundle
        )
        assert res["video"] is not None and Path(res["video"]).exists(), "seg sin mp4"
        assert Path(res["json"]).exists(), "seg sin JSON"
        assert res["index"] is None, "seg: index deberia ser None"
        print("  [ok] segmentation auto -> mp4 + JSON, index=None")

        # 2) segmentation sampling='all' + render OFF: sin mp4, index None, JSON si.
        res = run_inference(
            video,
            mode="segmentation",
            sampling="all",
            output_path=tmp / "seg_all.mp4",
            render_video=False,
            bundle=bundle,
        )
        assert res["video"] is None and res["index"] is None, "seg all/OFF mal retorno"
        assert not (tmp / "seg_all.mp4").exists(), "seg OFF no debe escribir mp4"
        assert Path(res["json"]).exists(), "seg all/OFF sin JSON"
        print("  [ok] segmentation all + render OFF -> JSON, sin mp4, index=None")

        # 3) tracking auto con cap: index dict no vacio, JSON con frames+tracks.
        res = run_inference(
            video,
            mode="tracking",
            max_frames=6,
            output_path=tmp / "trk.mp4",
            bundle=bundle,
        )
        assert res["video"] is not None and Path(res["video"]).exists(), "trk sin mp4"
        assert isinstance(res["index"], dict), "trk: index deberia ser dict"
        data = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
        assert "frames" in data and "tracks" in data, "trk sin frames+tracks"
        print("  [ok] tracking auto (cap) -> mp4 + JSON (frames+tracks), index dict")

        # 4) tracking render OFF + include_masks ON: rle en JSON, sin mp4.
        res = run_inference(
            video,
            mode="tracking",
            max_frames=4,
            output_path=tmp / "trk_masks.mp4",
            render_video=False,
            include_masks=True,
            bundle=bundle,
        )
        assert res["video"] is None and not (tmp / "trk_masks.mp4").exists()
        data = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
        assert any("rle" in det for det in _iter_dets(data)), "OFF+masks sin rle"
        print("  [ok] tracking render OFF + include_masks ON -> rle sin mp4")

        print("  [ok] bundle precargado reutilizado en segmentation y tracking\n")


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
    part_a_local()

    if len(sys.argv) > 1 and sys.argv[1] == "pod":
        part_b_pod()
    else:
        print("(Parte B omitida: pasa 'pod' como argumento para correrla en GPU)\n")

    print("== Resultado ==")
    print("  OK: las pruebas de unified_inference pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
