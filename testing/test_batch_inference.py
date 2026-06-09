"""Prueba manual de la orquestación de lotes (tarea batch_inference).

Dos partes:

  A) LOCAL (sin GPU): firma de ``run_batch`` + lógica de orquestación (selección desde
     un CSV temporal, skip-done, aislamiento de errores, resumen) con ``run_inference``
     y ``load_sam3`` **monkeypatcheados** — no requiere SAM3.

  B) POD (GPU): lote de los **3 primeros videos del split reservado** (``split=0``) en
     ambos modos, con **video + JSON extendido** (``include_masks``): segmentación con
     cuota y tracking acotado a 300 frames; una segunda corrida verifica skip-done.
     Requiere modelo SAM3 + GPU.

Uso:
    python testing/test_batch_inference.py          # solo Parte A (local)
    python testing/test_batch_inference.py pod       # Parte A + Parte B (en el pod)
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.core.batch as batch  # noqa: E402
import src.core.sam3_loader as sam3_loader  # noqa: E402
import src.data.metadata as metadata  # noqa: E402
from src.core.batch import run_batch  # noqa: E402
from src.core.inference_schema import inference_paths  # noqa: E402

_CSV_HEADER = "id,ruta,nombre,duracion,ancho,alto,fps_average,split"
_CSV_ROWS = [
    "0,data/raw/A.MOV,A.MOV,10,1920,1080,30,0",
    "1,data/raw/B.MOV,B.MOV,10,1920,1080,30,2",
    "2,data/raw/C.MOV,C.MOV,10,1920,1080,30,2",
    "3,data/raw/FAIL.MOV,FAIL.MOV,10,1920,1080,30,2",
    "4,data/raw/D.MOV,D.MOV,10,1920,1080,30,1",
]


def _assert_value_error(call, label: str) -> None:
    try:
        call()
    except ValueError:
        print(f"  [ok] {label} -> ValueError")
        return
    raise AssertionError(f"{label}: se esperaba ValueError y no se levanto")


def part_a_local() -> None:
    """Parte A — firma + orquestación con run_inference/load_sam3 monkeypatcheados."""
    print("== Parte A — local (sin GPU) ==")

    # Firma con defaults esperados.
    params = inspect.signature(run_batch).parameters
    expected = {
        "mode": "segmentation",
        "split": 2,
        "videos": None,
        "sampling": "auto",
        "max_frames": None,
        "include_masks": False,
        "render_video": False,
        "overwrite": False,
    }
    for name, default in expected.items():
        assert name in params, f"run_batch no expone {name}"
        assert (
            params[name].default == default
        ), f"run_batch.{name} default {params[name].default!r} != {default!r}"
    print("  [ok] firma de run_batch con defaults esperados")

    # CSV temporal DENTRO de PROJECT_ROOT (get_abs_path exige rutas relativas).
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as d:
        dtmp = Path(d)
        csv = dtmp / "meta.csv"
        csv.write_text("\n".join([_CSV_HEADER, *_CSV_ROWS]) + "\n", encoding="utf-8")
        rel_csv = csv.relative_to(PROJECT_ROOT).as_posix()
        outdir = dtmp / "outs"

        # Monkeypatch: manifiesto -> CSV temporal, outputs -> dir temporal,
        # load_sam3 -> objeto dummy, run_inference -> fake que registra llamadas.
        orig_meta = metadata._load_metadata_config
        orig_outputs = batch._load_outputs_dir
        orig_sam3 = sam3_loader.load_sam3
        orig_infer = batch.run_inference

        metadata._load_metadata_config = lambda: (None, rel_csv, None, [])
        batch._load_outputs_dir = lambda: str(outdir)
        sam3_loader.load_sam3 = lambda: object()

        calls: list[str] = []

        def fake_run_inference(ruta, **kw):
            calls.append(ruta)
            if "FAIL" in ruta:
                raise RuntimeError("boom")
            jp, _ = inference_paths(Path(ruta).stem, str(outdir))
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text("{}", encoding="utf-8")
            return {"json": jp, "video": None, "index": None}

        batch.run_inference = fake_run_inference

        try:
            # 1) Selección por split (orden por id), lista explícita y errores.
            assert [r[0] for r in batch._select_videos(2, None)] == [1, 2, 3]
            assert [r[0] for r in batch._select_videos(0, None)] == [0]
            assert batch._select_videos(2, ["data/raw/A.MOV"]) == [
                (0, "data/raw/A.MOV")
            ]
            assert [r[0] for r in batch._select_videos(0, [1, 2])] == [1, 2]
            _assert_value_error(
                lambda: batch._select_videos(2, [999]), "id inexistente"
            )
            _assert_value_error(
                lambda: batch._select_videos(2, ["nope"]), "ruta inexistente"
            )
            print("  [ok] selección: split / lista explícita (prioritaria) / errores")

            # 2) Skip-done: pre-crear el JSON de B (id 1) -> skipped, no llama infer.
            jp_b, _ = inference_paths("B", str(outdir))
            jp_b.parent.mkdir(parents=True, exist_ok=True)
            jp_b.write_text("{}", encoding="utf-8")
            calls.clear()
            res = run_batch(videos=[1, 2])
            by_id = {r["id"]: r for r in res}
            assert by_id[1]["status"] == "skipped", "B deberia ser skipped"
            assert by_id[2]["status"] == "done", "C deberia ser done"
            assert "data/raw/B.MOV" not in calls, "no debe invocar infer en skipped"
            # overwrite fuerza reproceso de B.
            calls.clear()
            res = run_batch(videos=[1], overwrite=True)
            assert res[0]["status"] == "done" and "data/raw/B.MOV" in calls
            print("  [ok] skip-done (JSON existente) y overwrite")

            # 3) Aislamiento de errores: FAIL (id 3) -> failed, C (id 2) -> done.
            calls.clear()
            res = run_batch(videos=[2, 3], overwrite=True)
            by_id = {r["id"]: r for r in res}
            assert by_id[3]["status"] == "failed" and by_id[3]["error"]
            assert by_id[2]["status"] == "done"
            assert len(calls) == 2, "el lote debe continuar tras el fallo"
            print("  [ok] aislamiento de errores (un fallo no detiene el lote)")

            # 4) Resumen: forma de cada entrada.
            keys = {"id", "ruta", "status", "json", "video", "error"}
            assert all(keys <= set(r) for r in res), "entradas con forma incompleta"
            print("  [ok] resumen estructurado por video")
        finally:
            metadata._load_metadata_config = orig_meta
            batch._load_outputs_dir = orig_outputs
            sam3_loader.load_sam3 = orig_sam3
            batch.run_inference = orig_infer
    print()


def _pick_reserved_videos(n: int = 3) -> list[str]:
    """Rutas (relativas) de los primeros ``n`` videos del split reservado (0)."""
    import pandas as pd

    from src.data.metadata import _load_metadata_config
    from src.utils import get_abs_path

    _, metadata_csv, _, _ = _load_metadata_config()
    df = pd.read_csv(get_abs_path(metadata_csv)).sort_values("id")
    reserved = df[df["split"] == 0]["ruta"].tolist()
    if len(reserved) < n:
        raise RuntimeError(
            f"Se necesitan >= {n} videos reservados; hay {len(reserved)}."
        )
    return [str(r) for r in reserved[:n]]


def part_b_pod() -> None:
    """Parte B — lote de 3 reservados en el pod (GPU): seg cuota + tracking 300."""
    print("== Parte B — pod (GPU) ==")
    vids = _pick_reserved_videos(3)
    print(f"  videos reservados: {vids}")

    # 1) Segmentación con cuota: video + JSON extendido.
    res = run_batch(
        mode="segmentation",
        videos=vids,
        sampling="quota",
        include_masks=True,
        render_video=True,
    )
    assert all(r["status"] == "done" for r in res), "seg: no todos done"
    for r in res:
        assert r["video"] and Path(r["video"]).exists(), "seg: falta mp4"
        assert Path(r["json"]).exists(), "seg: falta JSON"
    print("  [ok] segmentación cuota -> 3 done con mp4 + JSON")

    # 2) Tracking acotado a 300 frames: video + JSON extendido.
    res = run_batch(
        mode="tracking",
        videos=vids,
        max_frames=300,
        include_masks=True,
        render_video=True,
        overwrite=True,
    )
    assert all(r["status"] == "done" for r in res), "trk: no todos done"
    for r in res:
        assert r["video"] and Path(r["video"]).exists(), "trk: falta mp4"
        assert Path(r["json"]).exists(), "trk: falta JSON"
    print("  [ok] tracking 300 frames -> 3 done con mp4 + JSON")

    # 3) Skip-done: segunda corrida idéntica (sin overwrite) -> 3 skipped.
    res = run_batch(mode="tracking", videos=vids, max_frames=300)
    assert all(r["status"] == "skipped" for r in res), "skip-done fallo"
    print("  [ok] segunda corrida -> 3 skipped (skip-done)\n")


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
    part_a_local()

    if len(sys.argv) > 1 and sys.argv[1] == "pod":
        part_b_pod()
    else:
        print("(Parte B omitida: pasa 'pod' como argumento para correrla en GPU)\n")

    print("== Resultado ==")
    print("  OK: las pruebas de batch_inference pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
