"""Smoke test del hub `main.py` — local, SIN GPU (no carga SAM3).

Ejercita la lógica de consola/orquestación que no requiere inferencia:
validación de entrada, selección por defecto, derivación de rutas nativas e
idempotencia (reuso) de la etapa de inferencia.

Uso:
    python testing/test_main_hub.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

from src.utils import PROJECT_ROOT

# ``main.py`` vive en la raíz (script, no paquete instalado): asegúralo en el path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import main as hub  # noqa: E402

console = Console()
OK, FAIL = "[green]OK[/]", "[red]FAIL[/]"


def _find_real_video() -> Path | None:
    ddir = PROJECT_ROOT / "data" / "raw"
    if not ddir.exists():
        return None
    vids = sorted({*ddir.rglob("*.MOV"), *ddir.rglob("*.mov")})
    return vids[0] if vids else None


def test_validate_rejects_missing():
    try:
        hub.validate_video("no/existe/__nope__.MOV", console)
    except SystemExit as e:
        assert e.code == 2
        console.print(f"{OK} validate_video rechaza ruta inexistente")
        return
    raise AssertionError("debió rechazar la ruta inexistente")


def test_validate_rejects_non_video(tmp_txt: Path):
    try:
        hub.validate_video(str(tmp_txt), console)
    except SystemExit as e:
        assert e.code == 2
        console.print(f"{OK} validate_video rechaza no-video (.txt)")
        return
    raise AssertionError("debió rechazar el .txt")


def test_validate_accepts_real_video():
    vid = _find_real_video()
    if vid is None:
        console.print("[yellow]~ sin .MOV en data/raw; salto el caso de aceptación[/]")
        return
    out = hub.validate_video(str(vid), console)
    assert out == vid and out.exists()
    console.print(f"{OK} validate_video acepta un .MOV real: {vid.name}")


def test_choose_default():
    choice = hub.choose_pipeline(
        default=True, vista_arg=None, config={}, console=console
    )
    assert choice.default is True
    assert choice.tracker == hub.DEFAULT_TRACKER
    assert choice.detector == hub.DEFAULT_DETECTOR_FALLBACK  # config vacío => fallback
    assert choice.want_overlays is False
    assert choice.vista == "superior"  # default de vista (P2)
    # --vista explícito prevalece incluso en --default:
    gen = hub.choose_pipeline(
        default=True, vista_arg="generica", config={}, console=console
    )
    assert gen.vista == "generica"
    console.print(
        f"{OK} choose_pipeline(--default) sin prompts: vista superior/generica"
    )


def test_plan_outputs():
    paths = hub.plan_outputs("CLIP_X", "sam3_text+bytetrack", "outputs")
    tj = str(paths["tracking_json"])
    assert tj.endswith("outputs/inference/sam3_text+bytetrack/CLIP_X/CLIP_X.json"), tj
    assert str(paths["obj_overlay"]).endswith("CLIP_X_obj_id.mp4")
    assert str(paths["seg_json"]).endswith(
        "outputs/inference/sam3_text+bytetrack/seg/CLIP_X/CLIP_X.json"
    )
    assert str(paths["broadcast_mp4"]).endswith(
        "outputs/eventos/CLIP_X/CLIP_X_broadcast.mp4"
    )
    console.print(f"{OK} plan_outputs arma las rutas nativas esperadas")


def test_stage_inference_reuse():
    """Con el JSON ya presente, stage_inference reusa SIN importar SAM3."""
    stem = "__smoke_main_hub_fixture__"
    run_label = "sam3_text+bytetrack"
    paths = hub.plan_outputs(stem, run_label, "outputs")
    tj = paths["tracking_json"]
    tj.parent.mkdir(parents=True, exist_ok=True)
    tj.write_text('{"mode": "tracking", "tracks": []}', encoding="utf-8")
    try:
        assert "sam3" not in sys.modules, "SAM3 no debería estar importado todavía"
        choice = hub.PipelineChoice("sam3_text", "bytetrack", False, True)
        res = hub.stage_inference(
            Path("dummy.MOV"),
            choice,
            run_label,
            paths,
            overwrite=False,
            console=console,
        )
        assert res.status == "reusado", res.status
        assert (
            "sam3" not in sys.modules
        ), "stage_inference no debe cargar SAM3 al reusar"
        console.print(f"{OK} stage_inference reusa sin cargar SAM3")
    finally:
        tj.unlink(missing_ok=True)
        # limpia el árbol temporal del fixture
        for parent in (tj.parent, tj.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                pass


def test_stage_broadcast_generica_skips():
    """Con vista 'generica', stage_broadcast omite sin calcular homografía."""
    paths = hub.plan_outputs("CLIP_X", "sam3_text+bytetrack", "outputs")
    choice = hub.PipelineChoice("sam3_text", "bytetrack", False, True, vista="generica")
    res = hub.stage_broadcast(
        Path("dummy.MOV"), choice, paths, overwrite=False, console=console
    )
    assert res.status == "omitido", res.status
    assert "genérica" in res.detail
    console.print(
        f"{OK} stage_broadcast con vista 'generica' ⇒ omitido (sin homografía)"
    )


def main() -> int:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        tmp_txt = Path(f.name)
    try:
        test_validate_rejects_missing()
        test_validate_rejects_non_video(tmp_txt)
        test_validate_accepts_real_video()
        test_choose_default()
        test_plan_outputs()
        test_stage_inference_reuse()
        test_stage_broadcast_generica_skips()
    finally:
        tmp_txt.unlink(missing_ok=True)
    console.print("\n[bold green]Todos los smoke checks pasaron.[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
