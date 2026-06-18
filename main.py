#!/usr/bin/env python3
"""Hub de consola del pipeline FutBot MX — corrida end-to-end sobre **un video**.

Punto de entrada único y reproducible que encadena el pipeline completo
(inferencia → [overlays] → homografía/eventos → video de espectador) sobre un
video de entrada. **Solo lee** el video (no recorta clips); por costo se
recomiendan clips cortos. Es **idempotente** (reusa lo ya corrido sin rehacer lo
caro) y **reporta** dónde quedó cada artefacto, sin mover nada.

Uso:
    python main.py <ruta_video>                # interactivo (pregunta piezas)
    python main.py <ruta_video> --default      # config por defecto, sin preguntar
    python main.py <ruta_video> --overwrite    # fuerza re-correr todo

Fijos del entregable (no se preguntan): homografía por líneas, Kalman ON, gol
estricto, broadcast layout 2. Salida destacada: el video de espectador.

SDD: ``.specs/main_hub/``. Los imports pesados (torch/cv2/SAM3) son perezosos.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.utils import PROJECT_ROOT, get_abs_path

# --- constantes del hub --------------------------------------------------------
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}
DEFAULT_TRACKER = "bytetrack"
DEFAULT_DETECTOR_FALLBACK = "sam3_text"
LONG_VIDEO_SECS = 60.0  # umbral de advertencia "el pipeline es costoso"


VISTAS = ("superior", "generica")
DEFAULT_VISTA = "superior"


@dataclass
class PipelineChoice:
    detector: str
    tracker: str
    want_overlays: bool
    default: bool
    vista: str = DEFAULT_VISTA  # "superior" habilita homografía/eventos/broadcast


@dataclass
class StageResult:
    status: str  # "generado" | "reusado" | "omitido" | "fallido"
    paths: dict[str, Path] = field(default_factory=dict)
    detail: str = ""


# --- consola (rich) ------------------------------------------------------------


def _console():
    from rich.console import Console

    return Console()


# --- config --------------------------------------------------------------------


def _load_env(env_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _load_config() -> dict:
    """Carga el config activo (vía CONFIG_FILENAME del .env). Cero rutas absolutas."""
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontró CONFIG_FILENAME en el .env.")
    return json.loads(get_abs_path(f"configs/{config_filename}").read_text("utf-8"))


def _outputs_dir(config: dict) -> str:
    return config.get("working_dirs", {}).get("outputs_dir", "outputs")


def _default_detector(config: dict) -> str:
    """Detector por defecto del proyecto: clave ``detector`` del config o fallback."""
    return config.get("detector") or DEFAULT_DETECTOR_FALLBACK


# --- T3: validación de la entrada ---------------------------------------------


def validate_video(video_arg: str, console) -> Path:
    """Valida que ``video_arg`` sea un video usable. Devuelve la ruta absoluta.

    Falla con ``SystemExit(2)`` y mensaje claro **antes** de cargar nada pesado.
    """
    p = Path(video_arg)
    try:
        abs_p = p if p.is_absolute() else get_abs_path(str(p))
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Ruta inválida:[/] {exc}")
        raise SystemExit(2)

    if not abs_p.is_file():
        console.print(f"[bold red]No es un archivo:[/] {abs_p}")
        raise SystemExit(2)
    if abs_p.suffix.lower() not in VIDEO_EXTS:
        console.print(
            f"[bold red]Extensión no soportada:[/] '{abs_p.suffix}' "
            f"(usa una de {sorted(VIDEO_EXTS)})"
        )
        raise SystemExit(2)

    # Abrir con cv2 (perezoso) y comprobar ≥1 frame.
    from src.core.frame_extraction import get_frame_count, get_video_fps

    try:
        n_frames = get_frame_count(abs_p)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]No se pudo abrir el video con cv2:[/] {exc}")
        raise SystemExit(2)
    if n_frames < 1:
        console.print(f"[bold red]El video no tiene frames legibles:[/] {abs_p}")
        raise SystemExit(2)

    try:
        fps = get_video_fps(abs_p)
        secs = n_frames / fps if fps else 0.0
    except Exception:  # noqa: BLE001
        secs = 0.0
    if secs > LONG_VIDEO_SECS:
        console.print(
            f"[yellow]⚠ Video largo (~{secs:.0f}s, {n_frames} frames). El pipeline es "
            f"costoso; se recomiendan clips cortos. Continuando…[/]"
        )
    return abs_p


# --- T4: selección de piezas ---------------------------------------------------


def choose_pipeline(
    default: bool, vista_arg: str | None, config: dict, console
) -> PipelineChoice:
    """Decide detector/tracker/vista/overlays. ``--default`` no pregunta nada.

    ``vista_arg`` (de ``--vista``) prevalece si viene; si no, en modo interactivo se
    pregunta y en ``--default``/no-TTY se asume ``superior``.
    """
    if default:
        return PipelineChoice(
            detector=_default_detector(config),
            tracker=DEFAULT_TRACKER,
            want_overlays=False,
            default=True,
            vista=vista_arg or DEFAULT_VISTA,
        )

    if not sys.stdin.isatty():
        console.print(
            "[bold red]Terminal no interactiva:[/] usa [bold]--default[/] para correr "
            "sin preguntas."
        )
        raise SystemExit(2)

    import questionary

    from src.core.detectors import _DETECTORS
    from src.core.trackers import KNOWN_TRACKERS

    detector = questionary.select(
        "Detector / segmentador:", choices=list(_DETECTORS)
    ).ask()
    tracker = questionary.select("Tracker:", choices=list(KNOWN_TRACKERS)).ask()
    if vista_arg is not None:
        vista = vista_arg
    else:
        vista = questionary.select(
            "Vista de cámara (superior habilita homografía/eventos/broadcast):",
            choices=list(VISTAS),
        ).ask()
    want_overlays = questionary.confirm(
        "¿Generar overlays individuales (segmentación + tracking)?", default=False
    ).ask()

    if detector is None or tracker is None or vista is None or want_overlays is None:
        console.print("[bold red]Selección cancelada.[/]")
        raise SystemExit(130)

    return PipelineChoice(
        detector=detector,
        tracker=tracker,
        want_overlays=bool(want_overlays),
        default=False,
        vista=vista,
    )


# --- T5: rutas nativas ---------------------------------------------------------


def derive_run_label(choice: PipelineChoice) -> str:
    return f"{choice.detector}+{choice.tracker}"


def plan_outputs(stem: str, run_label: str, outputs_dir: str) -> dict[str, Path]:
    """Rutas **nativas** esperadas de cada artefacto (no crea carpetas)."""
    from src.core.events_schema import events_paths
    from src.core.inference_schema import inference_paths

    tracking_json, tracking_video = inference_paths(
        stem, outputs_dir, namespace=run_label
    )
    seg_json, seg_video = inference_paths(
        stem, outputs_dir, namespace=f"{run_label}/seg"
    )
    return {
        "tracking_json": tracking_json,
        "tracking_video": tracking_video,
        "obj_overlay": tracking_json.with_name(f"{stem}_obj_id.mp4"),
        "seg_json": seg_json,
        "seg_video": seg_video,
        "broadcast_mp4": events_paths(
            stem, "broadcast", "mp4", outputs_dir=outputs_dir
        ),
        "broadcast_png": events_paths(
            stem, "broadcast", "png", outputs_dir=outputs_dir
        ),
    }


# --- T6: etapa inferencia (tracking) ------------------------------------------


def stage_inference(
    video: Path,
    choice: PipelineChoice,
    run_label: str,
    paths: dict,
    overwrite: bool,
    console,
) -> StageResult:
    tracking_json = paths["tracking_json"]
    if tracking_json.exists() and not overwrite:
        return StageResult(
            "reusado",
            {"tracking_json": tracking_json, "tracking_video": paths["tracking_video"]},
            "JSON de tracking ya existe (no se re-infiere).",
        )

    console.print("[bold]▶ Inferencia (tracking)…[/]")
    from src.core.inference import run_inference

    res = run_inference(
        video,
        mode="tracking",
        detector=choice.detector,
        tracker=choice.tracker,
        run_label=run_label,
        include_masks=False,
        render_video=True,
        progress=True,
    )
    return StageResult(
        "generado",
        {"tracking_json": Path(res["json"]), "tracking_video": Path(res["video"])},
    )


# --- T7: overlays individuales (opcional) -------------------------------------


def stage_individual_overlays(
    video: Path,
    choice: PipelineChoice,
    run_label: str,
    paths: dict,
    overwrite: bool,
    console,
) -> StageResult:
    if not choice.want_overlays:
        return StageResult("omitido", {}, "no solicitado")

    out: dict[str, Path] = {}

    # Overlay de tracking (usa el video CRUDO como fuente).
    console.print("[bold]▶ Overlay individual de tracking…[/]")
    from src.core.track_overlay import render_obj_id_overlay

    if paths["obj_overlay"].exists() and not overwrite:
        out["obj_overlay"] = paths["obj_overlay"]
    else:
        out["obj_overlay"] = render_obj_id_overlay(
            paths["tracking_json"], video_path=video
        )

    # Overlay de segmentación (corrida per-frame en su propio namespace).
    console.print("[bold]▶ Overlay individual de segmentación…[/]")
    if paths["seg_json"].exists() and not overwrite:
        out["seg_json"] = paths["seg_json"]
        out["seg_video"] = paths["seg_video"]
    else:
        from src.core.inference import run_inference

        seg = run_inference(
            video,
            mode="segmentation",
            detector=choice.detector,
            run_label=f"{run_label}/seg",
            render_video=True,
            progress=True,
        )
        out["seg_json"] = Path(seg["json"])
        out["seg_video"] = Path(seg["video"])

    return StageResult("generado", out)


# --- T8: etapa broadcast (entregable) -----------------------------------------


def stage_broadcast(
    video: Path, choice: PipelineChoice, paths: dict, overwrite: bool, console
) -> StageResult:
    # Gate por vista: homografía/eventos/broadcast solo aplican a cámara superior.
    if choice.vista != "superior":
        return StageResult(
            "omitido", {}, "vista genérica: homografía/eventos/broadcast no aplican"
        )

    broadcast_mp4 = paths["broadcast_mp4"]
    if broadcast_mp4.exists() and not overwrite:
        return StageResult(
            "reusado",
            {"broadcast_mp4": broadcast_mp4, "broadcast_png": paths["broadcast_png"]},
            "video de espectador ya existe.",
        )

    # Validación: aunque se declare 'superior', si la homografía sale degradada el clip no
    # es realmente cámara superior → se omite (no se produce un broadcast degradado).
    console.print("[bold]▶ Homografía / métrica (validación de vista superior)…[/]")
    from src.core.metric_positions import compute_metric_positions

    try:
        metric = compute_metric_positions(paths["tracking_json"], video=video)
    except Exception as exc:  # noqa: BLE001
        return StageResult(
            "omitido",
            {},
            f"homografía no calculable ({exc}): el clip no parece superior",
        )
    if metric is None or not any(p.xy_cm is not None for p in metric.posiciones):
        return StageResult(
            "omitido",
            {},
            "homografía degradada: el clip no parece cámara superior (eventos omitidos)",
        )

    console.print("[bold]▶ Broadcast (video de espectador)…[/]")
    from src.core.event_broadcast_overlay import render_broadcast_overlay

    # clip=video (crudo) evita el <stem>.mp4 segmentado; fijos del entregable.
    res = render_broadcast_overlay(
        paths["tracking_json"],
        clip=video,
        layout=2,
        goal_source="strict",
        use_kalman=True,
        progress=True,
    )
    out = {"broadcast_mp4": Path(res.video)}
    if res.sample_png is not None:
        out["broadcast_png"] = Path(res.sample_png)
    return StageResult("generado", out)


# --- T9: reporte ---------------------------------------------------------------

_STATUS_STYLE = {
    "generado": "green",
    "reusado": "cyan",
    "omitido": "dim",
    "fallido": "red",
}


def report(stages: dict[str, StageResult], console) -> None:
    from rich.table import Table

    table = Table(title="Artefactos de la corrida")
    table.add_column("Etapa")
    table.add_column("Estado")
    table.add_column("Ruta")
    for stage_name, result in stages.items():
        style = _STATUS_STYLE.get(result.status, "white")
        if result.paths:
            first = True
            for path in result.paths.values():
                table.add_row(
                    stage_name if first else "",
                    f"[{style}]{result.status}[/]" if first else "",
                    str(path),
                )
                first = False
        else:
            table.add_row(
                stage_name, f"[{style}]{result.status}[/]", result.detail or "—"
            )
    console.print(table)


# --- orquestación --------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Hub end-to-end del pipeline FutBot MX sobre un video.",
    )
    parser.add_argument(
        "video", help="ruta del video (relativa a la raíz del proyecto o absoluta)"
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="corre la config por defecto del proyecto sin preguntar nada",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="fuerza re-correr todas las etapas aunque existan artefactos previos",
    )
    parser.add_argument(
        "--vista",
        choices=VISTAS,
        default=None,
        help="vista de cámara; 'superior' habilita homografía/eventos/broadcast "
        "(default: pregunta en interactivo, 'superior' con --default)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    console = _console()
    config = _load_config()
    outputs_dir = _outputs_dir(config)

    video = validate_video(args.video, console)
    choice = choose_pipeline(args.default, args.vista, config, console)
    run_label = derive_run_label(choice)
    paths = plan_outputs(video.stem, run_label, outputs_dir)

    console.rule("[bold]FutBot MX — hub de pipeline")
    console.print(
        f"Video: [bold]{video}[/]  |  modo: "
        f"[bold]{'--default' if choice.default else 'interactivo'}[/]\n"
        f"detector=[bold]{choice.detector}[/]  tracker=[bold]{choice.tracker}[/]  "
        f"vista=[bold]{choice.vista}[/]  overlays={'sí' if choice.want_overlays else 'no'}"
        f"  |  run_label=[bold]{run_label}[/]"
    )

    stages: dict[str, StageResult] = {}
    exit_code = 0
    pipeline = [
        (
            "Inferencia",
            lambda: stage_inference(
                video, choice, run_label, paths, args.overwrite, console
            ),
        ),
        (
            "Overlays individuales",
            lambda: stage_individual_overlays(
                video, choice, run_label, paths, args.overwrite, console
            ),
        ),
        (
            "Broadcast",
            lambda: stage_broadcast(video, choice, paths, args.overwrite, console),
        ),
    ]
    for name, fn in pipeline:
        try:
            stages[name] = fn()
        except Exception as exc:  # noqa: BLE001
            stages[name] = StageResult("fallido", {}, str(exc))
            console.print(f"[bold red]✗ {name} falló:[/] {exc}")
            exit_code = 1
            break  # no seguir si una etapa falla

    report(stages, console)
    if exit_code == 0:
        bc = stages.get("Broadcast")
        if bc and bc.paths.get("broadcast_mp4"):
            console.print(f"\n[bold green]✓ Entregable:[/] {bc.paths['broadcast_mp4']}")
    return exit_code


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
