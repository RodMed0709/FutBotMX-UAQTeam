"""Orquestación de inferencia por lotes (tarea batch_inference).

Capa **delgada y secuencial** sobre la fachada ``src.core.inference.run_inference``:
selecciona N videos del manifiesto ``db_metadata.csv``, **carga SAM3 una sola vez**,
**salta lo ya procesado** (skip-done por JSON de salida existente), **aísla errores**
(un video que falla no detiene el lote) y devuelve un **resumen estructurado** por
video. No reimplementa inferencia ni toca ``run_inference``, el esquema o ``src/data``.

API pública:
- ``run_batch``: itera un subconjunto de videos (por ``split`` o lista explícita) y
  devuelve ``list[dict]`` con el estado de cada uno (``done``/``skipped``/``failed``).

Importa ``run_inference`` a nivel de módulo (barato); ``pandas``, ``load_sam3`` y los
loaders de metadata se importan de forma perezosa dentro de la función.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.core.inference import run_inference
from src.core.inference_schema import inference_paths
from src.utils import PROJECT_ROOT, get_abs_path

# Campos de medición en None: para skip-done y fallos (no hubo inferencia medible).
_TIMING_NULL = {"elapsed_s": None, "peak_vram_mb": None, "fps": None}


def _reset_peak_vram() -> None:
    """Resetea el contador de pico de VRAM (no-op sin CUDA). torch perezoso."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _read_peak_vram_mb() -> float | None:
    """Pico de VRAM (MB) desde el ultimo reset, o ``None`` sin CUDA. torch perezoso."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e6


def _read_num_frames(json_path: Path | str) -> int | None:
    """``num_frames`` del header del JSON de salida; ``None`` si falta o falla la lectura."""
    try:
        doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return int(doc["num_frames"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _load_env(env_path: Path) -> dict[str, str]:
    """Parseo simple de un archivo .env (KEY = value), aplicando strip()."""
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _load_outputs_dir() -> str:
    """Lee ``working_dirs.outputs_dir`` de la configuración activa.

    Solo se necesita para derivar la ruta canónica del JSON de salida (skip-done);
    no se leen clases ni otros parámetros.

    Raises:
        ValueError: si CONFIG_FILENAME no está en el .env.
        KeyError: si falta ``working_dirs.outputs_dir``.
        FileNotFoundError: si el archivo de configuración no existe.
    """
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")
    config = json.loads(get_abs_path(f"configs/{config_filename}").read_text("utf-8"))
    working_dirs = config.get("working_dirs", {})
    if "outputs_dir" not in working_dirs:
        raise KeyError("Falta 'working_dirs.outputs_dir' en la configuracion.")
    return working_dirs["outputs_dir"]


def _validate_detector_tracker(detector: str | None, tracker: str | None) -> None:
    """Valida nombres de detector/tracker SIN cargar modelos.

    Se llama al inicio de ``run_batch``, antes de ``load_sam3()``, para fallar barato
    ante un nombre invalido. ``None`` se acepta en ambos (se usara el default del
    config mas adelante, en ``run_inference``/``track_video``).

    Imports perezosos para no arrastrar ``trackers``/``detectors`` al importar el
    modulo.

    Raises:
        ValueError: si ``tracker`` no esta en ``KNOWN_TRACKERS``, o si ``detector`` no
            esta registrado (delegado a ``get_detector``).
    """
    from src.core.detectors import get_detector
    from src.core.trackers import KNOWN_TRACKERS

    if tracker is not None and tracker not in KNOWN_TRACKERS:
        raise ValueError(
            f"tracker '{tracker}' no soportado (usa uno de {list(KNOWN_TRACKERS)})."
        )
    if detector is not None:
        get_detector(detector)  # levanta ValueError con su mensaje canonico


def _is_int(value: object) -> bool:
    """True si ``value`` puede interpretarse como entero (acepta ints y strings)."""
    try:
        int(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def _select_videos(split: int, videos: list[str | int] | None) -> list[tuple[int, str]]:
    """Selecciona ``(id, ruta)`` del manifiesto, ordenados de forma determinista por id.

    Si ``videos`` se pasa (rutas project-relative **o** ids), acota a ese conjunto y
    **tiene prioridad** sobre ``split``; un id/ruta inexistente levanta ``ValueError``.
    Si no, filtra por ``df["split"] == split``.
    """
    import pandas as pd

    from src.data.metadata import _load_metadata_config

    _, metadata_csv, _, _ = _load_metadata_config()
    df = pd.read_csv(get_abs_path(metadata_csv)).sort_values("id")

    if videos is not None:
        by_ruta = {str(v) for v in videos if not _is_int(v)}
        by_id = {int(v) for v in videos if _is_int(v)}
        sel = df[df["ruta"].isin(by_ruta) | df["id"].isin(by_id)]
        encontrados = set(sel["ruta"]) | {str(i) for i in sel["id"]}
        pedidos = by_ruta | {str(i) for i in by_id}
        faltan = pedidos - encontrados
        if faltan:
            raise ValueError(
                f"videos no encontrados en {metadata_csv}: {sorted(faltan)}"
            )
    else:
        sel = df[df["split"] == split]

    return [
        (int(vid), str(ruta)) for vid, ruta in sel[["id", "ruta"]].itertuples(False)
    ]


def run_batch(
    mode: str = "segmentation",
    split: int = 2,
    videos: list[str | int] | None = None,
    sampling: str = "auto",
    max_frames: int | None = None,
    include_masks: bool = False,
    render_video: bool = False,
    overwrite: bool = False,
    detector: str | None = None,
    tracker: str | None = None,
) -> list[dict]:
    """Corre la inferencia sobre un lote de videos reusando ``run_inference``.

    Itera **secuencialmente** los videos seleccionados, cargando SAM3 **una sola vez**
    y pasándolo a cada llamada. Salta los videos cuyo JSON de salida ya existe
    (``skip-done``) salvo ``overwrite``, y aísla errores: un video que falla se registra
    y el lote continúa.

    Args:
        mode: modo de inferencia para **todo** el lote (``"segmentation"`` o
            ``"tracking"``); se valida por video vía ``run_inference``.
        split: filtro del manifiesto (0=reserva, 1=fine-tuning, 2=testing). Default
            testing. Se ignora si se pasa ``videos``.
        videos: lista explícita de videos (rutas project-relative **o** ids). Si se
            pasa, acota el lote y **tiene prioridad** sobre ``split``. Un id/ruta que no
            exista en el manifiesto levanta ``ValueError``.
        sampling: estrategia de muestreo, propagada a ``run_inference`` (ver su
            docstring). Aplica a todo el lote.
        max_frames: tope de frames contiguos (solo tracking), propagado a
            ``run_inference``.
        include_masks: si ``True``, los JSON incluyen máscaras COCO-RLE.
        render_video: si ``True`` genera mp4 por video. **Default ``False``** (lote:
            el dato es el producto); sobreescribible.
        overwrite: si ``True``, reprocesa aunque el JSON ya exista (desactiva
            skip-done).
        detector: estrategia de deteccion por frame (``"sam3_text"`` | ``"yolo_sam3"``)
            para **todo** el lote. ``None`` (por defecto) usa el default del config.
            **Solo aplica en** ``mode="tracking"``; en segmentacion se **ignora**. Un
            nombre invalido levanta ``ValueError`` antes de cargar SAM3.
        tracker: tracker (``"bytetrack"`` | ``"botsort"``) para **todo** el lote.
            ``None`` (por defecto) usa el default del config. **Solo aplica en**
            ``mode="tracking"``; en segmentacion se **ignora**. Un nombre invalido
            levanta ``ValueError`` antes de cargar SAM3.

    Returns:
        ``list[dict]``, una entrada por video:
        ``{"id": int, "ruta": str, "status": "done"|"skipped"|"failed",
        "json": str | None, "video": str | None, "error": str | None,
        "elapsed_s": float | None, "peak_vram_mb": float | None, "fps": float | None}``.

        Campos de medición (solo con valor en ``done``; ``None`` en ``skipped``/
        ``failed``): ``elapsed_s`` es el wall-time (s) de la llamada ``run_inference``;
        ``peak_vram_mb`` es la VRAM pico (MB) durante esa llamada (``None`` sin CUDA);
        ``fps`` = ``num_frames / elapsed_s`` (``None`` si no se pudo leer ``num_frames``
        del JSON de salida).

    Raises:
        ValueError: ``videos`` con id/ruta inexistente; CONFIG_FILENAME ausente;
            ``detector``/``tracker`` con nombre invalido (antes de cargar SAM3).
        KeyboardInterrupt: se propaga (el lote es abortable).
    """
    from src.core.sam3_loader import load_sam3

    _validate_detector_tracker(detector, tracker)  # falla barato, antes de load_sam3
    outputs_dir = _load_outputs_dir()
    rows = _select_videos(split, videos)
    n = len(rows)
    print(f"== batch: {n} video(s), mode={mode}, render_video={render_video} ==")

    bundle = load_sam3()  # carga UNICA para todo el lote
    results: list[dict] = []

    for i, (vid, ruta) in enumerate(rows, start=1):
        json_path, _ = inference_paths(Path(ruta).stem, outputs_dir)
        if json_path.exists() and not overwrite:
            print(f"[{i}/{n}] {ruta} -> skipped")
            results.append(
                {
                    "id": vid,
                    "ruta": ruta,
                    "status": "skipped",
                    "json": str(json_path),
                    "video": None,
                    "error": None,
                    **_TIMING_NULL,
                }
            )
            continue

        try:
            _reset_peak_vram()  # aisla el pico de VRAM de este video
            t0 = time.perf_counter()
            res = run_inference(
                ruta,
                mode=mode,
                sampling=sampling,
                max_frames=max_frames,
                include_masks=include_masks,
                render_video=render_video,
                bundle=bundle,
                detector=detector,
                tracker=tracker,
            )
            elapsed = time.perf_counter() - t0
            peak_vram = _read_peak_vram_mb()
            num_frames = _read_num_frames(res["json"])
            fps = (
                num_frames / elapsed
                if (num_frames is not None and elapsed > 0)
                else None
            )
            entry = {
                "id": vid,
                "ruta": ruta,
                "status": "done",
                "json": str(res["json"]),
                "video": str(res["video"]) if res["video"] else None,
                "error": None,
                "elapsed_s": elapsed,
                "peak_vram_mb": peak_vram,
                "fps": fps,
            }
        except KeyboardInterrupt:
            raise  # abortable: no se traga
        except Exception as exc:  # aislamiento: registra y continúa
            entry = {
                "id": vid,
                "ruta": ruta,
                "status": "failed",
                "json": None,
                "video": None,
                "error": repr(exc),
                **_TIMING_NULL,
            }
        print(f"[{i}/{n}] {ruta} -> {entry['status']}")
        results.append(entry)

    done = sum(r["status"] == "done" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    failed = sum(r["status"] == "failed" for r in results)
    print(f"== batch: {done} done, {skipped} skipped, {failed} failed (de {n}) ==")
    return results
