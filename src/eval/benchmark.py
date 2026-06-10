"""Métricas sin ground-truth y tabla comparativa del benchmark (tarea `benchmark_metrics`).

Módulo **lector/agregador puro**: NO corre inferencia. Consume los JSON de inferencia
ya escritos por ``run_batch`` y el **timing** de su valor de retorno (``fps``,
``peak_vram_mb``), y produce una **tabla comparativa** de las configuraciones
detector × tracker.

Tres familias de métricas:
- **Eficiencia** (del resumen de ``run_batch``): ``fps``, ``peak_vram_mb``.
- **Trayectoria** (de la sección ``tracks`` del JSON, por ``obj_id``): longitud media
  de tracklet, tasa de fragmentación proxy, suavidad (varianza de la aceleración).
- **Máscara** (de la sección ``frames`` + ``rle``): IoU temporal y jitter del centro
  de masa. Requieren ``obj_id`` estable ⇒ solo configs con tracking.

``numpy``/``pandas``/``decode_rle`` se importan de forma **perezosa** dentro de las
funciones, para que ``import src.eval`` sea barato.

API pública: ``benchmark_videos``, ``video_metrics``, ``aggregate_config``,
``comparison_table``, ``write_table``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils import PROJECT_ROOT, get_abs_path

# Orden de columnas de la tabla comparativa.
_TABLE_COLUMNS = [
    "config",
    "fps",
    "peak_vram_mb",
    "tracklet_len",
    "frag_rate",
    "smoothness",
    "mask_iou",
    "com_jitter",
]

# Llaves de métricas por video (las de calidad; el timing se funde aparte).
_METRIC_KEYS = ("tracklet_len", "frag_rate", "smoothness", "mask_iou", "com_jitter")


def benchmark_videos(n: int = 5, seed: int = 42) -> list[str]:
    """Selecciona ``n`` videos del split de testing (2) de forma **reproducible**.

    Args:
        n: número de videos a muestrear.
        seed: semilla del muestreo (misma semilla → misma lista).

    Returns:
        Rutas project-relative de los videos, ordenadas por ``id``.
    """
    import pandas as pd

    from src.data.metadata import _load_metadata_config

    _, metadata_csv, _, _ = _load_metadata_config()
    df = pd.read_csv(get_abs_path(metadata_csv))
    testing = df[df["split"] == 2]
    sel = testing.sample(n=min(n, len(testing)), random_state=seed)
    return [str(r) for r in sel.sort_values("id")["ruta"]]


def _dist(a, b) -> float:
    """Distancia euclídea entre dos centroides ``[cx, cy]``."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _safe_mean(values):
    """Media ignorando ``None``/``nan``; ``None`` si no queda ninguna muestra."""
    import math

    xs = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return float(sum(xs) / len(xs)) if xs else None


def _trajectory_metrics(
    tracks: list[dict], width: float, frag_window: int, frag_radius_frac: float
) -> dict:
    """Longitud de tracklet, fragmentación proxy y suavidad desde ``tracks``."""
    import numpy as np

    if not tracks:
        return {"tracklet_len": None, "frag_rate": None, "smoothness": None}

    # Longitud media de tracklet.
    lengths = [len(t.get("observations", [])) for t in tracks]
    tracklet_len = _safe_mean(lengths)

    # Inicio/fin de cada track (por orden de frame_index).
    radius = frag_radius_frac * width if width else 0.0
    starts, ends = [], []  # (class, frame, centroid)
    for t in tracks:
        obs = sorted(t.get("observations", []), key=lambda o: o["frame_index"])
        if not obs:
            continue
        cls = t.get("class")
        starts.append((cls, obs[0]["frame_index"], obs[0]["centroid"]))
        ends.append((cls, obs[-1]["frame_index"], obs[-1]["centroid"]))

    # Fragmentación proxy: un track cuenta si OTRO de su clase arranca poco después y cerca.
    frag_count = 0
    for cls_a, f_a, c_a in ends:
        for cls_b, f_b, c_b in starts:
            if (
                cls_b == cls_a
                and f_a < f_b <= f_a + frag_window
                and _dist(c_a, c_b) < radius
            ):
                frag_count += 1
                break
    frag_rate = frag_count / len(tracks) if tracks else None

    # Suavidad: varianza de la magnitud de la aceleración de los centroides.
    smooth_per_track = []
    for t in tracks:
        obs = sorted(t.get("observations", []), key=lambda o: o["frame_index"])
        if len(obs) < 3:
            continue
        cents = np.array([o["centroid"] for o in obs], dtype=float)
        acc = np.diff(cents, n=2, axis=0)
        mag = np.linalg.norm(acc, axis=1)
        smooth_per_track.append(float(np.var(mag)))
    smoothness = _safe_mean(smooth_per_track)

    return {
        "tracklet_len": tracklet_len,
        "frag_rate": frag_rate,
        "smoothness": smoothness,
    }


def _mask_iou(a, b) -> float:
    """IoU entre dos máscaras booleanas."""
    import numpy as np

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def _centroid_of_mask(m):
    """Centro de masa ``(cx, cy)`` de los píxeles ``True``, o ``None`` si vacía."""
    import numpy as np

    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _mask_metrics(frames: list[dict], width: float) -> dict:
    """IoU temporal y jitter del centro de masa entre frames consecutivos.

    Compara cada ``(class, obj_id)`` entre un frame y el inmediatamente anterior
    (igual que el smoke). Requiere ``rle``; si ninguna detección lo trae ⇒ ``None``.
    """
    from src.core.inference_schema import decode_rle

    seen_rle = False
    ious, jitters = [], []
    prev: dict = {}  # (class, obj_id) -> (mask, centroid) del frame anterior
    for fr in frames:
        cur = {}
        for cls, dets in fr.get("detections", {}).items():
            for d in dets:
                if "rle" not in d:
                    continue
                seen_rle = True
                m = decode_rle(d["rle"])
                key = (cls, d["obj_id"])
                com = _centroid_of_mask(m)
                cur[key] = (m, com)
                if key in prev:
                    pm, pcom = prev[key]
                    ious.append(_mask_iou(pm, m))
                    if com is not None and pcom is not None and width:
                        jitters.append(_dist(com, pcom) / width)
        prev = cur

    if not seen_rle:
        return {"mask_iou": None, "com_jitter": None}
    return {"mask_iou": _safe_mean(ious), "com_jitter": _safe_mean(jitters)}


def video_metrics(
    doc: dict, *, frag_window: int = 5, frag_radius_frac: float = 0.05
) -> dict:
    """Métricas sin-GT de un JSON de inferencia **ya cargado**.

    Args:
        doc: JSON de inferencia (header + ``frames`` + opcional ``tracks``).
        frag_window: ventana (frames) para la fragmentación proxy.
        frag_radius_frac: radio espacial (fracción del ancho del frame) para la
            fragmentación proxy.

    Returns:
        ``{tracklet_len, frag_rate, smoothness, mask_iou, com_jitter}``. Trayectoria
        ``None`` si no hay ``tracks``; máscara ``None`` si no hay ``rle``.
    """
    width = float(doc.get("resolution", {}).get("width", 0) or 0)
    traj = _trajectory_metrics(
        doc.get("tracks") or [], width, frag_window, frag_radius_frac
    )
    mask = _mask_metrics(doc.get("frames", []), width)
    return {**traj, **mask}


def aggregate_config(
    label: str,
    entries: list[dict],
    *,
    frag_window: int = 5,
    frag_radius_frac: float = 0.05,
) -> dict:
    """Agrega las métricas de una **config** a partir del resumen de ``run_batch``.

    Args:
        label: etiqueta de la configuración (p. ej. ``"yolo_sam3+botsort"``).
        entries: lista-resumen de ``run_batch`` de **una** config (cada entry trae
            ``status``, ``json``, ``fps``, ``peak_vram_mb``...).
        frag_window, frag_radius_frac: se propagan a ``video_metrics``.

    Returns:
        Una fila ``{"config": label, fps, peak_vram_mb, ...métricas de calidad}``
        promediada sobre los videos ``done`` (ignorando ``None``). Configs sin
        tracking dejan trayectoria/máscara en ``None``.
    """
    rows = []
    for e in entries:
        if e.get("status") != "done":
            continue  # skipped/failed no aportan métricas
        doc = json.loads(Path(e["json"]).read_text(encoding="utf-8"))
        m = video_metrics(
            doc, frag_window=frag_window, frag_radius_frac=frag_radius_frac
        )
        m["fps"] = e.get("fps")
        m["peak_vram_mb"] = e.get("peak_vram_mb")
        rows.append(m)

    agg = {"config": label}
    for key in (*_METRIC_KEYS, "fps", "peak_vram_mb"):
        agg[key] = _safe_mean([r.get(key) for r in rows])
    return agg


def comparison_table(rows: list[dict]):
    """Construye el DataFrame comparativo (una fila por config, columnas ordenadas)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    return df.reindex(columns=_TABLE_COLUMNS)


def write_table(df, path: Path | str | None = None) -> Path:
    """Escribe el DataFrame comparativo a CSV (default ``outputs/benchmark/comparison.csv``)."""
    out = (
        Path(path)
        if path is not None
        else PROJECT_ROOT / "outputs" / "benchmark" / "comparison.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
