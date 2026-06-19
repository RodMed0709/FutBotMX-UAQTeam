"""T4 (fase_5 · Capa B) — velocidad y distancia en unidades métricas.

Consume las **posiciones en cm** de T3 (`metric_positions`) y deriva, por `obj_id`:
distancia recorrida (cm), velocidad media y máxima (cm/s), con **suavizado** y **rechazo de
saltos imposibles** (teleports por ID-switch del tracking o ruido de la homografía). Corre en
**CPU local**. Solo aplica a video de cámara superior (donde T3 tiene posiciones fiables).

Las cifras son **indicativas**: dependen de la calidad del tracking (ID-switches, no corregidos)
y de la homografía (~9–23 cm). Alimenta las métricas cuantitativas de la convocatoria 3.7.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.metric_positions import MetricResult, compute_metric_positions

# Umbral físico de velocidad (heurístico): estos robots de mesa no cruzan la cancha
# (243 cm) en una fracción de segundo; por encima = salto espurio (ID-switch / ruido H).
DEFAULT_MAX_SPEED_CMS = 300.0
DEFAULT_SMOOTH_WIN = 5  # ventana (frames-muestra) de la media móvil de velocidad


@dataclass
class ObjKinematics:
    obj_id: int
    cls: str
    n_muestras: int
    dur_s: float
    dist_cm: float
    v_media_cms: float
    v_max_cms: float
    serie: list[tuple[int, float]] | None  # (frame, v_cms) si with_series


@dataclass
class KinematicsResult:
    por_obj: list[ObjKinematics]
    resumen: dict


def _series_by_obj(result: MetricResult) -> dict[int, tuple[str, list]]:
    """Agrupa muestras con `xy_cm` válido por `obj_id`, ordenadas por `frame_index`."""
    by_obj: dict[int, tuple[str, list]] = {}
    for p in result.posiciones:
        if p.xy_cm is None:
            continue
        cls, samples = by_obj.setdefault(p.obj_id, (p.cls, []))
        samples.append((p.frame_index, p.xy_cm))
    for _obj, (_cls, samples) in by_obj.items():
        samples.sort(key=lambda s: s[0])
    return by_obj


def _smooth(values: list[float], win: int) -> list[float]:
    """Media móvil simple (ventana `win`, centrada) sobre una serie de velocidad."""
    if not values or win <= 1:
        return list(values)
    arr = np.asarray(values, dtype=float)
    k = min(win, len(arr))
    kernel = np.ones(k) / k
    return list(np.convolve(arr, kernel, mode="same"))


def _kinematics(
    cls: str, obj_id: int, samples: list, fps: float, max_speed_cms: float, win: int,
    with_series: bool,
) -> tuple[ObjKinematics, int]:
    """Cinemática de un objeto. Devuelve `(ObjKinematics, n_outliers)`."""
    n = len(samples)
    if n < 2:
        return ObjKinematics(obj_id, cls, n, 0.0, 0.0, 0.0, 0.0, [] if with_series else None), 0

    dist = 0.0
    outliers = 0
    serie: list[tuple[int, float]] = []  # (frame_medio, v_cms)
    for (f1, p1), (f2, p2) in zip(samples, samples[1:]):
        dt = (f2 - f1) / fps
        if dt <= 0:
            continue
        d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        v = d / dt
        if v > max_speed_cms:  # salto imposible -> descartar
            outliers += 1
            continue
        dist += d
        serie.append(((f1 + f2) // 2, v))

    dur_s = (samples[-1][0] - samples[0][0]) / fps
    v_vals = _smooth([v for _, v in serie], win)
    v_media = float(np.mean(v_vals)) if v_vals else 0.0
    v_max = float(np.max(v_vals)) if v_vals else 0.0
    out_serie = [(serie[i][0], round(v_vals[i], 2)) for i in range(len(serie))] if with_series else None
    return (
        ObjKinematics(obj_id, cls, n, round(dur_s, 2), round(dist, 1),
                      round(v_media, 1), round(v_max, 1), out_serie),
        outliers,
    )


def compute_kinematics(
    source: str | Path | MetricResult,
    *,
    fps: float | None = None,
    max_speed_cms: float = DEFAULT_MAX_SPEED_CMS,
    smooth_win: int = DEFAULT_SMOOTH_WIN,
    with_series: bool = False,
) -> KinematicsResult:
    """Velocidad/distancia por `obj_id`. `source` = ruta a tracks_json (llama a T3) o
    un `MetricResult` ya calculado."""
    result = source if isinstance(source, MetricResult) else compute_metric_positions(Path(source))
    fps = fps or result.resumen.get("fps")
    if not fps:
        raise ValueError("falta fps (ni en argumento ni en el resumen de T3)")

    por_obj: list[ObjKinematics] = []
    total_outliers = 0
    for obj_id, (cls, samples) in _series_by_obj(result).items():
        ok, n_out = _kinematics(cls, obj_id, samples, fps, max_speed_cms, smooth_win, with_series)
        por_obj.append(ok)
        total_outliers += n_out
    por_obj.sort(key=lambda o: o.dist_cm, reverse=True)

    por_clase: dict[str, dict] = {}
    for o in por_obj:
        agg = por_clase.setdefault(o.cls, {"n_obj": 0, "dist_cm": 0.0, "v_max_cms": 0.0})
        agg["n_obj"] += 1
        agg["dist_cm"] = round(agg["dist_cm"] + o.dist_cm, 1)
        agg["v_max_cms"] = max(agg["v_max_cms"], o.v_max_cms)

    resumen = {
        "fps": fps,
        "n_obj": len(por_obj),
        "por_clase": por_clase,
        "segmentos_outlier_descartados": total_outliers,
        "params": {"max_speed_cms": max_speed_cms, "smooth_win": smooth_win},
        "nota": "cifras indicativas (limitadas por ID-switches del tracking y por la homografía)",
    }
    return KinematicsResult(por_obj=por_obj, resumen=resumen)


def write_kinematics_json(result: KinematicsResult, path: str | Path) -> Path:
    """Escribe el resultado a JSON (resumen + métricas por objeto)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "por_obj": [
            {
                "obj_id": o.obj_id,
                "class": o.cls,
                "n_muestras": o.n_muestras,
                "dur_s": o.dur_s,
                "dist_cm": o.dist_cm,
                "v_media_cms": o.v_media_cms,
                "v_max_cms": o.v_max_cms,
                **({"serie": o.serie} if o.serie is not None else {}),
            }
            for o in result.por_obj
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
