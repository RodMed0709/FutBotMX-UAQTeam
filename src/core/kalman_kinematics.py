"""Fase 6 — driver del KF en cm sobre T3 (análogo a T4 ``metric_kinematics``).

Agrupa las posiciones de T3 (``metric_positions``) por ``obj_id``, construye una serie
DENSA por frame (oclusión = frame sin ``xy_cm``), corre ``run_kalman_on_track`` y produce
(i) estados por-frame y (ii) un resumen de cinemática (v_media/v_max/distancia) comparable
al de T4. Solo filtra clases móviles (balón, robots); las zonas/alfombra son anclas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.kalman_state import KFParams, KalmanState, run_kalman_on_track
from src.core.metric_positions import MetricPosition, MetricResult, compute_metric_positions

# Clases móviles (las estáticas green_floor/yellow_zone/blue_zone son anclas, se excluyen).
# NOTA: estos params estan tuneados para ESPACIO DE IMAGEN (px), porque T3 (cm) esta roto
# para los clips re-encodeados (ver 01_kalman_experiment.py). sigma_z=ruido de centroide (px),
# sigma_a=ruido de aceleracion (px/s^2). Calibrados para NIS medio ~2 (ver T6.5). En cm,
# re-tunear (sigma_z~15 del error de homografia 9-23 cm).
CLASS_PARAMS: dict[str, KFParams] = {
    "orange_ball": KFParams(sigma_a=300.0, sigma_z=8.0, max_gap_frames=15),
    "robot_a": KFParams(sigma_a=600.0, sigma_z=20.0, max_gap_frames=30),
    "robot_b": KFParams(sigma_a=600.0, sigma_z=20.0, max_gap_frames=30),
    "robot": KFParams(sigma_a=600.0, sigma_z=20.0, max_gap_frames=30),  # fallback single-robot
}


@dataclass
class ObjKalman:
    obj_id: int
    cls: str
    n_frames: int          # frames con estado (medidos + predichos)
    n_measured: int
    n_predicted: int       # frames de oclusión rellenados
    n_gated: int
    dur_s: float
    dist_cm: float
    v_media_cms: float
    v_max_cms: float
    estados: list[KalmanState]


@dataclass
class KalmanResult:
    por_obj: list[ObjKalman]
    resumen: dict


def load_metric_result_from_json(path: str | Path) -> MetricResult:
    """Lee un JSON de T3 (escrito por ``write_metric_positions_json``) a ``MetricResult``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pos = [
        MetricPosition(
            obj_id=p["obj_id"], cls=p["class"], frame_index=p["frame_index"],
            xy_cm=(tuple(p["xy_cm"]) if p["xy_cm"] is not None else None),
            status_H=p.get("status_H", "estimated"),
        )
        for p in data["posiciones"]
    ]
    return MetricResult(posiciones=pos, resumen=data.get("resumen", {}))


def _dense_samples_by_obj(result: MetricResult) -> dict[int, tuple[str, list]]:
    """Por obj_id: serie DENSA [min..max] de (frame, xy_cm|None, status_H)."""
    rows: dict[int, dict[int, tuple]] = {}
    cls_of: dict[int, str] = {}
    for p in result.posiciones:
        cls_of[p.obj_id] = p.cls
        if p.xy_cm is not None:
            rows.setdefault(p.obj_id, {})[p.frame_index] = (p.xy_cm, p.status_H)
    out: dict[int, tuple[str, list]] = {}
    for oid, fr in rows.items():
        if not fr:
            continue
        lo, hi = min(fr), max(fr)
        dense = [(f, fr[f][0] if f in fr else None, fr[f][1] if f in fr else "missing")
                 for f in range(lo, hi + 1)]
        out[oid] = (cls_of[oid], dense)
    return out


def _kinematics_of(estados: list[KalmanState], fps: float) -> tuple[float, float, float, float]:
    """(dur_s, dist_cm, v_media, v_max) de la serie de estados del KF."""
    if len(estados) < 2:
        return 0.0, 0.0, 0.0, 0.0
    dist = 0.0
    for a, b in zip(estados, estados[1:]):
        dist += float(np.hypot(b.xy_cm[0] - a.xy_cm[0], b.xy_cm[1] - a.xy_cm[1]))
    speeds = [s.speed_cms for s in estados]
    dur = (estados[-1].frame_index - estados[0].frame_index) / fps
    return dur, dist, float(np.mean(speeds)), float(np.max(speeds))


def compute_kalman_states(
    source: str | Path | MetricResult,
    *,
    fps: float | None = None,
    class_params: dict[str, KFParams] = CLASS_PARAMS,
) -> KalmanResult:
    """Corre el KF por obj_id sobre T3. ``source`` = MetricResult, JSON de T3, o JSON de
    tracking (se llama a T3). Devuelve estados + resumen de cinemática."""
    if isinstance(source, MetricResult):
        result = source
    else:
        p = Path(source)
        # heurística: el JSON de T3 tiene "posiciones"; el de tracking, no.
        head = json.loads(p.read_text(encoding="utf-8"))
        result = (load_metric_result_from_json(p) if "posiciones" in head
                  else compute_metric_positions(p))
    fps = fps or result.resumen.get("fps")
    if not fps:
        raise ValueError("falta fps (ni en argumento ni en el resumen de T3)")

    por_obj: list[ObjKalman] = []
    for oid, (cls, dense) in _dense_samples_by_obj(result).items():
        params = class_params.get(cls)
        if params is None:
            continue  # clase estática (ancla) o no configurada
        estados = run_kalman_on_track(dense, cls, oid, fps, params)
        if not estados:
            continue
        dur, dist, vm, vmax = _kinematics_of(estados, fps)
        por_obj.append(ObjKalman(
            obj_id=oid, cls=cls, n_frames=len(estados),
            n_measured=sum(1 for s in estados if s.source == "measured"),
            n_predicted=sum(1 for s in estados if s.source == "predicted"),
            n_gated=sum(1 for s in estados if s.source == "gated"),
            dur_s=round(dur, 2), dist_cm=round(dist, 1),
            v_media_cms=round(vm, 1), v_max_cms=round(vmax, 1), estados=estados,
        ))
    por_obj.sort(key=lambda o: o.dist_cm, reverse=True)

    por_clase: dict[str, dict] = {}
    for o in por_obj:
        agg = por_clase.setdefault(o.cls, {"n_obj": 0, "v_max_cms": 0.0, "n_predicted": 0})
        agg["n_obj"] += 1
        agg["v_max_cms"] = max(agg["v_max_cms"], o.v_max_cms)
        agg["n_predicted"] += o.n_predicted
    resumen = {
        "fps": fps,
        "n_obj": len(por_obj),
        "por_clase": por_clase,
        "frames_rellenados_oclusion": sum(o.n_predicted for o in por_obj),
        "frames_gated": sum(o.n_gated for o in por_obj),
        "nota": "KF velocidad-constante en cm; predict-only rellena oclusiones (<=max_gap)",
    }
    return KalmanResult(por_obj=por_obj, resumen=resumen)


def write_kalman_states_json(result: KalmanResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resumen": result.resumen,
        "por_obj": [
            {
                "obj_id": o.obj_id, "class": o.cls, "n_frames": o.n_frames,
                "n_measured": o.n_measured, "n_predicted": o.n_predicted, "n_gated": o.n_gated,
                "dur_s": o.dur_s, "dist_cm": o.dist_cm,
                "v_media_cms": o.v_media_cms, "v_max_cms": o.v_max_cms,
                "estados": [
                    {"frame_index": s.frame_index, "xy_cm": list(s.xy_cm),
                     "vxy_cms": list(s.vxy_cms), "speed_cms": round(s.speed_cms, 2),
                     "pos_sigma_cm": round(s.pos_sigma_cm, 2), "source": s.source}
                    for s in o.estados
                ],
            }
            for o in result.por_obj
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
