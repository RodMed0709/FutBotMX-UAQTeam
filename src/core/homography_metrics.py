"""Metricas de consistencia para la homografia campo->cenital (fase_4 v2).

Define el criterio cuantitativo con el que se comparan las variantes de homografia
(camino C actual, multi-feature, +circulo, +undistort, ...). Dos numeros + cobertura:

1. **Error de reproyeccion (cm)**: mediana de la distancia entre cada landmark
   detectado (mapeado a cm via ``H``) y su posicion real en el template. Exactitud.
2. **Jitter temporal (cm)**: para un landmark fisico estatico, su posicion cm
   reconstruida (``project(q_t, H_t)``) deberia ser constante porque el marco de
   mundo es el campo (fijo), independiente del movimiento de camara. Su desviacion
   estandar temporal mide cuanto "tiembla" la reconstruccion. Estabilidad.
3. **Cobertura**: fraccion de frames con ``H`` valida.

``H`` mapea **imagen (px) -> mundo (cm)** (convencion del repo: ``project_points``).

El runner ``run_variant`` recorre clips con un ``solver_fn`` intercambiable y
agrega las metricas; asi cada notebook de experimento solo aporta su ``solver_fn``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.field_landmarks import LANDMARK_POINTS


@dataclass
class FrameResult:
    """Salida del solver para un frame.

    ``detections`` son los landmarks que el solver USÓ para ajustar ``H`` (fit).
    ``eval_points`` son landmarks **held-out** (detectados pero NO usados en el
    ajuste): sobre estos se miden ``reproj``/``jitter`` para evitar la circularidad
    de medir el error sobre los mismos puntos que definieron ``H``. Si ``eval_points``
    está vacío, las métricas caen a ``detections`` (medición trivial, debe marcarse).
    """
    frame_idx: int
    H: np.ndarray | None  # (3,3) imagen->cm, o None si no se resolvio
    detections: dict[str, tuple[float, float]] = field(default_factory=dict)  # fit; nombre->(px,py)
    eval_points: dict[str, tuple[float, float]] = field(default_factory=dict)  # held-out; nombre->(px,py)

    def measure_points(self) -> dict[str, tuple[float, float]]:
        """Landmarks sobre los que se miden las métricas (held-out si existen)."""
        return self.eval_points if self.eval_points else self.detections


def project_img_to_world(H: np.ndarray, pts_img: np.ndarray) -> np.ndarray:
    """Mapea puntos de imagen (px) a mundo (cm) via ``H``.

    Args:
        H: matriz ``(3,3)`` imagen->cm.
        pts_img: ``(N,2)`` en pixeles.

    Returns:
        ``(N,2)`` en cm.
    """
    import cv2

    pts = np.asarray(pts_img, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))
    return out.reshape(-1, 2)


def reproj_error_cm(fr: FrameResult) -> float | None:
    """Error de reproyeccion (cm) de un frame: mediana sobre detecciones.

    Devuelve ``None`` si no hay ``H`` o no hay detecciones.
    """
    pts = fr.measure_points()
    if fr.H is None or not pts:
        return None
    names = list(pts.keys())
    img = np.array([pts[n] for n in names], dtype=np.float64)
    world_hat = project_img_to_world(fr.H, img)
    world_ref = np.array([LANDMARK_POINTS[n] for n in names], dtype=np.float64)
    d = np.linalg.norm(world_hat - world_ref, axis=1)
    return float(np.median(d))


def _clip_jitter_cm(frames: list[FrameResult], min_samples: int = 3) -> list[float]:
    """Jitter (cm) por landmark dentro de un clip: std temporal de su cm reconstruido.

    Para cada landmark detectado en >= ``min_samples`` frames con ``H`` valida,
    acumula ``project(q_t, H_t)`` y devuelve ``sqrt(var_x + var_y)``.
    """
    series: dict[str, list[tuple[float, float]]] = {}
    for fr in frames:
        if fr.H is None:
            continue
        for name, q in fr.measure_points().items():
            w = project_img_to_world(fr.H, np.array([q]))[0]
            series.setdefault(name, []).append((float(w[0]), float(w[1])))
    out = []
    for name, pts in series.items():
        if len(pts) < min_samples:
            continue
        arr = np.array(pts)
        out.append(float(np.sqrt(arr[:, 0].var() + arr[:, 1].var())))
    return out


def summarize(per_clip: dict[str, list[FrameResult]]) -> dict:
    """Agrega metricas sobre varios clips.

    Args:
        per_clip: ``nombre_clip -> lista de FrameResult`` (en orden temporal).

    Returns:
        dict con ``reproj_cm`` (mediana global), ``jitter_cm`` (mediana global de
        jitters por landmark/clip), ``coverage`` (frac. frames con H), ``n_frames``,
        y ``per_clip`` con el desglose por clip.
    """
    all_reproj: list[float] = []
    all_jitter: list[float] = []
    total = 0
    with_h = 0
    per_clip_out = {}
    for clip, frames in per_clip.items():
        total += len(frames)
        n_h = sum(1 for f in frames if f.H is not None)
        with_h += n_h
        reprojs = [r for r in (reproj_error_cm(f) for f in frames) if r is not None]
        jitters = _clip_jitter_cm(frames)
        all_reproj.extend(reprojs)
        all_jitter.extend(jitters)
        per_clip_out[clip] = {
            "n_frames": len(frames),
            "coverage": (n_h / len(frames)) if frames else 0.0,
            "reproj_cm": float(np.median(reprojs)) if reprojs else None,
            "jitter_cm": float(np.median(jitters)) if jitters else None,
        }
    return {
        "reproj_cm": float(np.median(all_reproj)) if all_reproj else None,
        "jitter_cm": float(np.median(all_jitter)) if all_jitter else None,
        "coverage": (with_h / total) if total else 0.0,
        "n_frames": total,
        "per_clip": per_clip_out,
    }


def run_variant(clips: dict[str, list[np.ndarray]], solver_fn) -> dict:
    """Corre un solver sobre clips ya cargados (frames RGB) y devuelve metricas.

    Args:
        clips: ``nombre_clip -> lista de frames RGB (H,W,3) uint8`` en orden.
        solver_fn: ``callable(frame_rgb, frame_idx) -> FrameResult``.

    Returns:
        El dict de ``summarize`` mas ``per_clip_frames`` (los FrameResult crudos,
        por si el notebook quiere graficar trayectorias o overlays).
    """
    per_clip: dict[str, list[FrameResult]] = {}
    for clip, frames in clips.items():
        results = [solver_fn(f, i) for i, f in enumerate(frames)]
        per_clip[clip] = results
    out = summarize(per_clip)
    out["per_clip_frames"] = per_clip
    return out
