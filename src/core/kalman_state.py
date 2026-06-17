"""Fase 6 — Filtro de Kalman explícito en cm (capa de análisis post-asociación).

KF de **velocidad constante** 2D (estado ``[px, py, vx, vy]`` en cm y cm/s) que corre
SOBRE los tracks ya asociados (T3 ``metric_positions``). No re-hace detección ni asociación.
Tres usos: (i) estado posición+velocidad principiado; (ii) **predict-only en oclusión**
(estima dónde está el objeto aunque no se vea); (iii) velocidad más suave/física que las
diferencias finitas de T4. ``numpy`` puro (matemática transparente para el paper).

Modelo (1D por eje, desacoplado; ruido de aceleración blanco):
  x⁻ = F(dt) x ;  P⁻ = F P Fᵀ + Q(dt)
  y = z - H x⁻ ;  S = H P⁻ Hᵀ + R ;  K = P⁻ Hᵀ S⁻¹
  x = x⁻ + K y ;  P = (I - K H) P⁻
  Oclusión (sin medición): x = x⁻, P = P⁻ (la incertidumbre crece).
R se calibra del error de homografía (~9–23 cm). Gating por distancia de Mahalanobis
(χ²₂(0.99)=9.21) reemplaza el corte duro de 300 cm/s de T4 (no tira el track).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHI2_2_099 = 9.21  # umbral de gating (2 gl, 99%)

# Matriz de medición: medimos posición (T3 da xy_cm, nunca velocidad).
_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@dataclass
class KFParams:
    sigma_a: float                    # raíz de PSD del ruido de aceleración, cm/s²
    sigma_z: float = 15.0             # std de medición, cm (de la homografía ~9–23)
    propagated_inflation: float = 2.0  # multiplica σ_z cuando status_H == "propagated"
    v0: float = 200.0                 # std inicial de velocidad, cm/s (prior amplio)
    gate_chi2: float = CHI2_2_099
    max_gap_frames: int = 15          # máximo de pasos predict-only para puentear oclusión


@dataclass
class KalmanState:
    """Salida por-frame (espeja MetricPosition)."""

    obj_id: int
    cls: str
    frame_index: int
    xy_cm: tuple[float, float]
    vxy_cms: tuple[float, float]
    speed_cms: float
    pos_sigma_cm: float
    source: str               # "measured" | "predicted" | "gated"
    nis: float | None


class KalmanCV:
    """KF de velocidad constante 2D. Estado [px, py, vx, vy] en cm / cm·s⁻¹."""

    def __init__(self, z0: tuple[float, float], params: KFParams):
        self.p = params
        self.x = np.array([z0[0], z0[1], 0.0, 0.0], dtype=float)
        sz2, v02 = params.sigma_z ** 2, params.v0 ** 2
        self.P = np.diag([sz2, sz2, v02, v02]).astype(float)

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        sa2 = self.p.sigma_a ** 2
        q11 = dt ** 4 / 4.0
        q13 = dt ** 3 / 2.0
        q33 = dt ** 2
        return sa2 * np.array([
            [q11, 0.0, q13, 0.0],
            [0.0, q11, 0.0, q13],
            [q13, 0.0, q33, 0.0],
            [0.0, q13, 0.0, q33],
        ])

    def predict(self, dt: float) -> None:
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, z: tuple[float, float], status_H: str = "estimated") -> dict:
        """Corrige con la medición. Hace gating (Mahalanobis); si supera el umbral
        NO actualiza (predict-only este frame) pero NO tira el track."""
        sz = self.p.sigma_z * (self.p.propagated_inflation if status_H == "propagated" else 1.0)
        R = (sz ** 2) * np.eye(2)
        zv = np.asarray(z, dtype=float)
        y = zv - _H @ self.x
        S = _H @ self.P @ _H.T + R
        Sinv = np.linalg.inv(S)
        nis = float(y @ Sinv @ y)
        if nis > self.p.gate_chi2:
            return {"nis": nis, "gated": True}
        K = self.P @ _H.T @ Sinv
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ _H) @ self.P
        return {"nis": nis, "gated": False}

    @property
    def pos(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def vel(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    @property
    def speed_cms(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))

    @property
    def pos_sigma_cm(self) -> float:
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))


def run_kalman_on_track(
    samples: list[tuple[int, tuple[float, float] | None, str]],
    cls: str,
    obj_id: int,
    fps: float,
    params: KFParams,
) -> list[KalmanState]:
    """Corre el KF sobre el rango de frames de un obj_id. ``samples`` debe ser DENSO
    (un item por frame del rango [min..max]); ``xy_cm=None`` = oclusión (predict-only).
    Si la oclusión supera ``max_gap_frames`` se termina el segmento y se re-inicializa
    en la siguiente detección. dt = (f2-f1)/fps por paso."""
    out: list[KalmanState] = []
    kf: KalmanCV | None = None
    gap = 0
    prev_f: int | None = None

    for fidx, xy, status in samples:
        if kf is None:
            if xy is not None:
                kf = KalmanCV(xy, params)
                gap = 0
                prev_f = fidx
                out.append(KalmanState(obj_id, cls, fidx, kf.pos, kf.vel,
                                       kf.speed_cms, kf.pos_sigma_cm, "measured", None))
            continue

        dt = (fidx - prev_f) / fps if prev_f is not None else 1.0 / fps
        prev_f = fidx
        kf.predict(dt)

        if xy is not None:
            info = kf.update(xy, status)
            gap = 0
            src = "gated" if info["gated"] else "measured"
            out.append(KalmanState(obj_id, cls, fidx, kf.pos, kf.vel,
                                   kf.speed_cms, kf.pos_sigma_cm, src, info["nis"]))
        else:
            gap += 1
            if gap > params.max_gap_frames:
                kf = None       # termina segmento; re-init en próxima detección
                prev_f = None
                continue
            out.append(KalmanState(obj_id, cls, fidx, kf.pos, kf.vel,
                                   kf.speed_cms, kf.pos_sigma_cm, "predicted", None))
    return out
