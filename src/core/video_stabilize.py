"""Estabilización de cámara (preproceso, ANTES de los modelos) — fase_4 v2.

La cámara siempre se mueve/tiembla; estabilizar los frames primero hace que SAM3,
YOLO y la homografía vean una escena más quieta → todo tiembla menos aguas abajo.

Técnica clásica (Nghia Ho): estimar la transformada afín entre frames consecutivos
por seguimiento de features (Lucas-Kanade), acumular la trayectoria de cámara,
**suavizarla** (media móvil) y warpear cada frame para seguir la trayectoria suave.
Quita el temblor de alta frecuencia conservando los paneos intencionales.

Dos pasadas (acceso aleatorio a frames vía decord): estima todas las transformadas,
suaviza, y luego warpea. Para clips de evaluación esto es suficiente.
"""

from __future__ import annotations

import numpy as np


def _moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    """Media móvil 1D con padding por borde (suaviza la trayectoria)."""
    k = 2 * radius + 1
    pad = np.pad(curve, (radius, radius), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(pad, kernel, mode="same")[radius:-radius]


def estimate_transforms(frames_gray: list[np.ndarray]) -> np.ndarray:
    """Transformadas afines (dx, dy, da) entre frames consecutivos.

    Devuelve ``(N-1, 3)`` con traslación y rotación de cada paso ``i -> i+1``.
    """
    import cv2

    out = []
    prev = frames_gray[0]
    last = (0.0, 0.0, 0.0)
    for cur in frames_gray[1:]:
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01,
                                     minDistance=30, blockSize=3)
        if p0 is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None)
            st = st.ravel().astype(bool)
            a, b = p0[st], p1[st]
            m = cv2.estimateAffinePartial2D(a, b)[0] if len(a) >= 6 else None
        else:
            m = None
        if m is None:
            dx, dy, da = last  # repetir el último (sin features fiables)
        else:
            dx, dy, da = float(m[0, 2]), float(m[1, 2]), float(np.arctan2(m[1, 0], m[0, 0]))
            last = (dx, dy, da)
        out.append((dx, dy, da))
        prev = cur
    return np.array(out, dtype=np.float64) if out else np.zeros((0, 3))


def stabilize_frames(frames_rgb: list[np.ndarray], smooth_radius: int = 15,
                     crop_ratio: float = 0.04) -> list[np.ndarray]:
    """Estabiliza una lista de frames RGB y la devuelve (mismo tamaño).

    Args:
        frames_rgb: frames ``(H,W,3)`` RGB en orden temporal.
        smooth_radius: radio de la media móvil sobre la trayectoria (mayor = más quieto,
            menos responsivo a paneos rápidos).
        crop_ratio: zoom para ocultar los bordes negros que deja el warp (4% por lado).

    Returns:
        Lista de frames RGB estabilizados.
    """
    import cv2

    if len(frames_rgb) < 3:
        return list(frames_rgb)
    H, W = frames_rgb[0].shape[:2]
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]

    tf = estimate_transforms(grays)                 # (N-1, 3)
    traj = np.cumsum(tf, axis=0)                     # trayectoria acumulada
    smooth = np.stack([_moving_average(traj[:, i], smooth_radius) for i in range(3)], axis=1)
    diff = smooth - traj                            # corrección por frame (para i>=1)

    # zoom para tapar bordes del warp
    z = 1.0 + 2 * crop_ratio
    zoom = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), 0, z)

    out = [cv2.warpAffine(frames_rgb[0], zoom, (W, H))]
    for i in range(1, len(frames_rgb)):
        dx, dy, da = tf[i - 1] + diff[i - 1]        # transformada suavizada de este paso
        m = np.array([[np.cos(da), -np.sin(da), dx],
                      [np.sin(da), np.cos(da), dy]], dtype=np.float64)
        stab = cv2.warpAffine(frames_rgb[i], m, (W, H))
        out.append(cv2.warpAffine(stab, zoom, (W, H)))
    return out
