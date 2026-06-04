"""Prueba headless del overlay multi-clase (tarea segmentation_overlay).

Valida ``overlay_detections`` con datos **sinteticos** (sin modelo ni graficos):
construye un frame y mascaras booleanas de prueba, compone y verifica:

  - la salida es ``uint8`` con forma ``(H, W, 3)``;
  - los pixeles bajo la mascara viran hacia el color de la clase;
  - los pixeles fuera de toda mascara no cambian;
  - el frame de entrada no se muta.

No usa matplotlib (no llama a ``show_overlay``), por lo que corre en entornos sin
graficos. No requiere SAM3 ni la configuracion (las clases se pasan explicitas).

Uso:
    python testing/test_overlay.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.core import overlay_detections  # noqa: E402


@dataclass
class _FakeDet:
    """Detection minima para la prueba (solo necesita ``mask``)."""

    mask: np.ndarray


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    H, W = 40, 60
    frame = np.full((H, W, 3), 30, dtype=np.uint8)  # gris oscuro uniforme

    # Clases sinteticas con colores de prueba.
    classes = [
        {"name": "robot", "color": [255, 0, 0]},
        {"name": "ball", "color": [0, 255, 0]},
    ]

    # Mascaras booleanas disjuntas.
    m_robot = np.zeros((H, W), dtype=bool)
    m_robot[5:15, 5:20] = True
    m_ball = np.zeros((H, W), dtype=bool)
    m_ball[25:35, 35:50] = True

    dets = {
        "robot": [_FakeDet(mask=m_robot)],
        "ball": [_FakeDet(mask=m_ball)],
    }

    frame_before = frame.copy()
    out = overlay_detections(frame, dets, classes=classes, alpha=0.5)

    print("== Verificaciones ==")
    assert out.shape == (H, W, 3), f"forma inesperada: {out.shape}"
    assert out.dtype == np.uint8, f"dtype inesperado: {out.dtype}"
    print(f"  forma={out.shape} dtype={out.dtype}  OK")

    # Pixeles bajo mascara viran al color: canal R sube en robot, G en ball.
    assert out[10, 10, 0] > frame_before[10, 10, 0], "robot no viro a rojo"
    assert out[30, 40, 1] > frame_before[30, 40, 1], "ball no viro a verde"
    print("  pixeles bajo mascara viran al color de clase  OK")

    # Pixel fuera de toda mascara no cambia.
    assert np.array_equal(out[0, 0], frame_before[0, 0]), "fondo modificado"
    print("  pixeles fuera de mascara sin cambios  OK")

    # No muta la entrada.
    assert np.array_equal(frame, frame_before), "overlay_detections muto el frame"
    print("  frame de entrada no mutado  OK")

    # Caso vacio: devuelve copia del frame.
    out_empty = overlay_detections(frame, {}, classes=classes, alpha=0.5)
    assert np.array_equal(out_empty, frame), "dict vacio no devolvio copia del frame"
    print("  dict vacio -> copia del frame  OK")

    print("\n== Resultado ==")
    print("  OK: demostracion headless de overlay_detections completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
