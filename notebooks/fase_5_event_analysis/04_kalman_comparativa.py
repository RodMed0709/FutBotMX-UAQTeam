# -*- coding: utf-8 -*-
"""Fase 5 / SDD-B (T7) — Comparativa de broadcast: SIN vs CON Kalman, en los dos layouts.

Renderiza 4 variantes del overlay de espectador sobre ``IMG_9933_5m30`` para que el equipo elija
cuál prefiere: {sin, con Kalman} × {layout 2, layout 1}. CPU local; consume el JSON de tracking +
el clip de al lado. ``render_broadcast_overlay`` siempre escribe a la misma ruta, así que cada
render se **mueve** a un nombre de variante (no se toca ``src``).

Nota: el clip que usa el broadcast es ``<stem>.mp4`` junto al JSON (segmentado). La diferencia que
importa aquí es el **efecto de Kalman**, idéntico clip en las 4 variantes. Para el entregable limpio
final, usar el clip crudo (``00_prepare_clips.py``) en el pod.

Uso (local o pod):  python notebooks/fase_5_event_analysis/04_kalman_comparativa.py
"""
from __future__ import annotations

import shutil

from src.core.event_broadcast_overlay import render_broadcast_overlay
from src.utils import PROJECT_ROOT

TRACKS = PROJECT_ROOT / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
OUT_DIR = PROJECT_ROOT / "outputs/eventos/IMG_9933_5m30"
# cap opcional para validación rápida (None = clip completo)
MAX_FRAMES: int | None = None

VARIANTES = [
    ("sin_kalman_L2", {"layout": 2, "use_kalman": False}),
    ("con_kalman_L2", {"layout": 2, "use_kalman": True}),
    ("sin_kalman_L1", {"layout": 1, "use_kalman": False}),
    ("con_kalman_L1", {"layout": 1, "use_kalman": True}),
]


def main() -> None:
    if not TRACKS.exists():
        raise FileNotFoundError(f"no está el JSON de tracking: {TRACKS}")
    resultados = []
    for nombre, kwargs in VARIANTES:
        print(f"\n[T7] render variante: {nombre}  ({kwargs})")
        res = render_broadcast_overlay(TRACKS, max_frames=MAX_FRAMES, **kwargs)
        destino = OUT_DIR / f"{TRACKS.stem}_cmp_{nombre}.mp4"
        shutil.move(str(res.video), str(destino))
        # mueve también el PNG de muestra si lo generó
        if res.sample_png is not None and res.sample_png.exists():
            shutil.move(str(res.sample_png), str(destino.with_suffix(".png")))
        resultados.append((nombre, destino))
        print(f"   -> {destino}")

    print("\n=== COMPARATIVA LISTA (IMG_9933_5m30) ===")
    for nombre, destino in resultados:
        print(f"  {nombre:16} {destino}")
    print("\nElige la variante preferida revisando los 4 mp4 anteriores.")


if __name__ == "__main__":
    main()
