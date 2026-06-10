"""Submódulo eval del proyecto FutBotMX.

Agrupa la **evaluación** del pipeline (no la inferencia, que vive en ``src/core``).
Pieza actual: el benchmark **sin ground-truth** de las configuraciones detector ×
tracker (tarea ``benchmark_metrics``) — métricas de eficiencia, trayectoria y máscara
a partir de los JSON de inferencia, más la tabla comparativa.

La evaluación con ground-truth manual (mIoU vs humano) es un proceso aparte (pausado).
"""

from __future__ import annotations

from src.eval.benchmark import (
    aggregate_config,
    benchmark_videos,
    comparison_table,
    video_metrics,
    write_table,
)

__all__ = [
    "aggregate_config",
    "benchmark_videos",
    "comparison_table",
    "video_metrics",
    "write_table",
]
