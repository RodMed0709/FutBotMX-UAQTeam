"""Submodulo data del proyecto FutBotMX.

Agrupa la preparacion y organizacion del dataset (no la logica de inferencia del
pipeline, que vive en ``src/core``). Piezas actuales: generacion y validacion del
manifiesto de metadatos del dataset (tarea csv_dataset_metadata) y exportacion del
set de frames de evaluacion (tarea eval_frame_export).
"""

from __future__ import annotations

from src.data.eval_frames import (
    export_testing_frames,
    validate_testing_frames_schema,
)
from src.data.metadata import build_metadata_csv, validate_metadata_schema

__all__ = [
    "build_metadata_csv",
    "validate_metadata_schema",
    "export_testing_frames",
    "validate_testing_frames_schema",
]
