"""Submodulo data del proyecto FutBotMX.

Agrupa la preparacion y organizacion del dataset (no la logica de inferencia del
pipeline, que vive en ``src/core``). Pieza actual: generacion y validacion del
manifiesto de metadatos del dataset (tarea csv_dataset_metadata).
"""

from __future__ import annotations

from src.data.metadata import build_metadata_csv, validate_metadata_schema

__all__ = [
    "build_metadata_csv",
    "validate_metadata_schema",
]
