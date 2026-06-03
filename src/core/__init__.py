"""Submodulo core del proyecto FutBotMX.

Contiene la logica central del pipeline. Primera pieza: extraccion de frames de
un video (tarea frame_extraction).
"""

from __future__ import annotations

from src.core.frame_extraction import extract_frames

__all__ = ["extract_frames"]
