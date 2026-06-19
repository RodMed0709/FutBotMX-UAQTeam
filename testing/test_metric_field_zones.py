"""Harness de T6 `metric_field_zones` (fase_5 · Capa B). Corre en CPU local (sin GPU).

Mide presencia y posesión por zona en cm (mitades/tercios), valida invariantes + casos borde y
escribe PNG por esquema.

    python testing/test_metric_field_zones.py [ruta/al/tracks.json]
"""

import json
import sys
from pathlib import Path

from src.core.events_schema import events_paths
from src.core.metric_field_zones import (
    compute_field_zones,
    write_field_zones_json,
    write_zones_png,
)
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT
    / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _edge_cases() -> None:
    """Esquema desconocido → error claro."""
    try:
        compute_field_zones(DEFAULT_TRACKS, schemes=("inexistente",))
    except ValueError:
        pass
    else:
        raise AssertionError("debió fallar con esquema desconocido")
    print("casos borde OK")


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    result = compute_field_zones(tracks)
    print("resumen:\n" + json.dumps(result.resumen, indent=2, ensure_ascii=False))
    print("por_esquema:\n" + json.dumps(result.por_esquema, indent=2, ensure_ascii=False))

    # --- invariantes ---
    for esquema, data in result.por_esquema.items():
        for cat in ("ball", "robot"):
            s = sum(data["presencia"][cat].values())
            assert abs(s - 100.0) <= 0.5 or s == 0.0, \
                f"presencia {cat} en {esquema} no suma ~100: {s}"
    # sesgo azul en el clip de gol (mitades): presencia del balón en azul > amarilla
    mit = result.por_esquema["mitades"]["presencia"]["ball"]
    assert mit["azul"] >= mit["amarilla"], f"el balón no se inclina a la mitad azul: {mit}"
    print("invariantes OK")

    _edge_cases()

    stem = tracks.stem
    for esquema in result.por_esquema:
        out = write_zones_png(
            result, esquema, events_paths(stem, f"field_zones_{esquema}", "png")
        )
        print(f"zonas {esquema}:", out)
    out_json = write_field_zones_json(
        result, events_paths(stem, "field_zones", "json")
    )
    print("escrito:", out_json)
    print("OK")


if __name__ == "__main__":
    main()
