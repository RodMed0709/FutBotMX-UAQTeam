"""Convención de rutas de salida para los productos de eventos (fase_5 y la ronda
de entregable de eventos).

Análogo a ``inference_schema.inference_paths`` pero para el dominio de **eventos**:
posesión, gol, violaciones de campo, heatmaps, zonas, overlay de espectador, etc.
Centraliza una sola convención —``outputs/eventos/[<namespace>/]<stem>/<stem>_<kind>.<ext>``—
para que cada video tenga **una carpeta dedicada** con todos sus productos juntos y
nada quede suelto en la raíz de ``outputs/``.

Es una **función pura**: construye la ruta, **no** crea carpetas (eso queda para el
escritor de cada módulo, que ya hace ``parent.mkdir``). Sin dependencias nuevas.
"""

from __future__ import annotations

from pathlib import Path

from src.utils import PROJECT_ROOT


def events_paths(
    stem: str,
    kind: str,
    ext: str,
    *,
    outputs_dir: str = "outputs",
    namespace: str | None = None,
) -> Path:
    """Ruta de un producto de eventos por video.

    Estructura: ``outputs_dir/eventos/[<namespace>/]<stem>/<stem>_<kind>.<ext>``.

    Args:
        stem: nombre base del video (``Path.stem``).
        kind: etiqueta del producto (``"goal_geometric"``, ``"possession"``,
            ``"heatmap_ball"``, ``"field_zones_mitades"``, ``"demo"``, …). La elige el
            llamador; queda como sufijo del archivo para distinguir productos.
        ext: extensión **sin punto** (``"json"``, ``"mp4"``, ``"png"``, …).
        outputs_dir: directorio raíz de salidas (relativo a ``PROJECT_ROOT``); default
            ``"outputs"`` (la convención del repo).
        namespace: subcarpeta opcional (clip/variante/config) insertada **antes** del
            ``<stem>``. ``None`` o cadena vacía ⇒ sin subcarpeta.

    Returns:
        ``Path`` **absoluto** (resuelto contra ``PROJECT_ROOT``). **No** crea carpetas:
        la ruta puede no existir todavía al volver.
    """
    base = PROJECT_ROOT / outputs_dir / "eventos"
    if namespace:
        base = base / namespace
    base = base / stem
    return base / f"{stem}_{kind}.{ext}"
