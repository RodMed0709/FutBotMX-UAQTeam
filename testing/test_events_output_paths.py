"""Prueba manual de events_paths (tarea events_output_paths).

Verifica la convención de rutas de salida para productos de eventos
(``outputs/eventos/[<namespace>/]<stem>/<stem>_<kind>.<ext>``) sin GPU ni disco:
es puramente construcción de rutas.

Uso:
    python testing/test_events_output_paths.py
"""

from __future__ import annotations

from src.core.events_schema import events_paths
from src.utils import PROJECT_ROOT


def _check(cond: bool, label: str) -> bool:
    print(f"  [{'ok' if cond else 'FALLO'}] {label}")
    return cond


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
    ok = True

    print("== Ruta básica ==")
    p = events_paths("IMG_9933_5m30", "goal_geometric", "json")
    print(f"  {p}")
    expected = (
        PROJECT_ROOT
        / "outputs"
        / "eventos"
        / "IMG_9933_5m30"
        / "IMG_9933_5m30_goal_geometric.json"
    )
    ok &= _check(p == expected, "estructura outputs/eventos/<stem>/<stem>_<kind>.<ext>")
    ok &= _check(p.is_absolute(), "ruta absoluta")
    ok &= _check(str(p).startswith(str(PROJECT_ROOT)), "bajo PROJECT_ROOT")
    ok &= _check(p.name == "IMG_9933_5m30_goal_geometric.json", "kind y ext en el nombre")

    print("\n== Otra extensión / kind ==")
    p_png = events_paths("clipX", "heatmap_ball", "png")
    print(f"  {p_png}")
    ok &= _check(p_png.suffix == ".png", "extensión .png")
    ok &= _check(p_png.parent.name == "clipX", "carpeta por <stem>")

    print("\n== Con namespace ==")
    p_ns = events_paths("IMG_9938", "demo", "mp4", namespace="clipA")
    print(f"  {p_ns}")
    expected_ns = (
        PROJECT_ROOT
        / "outputs"
        / "eventos"
        / "clipA"
        / "IMG_9938"
        / "IMG_9938_demo.mp4"
    )
    ok &= _check(p_ns == expected_ns, "namespace insertado antes del <stem>")

    print("\n== outputs_dir personalizado ==")
    p_out = events_paths("v", "possession", "json", outputs_dir="salidas")
    ok &= _check("salidas/eventos" in p_out.as_posix(), "respeta outputs_dir")

    print("\n== No crea carpetas ==")
    p_new = events_paths("__stem_inexistente__", "x", "json")
    ok &= _check(not p_new.parent.exists(), "el helper no crea la carpeta padre")

    print("\n== Resultado ==")
    print("  OK" if ok else "  HAY FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
