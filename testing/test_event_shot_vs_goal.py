"""Harness de tiro a gol vs gol (ronda de entregable de eventos). Corre en CPU local (sin GPU).

Clasifica cada lance cerca de una portería como ``tiro`` o ``gol``, en dos rutas:
- **cm** (cámara superior, autoridad): cruce de la línea de gol en cm + dirección a portería.
- **px** (proxy universal): bbox de la zona encogido + regla de 3/4 hacia la pared.

Valida invariantes, cruza la ruta cm con el gol geométrico (``compute_geometric_goals``) y con
los candidatos de T2 (``compute_goal_zone_events``), cubre casos borde y dibuja una línea de
tiempo tiro-vs-gol.

    python testing/test_event_shot_vs_goal.py [ruta/al/tracks.json]
"""

import json
import sys
from pathlib import Path

from src.core.event_goal_geometric import compute_geometric_goals
from src.core.event_shot_goal import (
    ShotGoalResult,
    _crossed_cm,
    _depth_frac,
    _in_approach,
    _wall_axis,
    compute_shot_vs_goal,
    write_shot_vs_goal_json,
)
from src.core.events_schema import events_paths
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _t2_candidates(tracks: Path) -> dict:
    """Candidatos a gol de T2 (Capa A, bbox en píxeles) para comparar."""
    from src.core.event_goals import compute_goal_zone_events
    from src.core.events_core import load_frame_objects

    r = compute_goal_zone_events(load_frame_objects(tracks))
    return r.resumen.get("eventos_por_zona", {})


def _print_result(tag: str, result: ShotGoalResult) -> None:
    print(f"\n[{tag}] resumen: tiros={result.resumen['tiros']} goles={result.resumen['goles']}")
    for e in result.eventos:
        xy = f"({e.xy_cm[0]:.0f},{e.xy_cm[1]:.0f})cm" if e.xy_cm else "px"
        print(f"  {e.tipo:4s} {e.zona:6s} frames {e.frame_inicio}-{e.frame_fin} "
              f"({e.dur_s}s) ref {xy}")


def _invariants(result: ShotGoalResult) -> None:
    for e in result.eventos:
        assert e.frame_fin >= e.frame_inicio
        assert e.tipo in ("tiro", "gol")
        assert e.zona in ("yellow", "blue")
    r = result.resumen
    assert r["total_eventos"] == r["tiros"]["total"] + r["goles"]["total"]
    assert r["total_eventos"] == len(result.eventos)


def _edge_cases() -> None:
    """Geometría cm y px: cruce estricto / banda de tiro y orientación de la pared."""
    depth, side = 15.0, 12.0
    # cm: gol azul = cruza la línea real (x>=231) dentro de la boca real [61,121].
    assert _crossed_cm((237.0, 91.0), "blue", 0.0) is True
    assert _crossed_cm((226.0, 91.0), "blue", 0.0) is False         # se queda corto -> no gol
    assert _crossed_cm((232.0, 55.0), "blue", 0.0) is False         # pasó la línea pero al poste
    assert _crossed_cm((6.0, 91.0), "yellow", 0.0) is True          # gol amarillo
    # cm: banda de tiro (no gol) — corto en x dentro de boca, o al poste pasado la línea.
    assert _in_approach((226.0, 91.0), "blue", depth, side) is True   # corto = tiro
    assert _in_approach((232.0, 55.0), "blue", depth, side) is True   # al poste = tiro
    assert _in_approach((217.0, 85.0), "blue", depth, side) is True   # parado cerca = tiro
    assert _in_approach((200.0, 91.0), "blue", depth, side) is False  # demasiado lejos
    # px: pared = lado más lejano del centro. bbox abajo (cy>H/2) -> eje y, signo +.
    axis, sign = _wall_axis((290.0, 1646.0, 413.0, 166.0), 1080.0, 1920.0)
    assert axis == 1 and sign == 1.0
    # px: entrada por el costado (balón fuera del rango transversal del bbox) -> no cuenta.
    bbox = (290.0, 1646.0, 413.0, 166.0)
    _, within_other, _ = _depth_frac((900.0, 1730.0), bbox, axis, sign, 0.0)
    assert within_other is False  # x=900 fuera de [290,703]: ni tiro ni gol
    # px: balón en el borde interno (no penetra) -> frac ~0 (< 3/4, no es gol).
    frac, wo, _ = _depth_frac((450.0, 1650.0), bbox, axis, sign, 0.0)
    assert wo is True and frac < 0.75
    print("casos borde OK")


def _empty_px() -> None:
    """Ruta px sobre un JSON sin objetos -> sin eventos (no revienta)."""
    import tempfile

    payload = {"resolution": {"width": 1080, "height": 1920}, "fps": 30, "tracks": []}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        tmp = Path(fh.name)
    try:
        r = compute_shot_vs_goal(tmp, route="px")
        assert r.eventos == []
        assert r.resumen["total_eventos"] == 0
    finally:
        tmp.unlink(missing_ok=True)
    print("ruta px vacía OK")


def _plot(cm: ShotGoalResult, px: ShotGoalResult, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    color = {"gol": "#d62728", "tiro": "#ff7f0e"}
    rows = [("cm (autoridad)", cm.eventos), ("px (proxy)", px.eventos)]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for i, (label, evs) in enumerate(rows):
        for e in evs:
            ax.barh(i, e.frame_fin - e.frame_inicio + 1, left=e.frame_inicio,
                    color=color[e.tipo], edgecolor="black", height=0.5)
            ax.text(e.frame_inicio, i + 0.32, f"{e.tipo[0].upper()}·{e.zona[0]}",
                    fontsize=7, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("frame")
    ax.set_title("Tiro a gol vs gol — línea de tiempo")
    ax.legend(handles=[Patch(color=c, label=t) for t, c in color.items()], loc="upper right")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print("viz:", png_path)


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    cm = compute_shot_vs_goal(tracks, route="cm")
    _print_result("cm", cm)
    _invariants(cm)

    px = compute_shot_vs_goal(tracks, route="px")
    _print_result("px", px)
    _invariants(px)
    print("invariantes OK")

    # --- comparación informativa: el refinamiento es MÁS estricto que las rutas previas ---
    # El gol geométrico laxo (event_goal_geometric) y T2 (bbox px) sobre-cuentan: un tiro al
    # poste o un tiro corto se contaban como gol. La ruta cm estricta los reclasifica como tiro.
    n_gol_cm = cm.resumen["goles"]["total"]
    n_geo = compute_geometric_goals(tracks).resumen["total_eventos"]
    n_t2 = sum(_t2_candidates(tracks).values())
    print(f"\ncomparación (informativa): goles cm (estricto)={n_gol_cm} | "
          f"gol geométrico laxo={n_geo} | candidatos T2 (bbox px)={n_t2}")
    assert n_gol_cm <= n_geo, "el refinamiento estricto no debería contar MÁS goles que el laxo"

    # --- regresión sobre el clip validado a mano (ground truth del equipo) ---
    if tracks.stem == "IMG_9933_5m30":
        g = cm.resumen["goles"]["total"]
        t = cm.resumen["tiros"]["total"]
        print(f"ground truth IMG_9933_5m30: esperado 1 gol + 3 tiros | obtenido {g} gol + {t} tiros")
        assert g == 1, f"se esperaba 1 gol real, hay {g}"
        assert t == 3, f"se esperaban 3 tiros, hay {t}"

    _edge_cases()
    _empty_px()

    stem = tracks.stem
    _plot(cm, px, events_paths(stem, "shot_vs_goal", "png"))
    out = write_shot_vs_goal_json(cm, events_paths(stem, "shot_vs_goal", "json"))
    print("escrito:", out)
    print("OK")


if __name__ == "__main__":
    main()
