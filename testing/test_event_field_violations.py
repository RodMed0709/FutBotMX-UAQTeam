"""Harness de violaciones de campo (ronda de entregable de eventos). Corre en CPU local (sin GPU).

Detecta ``fuera`` (cm: salida del campo / entrada al área chica), ``lack_of_progress`` (px,
prob.) y ``pushing`` (px, prob., solo en área chica). Valida invariantes, casos borde de
geometría, la coherencia esperada (el balón parado sale como lack_of_progress) y dibuja una viz
(cancha con los `fuera` + línea de tiempo por tipo).

    python testing/test_event_field_violations.py [ruta/al/tracks.json]
"""

import sys
from pathlib import Path

from src.core import field_template as ft
from src.core.event_field_violations import (
    FieldViolationsResult,
    _classify_fuera,
    _in_penalty,
    _out_of_field,
    _penalty_polys,
    compute_field_violations,
    write_field_violations_json,
)
from src.core.events_schema import events_paths
from src.utils import PROJECT_ROOT

DEFAULT_TRACKS = (
    PROJECT_ROOT / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json"
)


def resolve_tracks() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS


def _print(result: FieldViolationsResult) -> None:
    s = result.resumen
    print(f"fuera_disponible={s['fuera_disponible']}  conteo={s['conteo']}  "
          f"total={s['total_eventos']}")
    for e in result.eventos:
        ref = f"({e.ref[0]:.0f},{e.ref[1]:.0f})" if e.ref else "-"
        extra = f"causa={e.causa}" if e.tipo == "fuera" else f"p={e.probabilidad}"
        print(f"  {e.tipo:17s} zona={str(e.zona):5s} obj={e.obj_ids} "
              f"f{e.frame_inicio}-{e.frame_fin} ({e.dur_s}s) {extra} ref={ref}")


def _invariants(result: FieldViolationsResult) -> None:
    for e in result.eventos:
        assert e.frame_fin >= e.frame_inicio
        assert e.tipo in ("fuera", "lack_of_progress", "pushing")
        # causa <=> fuera
        assert (e.causa is not None) == (e.tipo == "fuera"), f"causa inconsistente: {e}"
        # probabilidad 1.0 <=> fuera (geométrico); probabilístico en (0,1)
        assert (e.probabilidad == 1.0) == (e.tipo == "fuera"), f"prob inconsistente: {e}"
        assert 0.0 < e.probabilidad <= 1.0
        if e.tipo == "fuera":
            assert e.causa in ("salida_campo", "area_chica")
            assert len(e.obj_ids) == 1
        if e.tipo == "pushing":
            assert e.zona in ("yellow", "blue"), "pushing siempre dentro de un área chica"
            assert len(e.obj_ids) == 2
        if e.tipo == "lack_of_progress":
            assert e.obj_ids == []
    print("invariantes OK")


def _edge_cases() -> None:
    """Geometría (sin homografía): boca, líneas, área chica."""
    m = 3.0
    polys = _penalty_polys()
    # centro de la cancha: ni fuera ni área chica
    assert _out_of_field((121.0, 91.0), m) is False
    assert _in_penalty((121.0, 91.0), polys, m) is None
    # robot pasado la línea derecha, fuera de la boca (y=151) -> fuera
    assert _out_of_field((245.0, 151.0), m) is True
    # robot pasado la línea derecha pero dentro de la boca (y=91) -> NO fuera (va al gol)
    assert _out_of_field((240.0, 91.0), m) is False
    # punto dentro del área chica azul -> causa area_chica, zona blue
    causa, zona = _classify_fuera((220.0, 91.0), polys, m)
    assert causa == "area_chica" and zona == "blue", (causa, zona)
    # punto fuera del campo (esquina) -> salida_campo
    causa, zona = _classify_fuera((5.0, 5.0), polys, m)
    assert causa == "salida_campo" and zona is None, (causa, zona)
    print("casos borde OK")


def _coherence(result: FieldViolationsResult) -> None:
    """El balón parado (≈ f502-676 visto en event_shot_vs_goal) debe salir como lack_of_progress."""
    lops = [e for e in result.eventos if e.tipo == "lack_of_progress"]
    hit = [e for e in lops
           if e.frame_inicio <= 650 and e.frame_fin >= 520 and e.probabilidad >= 0.9]
    assert hit, "no se detectó lack_of_progress en el tramo del balón parado (≈f502-676)"
    print(f"coherencia OK: lack_of_progress f{hit[0].frame_inicio}-{hit[0].frame_fin} "
          f"p={hit[0].probabilidad} (balón parado)")


def _plot(result: FieldViolationsResult, png_path: Path) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (axf, axt) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1.4]})

    # panel A: cancha con las posiciones de `fuera`
    canvas, to_px = ft.render_field(scale=2.2, margin_cm=10.0)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    col = {"salida_campo": (0, 0, 255), "area_chica": (0, 165, 255)}
    for e in result.eventos:
        if e.tipo == "fuera" and e.ref is not None:
            cv2.circle(canvas, to_px(e.ref), 6, col[e.causa], -1)
    axf.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axf.set_title("Fuera (cm): salida vs área chica")
    axf.axis("off")
    axf.legend(handles=[Patch(color="red", label="salida_campo"),
                        Patch(color="orange", label="area_chica")], loc="lower center", ncol=2)

    # panel B: línea de tiempo por tipo
    rows = {"fuera": 0, "lack_of_progress": 1, "pushing": 2}
    tcol = {"fuera": "#d62728", "lack_of_progress": "#1f77b4", "pushing": "#9467bd"}
    for e in result.eventos:
        axt.barh(rows[e.tipo], e.frame_fin - e.frame_inicio + 1, left=e.frame_inicio,
                 color=tcol[e.tipo], alpha=max(0.35, e.probabilidad), edgecolor="black", height=0.6)
    axt.set_yticks(list(rows.values()))
    axt.set_yticklabels(list(rows.keys()))
    axt.set_xlabel("frame")
    axt.set_title("Violaciones de campo — línea de tiempo (alpha = probabilidad)")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print("viz:", png_path)


def main() -> None:
    tracks = resolve_tracks()
    if not tracks.exists():
        raise FileNotFoundError(f"No hay JSON de tracking: {tracks}")

    result = compute_field_violations(tracks)
    _print(result)
    _invariants(result)
    _edge_cases()
    if tracks.stem == "IMG_9933_5m30":
        _coherence(result)

    stem = tracks.stem
    _plot(result, events_paths(stem, "field_violations", "png"))
    out = write_field_violations_json(result, events_paths(stem, "field_violations", "json"))
    print("escrito:", out)
    print("OK")


if __name__ == "__main__":
    main()
