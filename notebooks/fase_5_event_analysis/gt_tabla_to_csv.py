"""Convierte la tabla MD de GT de eventos (en el libro mayor) -> CSV para el evaluador.

Lee el archivo markdown indicado, extrae la tabla delimitada entre los marcadores
``<!-- GT_EVENTOS_START -->`` y ``<!-- GT_EVENTOS_END -->`` y escribe un CSV
``clip,t_inicio,t_fin,tipo,calificador,note`` (segundos) que consume
``eventos_discretos_eval.py``.

La tabla debe tener columnas (en este orden):
    | # | clip | t_inicio | t_fin | tipo | calificador | nota |
- ``t_inicio``/``t_fin`` en segundos (admite ``mm:ss`` o ``s.s``); ``t_fin`` vacío => puntual.
- ``tipo`` ∈ gol|tiro|fuera|lack_of_progress|pushing.
- ``calificador``: portería yellow|blue (gol/tiro/pushing) o causa salida_campo|area_chica (fuera).
Las filas vacías / de plantilla (sin clip o sin tipo) se ignoran.

Uso
---
    python notebooks/fase_5_event_analysis/gt_tabla_to_csv.py \
        --md .specs/drafts/paper_hallazgos_evidencia.md \
        --out outputs/eventos_gt/events_gt.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from src.utils import PROJECT_ROOT

START, END = "<!-- GT_EVENTOS_START -->", "<!-- GT_EVENTOS_END -->"
COLS = ["clip", "t_inicio", "t_fin", "tipo", "calificador", "note"]

# normalización (la anotación humana llega con mayúsculas / sinónimos / sufijos)
TIPO_MAP = {"gol": "gol", "tiro": "tiro", "fuera": "fuera", "pushing": "pushing",
            "lack": "lack_of_progress", "lop": "lack_of_progress",
            "lack_of_progress": "lack_of_progress"}
CAL_MAP = {"yellow": "yellow", "amarillo": "yellow", "amarilla": "yellow",
           "blue": "blue", "azul": "blue",
           "salida_campo": "salida_campo", "salida": "salida_campo",
           "area_chica": "area_chica", "area": "area_chica", "": ""}


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def _to_seconds(s: str) -> str:
    """Acepta '2.5', '2.5s', '0:12.5' (con o sin sufijo) -> segundos como string. Vacío -> ''."""
    s = re.sub(r"[^0-9:.]", "", s.strip())  # quita 's', 'seg', espacios, etc.
    if not s:
        return ""
    if ":" in s:
        m, sec = s.split(":", 1)
        return str(round(int(m) * 60 + float(sec), 3))
    return str(float(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = _abs(args.md).read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"no encuentro los marcadores {START} / {END} en {args.md}")
    block = text.split(START, 1)[1].split(END, 1)[0]

    rows: list[dict] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        _, clip, t0, t1, tipo, cal, nota = cells[:7]
        if clip.lower() in ("clip", "") or tipo.lower() in ("tipo", "") or set(clip) <= {"-", ":"}:
            continue  # header, separador o fila plantilla
        tipo_n = TIPO_MAP.get(tipo.strip().lower(), tipo.strip().lower())
        cal_n = CAL_MAP.get(cal.strip().lower(), cal.strip().lower())
        rows.append({"clip": clip.strip(), "t_inicio": _to_seconds(t0), "t_fin": _to_seconds(t1),
                     "tipo": tipo_n, "calificador": cal_n, "note": nota})

    if not rows:
        raise SystemExit("no se extrajo ninguna fila de evento (¿tabla vacía / plantilla?)")

    out = _abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"guardado: {out}  ({len(rows)} eventos)")
    by = {}
    for r in rows:
        by[r["tipo"]] = by.get(r["tipo"], 0) + 1
    print("por tipo:", ", ".join(f"{k}={v}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    main()
