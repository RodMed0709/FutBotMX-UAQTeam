"""Precision/Recall de EVENTOS DISCRETOS: detectores desplegados vs GT manual (por intervalos).

Mide la calidad de los detectores de eventos **desplegados** (capa métrica cm, la autoridad
cámara superior que alimenta el overlay) contra un GT ligero anotado a mano como **intervalos
de tiempo** (un evento es una duración, no un frame). Cubre el set completo:

    gol, tiro                 -> event_shot_goal.compute_shot_vs_goal (cm si hay homografía,
                                 px proxy si no — espeja el detector desplegado)
    fuera, lack_of_progress,  -> event_field_violations.compute_field_violations
    pushing

GT (CSV en segundos, lo produce ``gt_tabla_to_csv.py`` desde la tabla del libro mayor):
    clip, t_inicio, t_fin, tipo, calificador, note
(``t_fin`` vacío => evento puntual = ``t_inicio``).

Procedimiento (measurement-only, sin GPU/SAM3 nuevo; solo lee tracking JSON + CSV de GT):
1. Por clip, corre ambos detectores -> eventos predichos ``[frame_inicio, frame_fin]``.
2. Convierte cada intervalo del GT de segundos a frames con el fps del clip (del JSON).
3. Empareja, **por tipo**, GT<->predicho 1-a-1 por **solape temporal** (con holgura ``--tol``
   frames; ``--strict-qual`` exige además que coincida zona/causa). TP/FP/FN -> P, R, F1.

NOTA de alcance (honestidad): gol/tiro usan la ruta cm (autoridad) en clips de cámara superior
con homografía (JSON con ``include_masks=True``), y caen a la ruta px (proxy universal,
subdetecta) cuando la homografía no es fiable — igual que ``event_broadcast_overlay`` desplegado.
``fuera`` es cm-only; ``lack_of_progress``/``pushing`` son px. NO toca el pipeline ni ``src/``. LOCAL.

Uso
---
    python notebooks/fase_5_event_analysis/eventos_discretos_eval.py \
        --gt outputs/eventos_gt/events_gt.csv \
        --clips IMG_9933_5m30=outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json \
                IMG_9933_min1=outputs/inference/fase5_clips/IMG_9933_min1/IMG_9933_min1.json \
                IMG_9938_min1=outputs/inference/fase5_clips/IMG_9938_min1/IMG_9938_min1.json \
        --out outputs/eventos_gt/eventos_pr.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils import PROJECT_ROOT

EVENT_TYPES = ["gol", "tiro", "fuera", "lack_of_progress", "pushing"]


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def _clip_fps(json_path: Path) -> float:
    d = json.load(open(json_path))
    return float(d.get("fps") or 30.0)


def _predicted_events(stem: str, json_path: Path) -> list[dict]:
    """Eventos predichos por los detectores desplegados, normalizados a un esquema común."""
    from src.core.event_shot_goal import compute_shot_vs_goal
    from src.core.event_field_violations import compute_field_violations
    from src.core.metric_positions import compute_metric_positions

    out: list[dict] = []
    # La homografía necesita el clip crudo. El JSON guarda una ruta del pod (.MOV) que no
    # resuelve en local; el `<stem>.mp4` co-localizado SÍ es el clip crudo (green_floor se
    # excluye del render, las líneas blancas quedan visibles). Lo pasamos explícito.
    clip = json_path.parent / (json_path.stem + ".mp4")
    metric = compute_metric_positions(json_path, video=clip) if clip.exists() else None
    # Espeja el detector DESPLEGADO (event_broadcast_overlay.py:351-352): cm cuando hay
    # homografía, px (proxy universal) cuando no. Así clips no-cenitales (homografía
    # degradada) sí reportan gol/tiro por px en vez de salir todos FN.
    sg = (
        compute_shot_vs_goal(metric, route="cm")
        if metric is not None
        else compute_shot_vs_goal(json_path, route="px")
    )
    for e in sg.eventos:  # tipo gol|tiro, calificador = zona
        out.append({"clip": stem, "tipo": e.tipo, "calificador": e.zona or "",
                    "frame_inicio": e.frame_inicio, "frame_fin": e.frame_fin})
    fv = compute_field_violations(json_path)
    for e in fv.eventos:  # fuera (causa), lack_of_progress, pushing (zona)
        cal = e.causa if e.tipo == "fuera" else (e.zona or "")
        out.append({"clip": stem, "tipo": e.tipo, "calificador": cal or "",
                    "frame_inicio": e.frame_inicio, "frame_fin": e.frame_fin})
    return out


def _match(gt: list[dict], pred: list[dict], tol: int, strict_qual: bool) -> tuple[int, int, int, list]:
    """Empareja GT<->predichos 1-a-1 por solape temporal (mismo tipo). Devuelve TP, FP, FN, detalle."""
    used = [False] * len(pred)
    tp = 0
    detail: list[dict] = []
    for g in gt:
        g0, g1, gq = g["_f0"], g["_f1"], g.get("calificador", "")
        best, best_ov = None, None
        for i, p in enumerate(pred):
            if used[i]:
                continue
            if strict_qual and gq and p["calificador"] and p["calificador"] != gq:
                continue
            # solape (con holgura tol): los intervalos se tocan dentro de tol frames
            if g0 <= p["frame_fin"] + tol and g1 >= p["frame_inicio"] - tol:
                ov = min(g1, p["frame_fin"]) - max(g0, p["frame_inicio"])  # frames solapados
                if best_ov is None or ov > best_ov:
                    best, best_ov = i, ov
        if best is not None:
            used[best] = True
            tp += 1
            detail.append({**g, "match": "TP",
                           "pred_frames": f"[{pred[best]['frame_inicio']},{pred[best]['frame_fin']}]",
                           "pred_cal": pred[best]["calificador"]})
        else:
            detail.append({**g, "match": "FN", "pred_frames": "", "pred_cal": ""})
    fn = len(gt) - tp
    fp = used.count(False)
    for i, p in enumerate(pred):
        if not used[i]:
            detail.append({"clip": p["clip"], "t_inicio": "", "t_fin": "", "tipo": p["tipo"],
                           "calificador": "", "note": "", "match": "FP",
                           "pred_frames": f"[{p['frame_inicio']},{p['frame_fin']}]",
                           "pred_cal": p["calificador"]})
    return tp, fp, fn, detail


def _pr(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="CSV en segundos (clip,t_inicio,t_fin,tipo,calificador,note)")
    ap.add_argument("--clips", nargs="+", required=True, help="pares stem=ruta_tracking_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=int, default=15, help="holgura en frames para el solape temporal")
    ap.add_argument("--strict-qual", action="store_true",
                    help="exigir que coincida el calificador (zona/causa)")
    args = ap.parse_args()

    gt_rows = list(csv.DictReader(open(_abs(args.gt))))
    clip_paths = dict(c.split("=", 1) for c in args.clips)

    by_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in EVENT_TYPES}
    rows_out: list[dict] = []
    print(f"\n=== P/R eventos discretos (solape temporal, tol={args.tol} frames"
          f"{', calificador estricto' if args.strict_qual else ''}) ===")
    hdr = f"{'tipo':<17} {'GT':>3} {'pred':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>5} {'R':>5} {'F1':>5}"

    for stem, json_path in clip_paths.items():
        jp = _abs(json_path)
        fps = _clip_fps(jp)
        pred_all = _predicted_events(stem, jp)
        # GT de este clip -> frames
        gt_clip = []
        for r in gt_rows:
            if r["clip"] != stem:
                continue
            t0 = float(r["t_inicio"])
            t1 = float(r["t_fin"]) if str(r.get("t_fin", "")).strip() else t0
            gt_clip.append({**r, "_f0": round(t0 * fps), "_f1": round(t1 * fps)})
        print(f"\n[{stem}]  (fps={fps:.2f})")
        print(hdr)
        for t in EVENT_TYPES:
            gt_t = [r for r in gt_clip if r["tipo"] == t]
            pred_t = [p for p in pred_all if p["tipo"] == t]
            if not gt_t and not pred_t:
                continue
            tp, fp, fn, detail = _match(gt_t, pred_t, args.tol, args.strict_qual)
            p, r, f1 = _pr(tp, fp, fn)
            by_type[t]["tp"] += tp
            by_type[t]["fp"] += fp
            by_type[t]["fn"] += fn
            rows_out.extend(detail)
            print(f"{t:<17} {len(gt_t):>3} {len(pred_t):>4} {tp:>3} {fp:>3} {fn:>3} "
                  f"{p:>5.2f} {r:>5.2f} {f1:>5.2f}")

    out = _abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "t_inicio", "t_fin", "tipo", "calificador",
                                          "note", "match", "pred_frames", "pred_cal"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    print(f"\n=== AGREGADO por tipo (todos los clips) ===")
    print(f"{'tipo':<17} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>5} {'R':>5} {'F1':>5}")
    tot = {"tp": 0, "fp": 0, "fn": 0}
    for t in EVENT_TYPES:
        c = by_type[t]
        if c["tp"] + c["fp"] + c["fn"] == 0:
            continue
        p, r, f1 = _pr(c["tp"], c["fp"], c["fn"])
        for k in tot:
            tot[k] += c[k]
        print(f"{t:<17} {c['tp']:>3} {c['fp']:>3} {c['fn']:>3} {p:>5.2f} {r:>5.2f} {f1:>5.2f}")
    P, R, F1 = _pr(tot["tp"], tot["fp"], tot["fn"])
    print("-" * 48)
    print(f"{'TOTAL':<17} {tot['tp']:>3} {tot['fp']:>3} {tot['fn']:>3} {P:>5.2f} {R:>5.2f} {F1:>5.2f}")
    print(f"\nguardado detalle: {out}")


if __name__ == "__main__":
    main()
