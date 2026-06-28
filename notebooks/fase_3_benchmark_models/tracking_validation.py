"""Validación de tracking SIN GT (Tabla 10): ByteTrack vs BoT-SORT, objetivo y reproducible.

Mide consistencia de identidad a partir de los **tracks** del JSON de tracking (campo
``tracks``: ``{obj_id, class, observations:[{frame_index, bbox, centroid, score}]}``), sin
ground-truth de tracking (que no existe para este dataset). Es **measurement-only**: no toca
``src/`` ni corre modelos; lee los JSON ya generados por el benchmark.

Métricas (proxies objetivos, declarados como tales en el paper):

- **Oclusión recuperada (intra-track):** un track tiene un hueco de ``>= k`` frames en su
  ``frame_index`` y **reaparece con el MISMO ``obj_id``** -> el tracker mantuvo la identidad a
  través de la oclusión.
- **ID switch (hand-off):** un track termina en el frame ``fe`` cerca de una posición, y otro
  track de la **misma clase** arranca en ``(fe, fe+W]`` a ``< D`` px -> el objeto continuó pero
  con un ``obj_id`` NUEVO (identidad perdida).
- **Occlusion-recovery rate** = recuperadas / (recuperadas + switches).
- **ID consistency** = cobertura intra-track media = media de ``n_obs / span`` por track
  (qué tan contiguo es cada track, ignorando hand-offs).
- Auxiliares: nº de tracks y longitud media de tracklet (fragmentación).

Solo se consideran clases **dinámicas** (robot, orange_ball); zonas y green_floor se excluyen
(estáticas, una sola identidad).

Uso
---
    python notebooks/fase_3_benchmark_models/tracking_validation.py \
        --root outputs/inference/trackers \
        --detector yolo_sam3 --k 5 --window 30 --dist 200 \
        --out outputs/benchmark/tracking_validation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from src.utils import PROJECT_ROOT

DYNAMIC = {"robot", "robot_a", "robot_b", "orange_ball"}


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _score_clip(json_path: Path, k: int, window: int, dist: float) -> dict:
    d = json.loads(json_path.read_text())
    fps = float(d.get("fps") or 30.0)
    nframes = int(d.get("num_frames") or 0)
    tracks = [t for t in d["tracks"] if (t.get("class") in DYNAMIC)]

    recovered = 0          # huecos intra-track (mismo id sobrevive la oclusión)
    span_cov: list[float] = []  # cobertura n_obs/span por track
    tracklet_lens: list[int] = []

    # endpoints por clase para detectar hand-offs
    ends: dict[str, list] = {}
    starts: dict[str, list] = {}

    for t in tracks:
        obs = sorted(t["observations"], key=lambda o: o["frame_index"])
        if not obs:
            continue
        f0, f1 = obs[0]["frame_index"], obs[-1]["frame_index"]
        span = f1 - f0 + 1
        tracklet_lens.append(span)
        span_cov.append(len(obs) / span if span else 1.0)
        # huecos intra-track
        for a, b in zip(obs, obs[1:]):
            gap = b["frame_index"] - a["frame_index"] - 1
            if gap >= k:
                recovered += 1
        cls = t["class"]
        ends.setdefault(cls, []).append((f1, obs[-1]["centroid"]))
        starts.setdefault(cls, []).append((f0, obs[0]["centroid"]))

    # hand-offs: track que termina (no al final del video) y otro de misma clase
    # arranca poco después y cerca -> ID switch
    switches = 0
    for cls, end_list in ends.items():
        start_list = sorted(starts.get(cls, []), key=lambda s: s[0])
        used = [False] * len(start_list)
        for fe, ce in end_list:
            if nframes and fe >= nframes - 2:
                continue  # terminó al final del video (probable salida real)
            best = None
            for i, (fs, cs) in enumerate(start_list):
                if used[i] or fs <= fe or fs > fe + window:
                    continue
                if _dist(ce, cs) <= dist:
                    if best is None or fs < start_list[best][0]:
                        best = i
            if best is not None:
                used[best] = True
                switches += 1

    occl_total = recovered + switches
    return {
        "n_tracks": len(tracks),
        "mean_tracklet": (sum(tracklet_lens) / len(tracklet_lens)) if tracklet_lens else 0.0,
        "recovered": recovered,
        "switches": switches,
        "recovery_rate": (recovered / occl_total) if occl_total else float("nan"),
        "id_consistency": (sum(span_cov) / len(span_cov)) if span_cov else float("nan"),
        "fps": fps,
        "nframes": nframes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/inference/trackers")
    ap.add_argument("--detector", default="yolo_sam3", help="detector fijo (aísla el tracker)")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort"])
    ap.add_argument("--k", type=int, default=5, help="frames de hueco que cuentan como oclusión")
    ap.add_argument("--window", type=int, default=30, help="ventana (frames) de búsqueda de hand-off")
    ap.add_argument("--dist", type=float, default=200.0, help="dist px máx para hand-off")
    ap.add_argument("--out", default="outputs/benchmark/tracking_validation.csv")
    args = ap.parse_args()

    root = _abs(args.root)
    rows: list[dict] = []
    agg: dict[str, dict] = {}
    for trk in args.trackers:
        cfg = f"{args.detector}+{trk}"
        cdir = root / cfg
        jsons = sorted(cdir.glob("*/*.json"))
        if not jsons:
            print(f"[!] sin JSON en {cdir}")
            continue
        A = {"n_tracks": 0, "recovered": 0, "switches": 0,
             "mean_tracklet": [], "id_consistency": []}
        for jp in jsons:
            s = _score_clip(jp, args.k, args.window, args.dist)
            rows.append({"config": cfg, "clip": jp.stem, **{kk: round(vv, 4)
                        if isinstance(vv, float) else vv for kk, vv in s.items()}})
            A["n_tracks"] += s["n_tracks"]
            A["recovered"] += s["recovered"]
            A["switches"] += s["switches"]
            A["mean_tracklet"].append(s["mean_tracklet"])
            if not math.isnan(s["id_consistency"]):
                A["id_consistency"].append(s["id_consistency"])
        occl = A["recovered"] + A["switches"]
        agg[cfg] = {
            "config": cfg, "clip": "ALL", "n_clips": len(jsons),
            "n_tracks": A["n_tracks"], "recovered": A["recovered"], "switches": A["switches"],
            "recovery_rate": round(A["recovered"] / occl, 4) if occl else float("nan"),
            "id_consistency": round(sum(A["id_consistency"]) / len(A["id_consistency"]), 4)
                if A["id_consistency"] else float("nan"),
            "mean_tracklet": round(sum(A["mean_tracklet"]) / len(A["mean_tracklet"]), 1)
                if A["mean_tracklet"] else 0.0,
        }

    out = _abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "clip", "n_tracks", "mean_tracklet", "recovered",
                    "switches", "recovery_rate", "id_consistency", "fps", "nframes"])
        for r in rows:
            w.writerow([r["config"], r["clip"], r["n_tracks"], r["mean_tracklet"],
                        r["recovered"], r["switches"], r["recovery_rate"],
                        r["id_consistency"], round(r["fps"], 2), r["nframes"]])

    print(f"\n=== Tracking validation (k={args.k}, window={args.window}, dist={args.dist}px) ===")
    print(f"detector fijo = {args.detector}; clases dinámicas {sorted(DYNAMIC)}\n")
    hdr = f"{'config':22} {'clips':>5} {'tracks':>7} {'tracklet':>9} {'occl.recov':>11} {'switches':>9} {'recov.rate':>11} {'IDcons':>8}"
    print(hdr); print("-" * len(hdr))
    for cfg, a in agg.items():
        rr = "nan" if isinstance(a["recovery_rate"], float) and math.isnan(a["recovery_rate"]) else f"{a['recovery_rate']*100:.1f}%"
        ic = "nan" if isinstance(a["id_consistency"], float) and math.isnan(a["id_consistency"]) else f"{a['id_consistency']*100:.1f}%"
        print(f"{a['config']:22} {a['n_clips']:>5} {a['n_tracks']:>7} {a['mean_tracklet']:>9.1f} "
              f"{a['recovered']:>11} {a['switches']:>9} {rr:>11} {ic:>8}")
    print(f"\nguardado: {out}  ({len(rows)} filas por-clip)")


if __name__ == "__main__":
    main()
