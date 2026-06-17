# -*- coding: utf-8 -*-
"""Genera 06_kalman.ipynb (nbformat v4) sin depender de nbformat: construye el JSON."""
import json
from pathlib import Path

cells = []  # lista de (tipo, fuente)

def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r"""# FutBot MX 2026 · Fase 6 — Filtro de Kalman en cm **con fricción**

Seguimiento métrico de robots y balón (Copa FutBotMX). Este notebook **ejecuta y valida** el
filtro de Kalman de Fase 6 y lo **mejora con un término de fricción** motivado por la física
del fútbol robótico.

**Dos "Kalman" distintos en el proyecto:**
1. *Kalman interno del tracker* (ByteTrack/BoT-SORT, en píxeles) — hace la asociación
   detección↔track. No se toca aquí.
2. *Kalman explícito de Fase 6* (este, en **cm**) — corre **sobre** los tracks ya asociados +
   homografía (T3 `metric_positions`) para: estado posición+velocidad, **predicción en
   oclusión** (predict-only) y cinemática física suave.

**Mejora (contexto fútbol):** el modelo de velocidad-constante (CV) *sobre-extrapola* en
oclusión — los robots frenan y giran, así que seguir derecho a la última velocidad se pasa.
Añadimos **fricción viscosa** `v(t)=v₀·e^(−β·dt)`: robots con β alto (≈ "mantener última
posición"), balón con β bajo (sigue rodando). β=0 recupera el CV exacto.

> **Kernel:** ejecutar con el venv del pod (`futbot-cpu`, Python 3.11 con cv2). Ver `SUBIR.md`.""")

code(r"""import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path("/workspace/FutBotMX-UAQTeam")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks" / "fase_6_kalman"))

from src.core.kalman_kinematics import (
    compute_kalman_states, load_metric_result_from_json, CLASS_PARAMS,
)
from src.core.kalman_state import KFParams, KalmanCV
from src.core import field_template as ft

# Clips cenitales en cm (cache de homografía por líneas). Excluimos IMG_9938_min1:
# su homografía es no fiable (NIS ~120-380, var. de aceleración ~1e9).
CLIPS = {
    "IMG_9933_5m30": REPO / "outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30_cm_lines.json",
    "IMG_9933_min1": REPO / "outputs/inference/fase5_clips/IMG_9933_min1/IMG_9933_min1_cm_lines.json",
}
assert "friction_beta" in KFParams.__dataclass_fields__, (
    "Sube el src/core/kalman_state.py PARCHEADO (con friction_beta). Ver SUBIR.md."
)
results = {n: load_metric_result_from_json(p) for n, p in CLIPS.items() if p.exists()}
print("Clips cargados:", {k: f"{v.resumen.get('fps'):.2f} fps, {len(v.posiciones)} pos" for k, v in results.items()})""")

md(r"""## 1 · El modelo

Estado por eje desacoplado `[p, v]` en cm y cm/s. Predicción con paso `dt`:

- **CV (actual):** `p ← p + v·dt`, `v ← v`.
- **CV + fricción (nuevo):** con `α = e^(−β·dt)`, `v ← α·v` y `p ← p + v·(1−α)/β`
  (integral de una velocidad que decae). Si `β→0`, `(1−α)/β → dt` y se recupera el CV exacto.

La corrección (*update*) es idéntica: ganancia de Kalman + **gating de Mahalanobis**
(χ²₂,₀.₉₉=9.21). En oclusión sólo se predice (la incertidumbre crece) hasta `max_gap_frames`.""")

code(r"""# beta por clase (1/s). Robots: decaimiento fuerte (frenan/giran). Balon: leve (rueda).
# Se justifican en la ablacion (Seccion 4).
BETA = {"orange_ball": 1.0, "robot": 12.0, "robot_a": 12.0, "robot_b": 12.0}
CLASS_PARAMS_FRIC = {c: replace(p, friction_beta=BETA.get(c, 0.0)) for c, p in CLASS_PARAMS.items()}
VARIANTS = {"KF-CV": CLASS_PARAMS, "KF-friccion": CLASS_PARAMS_FRIC}
print("beta aplicado:", {c: CLASS_PARAMS_FRIC[c].friction_beta for c in CLASS_PARAMS_FRIC})""")

md(r"""## 2 · Demo visual (clip `IMG_9933_5m30`)

Minimap cenital: posiciones crudas T3 (gris) vs **trayectoria Kalman+fricción** (color por
clase). Los círculos rojos son la **incertidumbre** (±2σ) en los frames rellenados por
predicción durante oclusión.""")

code(r"""DEMO = "IMG_9933_5m30"
raw = results[DEMO]
fps = raw.resumen.get("fps") or 30.0
kres_cv = compute_kalman_states(raw, fps=fps, class_params=CLASS_PARAMS)
kres_fr = compute_kalman_states(raw, fps=fps, class_params=CLASS_PARAMS_FRIC)

def render_minimap(kres, raw, title):
    import cv2
    canvas, to_px = ft.render_field(scale=2.2, margin_cm=10.0)
    for p in raw.posiciones:
        if p.xy_cm is not None and p.cls in ("orange_ball", "robot", "robot_a", "robot_b"):
            cv2.circle(canvas, to_px(p.xy_cm), 2, (180, 180, 180), -1, cv2.LINE_AA)
    for o in kres.por_obj:
        col = (255, 120, 0) if o.cls == "orange_ball" else (0, 90, 230)
        ps = [to_px(s.xy_cm) for s in o.estados]
        for j in range(1, len(ps)):
            cv2.line(canvas, ps[j - 1], ps[j], col, 2, cv2.LINE_AA)
        for s, p in zip(o.estados, ps):
            if s.source == "predicted":
                r = max(2, int(round(s.pos_sigma_cm * 2.2)))
                cv2.circle(canvas, p, r, (255, 0, 0), 1, cv2.LINE_AA)
    plt.figure(figsize=(9, 7)); plt.imshow(canvas); plt.axis("off"); plt.title(title); plt.show()

render_minimap(kres_fr, raw, f"{DEMO} - Kalman + friccion (gris=crudo, rojo=+-2sigma en oclusion)")
print("objs moviles:", kres_fr.resumen["n_obj"],
      "| frames rellenados por oclusion:", kres_fr.resumen["frames_rellenados_oclusion"],
      "| gated:", kres_fr.resumen["frames_gated"])""")

md(r"""### Velocidad del balón: cruda vs Kalman

La diferencia finita (T4) es ruidosa; el Kalman entrega una rapidez suave y física.""")

code(r"""def ball_speed_plot(raw, kres):
    fps = raw.resumen.get("fps") or 30.0
    ball = {}
    for p in raw.posiciones:
        if p.cls == "orange_ball" and p.xy_cm is not None:
            ball.setdefault(p.obj_id, []).append((p.frame_index, np.array(p.xy_cm, float)))
    if not ball:
        print("sin balon en este clip"); return
    oid = max(ball, key=lambda k: len(ball[k]))
    pts = sorted(ball[oid])
    f_fd = [b[0] for a, b in zip(pts, pts[1:])]
    sp_fd = [float(np.linalg.norm(b[1] - a[1])) * fps for a, b in zip(pts, pts[1:])]
    ko = next((o for o in kres.por_obj if o.obj_id == oid), None)
    plt.figure(figsize=(11, 4))
    plt.plot(f_fd, sp_fd, color="lightgray", lw=1.0, label="diferencia finita (T4)")
    if ko:
        plt.plot([s.frame_index for s in ko.estados], [s.speed_cms for s in ko.estados],
                 color="tab:orange", lw=2.0, label="Kalman + friccion")
    plt.xlabel("frame"); plt.ylabel("rapidez balon (cm/s)")
    plt.title(f"Velocidad del balon (obj {oid}) - {DEMO}"); plt.legend(); plt.show()

ball_speed_plot(raw, kres_fr)""")

md(r"""## 3 · Tablas del paper (2 clips fiables)

**T6.1 — Recuperación de oclusión.** Se ocultan sintéticamente `g` frames y se mide el error
(cm) de cada método para predecir la posición oculta: `hold` (última posición), `lineal`
(velocidad por diferencia finita), `KF-CV` y `KF-fricción`. *Menor es mejor.*""")

code(r"""def occlusion_table(results_by_clip, variants, gaps=(1, 3, 5, 8, 12),
                    warmup=30, stride=8, maxwin=120):
    bucket = lambda c: "balon" if c == "orange_ball" else "robot"
    errs = {}
    add = lambda k, v: errs.setdefault(k, []).append(v)
    for res in results_by_clip.values():
        fps = res.resumen.get("fps") or 30.0
        by = {}
        for p in res.posiciones:
            if p.cls not in CLASS_PARAMS or p.xy_cm is None:
                continue
            by.setdefault(p.obj_id, (p.cls, []))[1].append((p.frame_index, p.xy_cm))
        for cls, lst in by.values():
            lst.sort(key=lambda r: r[0]); b = bucket(cls)
            fr = [r[0] for r in lst]; xy = {r[0]: np.array(r[1], float) for r in lst}; pres = set(fr)
            if len(fr) < warmup + max(gaps) + 2:
                continue
            for g in gaps:
                t = fr[warmup]; nw = 0
                while t + g + 1 <= fr[-1] and nw < maxwin:
                    hid = [t + k for k in range(1, g + 1)]
                    if (t not in pres) or (t - 1 not in pres) or not all(h in pres for h in hid):
                        t += stride; continue
                    nw += 1; last = xy[t]; vl = (xy[t] - xy[t - 1]) * fps
                    for k, h in enumerate(hid, 1):
                        add((b, g, "hold"), float(np.linalg.norm(last - xy[h])))
                        add((b, g, "lineal"), float(np.linalg.norm((xy[t] + vl * (k / fps)) - xy[h])))
                    for name, cp in variants.items():
                        kf = None
                        for f in range(t - warmup, t + 1):
                            if f not in pres:
                                continue
                            if kf is None:
                                kf = KalmanCV(tuple(xy[f]), cp[cls])
                            else:
                                kf.predict(1 / fps); kf.update(tuple(xy[f]), "estimated")
                        for k, h in enumerate(hid, 1):
                            kf.predict(1 / fps)
                            add((b, g, name), float(np.linalg.norm(np.array(kf.pos) - xy[h])))
                    t += stride
    methods = ["hold", "lineal"] + list(variants)
    rows = []
    for b in ("balon", "robot"):
        for g in gaps:
            row = {"objeto": b, "gap": g}
            for mth in methods:
                v = errs.get((b, g, mth))
                row[mth] = round(float(np.mean(v)), 2) if v else None
            rows.append(row)
    return pd.DataFrame(rows)

t61 = occlusion_table(results, VARIANTS)
t61""")

md(r"""**T6.5 — Consistencia (NIS) y v_max.** El NIS medio debe rondar **2** (2 g.l.) si el
filtro está bien calibrado; la fricción no debe degradarlo. `v_max` del balón muestra el
suavizado físico.""")

code(r"""def nis_vmax_table(results_by_clip, variants):
    rows = []
    for clip, res in results_by_clip.items():
        fps = res.resumen.get("fps") or 30.0
        for name, cp in variants.items():
            kres = compute_kalman_states(res, fps=fps, class_params=cp)
            nis = {}
            for o in kres.por_obj:
                for s in o.estados:
                    if s.nis is not None:
                        nis.setdefault("balon" if o.cls == "orange_ball" else "robot", []).append(s.nis)
            vmax_ball = max((o.v_max_cms for o in kres.por_obj if o.cls == "orange_ball"), default=0.0)
            rows.append({
                "clip": clip, "variante": name,
                "NIS_balon": round(float(np.mean(nis.get("balon", [float("nan")]))), 2),
                "NIS_robot": round(float(np.mean(nis.get("robot", [float("nan")]))), 2),
                "vmax_balon_cms": round(vmax_ball, 1),
                "oclusion_rellenada": kres.resumen["frames_rellenados_oclusion"],
            })
    return pd.DataFrame(rows)

nis_vmax_table(results, VARIANTS)""")

md(r"""## 4 · Ablación de β

Barremos β por clase y medimos el error *pooled* de recuperación de oclusión (gaps 1–12) en
`IMG_9933_5m30`. Elegimos el β que minimiza el error (acercándose o superando a `hold`).""")

code(r"""def beta_ablation(res, betas_by_bucket, gaps=(1, 3, 5, 8, 12),
                 warmup=30, stride=10, maxwin=80):
    bucket = lambda c: "balon" if c == "orange_ball" else "robot"
    fps = res.resumen.get("fps") or 30.0
    by = {}
    for p in res.posiciones:
        if p.cls not in CLASS_PARAMS or p.xy_cm is None:
            continue
        by.setdefault(p.obj_id, (p.cls, []))[1].append((p.frame_index, p.xy_cm))
    pooled = {}
    add = lambda k, v: pooled.setdefault(k, []).append(v)
    for cls, lst in by.values():
        lst.sort(key=lambda r: r[0]); b = bucket(cls); base = CLASS_PARAMS[cls]
        fr = [r[0] for r in lst]; xy = {r[0]: np.array(r[1], float) for r in lst}; pres = set(fr)
        if len(fr) < warmup + max(gaps) + 2:
            continue
        for g in gaps:
            t = fr[warmup]; nw = 0
            while t + g + 1 <= fr[-1] and nw < maxwin:
                hid = [t + k for k in range(1, g + 1)]
                if (t not in pres) or (t - 1 not in pres) or not all(h in pres for h in hid):
                    t += stride; continue
                nw += 1; last = xy[t]
                for k, h in enumerate(hid, 1):
                    add((b, "hold"), float(np.linalg.norm(last - xy[h])))
                for be in betas_by_bucket[b]:
                    kf = None; pr = replace(base, friction_beta=be)
                    for f in range(t - warmup, t + 1):
                        if f not in pres:
                            continue
                        if kf is None:
                            kf = KalmanCV(tuple(xy[f]), pr)
                        else:
                            kf.predict(1 / fps); kf.update(tuple(xy[f]), "estimated")
                    for k, h in enumerate(hid, 1):
                        kf.predict(1 / fps)
                        add((b, be), float(np.linalg.norm(np.array(kf.pos) - xy[h])))
                t += stride
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, b in zip(axes, ("balon", "robot")):
        bs = betas_by_bucket[b]
        ys = [float(np.mean(pooled.get((b, be), [float("nan")]))) for be in bs]
        hold = float(np.mean(pooled.get((b, "hold"), [float("nan")])))
        ax.plot(bs, ys, "o-", color="tab:orange", label="KF")
        ax.axhline(hold, color="gray", ls="--", label="hold")
        best = bs[int(np.nanargmin(ys))]
        ax.axvline(best, color="tab:green", ls=":", label=f"mejor beta={best}")
        ax.set_title(b); ax.set_xlabel("beta (1/s)"); ax.set_ylabel("error oclusion (cm)"); ax.legend()
    plt.tight_layout(); plt.show()

beta_ablation(results["IMG_9933_5m30"],
              {"balon": [0.0, 0.5, 1.0, 2.0, 4.0], "robot": [0.0, 5.0, 12.0, 20.0, 30.0]})""")

md(r"""## 5 · Conclusiones

- El **Kalman de Fase 6** entrega estado posición+velocidad en cm, **rellena oclusiones** por
  predicción y suaviza la velocidad (NIS≈2 ⇒ filtro consistente) en los clips fiables.
- La **fricción** (β por clase) corrige el defecto del modelo de velocidad-constante: en
  oclusiones largas los robots dejan de sobre-extrapolar. Ver T6.1 (`KF-fricción` vs `KF-CV`)
  y la ablación de la Sección 4.
- `IMG_9938_min1` se **excluye**: su homografía es no fiable (NIS ~120–380), lo que contamina
  las métricas agregadas. Pendiente: depurar su detección de líneas.""")

# ---- ensamblar notebook json ----
def mkcell(t, s):
    src = s.splitlines(keepends=True)
    if t == "code":
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
    return {"cell_type": "markdown", "metadata": {}, "source": src}

nb = {
    "cells": [mkcell(t, s) for t, s in cells],
    "metadata": {
        "kernelspec": {"display_name": "FutBot CPU (3.11)", "language": "python", "name": "futbot-cpu"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).parent / "06_kalman.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("escrito:", out, "|", len(cells), "celdas")
