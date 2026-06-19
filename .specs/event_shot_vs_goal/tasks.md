# Tasks — Tiro a gol vs gol (`event_shot_vs_goal`)

- **Tarea atómica:** `event_shot_vs_goal`
- **Paso de la metodología:** 4 (Descomposición en tareas) → habilita el paso 5
  (implementación).
- **Spec/plan de referencia:** `.specs/event_shot_vs_goal/{spec,plan}.md`.

> A partir de aquí (y **solo** aquí) se autoriza escribir/modificar código.

---

## T1 · Estructuras y geometría base

- [x] Crear `src/core/event_shot_goal.py` con `ShotGoalEvent` y `ShotGoalResult`
      (campos del plan §2).
- [x] Importar la geometría de `field_template` (líneas de gol, boca, `PENALTY_DEPTH_CM`)
      y los helpers a reutilizar (`_events_from_series`, `_ball_by_frame`,
      `compute_metric_positions`/`MetricResult`, `load_frame_objects`).
- [x] Docstring de módulo en español (rol: clasificar tiro vs gol; cm = autoridad, px =
      proxy).

**Verificación:** `python -c "import src.core.event_shot_goal"` sin error.

---

## T2 · Ruta cm (autoridad)

- [x] Predicados **estrictos** por muestra (cm): `_crossed_cm(xy)` (cruce de la línea real
      **dentro de la boca real**, sin ensanchar; `goal_margin_cm` opcional) e `_in_approach(xy)`
      (banda `tiro_depth_cm` frente a la línea, boca ±`side_cm` para postes).
- [x] **Sin gate de dirección** (un tiro al poste es estático y aun así es tiro).
- [x] **Relleno de huecos** `_fill_gaps`: la ausencia ≤ `gap_frames` no cierra el lance (balón
      parpadeante) → un balón parado es un solo lance.
- [x] Serie `near = _fill_gaps([(f, present, in_approach)])` por zona sobre el timeline
      contiguo → `_events_from_series` → intervalos.
- [x] **Clasificación del intervalo**: `gol` si algún frame cumplió `_crossed_cm`, si no
      `tiro`; `xy_cm` = muestra que cruzó (gol) o primera presente del intervalo (tiro).

**Verificación:** sobre `IMG_9933_5m30` el resultado es **1 gol + 3 tiros** (ground truth del
equipo); el gol geométrico laxo daba 3 (dos eran tiro al poste / tiro corto).

---

## T3 · Ruta px (proxy universal)

- [x] Centroide del balón vs bbox de `yellow_zone`/`blue_zone` con bbox **encogido** por
      `margin_px`.
- [x] **Eje de profundidad** hacia la pared (lado de la zona más alejado del centro de la
      imagen, eje auto x/y); regla `three_quarter_frac` → `crossed_px` (gol) vs `in_zone_px`
      (tiro).
- [x] Misma maquinaria (`_fill_gaps` + `_events_from_series` + clasificación del intervalo,
      sin dirección); `xy_cm=None`. Proxy conservador: subdetecta goles.

**Verificación:** un balón que entra al bbox **por el costado** (sin cruzar / sin
dirección) **no** se clasifica como gol.

---

## T4 · API pública y resumen

- [x] `compute_shot_vs_goal(source, *, route="cm", margin_cm, tiro_depth_cm,
      three_quarter_frac, margin_px, min_dir_cms, min_frames, exit_frames,
      cooldown_frames, fps)` → `ShotGoalResult` (firma del plan §5).
- [x] `route="cm"`: acepta tracks_json (llama a `compute_metric_positions`) o
      `MetricResult`. `route="px"`: tracks_json (usa `load_frame_objects`).
- [x] `resumen`: conteos por tipo (`tiros`/`goles`) y por zona, fps, `params`, ruta usada.
- [x] `write_shot_vs_goal_json(result, path)` (estilo `write_*` de fase_5).
- [x] Imports perezosos; sin GPU.

**Verificación:** `compute_shot_vs_goal(<json>)` y `...(route="px")` devuelven resultado
coherente; el JSON se escribe.

---

## T5 · Test manual + viz

- [x] Crear `testing/test_event_shot_vs_goal.py` (script directo, sin pytest, sin GPU),
      default sobre `IMG_9933_5m30`.
- [x] Ground truth del clip (aserción dura): `IMG_9933_5m30` ⇒ **1 gol + 3 tiros**.
      Comparación **informativa** con el gol geométrico laxo (3) y T2 (2): el estricto nunca
      cuenta MÁS goles (`#goles_cm ≤ #goles_laxo`).
- [x] Casos borde: balón ausente, sin H, entrada lateral (ruta px) ⇒ no gol.
- [x] Viz: línea de tiempo tiro-vs-gol → `events_paths(stem, "shot_vs_goal", "png")`;
      JSON → `events_paths(stem, "shot_vs_goal", "json")`.

**Verificación:** `python testing/test_event_shot_vs_goal.py` termina OK (local).

---

## T6 · Cierre

- [x] `ruff check` limpio en los archivos nuevos.
- [x] `black` aplicado (donde esté disponible).
- [x] Confirmar que `event_goals.py`/`event_goal_geometric.py`/`metric_positions.py`
      quedaron intactos (solo importados).
- [x] Confirmar con el usuario antes de cualquier commit (constitución §7.1/§11).

---

## Orden sugerido

T1 → T2 (ruta cm, la autoridad; validable en local) → T4 (API/resumen) → T5 (test) → T3
(ruta px, proxy) → T6. T3 puede ir antes de T5 si se quiere cubrir px en el mismo test.

---

## Fuera de alcance (recordatorio del spec)

- No conecta el resultado al overlay (es `event_broadcast_overlay`).
- No deprecia ni cambia la API de `event_goals`/`event_goal_geometric`.
- No detecta otros eventos (fuera, área chica → `event_field_violations`).
