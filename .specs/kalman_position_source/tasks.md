# Tasks — `kalman_position_source` (SDD-B)

Lista ejecutable derivada del `plan.md`. Orden de menor a mayor acoplamiento. Cada tarea es
auto-contenida y verificable. **No se escribe código fuera de estas tareas.**

---

## T1 — Campo `source` en `MetricPosition`

**Archivo:** `src/core/metric_positions.py`

- [ ] Añadir a la dataclass `MetricPosition` el campo opcional `source: str | None = None`
      (al final, con default → retrocompatible).
- [ ] `write_metric_positions_json`: incluir `"source": p.source` en el dict de cada posición
      **solo si** `p.source is not None` (no altera los JSON actuales).

**Aceptación:**
- `MetricPosition(obj_id, cls, frame_index, xy_cm, status_H)` sigue construyéndose sin pasar `source`.
- Un `MetricResult` sin Kalman serializa **idéntico** a hoy (ningún `source` en el JSON).

---

## T2 — Refinador `apply_kalman_to_metric` + `refine_with_kalman`

**Archivo:** `src/core/kalman_kinematics.py`

- [ ] `apply_kalman_to_metric(metric, kres, *, ball_only=True) -> MetricResult`:
  - [ ] Construir `kf_by_key = {(obj_id, frame_index): KalmanState}` solo para `cls in BALL_CLASSES`.
  - [ ] Recorrer `metric.posiciones`: si la clave está en `kf_by_key`, emitir `MetricPosition` con
        `xy_cm=s.xy_cm` y `source=s.source`; si no, copiar la posición tal cual.
  - [ ] Añadir las posiciones de balón **rellenadas por oclusión** (estados Kalman cuyo
        `(obj_id, frame)` no existía con `xy_cm` en el original) con `source="predicted"`.
  - [ ] Devolver `MetricResult(posiciones=nuevas, resumen=metric.resumen, H_por_frame=metric.H_por_frame)`.
- [ ] `refine_with_kalman(metric, *, ball_only=True) -> MetricResult` (wrapper que corre
      `compute_kalman_states` y llama a `apply_kalman_to_metric`).
- [ ] Importar `BALL_CLASSES` desde donde ya se define (reusar, no redefinir).

**Aceptación:**
- `H_por_frame` y `resumen` de salida son los **mismos objetos/valores** que la entrada.
- Robots/zonas idénticos; balón con `xy_cm` distinto y/o `source` no `None`.
- Frames de oclusión del balón quedan rellenados con `source="predicted"`.

---

## T3 — Guarda conservadora en `event_shot_goal`

**Archivo:** `src/core/event_shot_goal.py`

- [ ] En la ruta cm, al confirmar un **gol** del balón, exigir **≥1 frame con
      `source != "predicted"`** dentro de la ventana del evento; si el cruce se apoya solo en
      frames `predicted`, no declarar gol (degradar/descartar según la lógica de cruce existente).
- [ ] Leer `p.source` donde se obtienen las posiciones del balón por frame; con `source=None`
      (flag off) el comportamiento debe ser **exactamente** el actual.

**Aceptación:**
- Con un `MetricResult` sin Kalman (todas `source=None`), los goles detectados son **idénticos**
      a los de hoy (regresión).
- Con un `MetricResult` refinado, un cruce solo-predicho **no** produce gol.

---

## T4 — Flag `use_kalman` + panel de velocidad en el broadcast

**Archivo:** `src/core/event_broadcast_overlay.py`

- [ ] Añadir `use_kalman: bool = False` a `render_broadcast_overlay`.
- [ ] Tras `metric = compute_metric_positions(...)`: si `use_kalman and metric and not degradado`,
      `kres = compute_kalman_states(metric, fps=fps)` y `metric = apply_kalman_to_metric(metric, kres)`.
      Conservar `kres` para el panel.
- [ ] Helper `_ball_speed_by_frame(kres) -> dict[int, float]` y `v_max` global del balón.
- [ ] `_draw_velocity_panel(width, h, vdata) -> np.ndarray` (estilo de los paneles existentes).
- [ ] `_compose_layout2`: columna izquierda a **3 zonas** cuando hay `vdata`
      (posesión arriba · **velocidad medio** · feed abajo); sin `vdata`, split de 2 zonas actual.
- [ ] `_compose_layout1`: panel de velocidad como bloque opcional (cuando hay `vdata`).
- [ ] Pasar `vdata` (o `None` si flag off) desde el loop a `compose(...)`; ambas firmas
      `_compose_layout{1,2}` ganan `vdata`.

**Aceptación:**
- `use_kalman=False`: video **idéntico** al actual (sin panel de velocidad).
- `use_kalman=True`: estela de balón continua + panel `Ball speed | v_max` entre posesión y feed.
- La homografía embebida (`draw_field_overlay_on_frame`) sigue dibujándose igual.

---

## T5 — Smoke test

**Archivo:** `testing/test_kalman_position_source.py` (script manual, sin GPU)

- [ ] Sobre un `MetricResult` de fixture o un JSON de T3 existente:
  - [ ] `refine_with_kalman` conserva `H_por_frame`/`resumen`.
  - [ ] robots/zonas sin cambios; balón cambiado; oclusión rellenada con `source="predicted"`.
  - [ ] regresión: `event_shot_goal` sobre el `metric` **crudo** == resultado de hoy.

**Aceptación:** corre con `python testing/test_kalman_position_source.py` y pasa todas las aserciones.

---

## T6 — Ablación con vs sin Kalman

**Archivo:** `notebooks/fase_6_kalman/` (script `.py`)

- [ ] Sobre `IMG_9933_5m30`: tabla/figura de goles detectados y `v_max` del balón con vs sin Kalman
      (reusa `compute_shot_vs_goal` + `compute_kalman_states`). Salida a `assets/fase6/`.

**Aceptación:** genera la tabla/figura reproducible.

---

## T7 — Notebook comparativo (test final)

**Archivo:** `notebooks/fase_5_event_analysis/` (`.ipynb`)

- [ ] Renderiza (a) `use_kalman=False`, (b) `use_kalman=True`, (c) `use_kalman=True` con `layout=1`
      y `layout=2`. CPU local; clip **crudo** + JSON de tracking.

**Aceptación:** notebook reproducible que produce los videos para que el equipo elija.

---

## T8 — Verificación en pod

- [ ] Re-render del broadcast con `use_kalman=True` (GPU/clip) para revisión visual.
- [ ] Confirmar que `use_kalman=False` es idéntico al entregable actual.

**Aceptación:** video con balón Kalman + panel de velocidad revisado; off == baseline.

---

## Notas

- T1–T4 = código `src/`; T5 = test; T6–T7 = notebooks; T8 = pod.
- Para el **19**: T1–T5 dan el flag desmontable de bajo riesgo (default-off); T6–T7 son el material
  de validación/elección; T8 cierra la verificación visual.
- Fuera de esta tarea (recordatorio): posesión/violaciones, tracking, elipse de incertidumbre,
  overlays seg/tracking (SDD-A/SDD-C).
