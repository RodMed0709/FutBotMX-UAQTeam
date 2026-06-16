# Plan — `event_goal_geometric` (gol geométrico)

## Enfoque

Módulo nuevo `src/core/event_goal_geometric.py` que consume la salida de T3 y emite eventos de
gol en cm. Reusa lo existente:

- `metric_positions.compute_metric_positions` / `MetricResult` (insumo: balón en cm por frame).
- `event_goals._events_from_series` (máquina de estados debounce/cierre/cooldown) — **el mismo
  motor que T2**, alimentado con un booleano por frame.
- `field_template` (líneas de gol, boca, `render_field` para la viz).
- `events_core.BALL_CLASSES`.

## Pasos

1. **Serie booleana por portería** (`_ball_in_goal_series`): de `result.posiciones`, para cada
   `frame_index` con balón, evaluar si **alguna** muestra del balón cae en la región de cada
   portería:
   - amarilla: `x <= GOAL_LINE_X_LEFT_CM + margin` y `_GOAL_TOP_Y_CM - margin <= y <=
     _GOAL_BOTTOM_Y_CM + margin`;
   - azul: `x >= GOAL_LINE_X_RIGHT_CM - margin` y boca con margen.
   Devuelve `{"yellow": {frame: bool}, "blue": {frame: bool}}` sobre el rango de frames con balón.
2. **Eventos por portería** (`compute_geometric_goals`): aplicar `_events_from_series` a cada
   serie booleana (reutilizar firma/params: `min_frames`, `exit_frames`, `cooldown_frames`).
   Para cada evento, registrar `xy_cm` = posición del balón en `frame_inicio`.
3. **Salida** (`GeometricGoalResult`): `eventos` (lista de `GoalEventGeo`) + `resumen`
   (`eventos_por_zona`, total, fps, params, `zonas_evaluadas`). `write_geometric_goals_json`.
4. **API pública**: `compute_geometric_goals(source, *, margin_cm=..., min_frames=...,
   exit_frames=..., cooldown_frames=..., fps=None) -> GeometricGoalResult`; `source` = ruta
   tracks_json (llama a T3) o `MetricResult`.
5. **Test/harness** `testing/test_event_goal_geometric.py`:
   - corre sobre `IMG_9933_5m30.json` (CPU local);
   - imprime resumen + eventos; **invariantes** (frame_fin ≥ frame_inicio; eventos dentro del
     rango; conteo ≥ 0); **comparación con T2** (debe detectar gol(es) en azul como T2);
   - **casos borde** (serie sin balón en boca → 0 eventos; sostenido → 1 evento);
   - **viz**: timeline por portería + marca de `xy_cm` sobre `render_field` → `.png`.

## Decisiones técnicas

- **Reuso del motor de T2**: `_events_from_series` ya implementa el debounce/cierre/cooldown y
  está probado. El gol geométrico solo cambia **cómo se construye el booleano** (línea en cm vs
  bbox en píxeles). Esto deja claro que es un *refinamiento*, no un sistema paralelo.
  - Verificar la firma exacta de `_events_from_series` en `event_goals.py` y, si conviene,
    importarla (es "privada" pero el reuso intencional está aceptado en el plan).
- **margin_cm**: pequeño (p. ej. 5–10 cm) para absorber ruido de H sin inflar; el balón "dentro"
  (x=237) cae holgado.
- **fps**: de `MetricResult.resumen["fps"]` o argumento; si falta, error claro.
- **Varias muestras del balón** (ID-switch): `any(...)` sobre las muestras del frame.

## Riesgos / validación

- Detección del balón con huecos cerca de la portería → el debounce con `min_frames` bajo y un
  `exit_frames`/`cooldown` razonable evita perder el evento o duplicarlo. Calibrar con el clip
  real (los eventos de T2 estaban en frames 840–848 y 1197–1225).
- Ruido de H que mete el balón en la boca por 1–2 frames → `min_frames` lo filtra.

## Estructura de archivos

- `src/core/event_goal_geometric.py` (nuevo).
- `testing/test_event_goal_geometric.py` (nuevo).
