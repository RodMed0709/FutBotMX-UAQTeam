# Plan — `metric_field_zones` (T6)

## Enfoque

Módulo nuevo `src/core/metric_field_zones.py` que combina T3 (posiciones cm) y T1 (posesión).
Reusa:

- `metric_positions.compute_metric_positions` / `MetricResult` (posiciones cm).
- `events.compute_possession` + `events_core.load_frame_objects` (poseedor por frame, píxeles).
- `field_template` (`LENGTH_CM`, `WIDTH_CM`, `PENALTY_*`, `render_field`).
- `events_core.BALL_CLASSES`.

## Pasos

1. **Esquemas de zona** (`_SCHEMES`): cada esquema = `(labels, fn(x, y) -> label)`:
   - `mitades`: x < L/2 → "amarilla", si no "azul";
   - `tercios`: x < L/3 → "amarillo", < 2L/3 → "medio", si no "azul";
   - (opcional `areas`).
2. **Presencia** (`_presence`): por esquema y categoría (ball/robot), contar muestras `xy_cm`
   válidas por zona (clip a la cancha) → %.
3. **Balón por frame en cm** (`_ball_cm_by_frame`): primera muestra del balón por frame (T3).
4. **Posesión por zona** (`_possession_by_zone`): de `possession.por_frame`, para cada frame con
   owner ≠ None que tenga balón en cm, sumar a la zona del balón → %.
5. **API** `compute_field_zones(tracks_json, *, schemes=("mitades","tercios"), fps=None,
   metric=None) -> FieldZonesResult`: si `metric` None, llama a T3; carga `by_frame` y T1 desde
   el mismo JSON. `write_field_zones_json`.
6. **Render** `render_zones(scheme, presence_ball, posesion, *, scale, margin_cm)` — cancha con
   fronteras + texto de %; `write_zones_png`.
7. **Test** `testing/test_metric_field_zones.py`: corre sobre `IMG_9933_5m30.json`; resumen;
   **invariantes** (presencia suma ~100% por categoría/esquema; sesgo azul en el clip de gol);
   **casos borde** (esquema desconocido → error; sin posesión → 0%); **viz** PNG por esquema.

## Decisiones técnicas

- **Posesión en píxeles, zona en cm**: T1 decide el poseedor por proximidad (robusto en
  píxeles); la **zona** se decide con la posición del balón en cm (T3). Combinación limpia.
- **Esquemas como datos**: añadir un esquema = añadir una entrada a `_SCHEMES`, sin tocar la
  lógica (mismo espíritu config-driven del repo).
- **Clip a la cancha**: igual que T5, posiciones fuera se recortan a `[0,L]×[0,W]`.

## Riesgos / validación

- Balón fragmentado (tracking) → menos frames de posesión; se reporta el nº de frames usados.
- Asignación dura en la frontera (x≈L/2) → aceptable; el sesgo agregado es estable.

## Estructura de archivos

- `src/core/metric_field_zones.py` (nuevo).
- `testing/test_metric_field_zones.py` (nuevo).
