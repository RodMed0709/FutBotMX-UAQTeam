# Spec — `event_possession`: posesión por cercanía (fase_5, Capa A)

## Contexto

Primera tarea de **fase_5 (análisis de eventos)**, **Capa A — relacional (en píxeles),
universal**: usa relaciones objeto-a-objeto dentro del mismo frame, así que funciona sobre
el JSON de **cualquier** video (cámara superior o Meta-Glasses), sin homografía ni GPU.
Absorbe la base de carga del JSON (la futura `events_core` se extraerá cuando T2 la reuse).

## Objetivo

A partir del JSON de tracking, determinar **quién posee el balón** en cada frame
(robot más cercano dentro de un gate) y derivar **métricas temporales de posesión**:
tiempo por robot, cambios de posesión y % de balón controlado vs libre.

## Entrada

- JSON de tracking de `run_inference(mode="tracking", ...)` (formato `Track`/
  `TrackObservation`: por track `obj_id`, `class`, y `observations[{frame_index, bbox,
  centroid, score}]`). De **cualquier** config 2×2. **No** re-infiere modelos.
- Parámetros (con defaults razonables):
  - `gate_k`: factor del gate de cercanía, relativo al **tamaño del robot**
    (gate = `gate_k × diagonal del bbox del robot`), para ser robusto a zoom/cámara.
  - `min_frames`: histéresis — frames consecutivos mínimos para **confirmar** un cambio
    de posesión (evita parpadeo).
  - `fps`: para convertir frames a segundos (se toma del JSON si está).

## Salida

Estructura Python (dict/dataclass) con:
- `por_frame`: `frame_index -> obj_id | None` (None = balón libre o no visible).
- `resumen`:
  - tiempo de posesión por `obj_id` (frames **y** segundos),
  - nº de cambios de posesión,
  - % de tiempo con balón **controlado** vs **libre**,
  - frames con balón no visible.
- Opcionalmente se **escribe a un JSON** junto a la salida de tracking.

## Método

Por cada `frame_index` del JSON:
1. **Balón**: `centroid` del track `orange_ball` (si hay varios, el de mayor `score`).
   Si no hay balón ese frame → posesión `None` ("balón no visible").
2. **Robots**: `centroid` de cada track `robot` presente en el frame.
3. **Cercanía**: distancia **euclidiana en píxeles** balón↔cada robot.
4. **Asignación**: el robot de **distancia mínima**, si esa distancia < `gate`
   (`gate_k × diagonal del bbox del robot`); si ninguno cumple → `None` (balón libre).
5. **Histéresis**: un cambio de poseedor se confirma solo tras `min_frames` consecutivos
   con el nuevo poseedor (suaviza el parpadeo del frame-a-frame).
6. **Agregación**: a partir de la serie por-frame, calcular el resumen.

Posesión medida **por `obj_id`** (no por equipo; la asignación aliado/rival no existe aún).
Todo es numpy/CPU; reutiliza el patrón de carga del JSON ya presente en el repo
(`_load_tracks_from_json`).

## No-objetivos

- **Posesión métrica (cm)** o por zona del campo — eso es Capa B (T3/T4/T6), cámara superior.
- **Asignación de equipo** (aliado/rival) — extensión futura (p. ej. DINOv3/color).
- **Eventos de gol** — T2 (`event_goal_zone`).
- **Video overlay pulido** — T7. (Aquí solo viz de validación en el test.)

## Verificación

- **Smoke funcional** sobre un JSON existente en `outputs/` (p. ej.
  `outputs/inference/fase3_eventos/IMG_9780/IMG_9780.json`), en local sin GPU:
  - corre sin error, devuelve `por_frame` y `resumen` coherentes (suma de tiempos =
    frames procesados; % controlado + % libre + % no-visible ≈ 100%).
  - casos borde: frames sin balón → `None`; un solo robot; ningún robot dentro del gate.
- **Visualización de validación**: línea de tiempo de posesión (quién posee por frame) como
  resumen impreso y/o gráfica matplotlib simple. Sin overlay sobre el video (eso es T7).
- Lint/format limpios (`ruff`, `black`).
