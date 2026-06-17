# Plan — `event_possession`

## Archivos

### Nuevo: `src/core/events.py`
Módulo de análisis de eventos (Capa A). Arranca con la base de carga del JSON (semilla de
`events_core`) + la lógica de posesión. `numpy` perezoso; sin GPU, sin homografía.

- `@dataclass FrameObject` — `obj_id: int`, `class_name: str`, `bbox: tuple` (`[x,y,w,h]`),
  `centroid: tuple` (`[x,y]`), `score: float`.
- `load_frame_objects(tracks_json) -> dict[int, list[FrameObject]]`
  Invierte el JSON (`tracks[].observations[]`) a estructura **por frame**. Base reusable por
  T2+ (se promoverá a `events_core` cuando T2 la consuma). Valida formato de `bbox`/`centroid`.
- `@dataclass PossessionResult` — `por_frame: dict[int, int | None]`, `resumen: dict`.
- `compute_possession(frame_objects, *, gate_k=DEFAULT, min_frames=DEFAULT, fps=None) -> PossessionResult`
  Núcleo: balón→robots→cercanía→gate→histéresis→agregación.
- `write_possession_json(result, path) -> Path` — opcional, persiste el resultado.

Helpers privados:
- `_bbox_diagonal(bbox)` — diagonal del bbox (para el gate relativo al tamaño del robot).
- `_ball_centroid(objs)` — centroide del `orange_ball` de mayor `score` (o `None`).
- `_nearest_robot(ball_xy, robots)` — `(obj_id, dist)` del robot más cercano, o `None`.
- `_apply_hysteresis(serie_cruda, min_frames)` — confirma cambios tras `min_frames`.
- `_summarize(por_frame, n_frames, fps)` — tiempos por `obj_id`, cambios, % controlado/libre/no-visible.

Constantes de default (configurables por parámetro): `gate_k`, `min_frames`.

### Nuevo: `testing/test_event_possession.py`
Script manual (estilo del repo, se corre con `python`). Sobre un JSON real de `outputs/`:
- corre `load_frame_objects` + `compute_possession`,
- imprime el resumen y valida invariantes (% controlado+libre+no_visible ≈ 100; suma de
  tiempos = frames),
- ejerce casos borde (frame sin balón, un solo robot, nadie en el gate),
- **viz de validación**: línea de tiempo de posesión (matplotlib → guarda un `.png` en
  `outputs/`), sin overlay sobre el video.

### No se tocan
`src/core/minimap_pipeline.py` etc. quedan intactos. (Más adelante, T2 podría promover
`load_frame_objects` a `events_core.py`; no en esta tarea.)

## Decisiones de diseño

- **Centroides, no foot-points**: para posesión basta proximidad; el `centroid` del JSON
  sirve directo. (El foot-point es para proyección métrica, Capa B.)
- **Gate relativo al tamaño del robot** (`gate_k × diagonal del bbox`): robusto a zoom/cámara,
  a diferencia de un umbral absoluto en píxeles.
- **Histéresis sobre la serie cruda** (no sobre distancias): primero se calcula el poseedor
  crudo por frame, luego se confirman cambios con `min_frames`. Simple y depurable.
- **Por `obj_id`** (no equipo): la asignación aliado/rival no existe; se deja fuera.
- **Estructura por-frame reusable**: `load_frame_objects` es la semilla de `events_core`
  para T2 (gol en zona), que necesitará balón + bboxes de zonas por frame.

## Pasos de ejecución (detalle en `tasks.md`)
1. `events.py`: `FrameObject` + `load_frame_objects` (carga por frame).
2. `events.py`: helpers (`_bbox_diagonal`, `_ball_centroid`, `_nearest_robot`).
3. `events.py`: `compute_possession` (gate + histéresis) + `_summarize`.
4. `events.py`: `PossessionResult` + `write_possession_json`.
5. `testing/test_event_possession.py`: smoke + invariantes + casos borde + viz timeline.
6. Lint/format (`ruff`, `black`).

## Verificación
Ver `spec.md` §Verificación. Gate: el smoke corre en local sin GPU sobre un JSON existente,
invariantes OK, y la gráfica de timeline se genera.
