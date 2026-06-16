# Plan — `event_goal_zone`

## Archivos

### Nuevo: `src/core/events_core.py`  (promoción de la base compartida)
Mueve aquí lo común que hoy vive en `events.py`:
- `FrameObject` (dataclass).
- `load_frame_objects(tracks_json) -> dict[int, list[FrameObject]]`.
- `_ball_centroid(objs)` (balón de mayor score) — compartido por T1 y T2.
- `BALL_CLASSES`, `ROBOT_CLASS` y constantes de clases.

### Modificar: `src/core/events.py`  (posesión, T1)
- Importar `FrameObject`, `load_frame_objects`, `_ball_centroid`, constantes desde
  `events_core` (y **re-exportarlos** para no romper imports existentes:
  `from src.core.events import FrameObject` debe seguir funcionando).
- El resto (`compute_possession`, etc.) intacto.

### Nuevo: `src/core/event_goals.py`  (T2)
- `YELLOW_ZONE="yellow_zone"`, `BLUE_ZONE="blue_zone"`; mapa zona→etiqueta corta.
- `@dataclass GoalEvent` — `zona: str`, `frame_inicio: int`, `frame_fin: int`,
  `dur_frames: int`, `dur_s: float | None`.
- `@dataclass GoalZoneResult` — `eventos: list[GoalEvent]`, `resumen: dict`.
- Helpers privados:
  - `_point_in_bbox(pt, bbox, margin)` — punto-en-rectángulo (bbox `[x,y,w,h]`).
  - `_ball_in_zone(ball_xy, zone_objs, margin)` — dentro de **cualquier** bbox de la zona.
  - `_zones_present(by_frame)` — qué zonas (`yellow`/`blue`) aparecen en el JSON.
  - `_events_from_series(serie_dentro, zona, *, min_frames, exit_frames, cooldown, fps)`
    — máquina de estados (debounce/cierre/cooldown) → lista de `GoalEvent`.
- `compute_goal_zone_events(by_frame, *, margin, min_frames, exit_frames, cooldown_frames,
  fps) -> GoalZoneResult` — orquesta: por cada zona presente arma la serie dentro/fuera y
  extrae eventos; agrega el resumen.
- `write_goal_events_json(result, path)` — opcional.

### Modificar: `testing/test_event_possession.py` → harness reusable
Ya imprime video/duración y resuelve el JSON. Se **extiende** (o se añade
`testing/test_event_goal_zone.py` que reusa su resolución de JSON) para:
- correr `compute_goal_zone_events`, validar (zona ausente no rompe; debounce no duplica),
- **viz**: timeline de intervalos balón-en-zona por color de zona.

Decisión: **nuevo `testing/test_event_goal_zone.py`** (un test por tarea), reutilizando la
función `resolve_tracks()` del harness de T1 (se importa o se duplica mínima).

## Decisiones de diseño

- **Módulo propio `event_goals.py`** (no meter todo en `events.py`): cada tipo de evento en
  su archivo, sobre la base común `events_core`. Mantiene `events.py` enfocado en posesión.
- **Máquina de estados por zona** (no umbral por frame): un evento = entrada sostenida;
  debounce/exit/cooldown evitan parpadeo y doble conteo.
- **"Dentro de cualquiera"** ante fragmentación de la zona (varios tracks del mismo color).
- **Zonas presentes dinámicas**: si `blue_zone` no está en el JSON, no se procesa (sin error).
- **Punto-en-bbox** (centroide del balón), no IoU: simple y suficiente para "balón en zona".

## Pasos de ejecución (detalle en `tasks.md`)
1. Crear `events_core.py` (mover base) + ajustar `events.py` (import/re-export); verificar
   que T1 sigue corriendo.
2. `event_goals.py`: helpers (punto-en-bbox, balón-en-zona, zonas presentes).
3. `event_goals.py`: máquina de estados `_events_from_series` + `compute_goal_zone_events`.
4. `event_goals.py`: `GoalEvent`/`GoalZoneResult` + `write_goal_events_json`.
5. `testing/test_event_goal_zone.py`: smoke + casos borde + viz timeline.
6. Lint (`ruff`).

## Verificación
Ver `spec.md` §Verificación. Gate: smoke local sin GPU sobre un JSON existente; zona ausente
no rompe; T1 intacto.
