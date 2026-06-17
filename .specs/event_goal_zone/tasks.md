# Tasks — `event_goal_zone`

> Segunda tarea de fase_5 (Capa A, universal). Código solo a partir de aquí.

## Implementación

- [x] `src/core/events_core.py`: `FrameObject`, `load_frame_objects`, `ball_centroid`,
      `ROBOT_CLASS`/`BALL_CLASSES` (base compartida; `_ball_centroid` → público `ball_centroid`).
- [x] `src/core/events.py`: importa y **re-exporta** desde `events_core` (`__all__`); no rompe
      `from src.core.events import FrameObject`; `compute_possession` intacto.
- [x] **T1 sigue corriendo** verificado (`test_event_possession.py`: invariantes/casos borde OK).
- [x] `src/core/event_goals.py`: helpers `_point_in_bbox`, `_ball_in_zone` (cualquier track de
      la zona), `_zones_present`.
- [x] `src/core/event_goals.py`: `_events_from_series` (debounce `min_frames` / cierre
      `exit_frames` / `cooldown_frames`).
- [x] `src/core/event_goals.py`: `compute_goal_zone_events(...)` →
      `GoalZoneResult(eventos, resumen)`; `GoalEvent`; `write_goal_events_json`.
- [x] `testing/test_event_goal_zone.py`: smoke (reusa `resolve_tracks` de T1); casos borde;
      **viz timeline** (broken_barh por zona) → `.png`.
- [x] `ruff check` limpio.

## Verificación

- [x] Smoke en **local sin GPU** sobre `IMG_9780.json` (solo `yellow_zone`):
      `zonas_presentes=["yellow"]`, no falla por `blue_zone` ausente; **0 eventos**
      CONFIRMADO COMO DATO REAL (el balón nunca llegó a <202px de la zona en el clip; la
      lógica sí detecta — caso borde sintético da 1 evento).
- [x] Debounce/máquina de estados validados en casos borde (dentro sostenido = 1 evento;
      nunca dentro = 0).
- [x] T1 (`event_possession`) sigue funcionando tras mover la base a `events_core`.

## Observación

Para **ver eventos de gol reales** hace falta un clip donde el balón sí entre a una zona
(p. ej. el clip de 35 s generado con `--video`, o un tramo con jugada de gol). `IMG_9780`
(10 s) no tiene jugada hacia la portería amarilla.

## Fuera de alcance (recordatorio)

Gol geométrico en cm (Capa B), atribución (quién marcó), overlay de video (T7).
