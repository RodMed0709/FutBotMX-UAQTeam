# Tasks — `event_possession`

> Primera tarea de fase_5 (Capa A, universal). Código solo a partir de aquí.

## Implementación

- [x] `src/core/events.py`: `FrameObject` (dataclass) + `load_frame_objects(tracks_json)` →
      `dict[frame_index -> [FrameObject]]` (invierte `tracks[].observations[]`; valida
      `bbox`=[x,y,w,h] y `centroid`=[x,y]).
- [x] `src/core/events.py`: helpers `_bbox_diagonal`, `_ball_centroid` (orange_ball de mayor
      score), `_nearest_robot` (obj_id + distancia mínima).
- [x] `src/core/events.py`: `_apply_hysteresis(serie, min_frames)` + `_summarize(por_frame,
      ball_visible, fps)` (tiempos por obj_id, cambios, % controlado/libre/no-visible).
- [x] `src/core/events.py`: `compute_possession(frame_objects, *, gate_k, min_frames, fps)`
      → `PossessionResult(por_frame, resumen)`; `write_possession_json(result, path)`.
- [x] `testing/test_event_possession.py`: smoke sobre un JSON real de `outputs/`; valida
      invariantes (% controlado+libre+no_visible ≈ 100; suma de tiempos = frames); casos
      borde (sin balón, un robot, robot pegado al balón); **viz timeline** matplotlib → `.png`.
- [x] `ruff check` limpio (`black` no instalado en el env local; estilo conforme).

## Verificación

- [x] Smoke corre en **local sin GPU** sobre `outputs/inference/fase3_eventos/IMG_9780/
      IMG_9780.json`: 299 frames, fps 30; posesión por robot (#2 domina 2.47s), 8 cambios;
      timeline generado (`outputs/event_possession_timeline.png`).
- [x] Invariantes numéricas OK (controlado 38.5 + libre 22.4 + no_visible 39.1 = 100).

## Observación

- `pct_no_visible` ≈ 39% en este video: la detección de `orange_ball` es **intermitente**
  (el balón no se detecta en ~4 de cada 10 frames). No es bug del análisis; es calidad de
  detección del JSON. La histéresis ya cubre huecos breves. Si molesta, se mitiga con mejor
  detección de balón o interpolación (futuro, no en esta tarea).

## Fuera de alcance (recordatorio)

Posesión métrica/cm, equipos, gol en zona (T2), overlay de video (T7). `load_frame_objects`
se promoverá a `events_core` cuando T2 lo reuse (no aquí).
