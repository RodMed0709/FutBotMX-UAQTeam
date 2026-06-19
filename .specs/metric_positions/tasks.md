# Tasks — `metric_positions` (T3)

> Primera tarea de la Capa B (métrica, cm). Código solo a partir de aquí.
> Insumo de referencia: `outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json`
> (+ su clip `IMG_9933_5m30.mp4`).

## Implementación

- [x] `src/core/metric_positions.py`: carga del JSON extendido; objetos estables por frame vía
      `minimap_pipeline._load_tracks_from_json` (reuso: ya da `(obj_id, cls, foot_xy)` solo
      robots/balón); por frame, `carpet_rle` de `green_floor` (mayor bbox) + centroides de
      `yellow_zone`/`blue_zone` (`_load_field_anchors`, campo `centroid` del JSON).
- [x] `metric_positions.py`: limpieza de alfombra reusando `minimap_pipeline._largest_component`
      (no se duplicó) tras `inference_schema.decode_rle`.
- [x] `metric_positions.py`: `_solve_homographies(...)` con `VideoHomography.update_masks`,
      leyendo el frame del clip (cv2) para las líneas blancas; `(H, status)` por `frame_index`;
      `green_floor` ausente → máscara vacía → propaga H previa.
- [x] `metric_positions.py`: foot-point ya viene de `_load_tracks_from_json` (`_foot_point`);
      proyección con `homography.project_points` (cv2.perspectiveTransform) → cm + `status_H`.
- [x] `metric_positions.py`: `compute_metric_positions(tracks_json, video=None, *,
      smooth_beta=0.4) -> MetricResult` + `write_metric_positions_json`.
- [x] `metric_positions.py`: `_resolve_clip` (override `video=`; default = clip junto al JSON,
      NO la ruta `/workspace/...` del pod).
- [x] `testing/test_metric_positions.py`: corre sobre `IMG_9933_5m30.json` (CPU local);
      resumen; **invariantes** (estados suman n_frames; ≥80% dentro de la cancha; pct_H>0);
      **casos borde** (sin `include_masks` → ValueError; sin frames → anchors vacío);
      **viz** trayectorias en cm sobre `field_template.render_field` → `.png`.
- [x] `ruff check` limpio.

## Verificación

- [x] Corre en **local sin GPU** (no carga SAM3/YOLO; solo cv2 + pycocotools).
- [x] Calidad de H coherente con el minimap: 1799 frames → 103 estimadas / 1672 propagadas /
      24 rechazadas → **96.1% H válida**; 6386/6714 posiciones con cm.
- [x] **100% de posiciones dentro de la cancha** (±40 cm). Balón en los frames de gol:
      **frame 844 → (225, 62) cm** (boca portería azul), **frame 1210 → (237, 110) cm**
      (dentro de la portería azul, x=237=`BLUE_GOAL_X_CM`) — coherente con T2.
- [x] Casos borde no rompen.

## Fuera de alcance (recordatorio)

Velocidad/distancia (T4), heatmap (T5), zonas del campo (T6), gol geométrico (refina T2),
overlay de video (T7), gate automático de elegibilidad Capa B.

## Notas

- Si `solve_masks` resulta NO necesitar el `img` real para algún clip (líneas detectables solo
  de la máscara), se puede evitar leer el video; pero por defecto el plan lee el frame
  (las líneas blancas se detectan sobre la imagen dentro de la alfombra).
- `start_frame`/`frame_step` en tracking siguen pendientes (`frame_window_sampling`); T3 no los
  necesita (opera sobre el clip ya cortado, índices 1:1).
