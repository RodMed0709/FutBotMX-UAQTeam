# Plan — Ventana y submuestreo de frames

> SDD retroactiva (capacidad ya implementada). Documenta el diseño realizado.

## Archivos afectados

1. `src/core/frame_extraction.py` — `iter_frames`: añadir `start_frame` y `frame_step`
   (keyword, con defaults), validarlos, e iterar `range(start_frame, total, frame_step)`
   cortando al alcanzar `max_frames` frames entregados. El `frame_index` emitido sigue
   siendo el índice del **video fuente**.
2. `src/core/minimap_pipeline.py` — `render_minimap_video`: exponer `start_frame`/`frame_step`,
   pasarlos a `iter_frames`, dimensionar la barra con el conteo *strided*, escribir a
   `fps/frame_step`, y limitar el default de `max_frames` por `tracks_json` al recorrido
   completo (`start_frame=0`, `frame_step=1`).

## Decisiones de diseño

- **`frame_index` = índice fuente** (no un contador 0..N): así sigue casando con
  `tracks_json`, cuyas observaciones se indexan por frame fuente.
- **`max_frames` = cantidad entregada**, no rango. Mantiene la semántica previa ("primeros
  N") y se compone bien con `start_frame`/`frame_step`.
- **Defaults neutros** (`0`/`1`) → retrocompatibilidad total; no se tocan otros consumidores.
- **fps de salida = `fps_fuente / frame_step`** en el driver (no en `iter_frames`, que no
  escribe video): conserva la velocidad real del clip submuestreado.

## Pasos de ejecución (hechos)

1. Extender `iter_frames` (parámetros + validación + iteración strided). 
2. Exponer y propagar en `render_minimap_video` (barra, fps, default de tracks).
3. Verificar índices/validaciones en local; integración (clip `IMG_9933_c`) en pod.

## Verificación

Ver `spec.md` §Verificación.
