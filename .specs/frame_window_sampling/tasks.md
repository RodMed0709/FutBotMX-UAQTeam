# Tasks — Ventana y submuestreo de frames

> SDD retroactiva: implementado en la rama `feat/consolidate-homography-path-c`
> (commit `922a5c2`), como capacidad surgida al consolidar fase_4 homografía.

## Hecho

- [x] `iter_frames` acepta `start_frame` (≥0) y `frame_step` (≥1), con defaults `0`/`1`
      (retrocompatible); itera `range(start_frame, total, frame_step)` y corta al entregar
      `max_frames` frames.
- [x] Validación: `start_frame<0` y `frame_step<1` → `ValueError` con mensaje claro.
- [x] `frame_index` emitido = índice del video fuente (casa con `tracks_json`).
- [x] `render_minimap_video` expone `start_frame`/`frame_step` y los propaga a `iter_frames`.
- [x] Barra de progreso dimensionada con el conteo *strided*.
- [x] fps de salida = `fps_fuente / frame_step` (velocidad real del clip submuestreado).
- [x] Default de `max_frames` por `tracks_json` solo en recorrido completo (`start_frame=0`,
      `frame_step=1`).
- [x] Verificación local (índices `1800,1802,…` + validaciones) e integración en pod
      (clip `IMG_9933_c`, duración real 10 s).
- [x] `ruff`/`py_compile` limpios.

## Pendiente

- [ ] (opcional) Exponer `start_frame`/`frame_step` en otros drivers que usan `iter_frames`
      (`track_video`/`run_inference`) si fase_5 lo necesita para acotar tracking a un tramo.

## Notas

- Capacidad **general de `frame_extraction`**, no de homografía; por eso vive en su propia
  SDD y no en `field_homography` (que solo la consume vía `render_minimap_video`).
- Será útil en **fase_5** para extraer clips de eventos (goles/jugadas) sin procesar el
  video completo.
