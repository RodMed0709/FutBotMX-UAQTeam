# Plan — `metric_positions` (T3)

## Enfoque

Módulo nuevo `src/core/metric_positions.py` que consume el **JSON de tracking extendido** y
emite **posiciones en cm**. Reusa al máximo lo existente:

- `auto_homography.VideoHomography.update_masks` (homografía camino C, gate + EMA + propagación).
- `inference_schema.decode_rle` (rle COCO → máscara binaria).
- `minimap_pipeline._largest_component` (limpiar `green_floor` a su componente conexa mayor)
  — extraer/compartir si conviene, o replicar la lógica mínima.
- `field_template` (geometría cm + `render_field` para la viz del test).
- `events_core.load_frame_objects` / patrón de carga de `tracks[]` (obj_id estable por frame).

**Diferenciador**: `update_masks` espera `img` solo para el solver HSV interno; en el camino
de máscaras necesita la `carpet_mask` y los centroides de portería. Como los tomamos del JSON
(`frames[].detections`), **no hace falta el frame de imagen ni un detector** → CPU local.
Verificar que `solve_masks`/`_orient_and_build` no requieran el `img` real para nada esencial
(las esquinas interiores salen de la máscara de alfombra + líneas blancas dentro de ella).
> Riesgo a validar en implementación: `solve_masks(img, carpet_mask, ...)` usa `img` para
> detectar las líneas blancas DENTRO de la alfombra (`_white_in_carpet`). Entonces SÍ se
> necesita el frame de imagen. Decisión: **leer el frame del clip** (`iter_frames`/cv2) para
> alimentar `solve_masks`, pero NO correr ningún modelo. Sigue siendo CPU local (solo I/O de
> video). La alfombra/centroides vienen del JSON; el frame solo aporta los píxeles para las
> líneas blancas.

## Pasos

1. **Carga del JSON** (`_load_extended_json`): leer `fps`, `resolution`, `num_frames`;
   construir (a) objetos estables por frame desde `tracks[].observations`
   (`frame -> [(obj_id, class, bbox, centroid)]`, solo robots y balón) y (b) por frame, la
   `carpet_mask` (decode del `rle` de `green_floor`) + centroides de `yellow_zone`/`blue_zone`
   desde `frames[].detections`.
2. **Resolver H por frame** (`_solve_homographies`): instanciar `VideoHomography`; por frame en
   orden, leer el frame del clip (ruta del video del JSON o pasada como arg), llamar
   `update_masks(frame, carpet_mask_limpia, yc, bc)`; guardar `(H, status)` por `frame_index`.
   - `carpet_mask` limpiada con componente conexa mayor.
   - Manejar `green_floor` ausente en un frame → no llamar solve; `update_masks` con máscara
     vacía o saltar e invocar propagación (replicar el comportamiento del minimap).
3. **Proyección** (`_project`): por frame con H disponible, para cada objeto calcular el punto
   fuente (foot-point robots / centroide balón) y `cv2.perspectiveTransform(pt, H)` → cm.
   Etiquetar con `status_H` del frame. Objetos en frames `init` (sin ancla aún) → marcar y
   omitir de la salida métrica o incluir con `xy_cm=null`.
4. **Salida** (`compute_metric_positions(...) -> MetricResult`): dataclass con
   `posiciones` (lista de `{obj_id, class, frame_index, xy_cm, status_H}`) y `resumen`
   (calidad de H + conteos). `write_metric_positions_json(result, path)`.
5. **API pública**: `compute_metric_positions(tracks_json, video=None, *, smooth_beta=0.4)`;
   `video` por defecto se resuelve del campo `video` del JSON (con `get_abs_path`/ruta del clip
   local), permitiendo override.
6. **Test/harness** `testing/test_metric_positions.py`:
   - corre sobre `IMG_9933_5m30.json` (CPU local);
   - imprime resumen de calidad; **invariantes** (posiciones dentro de la cancha salvo margen;
     conteos coherentes); **casos borde** (frame vacío, objeto antes del ancla);
   - **viz**: trayectorias en cm sobre `render_field` → `.png` en `outputs/`.

## Decisiones técnicas

- **Reusar, no reimplementar** el solver de homografía (`VideoHomography`). T3 solo orquesta:
  alimentarlo desde el JSON y proyectar objetos.
- **Frame de imagen**: necesario para `solve_masks` (líneas blancas), NO para inferencia.
  Se lee del clip con `iter_frames`/cv2 (I/O, no modelo). Mantiene "sin GPU".
- **Alineación de índices**: los `frame_index` de `tracks[]` y de `frames[]` casan (mismo clip);
  el frame leído del video debe usar el mismo índice (sin `frame_step`).
- **Foot-point vs centroide**: documentado en spec (supuesto 3). Robots = foot-point; balón =
  centroide.
- **Sistema cm**: `field_template` (sin redefinir). La H de `update_masks` ya mapea imagen→cm
  en ese sistema (es el que usa el minimap).

## Riesgos / validación

- Que `solve_masks` dependa del `img` real (confirmado: sí, por las líneas blancas) → el plan
  ya lo contempla leyendo el frame del clip.
- Resolución del JSON vs del clip descargado: el video del JSON apunta a la ruta del pod
  (`/workspace/...`); el test debe resolver al clip **local**
  (`outputs/inference/fase5_clips/<stem>/<stem>.mp4`) o aceptar `video=` explícito.
- `green_floor` fragmentado → `_largest_component` (igual que el minimap).

## Estructura de archivos

- `src/core/metric_positions.py` (nuevo).
- `testing/test_metric_positions.py` (nuevo).
- Posible utilería compartida: si `_largest_component` se reusa, exponerla (mover a un sitio
  común o importarla de `minimap_pipeline`); decidir en implementación sin duplicar.
