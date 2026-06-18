# plan.md — `overlay_mascaras_rle`

## Enfoque

Reusar la maquinaria existente: `track_overlay._compose_frame` ya decodifica `rle`
(`decode_rle`) y mezcla el relleno con color de clase cuando `draw_masks=True`. El
único faltante es un modo que **suprima** cajas/etiquetas/estela para dejar **solo
máscara**, y cablearlo en `main.py` apoyado en un JSON de tracking que **sí** traiga
`rle`.

No se escribe motor de pintado nuevo ni se toca `track_video`, `overlay_detections`,
la fachada `run_inference` ni el broadcast.

## Cambios por archivo

### 1. `src/core/track_overlay.py`

- Helper `_payload_has_rle(payload) -> bool`: True si alguna detección de algún frame
  trae la clave `rle` (corta en el primer hallazgo).
- `_compose_frame(...)`: nuevo parámetro keyword `masks_only: bool`. Cuando es `True`:
  - se fuerza el pintado de máscara (paso 1) y
  - se **omiten** los pasos 2 (trayectorias), 3 (cajas) y 4 (etiquetas).
- `render_obj_id_overlay(...)`: nuevo parámetro `masks_only: bool = False`.
  - Si `masks_only` ⇒ `draw_masks` efectivo = `True`.
  - Si `masks_only` y el JSON **no** trae `rle` ⇒ `ValueError` con mensaje accionable
    ("re-correr la inferencia con `include_masks=True`/`--overwrite`").
  - Propagar `masks_only` a `_compose_frame`.
  - Docstring actualizado.

### 2. `main.py`

- `stage_inference`: `run_inference(..., include_masks=True, ...)` (era `False`).
- `plan_outputs`: sustituir las claves `seg_json`/`seg_video` (ruta en
  `<run_label>/seg/<stem>/`) por una sola `seg_overlay = tracking_json.with_name(
  f"{stem}_seg.mp4")` (junto al JSON de tracking y al `*_obj_id.mp4`).
- `stage_individual_overlays`: reemplazar el bloque de segmentación
  (`run_inference(mode="segmentation")`) por:
  ```python
  if paths["seg_overlay"].exists() and not overwrite:
      out["seg_overlay"] = paths["seg_overlay"]
  else:
      out["seg_overlay"] = render_obj_id_overlay(
          paths["tracking_json"],
          video_path=video,
          output_path=paths["seg_overlay"],
          draw_masks=True,
          masks_only=True,
      )
  ```
  Sin import de `run_inference` en esta etapa (ya no se re-infiere).

## Riesgos y mitigaciones

- **JSON sin `rle` (corridas viejas / `--overwrite` no usado):** `render_obj_id_overlay`
  con `masks_only=True` levanta `ValueError` claro; `main.py` lo captura en su bucle de
  etapas y reporta `fallido` con el mensaje. Se documenta en `tasks.md` que para datos
  ya generados hay que `--overwrite`.
- **Tamaño del JSON:** asumido y documentado en `spec.md` (costo aceptado).
- **Consumidores del JSON (broadcast/metric):** ignoran campos extra; el JSON de
  `fase5_clips` ya traía `rle` y el broadcast funcionó. Sin cambios.

## Validación

- Smoke local sin GPU: tomar un JSON de tracking **con `rle`** ya existente
  (`outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json`) y correr
  `render_obj_id_overlay(..., masks_only=True)`; verificar que el mp4 resultante tiene
  el **mismo número de frames** que el `*_obj_id.mp4` (1799) y que solo muestra máscara.
- Smoke de error: correr `masks_only=True` sobre un JSON sin `rle` ⇒ `ValueError`.
- (Pod, fuera de este cambio) re-correr `main.py --overwrite` sobre un clip y
  comprobar las tres salidas + una sola pasada de SAM3.
