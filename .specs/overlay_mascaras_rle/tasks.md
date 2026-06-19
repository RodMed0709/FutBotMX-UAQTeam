# tasks.md — `overlay_mascaras_rle`

Implementar la "opción A": overlay de segmentación como post-pase sobre el `rle` del
JSON de tracking (sin SAM3), a video completo. Orden de ejecución:

- [x] **T1 — `track_overlay`: modo solo-máscara.**
  - Añadir `_payload_has_rle(payload) -> bool`.
  - Añadir keyword `masks_only: bool` a `_compose_frame`; cuando es `True` pinta solo
    el relleno de máscara (omite trayectorias, cajas y etiquetas).
  - Añadir `masks_only: bool = False` a `render_obj_id_overlay`: implica `draw_masks`,
    valida presencia de `rle` (si falta ⇒ `ValueError` accionable), propaga el flag.
  - Actualizar docstrings.

- [x] **T2 — `main.py`: guardar máscaras en la inferencia.**
  - `stage_inference`: `include_masks=True`.

- [x] **T3 — `main.py`: rutas del artefacto de segmentación.**
  - `plan_outputs`: reemplazar `seg_json`/`seg_video` por
    `seg_overlay = tracking_json.with_name(f"{stem}_seg.mp4")`.

- [x] **T4 — `main.py`: overlay de segmentación como post-pase.**
  - `stage_individual_overlays`: sustituir la corrida `run_inference(mode="segmentation")`
    por `render_obj_id_overlay(..., draw_masks=True, masks_only=True)` con
    `output_path=paths["seg_overlay"]`; respetar idempotencia (`exists() and not
    overwrite`).

- [x] **T5 — Smoke local (sin GPU).**
  - Sobre `outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json` (trae `rle`):
    `render_obj_id_overlay(..., masks_only=True)` ⇒ mp4 con 1799 frames y solo máscara.
  - `masks_only=True` sobre un JSON sin `rle` ⇒ `ValueError`.
  - `ruff check .` y `black .` sobre lo tocado.

## Trabajo futuro / notas

- Para repoblar con `rle` los JSON ya generados en el pod (hechos con
  `include_masks=False`), hay que correr `main.py --overwrite` (re-ejecuta SAM3 **una**
  vez). Un re-run normal los deja `reusado` sin máscaras.
- El fix de `sampling="all"` en la ruta `mode="segmentation"` de la fachada deja de ser
  necesario para el hub (ya no se usa esa ruta). La fachada se conserva intacta por si
  otros llamadores la usan.
