# spec.md — `overlay_mascaras_rle`

## Contexto

El hub `main.py` genera tres salidas visuales sobre un video: overlay de
**tracking** (`obj_id`), overlay de **segmentación** y el video de **broadcast**.
Hoy el overlay de segmentación se produce **re-corriendo SAM3** desde cero
(`run_inference(mode="segmentation")` en `stage_individual_overlays`), lo que tiene
dos problemas observados en producción:

1. **Re-cómputo caro e innecesario.** El relleno de máscara por clase solo necesita
   las **máscaras**, no a SAM3. SAM3 únicamente hace falta para *producir* las
   máscaras la primera vez. Si las máscaras ya están guardadas como RLE en el JSON de
   tracking, pintarlas es puro `cv2`, sin GPU. La primitiva de pintado **ya existe**
   (`track_overlay._compose_frame` con `draw_masks=True`, vía `decode_rle`).

2. **Duración distinta a la del overlay de tracking.** La fachada resuelve
   `mode="segmentation"` con `sampling="auto"` a **cuota equiespaciada**
   (`preprocess.frame_quota = 30` frames), mientras el overlay de tracking recorre el
   video completo. Resultado: el `*_seg.mp4` dura ~1 s y el `*_obj_id.mp4` dura lo que
   el clip. El usuario confirmó que un overlay de segmentación a **video completo** se
   ve bien.

La causa de fondo de ambos: el JSON de tracking del hub se escribe con
`include_masks=False` (`main.py::stage_inference`), así que **no guarda `rle`**, y el
overlay de segmentación está implementado como **corrida nueva** en vez de
**post-pase** sobre el `rle`.

## Objetivo

Producir el overlay de segmentación como un **post-pase desacoplado sobre el JSON de
tracking** (sin SAM3, CPU, ejecutable en local), recorriendo el **video completo**
para que su duración coincida con el overlay de tracking. Una **sola** pasada de SAM3
(la inferencia de tracking, ahora con máscaras) alimenta **ambos** overlays.

Esta es la "opción A" acordada en la conversación.

## Alcance

- `src/core/track_overlay.py`: añadir un modo **solo-máscara** (`masks_only`) a
  `render_obj_id_overlay`/`_compose_frame` que pinte **únicamente** el relleno de
  máscara por clase (sin caja, sin etiqueta, sin estela), conservando el look del
  overlay de segmentación per-frame. Validar que el JSON traiga `rle` cuando se pide
  `masks_only` (error claro si no).
- `main.py`:
  - `stage_inference`: correr el tracking con `include_masks=True` para que el JSON
    cargue `rle` (base de ambos overlays).
  - `stage_individual_overlays`: reemplazar la corrida SAM3 de segmentación por una
    llamada a `render_obj_id_overlay(..., draw_masks=True, masks_only=True)` (post-pase
    sobre el JSON de tracking, video completo, sin GPU).
  - `plan_outputs`: el artefacto de segmentación pasa a ser un único mp4
    (`<stem>_seg.mp4`) junto al JSON de tracking; deja de existir el JSON/MP4 de
    segmentación en el namespace `<run_label>/seg/`.

## Fuera de alcance

- Migrar `minimap_pipeline` / `compose_demo` (mosaico) — intactos.
- El overlay de tracking (`obj_id`) conserva su look actual (cajas + `#id` + estela,
  **sin** máscara) para mantener las tres salidas visualmente distintas.
- El fix de `sampling` en la fachada para `mode="segmentation"` deja de ser necesario
  en el hub (ya no se invoca esa ruta); la fachada se deja como está.

## Comportamiento esperado

- `python main.py <clip>` (interactivo, overlays=sí) o el flujo equivalente produce:
  - `<stem>_obj_id.mp4` (tracking, sin cambios),
  - `<stem>_seg.mp4` (segmentación, **solo máscara**, **misma duración** que el de
    tracking, generado **sin SAM3**),
  - el broadcast como hasta ahora.
- Solo **una** pasada de SAM3 por corrida (la inferencia de tracking).
- El overlay de segmentación es ejecutable **en local sin GPU** a partir del JSON.

## Consideraciones / costos

- El JSON de tracking crece (~36 MB / 60 s) por guardar `rle` de cada detección por
  frame. Es el precio de evitar la segunda pasada de SAM3.
- **Idempotencia (importante para datos ya generados):** los JSON ya producidos en el
  pod con `include_masks=False` **no** se actualizan con solo re-correr `main.py`
  (la etapa los marca `reusado`). Para repoblarlos con `rle` hay que correr con
  `--overwrite`, lo que **vuelve a ejecutar SAM3 una vez**. Tras eso, ambos overlays
  salen del `rle` sin re-cómputo.
- Si se pide `masks_only` sobre un JSON sin `rle`, se levanta `ValueError` con un
  mensaje que indica re-correr la inferencia con máscaras.
