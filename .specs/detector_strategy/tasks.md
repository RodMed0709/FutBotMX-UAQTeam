# Tasks — Detector inyectable en el tracking (`detector_strategy`)

- **Tarea atómica:** `detector_strategy`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Detectores y registro

- [x] **T1 — Adaptador `src/core/detectors/sam3_text.py`**
  - Definir `detect(frame, classes=None, bundle=None)` que delega en
    `segmentation.detect_classes_in_frame` (la impl canónica **no** se mueve).
  - **Verificación:** `sam3_text.detect` tiene la firma del contrato y devuelve
    `{nombre: [Detection]}` reusando el camino actual.
  - **Plan:** §3.1, §3.2. **Spec:** AC-1, AC-2.

- [x] **T2 — Composer `src/core/detectors/yolo_sam3.py`**
  - Definir `detect(frame, classes=None, bundle=None) -> dict[str, list[Detection]]`:
    resolver `classes`/`bundle`; partir en `yolo_classes` (con `yolo_id`) y
    `text_classes` (sin `yolo_id`); `detect_boxes(frame, classes=yolo_classes)` y por
    clase `boxes_to_masks(frame, boxes, bundle=bundle, scores=scores)`; las
    `text_classes` (`green_floor`) vía `segment_with_text(frame, prompt, bundle)`.
  - Importar de los hermanos por ruta directa (`src.core.detectors.box_prompt`,
    `...yolo_boxes`) y de `segmentation`; mantener imports pesados perezosos.
  - **Verificación:** devuelve `{nombre: [Detection con máscara]}`; las clases con
    `yolo_id` salen del camino YOLO→box-prompt y `green_floor` del text-prompt.
  - **Plan:** §3.3. **Spec:** AC-4, AC-5.

- [x] **T3 — Registro + `get_detector` en `src/core/detectors/__init__.py`**
  - Mapa `{"sam3_text": sam3_text.detect, "yolo_sam3": yolo_sam3.detect}` y
    `get_detector(name)` que lanza `ValueError` si el nombre no existe. Sumar
    `get_detector` a `__all__` (junto a lo ya exportado).
  - **Verificación:** `get_detector("sam3_text")`/`"yolo_sam3"` resuelven; nombre
    desconocido ⇒ `ValueError`; `import src.core.detectors` no arrastra
    `torch`/`ultralytics`/`supervision`; sin import circular.
  - **Plan:** §3.4. **Spec:** AC-6.

---

## Fase B — Refactor del tracking

- [x] **T4 — `track_video` recibe `detector` (inyección + resolución temprana)**
  - Añadir parámetro `detector: str | Callable | None = None`. Resolver **antes** de
    `bundle = bundle or load_sam3()`: si `None` → default de config (`detector`) o
    `"sam3_text"`; si `str` → `get_detector(...)`; si callable → usarlo. Validación de
    nombre inválido **antes** de cargar modelos.
  - Sustituir en el bucle `detect_classes_in_frame(...)` por
    `detector_fn(frame, classes=classes, bundle=bundle)`. Nada más del bucle cambia.
  - **Verificación:** con `detector="sam3_text"` (default) el resultado es idéntico
    al actual; un nombre inválido lanza `ValueError` sin cargar SAM3.
  - **Plan:** §3.5, §3.7. **Spec:** AC-1, AC-2, AC-6.

- [x] **T5 — Leer la clave opcional `detector` de la config**
  - Que `_load_tracking_config` (o una lectura puntual) devuelva la clave `detector`
    (default `"sam3_text"`), usada cuando el parámetro de `track_video` es `None`.
  - **Verificación:** sin clave en config → default `"sam3_text"`; con clave
    `"yolo_sam3"` → se usa ese detector cuando no se pasa parámetro.
  - **Plan:** §3.5. **Spec:** AC-7.

---

## Fase C — Fachada

- [x] **T6 — `run_inference` propaga `detector`**
  - Añadir `detector: str | None = None`; propagarlo a `track_video` en
    `mode="tracking"`. En `mode="segmentation"` se ignora (documentado), sin romper.
  - **Verificación:** `run_inference(mode="tracking", detector="yolo_sam3")` enruta
    con ese detector; sin indicarlo conserva el comportamiento actual; segmentación
    no se afecta.
  - **Plan:** §3.6. **Spec:** AC-7.

---

## Fase D — Validación

- [ ] **T7 — Script smoke A/B `testing/test_detector_strategy.py` (pod, full frames)**
  - Pinear `data/raw/17Abril/Cámaras/IMG_9871.MOV`, **full frames**. Correr
    `track_video(detector="yolo_sam3", render_video=True)`; aserciones (resultado
    `{"json","video","index"}`; `tracks` no vacío; `obj_id` único y reaparece entre
    frames; `green_floor` presente). Opcional: `render_obj_id_overlay` sobre el JSON
    para A/B contra `demo_hybrid_IMG_9871.mp4`. Guarda rápida de no-regresión con
    `detector="sam3_text"` y `max_frames` corto.
  - **Verificación (pod):** corre end-to-end; aserciones pasan; el overlay/JSON es
    coherente y comparable al demo (máscaras/green_floor empatan; `obj_id`/colores
    difieren por ByteTrack).
  - **Plan:** §4. **Spec:** AC-3, AC-8.

---

## Fase E — Cierre

- [x] **T8 — Lint, formato y no-regresión**
  - `ruff check .` y `black .` limpios sobre lo nuevo. Confirmar: `import src.core`
    no arrastra pesados; default `"sam3_text"` reproduce el camino actual; schema,
    overlay y ByteTrack sin cambios.
  - **Verificación:** linters limpios; import barato; no-regresión confirmada.
  - **Plan:** §3.7, §5. **Spec:** AC-2.

---

## Trabajo futuro (fuera de esta tarea)

- `botsort_tracker` (tarea 4): segundo tracker sobre el mismo punto de composición.
- Optimización de `green_floor` "cada N frames" (`green_every`).
- Excluir `green_floor` del tracking (un follow-up del proceso).
- Cablear el detector en segmentación (`run_pipeline`).
