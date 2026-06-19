# Spec — Detector inyectable en el tracking (`detector_strategy`)

- **Tarea atómica:** `detector_strategy`
- **Paso de la metodología:** 2 (Especificación)
- **Proceso:** tercera tarea de la secuencia que integra el pipeline YOLO + SAM3
  (SAM3-céntrico) al módulo `src/`. Compone las dos piezas ya implementadas
  (`yolo_detector` → cajas, `sam3_box_prompt` → máscaras) dentro del tracking
  existente.
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline YOLO + SAM3,
> **quiero** poder **elegir qué detector** alimenta al tracking (el SAM3-text
> actual o el nuevo YOLO+SAM3 box-prompt) sin reimplementar el bucle de tracking,
> **para** obtener el pipeline SAM3-céntrico **con `obj_id` estables** reutilizando
> el ByteTrack, el esquema JSON y el overlay que ya existen, y dejar de depender del
> tracker IoU improvisado del notebook `hybrid.py`.

---

## 2. Motivación (por qué)

- El notebook `notebooks/fase_2_YOLO_SAM3/hybrid.py` ya junta YOLO + box-prompt +
  tracking, **pero** con un tracker IoU *greedy casero*: por eso los `obj_id`
  "bailan" entre frames. El tracking de `src` (`track_video`) ya resuelve eso con
  **ByteTrack** (`obj_id` estable y globalmente único), schema JSON unificado y
  overlay — solo que hoy su detector está **fijo** a SAM3-text
  (`detect_classes_in_frame`).
- El nuevo pipeline solo cambia **quién produce las detecciones** (YOLO localiza →
  SAM3 box-prompt segmenta), no el resto. Si el detector se vuelve una **estrategia
  inyectable**, el pipeline YOLO+SAM3 hereda *gratis* `obj_id` estable, JSON, overlay
  y batch, sin duplicar el bucle.
- Las dos piezas nuevas ya están listas (`detect_boxes`, `boxes_to_masks`); falta el
  **pegamento**: un detector que las componga y un punto donde `track_video` lo
  reciba.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Definir una interfaz de detector** como contrato simple: un callable
  `(frame, classes, bundle) -> dict[str, list[Detection]]` (la misma firma que ya
  tiene `detect_classes_in_frame`). Sin clase base abstracta.
- **Dos detectores** en `src/core/detectors/`:
  - `sam3_text`: adaptador delgado que expone el detector SAM3-text actual con esa
    firma (la implementación canónica **se queda** en `segmentation.py`).
  - `yolo_sam3`: nuevo composer que produce `Detection` con máscara — para clases
    con `yolo_id` usa `detect_boxes` → `boxes_to_masks`; para clases sin `yolo_id`
    (`green_floor`) usa el text-prompt existente (`segment_with_text`).
- **Registro y selección por nombre**: un mapa `{"sam3_text": ..., "yolo_sam3": ...}`
  y un resolvedor `get_detector(name)` en el subpaquete `detectors/`.
- **Refactor de `track_video`** para recibir un parámetro `detector` (nombre del
  registro **o** callable), con default `"sam3_text"` → **comportamiento idéntico al
  actual** (no-regresión). El resto del bucle (`mask→bbox→ByteTrack→obj_id`),
  el JSON y el overlay **no cambian**.
- **Cableado en la fachada** `run_inference`: propaga `detector` a `track_video`
  (modo tracking). Selección **opcional por config** (clave `detector`); si no se
  indica, el default de código es `"sam3_text"`.
- **Validación temprana**: un `detector` desconocido lanza `ValueError` **antes** de
  cargar modelos.
- **Carga de modelos** dentro del detector `yolo_sam3`: YOLO vía `load_yolo()`
  (cacheado); box-prompt/green_floor usan el `bundle` SAM3 recibido.
- **Test smoke** (script manual, pod) **pineado** al video canónico de los notebooks
  (`data/raw/17Abril/Cámaras/IMG_9871.MOV`), **full frames**, para A/B contra
  `demo_hybrid_IMG_9871.mp4`.

### 3.2 Fuera de alcance

- **BoT-SORT** (tarea `botsort_tracker`): se sigue usando ByteTrack.
- **Optimización de `green_floor` "cada N frames"** (el `green_every` del hybrid):
  se difiere; por ahora `green_floor` se recalcula por text-prompt **cada frame**.
- **Excluir `green_floor` del tracking**: queda fuera; se sigue trackeando como hoy
  (un ByteTrack por clase).
- **Cablear el detector en segmentación** (`run_pipeline`): se difiere; esta tarea
  cablea el detector solo en **tracking** (donde importan los IDs).
- **Propagación SAM3-video** (`propagation.py`): no se porta.
- Cambios al esquema JSON, a `overlay`/`track_overlay` o a la lógica de ByteTrack:
  el cambio es **solo aguas arriba** (quién genera las detecciones).
- La definición del **cómo técnico** (nombres/firmas exactos, registro concreto,
  forma del refactor): corresponde al `plan.md`.

---

## 4. Comportamiento esperado (criterios de aceptación)

1. **Detector inyectable**: `track_video` acepta `detector` (nombre o callable) y lo
   usa para producir las detecciones por frame, sin cambiar el resto del bucle.
2. **No-regresión**: con `detector="sam3_text"` (default), el resultado es
   **idéntico** al actual (mismo JSON/overlay/`obj_id`).
3. **Pipeline YOLO+SAM3 con IDs estables**: con `detector="yolo_sam3"`,
   `track_video`/`run_inference` devuelven la **misma forma** de resultado
   (`{"json","video","index"}`), con `obj_id` **estable** y `tracks` poblado. Esto
   resuelve los IDs inestables del `hybrid.py`.
4. **Composición correcta**: en `yolo_sam3`, las clases con `yolo_id` se segmentan
   vía YOLO→box-prompt y las clases sin `yolo_id` (`green_floor`) vía text-prompt;
   la salida es `{nombre_clase: [Detection con máscara]}`.
5. **green_floor presente**: `green_floor` sigue apareciendo (vía text-prompt) en la
   salida del detector `yolo_sam3`.
6. **Selección por nombre y validación**: `get_detector("yolo_sam3")`/`"sam3_text"`
   resuelven; un nombre desconocido lanza `ValueError` antes de cargar modelos.
7. **Fachada**: `run_inference(..., mode="tracking", detector=...)` propaga la
   selección; sin indicarla, conserva el comportamiento actual.
8. **Verificación A/B (pod)**: el smoke corre `track_video(detector="yolo_sam3")`
   sobre `IMG_9871.MOV` (full frames) y produce JSON con `tracks` + mp4; las
   máscaras y `green_floor` se ven equivalentes al demo de fase_2 (los `obj_id`/
   colores difieren por usar ByteTrack en vez del tracker casero — es la mejora).

---

## 5. Dependencias y relación con otras tareas

- **Depende de:** `yolo_detector` (`detect_boxes`, `load_yolo`) y `sam3_box_prompt`
  (`boxes_to_masks`, `ensure_tracker_loaded`), ambos implementados; el tracking
  existente (`track_video`, ByteTrack, `inference_schema`, overlay).
- **Cierra** la integración del pipeline YOLO+SAM3 en `src/`: tras esta tarea corre
  por `run_inference` y `run_batch` con `obj_id` estable.
- **Habilita:** `botsort_tracker` (tarea 4) — el segundo tracker se montará sobre el
  mismo punto de composición.
