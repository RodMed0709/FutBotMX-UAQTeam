# Spec — Esquema común del entregable de inferencia (`inference_schema`)

- **Tarea atómica:** `inference_schema`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Pipeline de inferencia unificado + batch (roadmap
  del pipeline unificado, tarea 1). **Cimiento** del que dependen
  `optional_render`, `unified_inference`, `batch_inference` y —de forma
  transversal— `prediction_export` del roadmap de evaluación (tarea 4).
- **Depende de:** los dos caminos de inferencia existentes
  (`pipeline.py::run_pipeline` y `tracking.py::track_video`) y de las piezas que ya
  componen (`detect_classes_in_frame`, `overlay_detections`, `frame_extraction`).

---

## 1. Requisito (historia de usuario)

> **Como** persona que construye el pipeline de análisis de fútbol robótico,
> **quiero** que los dos caminos de inferencia (per-frame/segmentación y tracking)
> emitan un **mismo esquema de entregable**: auto-describible, con geometría por
> detección y con **máscaras opcionales en COCO RLE**,
> **para** que el JSON sea el **producto real** del pipeline —reproducible,
> consumible por la evaluación y suficiente para reconstruir la visualización **sin
> re-invocar el modelo**— y para que la fachada unificada y la capa batch trabajen
> contra **un solo formato**.

---

## 2. Motivación (por qué)

- **El JSON actual es insuficiente y asimétrico.** El seg-only
  (`run_pipeline`) emite solo `{index, detections:{clase:[{obj_id, score}]}}`:
  **sin geometría** (ni `bbox`, ni `frame_index` real) → no permite localizar ni
  reconstruir nada. El de tracking sí lleva caja/centroide/frame por `obj_id`, pero
  **sin máscaras**. Ninguno es **auto-describible** (falta resolución, fps real,
  config que lo produjo).
- **El dato estructurado debe ser el producto, no el mp4.** Para lotes y para
  evaluación, generar video siempre es desperdicio; el entregable que importa es el
  JSON. Esta tarea fija ese contrato (el mp4 opcional es la tarea siguiente).
- **Las máscaras en RLE habilitan dos cosas a la vez:** (a) que la **evaluación**
  consuma máscaras predichas (mIoU/Boundary IoU/Dice se calculan sobre máscaras, y
  COCO-RLE es justo lo que `pycocotools` espera), y (b) que un futuro **overlay por
  `obj_id`** sea un **post-pase desacoplado** (recolorear desde el JSON, sin SAM3),
  porque RLE es una codificación **sin pérdida** de la máscara.
- **Un formato común** evita que la fachada unificada y la batch lidien con dos
  esquemas, y alinea el entregable de predicción con lo que `prediction_export`
  necesitará.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Definir e implementar un esquema JSON común** emitido por `run_pipeline` y
  `track_video`, con:
  - **Geometría por detección en ambos modos:** `bbox` `(x, y, w, h)` en píxeles
    absolutos, `centroid` `(cx, cy)` y **`frame_index` real** del video.
  - **Metadatos auto-descriptivos a nivel de corrida:** resolución `(H, W)`, **fps
    real** de la fuente, **snapshot del contenido completo de la config activa**,
    identificador de **versión del modelo**, **timestamp** y **`schema_version`**.
  - **Máscaras opcionales en COCO RLE**, embebidas por detección (campo `rle`),
    controladas por un **parámetro de función `include_masks` (default `False`)**.
- **Organización canónica frame-indexed:** lista de frames; cada frame con sus
  detecciones (`obj_id`, `bbox`, `centroid`, `score`, y `rle` si aplica). En **modo
  tracking** se **conserva además** el índice de tracks agnóstico actual
  (`obj_id → clase`, observaciones), **fundido en el mismo JSON**.
- **Un solo JSON por corrida**, con **nombre agnóstico al modo**, escrito bajo una
  **carpeta por video**:
  `outputs/inference/<video_stem>/<video_stem>.json` (git-ignored). Cuando se genere
  mp4, vive **junto** al JSON en esa misma carpeta.
- **Dependencia:** `pycocotools` para codificar RLE, **import perezoso** (estilo del
  repo) y declarada en `requirements.txt`.
- **Documentar** que `obj_id` es **inestable** en per-frame y **estable** en
  tracking (misma clave, distinta semántica por modo).
- **Script de prueba manual** (`testing/`) que valide el esquema (geometría +
  metadatos localmente; máscaras/RLE en GPU/pod).

### 3.2 Fuera de alcance

- **No** se añade el flag `render_video` (tarea `optional_render`): el mp4 se sigue
  escribiendo como hoy, pero **reubicado** a la nueva carpeta por video.
- **No** se unifican los dos caminos bajo una fachada (`unified_inference`): cada
  función sigue siendo su propia puerta de entrada; solo cambia **lo que emiten**.
- **No** se implementa `prediction_export` ni la exportación a **COCO estándar**
  (archivo aparte): esta tarea **define** el contrato/RLE; la proyección a COCO la
  hará esa tarea del roadmap de evaluación.
- **No** se cambia la lógica de detección/tracking (`detect_classes_in_frame`,
  asociación ByteTrack, muestreo de frames).
- **No** se persisten máscaras crudas (booleanas) nunca; solo RLE y solo si
  `include_masks=True`.
- **No** se mantiene retrocompatibilidad con los JSON viejos (outputs desechables,
  git-ignored).
- El **cómo técnico** (firmas exactas, estructura de claves, helper de codificación
  RLE, captura del id de versión del modelo, detalle del test) corresponde al
  `plan.md`.

---

## 4. Comportamiento esperado

### 4.1 Estructura del entregable (común a ambos modos)

Un JSON por corrida con dos secciones:

- **Cabecera (metadatos de corrida):** `schema_version`, `video` (ruta), `mode`
  (`"segmentation"` | `"tracking"`), `model_version`, `timestamp`, `fps` (real),
  `resolution` `{height, width}`, `num_frames`, `classes`, `include_masks` (bool) y
  `config` (snapshot completo de la config activa).
- **`frames`:** lista frame-indexed. Cada entrada: `frame_index` (real, del video
  fuente) y `detections: {clase: [ {obj_id, bbox:[x,y,w,h], centroid:[cx,cy],
  score, rle?} ]}`. `rle` presente **solo** si `include_masks=True`.

En **modo tracking** se añade además **`tracks`**: el índice agnóstico actual
(`obj_id`, `class`, `observations[{frame_index, bbox, centroid, score}]`), fundido
en este mismo archivo (ya no un `_tracks.json` aparte).

### 4.2 Máscaras (RLE)

- Si `include_masks=True`, cada detección incluye su máscara como **COCO RLE**
  (codificación sin pérdida): `RLE → decode → máscara (H, W)` idéntica.
- Default `False`: el JSON queda ligero (solo geometría + metadatos) — el caso de
  lotes/videos completos.
- La reconstrucción visual de un video (overlay de máscaras) es posible **desde el
  JSON + el video real**, sin SAM3.

### 4.3 Geometría

- `bbox` derivado de la máscara como caja envolvente `(x, y, w, h)` en **píxeles
  absolutos**; `centroid` como centro de esa caja. Coherente con el `TrackObservation`
  del tracking y con el formato COCO.
- El seg-only, que hoy no expone geometría, pasa a derivarla de la máscara igual que
  el tracking.

### 4.4 Salidas y ubicación

- **Carpeta por video** bajo `outputs/inference/<video_stem>/` (git-ignored):
  - `<video_stem>.json` — **siempre** (el entregable).
  - `<video_stem>.mp4` — mientras el render siga activo (hasta `optional_render`).
- Nombre del JSON **agnóstico al modo** (el `mode` va dentro de la cabecera).

### 4.5 Auto-descripción y reproducibilidad

- El `config` embebido permite saber **con qué** se produjo el resultado (prompts,
  umbrales, parámetros de tracking, control de frames) sin depender del `.env`/config
  vigentes al momento de leerlo.
- `resolution` hace interpretables las coordenadas en píxeles; `schema_version`
  permite evolucionar el formato de forma controlada.

---

## 5. Criterios de aceptación

1. **AC-1 — Esquema común:** `run_pipeline` y `track_video` emiten el mismo esquema
   base (cabecera + `frames` frame-indexed); tracking añade `tracks` en el **mismo**
   archivo.
2. **AC-2 — Geometría en seg-only:** el JSON de seg-only incluye, por detección,
   `bbox`, `centroid` y `frame_index` real (hoy ausentes).
3. **AC-3 — Metadatos:** la cabecera incluye `schema_version`, `model_version`,
   `timestamp`, `fps` real, `resolution (H,W)`, `classes`, `include_masks` y el
   `config` completo embebido.
4. **AC-4 — Máscaras opcionales en RLE:** con `include_masks=True`, cada detección
   trae su máscara en **COCO RLE** decodificable sin pérdida; con `False`, no hay
   campo `rle` y el JSON queda ligero.
5. **AC-5 — `include_masks` por parámetro:** se controla como **argumento de la
   función**, default `False`; nada hardcodeado.
6. **AC-6 — Un solo archivo / ubicación:** se escribe un único JSON por corrida en
   `outputs/inference/<video_stem>/<video_stem>.json`; en tracking ya **no** se emite
   `_tracks.json` aparte.
7. **AC-7 — mp4 reubicado:** cuando se genera, el mp4 vive junto al JSON en la
   carpeta del video (sin introducir aún el flag de render).
8. **AC-8 — `obj_id` documentado:** el código deja explícito que `obj_id` es
   inestable en per-frame y estable en tracking.
9. **AC-9 — Reconstrucción sin modelo:** a partir de un JSON con `include_masks=True`
   y el video real, se pueden recolorear/visualizar las máscaras sin invocar SAM3
   (validado en el test).
10. **AC-10 — Dependencia declarada:** `pycocotools` queda en `requirements.txt` e
    importado de forma perezosa.
11. **AC-11 — Verificación:** un script en `testing/` valida la forma del esquema y
    los metadatos (localmente) y la ida-vuelta RLE↔máscara (GPU/pod).
12. **AC-12 — Contrato para evaluación:** el RLE emitido es **COCO-RLE**, de modo que
    `prediction_export` pueda proyectarlo a COCO estándar sin re-codificar.

---

## 6. Supuestos y notas

- **El entregable es el dato, el mp4 es opcional** (el flag llega en
  `optional_render`); esta tarea ya **estructura** la salida para ello (carpeta por
  video, JSON siempre).
- **RLE como puente con la evaluación:** se elige COCO-RLE precisamente porque el GT
  llegará en COCO (Roboflow) y `pycocotools` calcula las métricas sobre ese formato;
  evita glue frágil y re-codificaciones.
- **RLE solo donde aporta:** default `False`. No se persisten máscaras de los 123
  videos completos; el caso con máscaras es evaluación / depuración puntual.
- **`include_masks` es parámetro, no config:** se decide **por llamada** desde
  código quién quiere conservar máscaras (p. ej. el futuro `prediction_export` lo
  prende para el set congelado; la batch lo deja apagado).
- **Un solo JSON por corrida:** el índice de tracks se funde en el mismo archivo; se
  elimina el `_tracks.json` separado para no fragmentar el entregable.
- **`inference/` nombra el proceso** (predicciones del modelo sobre videos),
  hermano de `eval/` (métricas) y un futuro `train/` (fine-tuning). El mp4, aunque es
  visualización, sale del run de inferencia y vive aquí por ahora.
- **Sin retrocompatibilidad:** los JSON previos son outputs git-ignored desechables;
  se cambia el formato directamente.
- Esta especificación **no** define el *cómo* técnico (firmas, estructura exacta de
  claves, helper de RLE, captura del id de versión del modelo, manejo de errores ni
  el detalle del test); todo ello corresponde al `plan.md`.
