# Spec — Detector inyectable en segmentación (`detector_in_segmentation`)

- **Tarea atómica:** `detector_in_segmentation`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Rediseño del benchmark sin-GT hacia una composición
  real **detector/segmentador → tracker** (Fase 1: eficiencia de detectores; Fase 2:
  eficiencia/consistencia de trackers en 2×2).
- **Depende de:** `detector_strategy` y `yolo_detector` (factory `get_detector` y la
  estrategia `yolo_sam3`, ya completas), que esta tarea reutiliza tal cual.
- **Habilita:** `config_aware_output_paths` (tarea 2, namespacing de salidas) y los
  notebooks de benchmark por fases, que necesitan correr `yolo_sam3` **como
  segmentador suelto** (sin tracker).

---

## 1. Requisito (historia de usuario)

> **Como** persona que evalúa qué motor de detección/segmentación conviene,
> **quiero** poder correr **cualquier detector** (`sam3_text` o `yolo_sam3`) en
> `mode="segmentation"`, sin tracker,
> **para** medir la eficiencia del detector de forma aislada (Fase 1 del benchmark) y
> para que el pipeline **realmente componga** detector → tracker en ambos modos, no
> solo en tracking.

---

## 2. Motivación (por qué)

- **Segmentación esquiva la abstracción de detector.** Hoy `run_pipeline`
  (`pipeline.py`) **no recibe** `detector`: llama directo a
  `detect_classes_in_frame` (`pipeline.py:174`), que fija SAM3-texto. En cambio
  `track_video` (`tracking.py`) **sí** recibe `detector` y lo resuelve con
  `get_detector(detector)`. La estrategia de detector ya existe, pero un solo modo la
  usa.
- **`yolo_sam3` no se puede usar como segmentador.** Por lo anterior, "YOLO→SAM3 sin
  tracking" no existe en el código; el benchmark tuvo que omitir esa fila tratándola
  como inexistente. Es un **artefacto del código**, no del diseño: detector y tracker
  son ejes ortogonales.
- **La fachada descarta `detector` en segmentación.** `run_inference` ya acepta
  `detector`, pero en la rama de segmentación (`inference.py:69-71`) **no lo
  propaga** a `run_pipeline` (su docstring incluso dice "se ignora"). Solo la rama de
  tracking lo pasa.
- **El benchmark por fases lo necesita.** La Fase 1 (elegir el mejor
  detector/segmentador por eficiencia, sin confundir con el tracker) exige correr
  cada detector **solo**, en segmentación. Sin este desacople, la Fase 1 no es
  medible de forma limpia.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **`run_pipeline` gana un parámetro `detector: str | None = None`**, espejando a
  `track_video`.
- **Resolución del detector**, hecha **una sola vez antes del loop de frames**:
  - `None` ⇒ default del config (clave `detector`, fallback `"sam3_text"`), mismo
    criterio que tracking.
  - Nombre inválido ⇒ **`ValueError` antes de cargar SAM3** (fail-cheap), vía el
    mismo factory `get_detector()` de `src.core.detectors`.
- **Sustitución de la llamada por-frame**: dentro del loop, en lugar de
  `detect_classes_in_frame(frame, ...)`, se invoca la estrategia resuelta
  `detector_fn(frame, classes=classes, bundle=bundle)`.
- **Propagación en la fachada**: `run_inference` pasa su `detector` a `run_pipeline`
  en la rama de segmentación; se corrige el docstring que hoy dice "se ignora".
- **Smoke test**: verificación funcional de que segmentación corre con
  `detector="yolo_sam3"` (extiende `testing/test_pipeline.py` o
  `test_unified_inference.py`).
- **Actualización de `CLAUDE.md`**: la descripción de `pipeline.py` refleja que ahora
  acepta `detector` (hoy menciona solo `classes`/`bundle`).

### 3.2 Fuera de alcance

- **Namespacing de salidas por config** (`outputs/inference/<config>/<stem>/…`) y el
  ajuste de `skip-done`: es la tarea 2 (`config_aware_output_paths`). Esta tarea
  **no** toca `inference_schema.py` ni las rutas de salida.
- **Los notebooks de benchmark** (Fase 1 detectores, Fase 2 trackers 2×2): consumen
  esta capacidad pero son entregables aparte.
- **Cambios a la estrategia `yolo_sam3`**: ya funciona por-frame en tracking; se
  reutiliza sin tocarla.
- **Cambios a los archivos de config**: la clave `detector` ya existe en
  `01_yolo_sam3_config.json`.
- **El esquema JSON**: se conserva idéntico; un detector u otro produce el mismo
  formato de salida.

---

## 4. Comportamiento esperado

### 4.1 `detector=None` (default) — retrocompatible

`run_pipeline(..., detector=None)` y `run_inference(mode="segmentation")` sin
`detector` se comportan **idéntico a hoy**: resuelven al default del config
(`sam3_text` salvo que el config diga otra cosa). Ningún llamador existente cambia de
comportamiento.

### 4.2 `detector="sam3_text"` — explícito

Igual que el default actual: SAM3 segmenta por *prompt de texto*. Equivalente a no
pasar nada cuando el config no redefine `detector`.

### 4.3 `detector="yolo_sam3"` — la capacidad nueva

`mode="segmentation"` + `detector="yolo_sam3"` corre segmentación **por-frame**
(`obj_id` inestable, propio del modo) usando YOLO→SAM3 box-prompt, y emite la
**misma estructura de JSON** que la de `sam3_text`. Es exactamente el detector que
usa tracking, pero sin asociación temporal.

### 4.4 Validación

Un `detector` con nombre no registrado levanta **`ValueError`** (mensaje canónico de
`get_detector`) **antes** de cargar SAM3, igual que en tracking/`run_batch`.

### 4.5 Propagación por la fachada

`run_inference(mode="segmentation", detector="yolo_sam3")` reenvía el detector a
`run_pipeline`. La asimetría histórica (tracking lo usa, segmentación lo ignora)
desaparece: el parámetro `detector` es **ortogonal al modo**.

---

## 5. Criterios de aceptación

1. `run_pipeline` acepta `detector: str | None = None` y lo resuelve con
   `get_detector()`; con `None` usa el default del config (`sam3_text` por fallback).
2. La detección por-frame en segmentación pasa por la **estrategia resuelta**, no por
   `detect_classes_in_frame` hardcodeado.
3. `run_inference` propaga `detector` a `run_pipeline` en segmentación; su docstring
   ya no dice que el detector "se ignora" en ese modo.
4. Un `detector` inválido levanta `ValueError` **antes** de cargar SAM3.
5. `mode="segmentation"` + `detector="yolo_sam3"` corre y produce un JSON con la misma
   estructura que `sam3_text` (verificado por smoke test).
6. Sin `detector` (o con `sam3_text`), la salida es idéntica a la de hoy
   (retrocompatibilidad).
7. No se modifican `inference_schema.py`, las rutas de salida, los configs ni la
   estrategia `yolo_sam3`.
8. `CLAUDE.md` refleja que `pipeline.py` acepta `detector`.

---

## 6. Supuestos y notas

- Lista completa de supuestos acordada con el usuario (categorías técnicas,
  funcionales y de proceso); **ninguno rechazado**.
- `run_batch` ya reenvía `detector` a `run_inference` sin importar el modo, por lo que
  esta capacidad empieza a tener efecto en segmentación **sin** tocar `batch.py`.
- Validación: **smoke test** (funcional) ahora; la validación visual con caso real se
  hará cuando se pida, según la filosofía de tests del repo.
- Esta tarea es el **primer eslabón** del rediseño del benchmark: desacopla el eje
  detector para que las Fases 1 y 2 sean medibles de forma limpia.
