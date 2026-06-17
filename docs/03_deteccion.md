# Fase 03 — Detección

> **Qué hay en cada frame.** La detección es el primer eje intercambiable del pipeline:
> se elige por nombre vía registro (`get_detector`) y devuelve siempre la misma moneda
> (`{clase: [Detection]}`), de modo que segmentación y tracking no saben qué detector
> corre debajo. Aquí vive **la innovación sobre SAM3**: el detector `yolo_sam3`.

- **Notebooks de referencia:** [`fase_2_YOLO_SAM3/`](../notebooks/fase_2_YOLO_SAM3/)
  (`01_inference_pipeline`, `02_video_propagation`, `03_hybrid_pipeline`)
- **Tareas SDD:** [`sam3_box_prompt`](../.specs/sam3_box_prompt/),
  [`yolo_detector`](../.specs/yolo_detector/), [`detector_strategy`](../.specs/detector_strategy/),
  [`detector_in_segmentation`](../.specs/detector_in_segmentation/)

---

## El contrato: registro de detectores

Un detector es un **callable** `detect(frame, classes=None, bundle=None, ...) ->
{nombre_clase: [Detection]}`. Se resuelve por nombre:

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `get_detector(name)` | [`detectors/__init__.py:31`](../src/core/detectors/__init__.py#L31) | Resuelve `"sam3_text"` o `"yolo_sam3"`. Lanza `ValueError` si no está registrado. |

Añadir un detector nuevo = registrarlo en `_DETECTORS` y respetar la firma. Segmentación
([04](04_segmentacion.md)) y tracking ([05](05_tracking.md)) lo reciben por parámetro
`detector=` y no cambian.

## `sam3_text` — SAM3 por prompt de texto (camino base)

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `detect(frame, classes, bundle)` | [`detectors/sam3_text.py:20`](../src/core/detectors/sam3_text.py#L20) | Delega en [`detect_classes_in_frame`](../src/core/segmentation.py#L163): una sesión SAM3 de **texto por clase** (`"robot"`, `"orange ball"`…). |

Cero entrenamiento: SAM3 segmenta directamente desde el prompt. Es el default cuando el
config no tiene clave `detector` (caso del config activo `01_yolo_sam3_config.json`).

## `yolo_sam3` — YOLO afinado → SAM3 box-prompt (la innovación)

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `detect(frame, classes, bundle, conf=None)` | [`detectors/yolo_sam3.py:29`](../src/core/detectors/yolo_sam3.py#L29) | Un **YOLO afinado** localiza cajas rápido; SAM3 segmenta **dentro de cada caja** (box-prompt). Las clases con `yolo_id` van por YOLO; el resto (p. ej. `green_floor`) por texto. Misma salida que `sam3_text`. |

Módulos de soporte:

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `load_yolo(weights=None, device=None)` | [`detectors/yolo_boxes.py:129`](../src/core/detectors/yolo_boxes.py#L129) | Carga el YOLO afinado (pesos desde la sección `yolo` del config; por defecto `assets/yolo/best.pt`). Cacheado. |
| `detect_boxes(...)` | [`detectors/yolo_boxes.py:152`](../src/core/detectors/yolo_boxes.py#L152) | Corre YOLO y devuelve `BoxDetection` (caja + clase + score). |
| `BoxDetection` | [`detectors/yolo_boxes.py:42`](../src/core/detectors/yolo_boxes.py#L42) | Caja cruda de YOLO antes de la segmentación. |
| `boxes_to_masks(...)` | [`detectors/box_prompt.py:33`](../src/core/detectors/box_prompt.py#L33) | Convierte cajas en máscaras vía SAM3 **box-prompt** → `Detection`. |

### ¿Por qué es innovación y no solo "otro modelo"?

El YOLO se afinó con **auto-etiquetas generadas por SAM3** sobre los videos NO-testing
(ver [02 Preliminares](02_preliminares.md)): SAM3 etiqueta, YOLO aprende a localizar
rápido, y SAM3 vuelve a segmentar con precisión dentro de esas cajas. Es un bucle
**SAM3-assisted labeling** que no requiere anotación manual. El split de testing queda
intocado para que el [benchmark](07_benchmark.md) sea honesto entre ambos detectores.

### Cuándo conviene cada uno

- En **segmentación pura**, `yolo_sam3` **no** acelera (SAM3 igual segmenta cada caja) y
  pesa ~1 GB más de VRAM — ver [benchmark](07_benchmark.md).
- La ventaja de YOLO aparece **con tracker**: cajas estables → menor fragmentación,
  sobre todo `yolo_sam3 + botsort`.

---

### Cómo encaja con el resto

La salida `{clase: [Detection]}` entra a la **segmentación** ([04](04_segmentacion.md))
tal cual, o al **tracking** ([05](05_tracking.md)), que deriva cajas de las máscaras y
las asocia en `obj_id` estables. La elección del detector se expone arriba en la fachada
`run_inference` ([06](06_pipeline_principal.md)) como parámetro `detector=`.
