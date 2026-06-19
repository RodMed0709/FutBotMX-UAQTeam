# Plan técnico — Detector inyectable en el tracking (`detector_strategy`)

- **Tarea atómica:** `detector_strategy`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso:** tercera tarea de la secuencia que integra el pipeline YOLO + SAM3 a
  `src/`; compone `detect_boxes` (YOLO) + `boxes_to_masks` (box-prompt) dentro del
  tracking existente.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo (a) fijar un **contrato de detector** (callable
`(frame, classes, bundle) -> dict[str, list[Detection]]`); (b) implementar los dos
detectores (`sam3_text` adaptador, `yolo_sam3` composer) y un **registro** con
resolución por nombre; (c) **refactorizar `track_video`** para inyectar el detector
sin tocar el resto del bucle (`mask→bbox→ByteTrack→obj_id`, JSON, overlay); y (d)
**cablearlo en `run_inference`**. Resuelve los `obj_id` inestables del `hybrid.py`
reutilizando ByteTrack. Además, definir el script de validación A/B (pod).

---

## 2. Stack técnico

- **Python:** 3.11.
- **Composición:** reutiliza `src.core.detectors.yolo_boxes.detect_boxes`/`load_yolo`,
  `src.core.detectors.box_prompt.boxes_to_masks`,
  `src.core.segmentation.segment_with_text`/`detect_classes_in_frame`/`Detection`.
- **Tracking:** `src.core.tracking.track_video` (se refactoriza); ByteTrack
  (`trackers`) y schema (`inference_schema`) **sin cambios**.
- **Tipos:** `typing.Callable`; `dataclasses` ya existentes.
- **Imports perezosos**: torch/ultralytics/supervision siguen dentro de funciones.

---

## 3. Diseño

### 3.1 Contrato de detector

Un detector es un **callable**:

```python
Detector = Callable[[np.ndarray, list[dict] | None, "Sam3Bundle | None"],
                    dict[str, list[Detection]]]
```

Firma concreta `detect(frame, classes=None, bundle=None) -> {nombre: [Detection]}`.
Es **exactamente** la de `detect_classes_in_frame`, así que el detector SAM3-text
encaja sin adaptarse.

### 3.2 `src/core/detectors/sam3_text.py` — adaptador

```python
from src.core.segmentation import detect_classes_in_frame

def detect(frame, classes=None, bundle=None):
    return detect_classes_in_frame(frame, classes=classes, bundle=bundle)
```

Adaptador **delgado**: la implementación canónica se queda en `segmentation.py`
(no se mueve, para no romper imports existentes). Solo da un punto de entrada con el
nombre del detector.

### 3.3 `src/core/detectors/yolo_sam3.py` — composer

```python
def detect(frame, classes=None, bundle=None) -> dict[str, list[Detection]]:
    classes = classes if classes is not None else _load_classes()
    bundle = bundle or load_sam3()

    yolo_classes = [c for c in classes if "yolo_id" in c]
    text_classes = [c for c in classes if "yolo_id" not in c]

    result: dict[str, list[Detection]] = {}

    # Clases YOLO: cajas (detect_boxes) -> mascaras (box-prompt).
    boxes_by_class = detect_boxes(frame, classes=yolo_classes)  # load_yolo() interno
    for c in yolo_classes:
        name = c["name"]
        bds = boxes_by_class.get(name, [])
        boxes = [bd.bbox for bd in bds]
        scores = [bd.score for bd in bds]
        result[name] = boxes_to_masks(frame, boxes, bundle=bundle, scores=scores)

    # Clases sin yolo_id (green_floor): text-prompt como hoy.
    for c in text_classes:
        result[c["name"]] = segment_with_text(frame, c["sam3_prompts"][0], bundle)

    return result
```

- YOLO se carga internamente vía `load_yolo()` (cacheado); box-prompt y green_floor
  usan el `bundle` SAM3 recibido (`boxes_to_masks` invoca `ensure_tracker_loaded`).
- El `obj_id` posicional que pone `boxes_to_masks` es **irrelevante**: `track_video`
  lo sobrescribe con el `obj_id` estable de ByteTrack (ver §3.5). Por eso no hace
  falta más bookkeeping aquí.
- Si `green_every` se quisiera optimizar luego, este es el punto natural (hoy: cada
  frame; diferido).

### 3.4 Registro y resolución — `src/core/detectors/__init__.py`

```python
from src.core.detectors import sam3_text, yolo_sam3

_DETECTORS = {"sam3_text": sam3_text.detect, "yolo_sam3": yolo_sam3.detect}

def get_detector(name):
    if name not in _DETECTORS:
        raise ValueError(
            f"detector '{name}' no soportado (usa uno de {sorted(_DETECTORS)})."
        )
    return _DETECTORS[name]
```

- Se exporta `get_detector` (y se mantienen `boxes_to_masks`, `detect_boxes`,
  `load_yolo`, `BoxDetection`) en `__all__`.
- Importar `detectors/__init__` no arrastra dependencias pesadas (los módulos solo
  importan `numpy`/`Detection` a nivel de módulo; torch/ultralytics son perezosos).
- **Sin ciclos**: los módulos importan a sus hermanos por ruta directa
  (`src.core.detectors.box_prompt`, etc.), no vía el `__init__`.

### 3.5 Refactor de `track_video`

- **Nueva firma** (parámetro añadido, resto igual):
  ```python
  def track_video(..., detector: str | Callable | None = None, ...):
  ```
- **Resolución temprana** (antes de cargar SAM3, para validar pronto):
  ```python
  if detector is None:
      detector = _resolved_default_detector  # config 'detector' o "sam3_text"
  detector_fn = detector if callable(detector) else get_detector(detector)
  ```
  Un nombre desconocido lanza `ValueError` **antes** de `bundle = bundle or load_sam3()`.
- **Único cambio en el bucle**: sustituir
  `dets = detect_classes_in_frame(frame, classes=classes, bundle=bundle)`
  por `dets = detector_fn(frame, classes=classes, bundle=bundle)`.
- Todo lo demás (un ByteTrack por clase, `mask→bbox`, asignación de `obj_id`
  estable, overlay, schema JSON) **no cambia**. El `det.obj_id` lo fija ByteTrack,
  así que el `obj_id` que traiga el detector es indiferente.
- **Config**: `_load_tracking_config` (o lectura puntual) devuelve también la clave
  opcional `detector` (default `"sam3_text"`), usada cuando el parámetro es `None`.

### 3.6 Cableado en `run_inference`

- Añadir `detector: str | None = None` a `run_inference` y propagarlo a
  `track_video` en el camino `mode="tracking"`.
- En `mode="segmentation"` el detector **no aplica** (fuera de alcance): se ignora
  (documentado), no rompe.
- `run_batch` no requiere cambios estructurales; si se desea, puede propagar
  `detector` como kwarg que ya pasa a `run_inference` (decisión menor de
  implementación, sin cambiar su contrato).

### 3.7 Imports perezosos / no-regresión

- `track_video` ya importa `supervision`/`trackers` dentro de la función; se
  mantiene. `get_detector` y los módulos de detectores no añaden imports pesados a
  nivel de módulo.
- Con `detector` por defecto (`"sam3_text"`), `track_video` reproduce el camino
  actual exactamente (mismo detector, mismo resultado).

---

## 4. Script de validación A/B (smoke, pod)

Archivo: `testing/test_detector_strategy.py` (manual, **no** pytest; corre en el
**pod**: requiere YOLO + SAM3 + GPU).

- **Video pineado**: `data/raw/17Abril/Cámaras/IMG_9871.MOV` (el canónico de
  `notebooks/fase_2_YOLO_SAM3`), **full frames** (sin cap), para A/B directo contra
  `demo_hybrid_IMG_9871.mp4`.
- Correr `track_video(video, detector="yolo_sam3", render_video=True)`.
- **Aserciones:**
  - el resultado tiene `{"json","video","index"}`; el JSON existe y su sección
    `tracks` no está vacía;
  - cada `Track` mantiene un `class_name` consistente y `obj_id` único; los
    `obj_id` reaparecen entre frames (estabilidad: nº de tracks ≪ nº de
    observaciones);
  - `green_floor` aparece en los registros frame-indexed (vía text-prompt).
- **Inspección visual A/B**: opcionalmente, generar el overlay por `obj_id` con
  `track_overlay.render_obj_id_overlay` sobre el JSON, para comparar lado a lado con
  el demo (recordando: máscaras/green_floor deben empatar; `obj_id`/colores no, por
  usar ByteTrack en vez del tracker casero).
- **Guarda de no-regresión** (rápida): correr también
  `track_video(..., detector="sam3_text", max_frames=<corto>)` y confirmar que sigue
  produciendo JSON+mp4 como antes.

---

## 5. Riesgos y consideraciones

- **Doble modelo en GPU**: con `yolo_sam3`, YOLO + SAM3 (video/text + tracker)
  coexisten en VRAM. Aceptado (es el diseño; corre en el pod).
- **Coste de green_floor cada frame**: al diferir `green_every`, el text-prompt de
  green_floor corre por frame (más lento que el demo, que lo hacía cada N). Correcto
  pero más caro; optimización futura.
- **Orden de validación**: resolver el detector **antes** de cargar modelos para que
  un nombre inválido falle barato.
- **No-regresión**: el default `"sam3_text"` y el cambio acotado a una sola línea del
  bucle garantizan que el camino actual no cambia; el smoke lo verifica.
- **`run_batch`**: si se propaga `detector`, es un kwarg pasante; no cambia su lógica
  de selección de videos ni de skip.

---

## 6. Qué NO incluye este plan

- BoT-SORT (tarea `botsort_tracker`), la optimización `green_every`, excluir
  green_floor del tracking, ni cablear el detector en segmentación (`run_pipeline`).
- Cambios al esquema JSON, a `overlay`/`track_overlay` o a la lógica de ByteTrack.
- La descomposición en pasos accionables y su checklist: corresponde a `tasks.md`.
