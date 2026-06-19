# Plan técnico — Detector de cajas YOLO (`yolo_detector`)

- **Tarea atómica:** `yolo_detector`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso:** segunda tarea de la secuencia que integra el pipeline YOLO + SAM3 a
  `src/`; sus cajas alimentan al box-prompt ya implementado (`sam3_box_prompt`).
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo (a) **cargar el modelo YOLO** (`best.pt`)
config-driven y cacheado, independiente del `Sam3Bundle`; (b) implementar un
building block que, dado un frame, devuelva las **cajas por clase del repo**
(estructura ligera caja+score), mapeando la clase YOLO vía `yolo_id`; y (c)
**crecer el config de la fase** con las claves persistentes que esto requiere.
Sustituye la inferencia YOLO suelta y hardcoded del notebook `fase_2_YOLO_SAM3`.
Además, definir el script de validación manual (smoke).

---

## 2. Stack técnico

- **Python:** 3.11.
- **Detector:** `ultralytics` (`from ultralytics import YOLO`), `from_pretrained`
  implícito al construir `YOLO(str(weights))`; inferencia con `model.predict(...)`.
  Import **perezoso** dentro de las funciones.
- **Imagen:** `PIL.Image` (se pasa PIL a `predict` para evitar la ambigüedad
  RGB/BGR de ultralytics con arrays numpy); `numpy` para leer las salidas.
- **Configuración:** `json` + lectura de `CONFIG_FILENAME` desde `.env` con
  `strip()`; rutas vía `src/utils.py::get_abs_path`. Lista de clases reutilizando
  `src.core.segmentation._load_classes`.
- **Caché:** `functools.lru_cache` (estándar), como `sam3_loader`.
- **Estructura de salida:** `dataclasses.dataclass`.

> Nota: `ultralytics`/`torch` son pesadas; los imports perezosos mantienen barato
> `import src.core` (ver §3.6).

---

## 3. Diseño

### 3.1 Ubicación y módulo

- **Archivo nuevo:** `src/core/detectors/yolo_boxes.py`.
- **Exportación:** en `src/core/detectors/__init__.py`, añadir
  `from src.core.detectors.yolo_boxes import BoxDetection, detect_boxes, load_yolo`
  y sumarlos a `__all__` (junto a `boxes_to_masks`).

### 3.2 Estructura de salida — `BoxDetection`

```python
@dataclass
class BoxDetection:
    bbox: tuple[float, float, float, float]  # xyxy en pixeles absolutos
    score: float
```

Estructura **ligera** caja+score (sin máscara): YOLO produce cajas, y es justo lo
que consume `boxes_to_masks(frame, boxes, scores=...)` en la tarea siguiente. El
nombre de clase no va en la dataclass porque la salida ya está **agrupada por
clase** (la clave del dict).

### 3.3 Carga del modelo — `load_yolo`

```python
def load_yolo(weights: str | Path | None = None, device: str | None = None):
    ...  # -> YOLO
```

- `weights=None` ⇒ se resuelve desde el config (`working_dirs.yolo_weights`) con
  `get_abs_path`. Un valor concreto fuerza la ruta.
- **Caché singleton** vía un `_cached_load_yolo(weights_str)` decorado con
  `lru_cache(maxsize=1)`; `load_yolo` resuelve la ruta y delega. Mismo patrón que
  `sam3_loader` (la 2ª llamada reutiliza el modelo).
- **Device**: ultralytics mueve el modelo al device en `predict(device=...)`, así
  que la carga es device-agnóstica; `device` se resuelve y se aplica en
  `detect_boxes` (§3.5). El parámetro `device` de `load_yolo` se acepta por simetría
  pero no fija el singleton (igual criterio que `load_sam3`).
- `ultralytics` se importa **dentro** de la función.

### 3.4 Lectura de config y mapeo de clases

- `_load_yolo_config()` (local, patrón de `tracking._load_tracking_config`): lee
  `CONFIG_FILENAME` del `.env`, resuelve `configs/<...>` con `get_abs_path`, parsea
  el JSON y devuelve lo necesario: ruta `working_dirs.yolo_weights`, sección `yolo`
  (`conf`, `imgsz` con defaults `0.4`/`960`).
- **Mapa `yolo_id → nombre de clase del repo`**: se construye desde las clases del
  config (`_load_classes()`), tomando solo las que tienen `yolo_id`:
  `{cls["yolo_id"]: cls["name"] for cls in classes if "yolo_id" in cls}`. Así
  `green_floor` (sin `yolo_id`) queda fuera de la salida de YOLO.

### 3.5 Building block `detect_boxes` — firma y flujo

```python
def detect_boxes(
    frame: np.ndarray,
    model=None,
    classes: list[dict] | None = None,
    conf: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
) -> dict[str, list[BoxDetection]]:
    ...
```

Flujo:

1. `model = model or load_yolo()`.
2. Resolver `classes` (`_load_classes()` si `None`) y construir el mapa
   `yolo_id → nombre`.
3. Resolver `conf`/`imgsz` (argumento → config → default) y `device`
   (`cuda` si disponible, si no `cpu`, salvo override).
4. `img = PIL.Image.fromarray(frame)` (RGB) → evita ambigüedad RGB/BGR.
5. `res = model.predict(img, imgsz=imgsz, conf=conf, device=device,
   verbose=False)[0]`.
6. Inicializar `out = {name: [] for name in mapa.values()}` (todas las clases YOLO,
   listas vacías por defecto → AC-6).
7. Recorrer `res.boxes`: para cada caja, `cls_id = int(box.cls)`; si `cls_id` no
   está en el mapa → **descartar** (AC-5); si está, `name = mapa[cls_id]`,
   `bbox = tuple(float(v) for v in box.xyxy[0])`, `score = float(box.conf)`;
   `out[name].append(BoxDetection(bbox, score))`.
8. `return out`.

> Lectura vectorizada alternativa (equivalente): `res.boxes.xyxy.cpu().numpy()`,
> `res.boxes.cls.cpu().numpy().astype(int)`, `res.boxes.conf.cpu().numpy()` y zip,
> como el notebook. Se elige la forma más legible en implementación.

### 3.6 Imports perezosos

`yolo_boxes.py`: a nivel de módulo solo `numpy`, `dataclasses`, `pathlib`,
`functools` y `Path`/typing. `ultralytics`, `torch` (para `cuda.is_available`) y
`PIL.Image` se importan **dentro** de las funciones. `import src.core` y
`import src.core.detectors` no deben arrastrar `ultralytics`.

### 3.7 Crecimiento del config de la fase

Editar `configs/01_yolo_sam3_config.json` (persistente; AC-8):

- En `working_dirs`, añadir `"yolo_weights": "assets/yolo/best.pt"`.
- En `classes`:
  - `robot`   → añadir `"yolo_id": 0`.
  - `orange_ball` → añadir `"yolo_id": 1`.
  - añadir clase **`yellow_zone`**:
    `{ "name": "yellow_zone", "sam3_prompts": ["yellow zone"], "color": [255, 230, 0], "coco_id": 4, "yolo_id": 2 }`.
  - `green_floor` → **sin** `yolo_id` (sigue siendo text-prompt).
- Añadir sección nueva:
  `"yolo": { "conf": 0.4, "imgsz": 960 }`.

> El `best.pt` es un **artefacto real** bajo `assets/yolo/` (git-ignored, como
> `assets/sam3`); poblarlo es setup de entorno (hoy manual; futuro `bootstrap_data`),
> no código de esta tarea.

---

## 4. Script de validación (smoke)

Archivo: `testing/test_yolo_detector.py` (script manual, **no** pytest). Requiere
`best.pt` en disco; admite **CPU** (YOLO es ligero), así que corre en el pod o donde
estén los pesos.

1. Localizar un `.MOV` real (rglob sobre `dataset_dir`) y extraer **1 frame**
   (patrón de `test_segmentation.py`/`test_box_prompt.py`).
2. `model = load_yolo()`.
3. `out = detect_boxes(frame, model=model)`.
4. **Aserciones:** `out` es `dict`; sus claves son las clases con `yolo_id`
   (`robot`, `orange_ball`, `yellow_zone`) y **no** incluye `green_floor`; cada
   `BoxDetection` tiene `bbox` de 4 valores y `score` en `[0, 1]`.
5. **Reporte:** conteo de cajas por clase y scores; opcionalmente dibujar las cajas
   con `cv2.rectangle` y guardar un PNG bajo `outputs/` para inspección visual.

---

## 5. Riesgos y consideraciones

- **RGB/BGR:** pasar `PIL.Image` (RGB) a `predict` evita el supuesto BGR de
  ultralytics sobre arrays numpy. Documentado en §3.5.
- **Doble modelo en memoria:** en el pipeline compuesto (tarea 3) YOLO y SAM3
  coexisten en GPU; aceptado (es el diseño). Aquí YOLO se carga aparte y cacheado.
- **`best.pt` ausente:** `load_yolo` falla temprano y claro vía `get_abs_path`
  (`FileNotFoundError`); el smoke reporta sin abortar el resto si no está.
- **`coco_id` vs `yolo_id`:** son numeraciones **distintas** (COCO 1-indexed del
  repo vs índice de clase YOLO 0-indexed). Por eso se añade un campo separado
  `yolo_id`, no se reutiliza `coco_id`.
- **No regresión:** el crecimiento del config solo **añade** claves; las existentes
  (incluido `green_floor` sin `yolo_id`) no cambian, así que los consumidores
  actuales del config siguen igual.

---

## 6. Qué NO incluye este plan

- La composición `detect_boxes` → `boxes_to_masks` → tracker (tarea
  `detector_strategy`).
- Máscaras, `obj_id` estable, mp4/JSON, `green_floor`, y el refactor del detector
  SAM3-text.
- La descomposición en pasos accionables y su checklist: corresponde a `tasks.md`.
