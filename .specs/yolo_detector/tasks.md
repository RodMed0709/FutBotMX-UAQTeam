# Tasks — Detector de cajas YOLO (`yolo_detector`)

- **Tarea atómica:** `yolo_detector`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Config de la fase

- [x] **T1 — Crecer `configs/01_yolo_sam3_config.json`**
  - En `working_dirs`, añadir `"yolo_weights": "assets/yolo/best.pt"`.
  - En `classes`: añadir `"yolo_id": 0` a `robot` y `"yolo_id": 1` a `orange_ball`;
    añadir la clase `{ "name": "yellow_zone", "sam3_prompts": ["yellow zone"],
    "color": [255, 230, 0], "coco_id": 4, "yolo_id": 2 }`; dejar `green_floor`
    **sin** `yolo_id`.
  - Añadir sección `"yolo": { "conf": 0.4, "imgsz": 960 }`.
  - **Verificación:** el JSON es válido; las claves existentes no cambian;
    `green_floor` sigue sin `yolo_id`.
  - **Plan:** §3.7. **Spec:** AC-8.

---

## Fase B — Carga del modelo

- [x] **T2 — `BoxDetection` y `load_yolo` en `src/core/detectors/yolo_boxes.py`**
  - Definir la dataclass `BoxDetection(bbox: tuple[float,float,float,float],
    score: float)`.
  - `_load_yolo_config()`: leer `CONFIG_FILENAME` del `.env` con `strip()`, resolver
    `configs/<...>` con `get_abs_path`, parsear JSON y devolver
    `working_dirs.yolo_weights` + sección `yolo` (`conf`/`imgsz` con defaults).
  - `load_yolo(weights=None, device=None)`: resolver ruta (config si `None`) con
    `get_abs_path`; delegar en `_cached_load_yolo(weights_str)` decorada con
    `lru_cache(maxsize=1)` que hace `from ultralytics import YOLO; return
    YOLO(str(weights))`. `ultralytics` importado **dentro**.
  - **Verificación:** `load_yolo()` resuelve la ruta del config y devuelve un modelo;
    2ª llamada por defecto ⇒ **mismo** objeto (caché); `best.pt` ausente ⇒
    `FileNotFoundError` temprano; `import src.core` no carga `ultralytics`.
  - **Plan:** §3.2, §3.3, §3.4, §3.6. **Spec:** AC-1, AC-2.

---

## Fase C — Inferencia y mapeo

- [x] **T3 — `detect_boxes` (inferencia + agrupado por clase)**
  - Firma del §3.5:
    `detect_boxes(frame, model=None, classes=None, conf=None, imgsz=None,
    device=None) -> dict[str, list[BoxDetection]]`.
  - Flujo: `model = model or load_yolo()`; resolver `classes` (`_load_classes()` si
    `None`) y construir el mapa `{yolo_id: name}` (solo clases con `yolo_id`);
    resolver `conf`/`imgsz`/`device`; `PIL.Image.fromarray(frame)`;
    `model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]`;
    inicializar `out` con listas vacías por clase del mapa; recorrer cajas y
    empaquetar `BoxDetection(bbox xyxy, score)` bajo el nombre mapeado.
  - **Verificación:** con un frame con objetos devuelve cajas agrupadas por nombre
    del repo; un frame sin objetos devuelve listas vacías; `torch`/`PIL` importados
    solo al invocar.
  - **Plan:** §3.4, §3.5, §3.6. **Spec:** AC-3, AC-4, AC-6, AC-7.

- [x] **T4 — Casos borde de mapeo (defensivo)**
  - Una clase YOLO (`cls_id`) **no** presente en el mapa se **descarta** sin error.
    `green_floor` (sin `yolo_id`) **no** aparece nunca en la salida.
  - **Verificación:** la salida nunca contiene `green_floor`; un `cls_id`
    desconocido no rompe.
  - **Plan:** §3.4, §3.5 (paso 7). **Spec:** AC-4, AC-5.

- [x] **T5 — Exportar en `src/core/detectors/__init__.py`**
  - Añadir `from src.core.detectors.yolo_boxes import BoxDetection, detect_boxes,
    load_yolo` y sumarlos a `__all__` (junto a `boxes_to_masks`).
  - **Verificación:** `from src.core.detectors import detect_boxes, load_yolo,
    BoxDetection` funciona desde cualquier cwd; `ruff check .` pasa.
  - **Plan:** §3.1.

---

## Fase D — Validación

- [ ] **T6 — Script smoke `testing/test_yolo_detector.py`**
  - Implementar el §4: localizar `.MOV` real (rglob) y extraer 1 frame;
    `load_yolo()`; `detect_boxes(frame)`; aserciones (claves = clases con `yolo_id`,
    **sin** `green_floor`; `bbox` de 4 valores; `score` en `[0,1]`); reportar conteo
    por clase y scores; opcional: dibujar cajas y guardar PNG bajo `outputs/`.
  - **Verificación:** el script corre end-to-end (admite CPU), las aserciones pasan
    y el reporte/overlay es coherente.
  - **Plan:** §4. **Spec:** AC-9.

---

## Fase E — Cierre

- [x] **T7 — Lint, formato y no-regresión**
  - `ruff check .` y `black .` limpios sobre lo nuevo. Confirmar que
    `import src.core` no arrastra `ultralytics`; que el crecimiento del config solo
    **añade** claves (consumidores actuales intactos).
  - **Verificación:** linters limpios; import barato confirmado; no-regresión del
    config.
  - **Plan:** §3.6, §5. **Spec:** AC-2.

---

## Trabajo futuro (fuera de esta tarea)

- Composición `detect_boxes` → `boxes_to_masks` → tracker en la tarea
  `detector_strategy`: ahí las cajas YOLO se vuelven máscaras y el `obj_id` se
  estabiliza.
- Selección de tracker (ByteTrack/BoT-SORT) y demás claves de config de la fase.
