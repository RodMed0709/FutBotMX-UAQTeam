# Tasks — Segmentación por caja con SAM3 (`sam3_box_prompt`)

- **Tarea atómica:** `sam3_box_prompt`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Extensión del loader (cara tracker)

- [x] **T1 — Añadir campos opcionales a `Sam3Bundle`**
  - En `src/core/sam3_loader.py`, sumar a la dataclass los campos
    `tracker_processor: Any = None` y `tracker_model: Any = None` (después de
    `device`, con default `None`).
  - **Verificación:** `load_sam3()` por defecto sigue devolviendo un bundle con
    ambos campos en `None`; los llamadores actuales no cambian; `import src.core`
    no carga `torch`/`transformers`.
  - **Plan:** §3.2. **Spec:** AC-2.

- [x] **T2 — Función `ensure_tracker_loaded(bundle)` (carga perezosa e idempotente)**
  - Definir en `sam3_loader.py`. Si `bundle.tracker_model is not None` → devolver
    el bundle sin recargar. Si no: importar `Sam3TrackerModel`/`Sam3TrackerProcessor`
    **dentro** de la función; resolver ruta con `_resolve_sam3_dir()`; cargar
    processor + model (`bfloat16`, `low_cpu_mem_usage=True`, `.to(bundle.device)`,
    `.eval()`); asignarlos al bundle; devolver el bundle.
  - Docstring: documentar el warning benigno `sam3_video → sam3_tracker` como
    esperado; **no** silenciarlo con código.
  - **Verificación:** primera llamada carga la cara tracker en el `device` del
    bundle; segunda llamada **no** recarga (idempotente, mismos objetos);
    `torch`/`transformers` se importan solo al invocarla.
  - **Plan:** §3.2, §3.6, §3.7. **Spec:** AC-1, AC-7.

---

## Fase B — Building block box-prompt

- [x] **T3 — Crear el subpaquete `src/core/detectors/`**
  - Crear `src/core/detectors/__init__.py`.
  - **Verificación:** `import src.core.detectors` funciona; el paquete no arrastra
    dependencias pesadas al importarse.
  - **Plan:** §3.1.

- [x] **T4 — Implementar `boxes_to_masks` en `src/core/detectors/box_prompt.py`**
  - Firma del §3.3:
    `boxes_to_masks(frame, boxes, bundle=None, scores=None) -> list[Detection]`,
    decorada con `@torch.no_grad()`. Importar `Detection` de
    `src.core.segmentation`; `numpy` a nivel de módulo; `torch` y `PIL.Image`
    **dentro** de la función.
  - Flujo del §3.4: resolver bundle (`load_sam3()` si `None`) + `ensure_tracker_loaded`;
    caso vacío → `[]` sin invocar al modelo; PIL desde el frame; `tracker_processor(
    images=[img], input_boxes=[boxes])` a `device`; castear **solo flotantes** a
    `bfloat16`; `tracker_model(..., multimask_output=False)`;
    `post_process_masks(out.pred_masks.cpu(), original_sizes)`; normalizar
    `ndim==4 → [:,0]`; `astype(bool)`.
  - Empaquetar 1:1: `Detection(obj_id=i, mask=mask, score=scores[i] if scores else 1.0)`.
  - **Verificación:** con N cajas válidas devuelve N `Detection` en orden, máscaras
    `bool (H,W)`; lista de cajas vacía → `[]` sin llamar al modelo.
  - **Plan:** §3.3, §3.4. **Spec:** AC-3, AC-4, AC-5.

- [x] **T5 — Casos borde: máscara vacía y desajuste N (defensivo)**
  - No filtrar máscaras vacías (se devuelve la `Detection` igual). Si el modelo
    devuelve menos máscaras que cajas, empaquetar las disponibles en orden sin
    excepción.
  - **Verificación:** una caja que produce máscara vacía sigue devolviendo su
    `Detection` (preserva 1:1); un desajuste no revienta.
  - **Plan:** §3.4 (nota defensiva), §3.5. **Spec:** AC-6.

- [x] **T6 — Exportar en `src/core/detectors/__init__.py`**
  - `from src.core.detectors.box_prompt import boxes_to_masks` y
    `__all__ = ["boxes_to_masks"]`.
  - **Verificación:** `from src.core.detectors import boxes_to_masks` funciona
    desde cualquier cwd; `ruff check .` y `black .` pasan sobre el código nuevo.
  - **Plan:** §3.1.

---

## Fase C — Validación

- [x] **T7 — Script smoke `testing/test_box_prompt.py` (corre en el pod)**
  - Implementar la estrategia del §4: `extract_frames(<video real no-forced_testing>)`
    → 1 frame; `load_sam3()`; producir cajas reales con
    `detect_classes_in_frame` + `mask_to_bbox_centroid` (clase con detecciones,
    p. ej. `robot`); llamar `boxes_to_masks`; aserciones
    (`len(out)==len(boxes)`, ≥1 máscara no vacía, formas/dtype); guardar un overlay
    PNG bajo `outputs/` para inspección visual.
  - **Verificación (en el pod):** el script corre end-to-end, las aserciones pasan
    y el overlay muestra máscaras coherentes con las cajas.
  - **Plan:** §4. **Spec:** AC-8.

---

## Fase D — Cierre

- [x] **T8 — Lint, formato y verificación de no-regresión**
  - `ruff check .` y `black .` limpios sobre lo nuevo. Confirmar que `load_sam3()`
    por defecto **no** carga la cara tracker (campos en `None`) y que los
    consumidores existentes de `Sam3Bundle` siguen funcionando.
  - **Verificación:** linters limpios; `import src.core` no arrastra torch;
    no-regresión del loader confirmada.
  - **Plan:** §3.6, §5. **Spec:** AC-2.

---

## Trabajo futuro (fuera de esta tarea)

- Detector YOLO que produce las cajas (`yolo_detector`).
- Cableado del box-prompt al tracking/pipeline/fachada (`detector_strategy`):
  ahí el `score` real vendrá del detector y el `obj_id` pasará a ser **estable**.
- Selección de tracker (ByteTrack/BoT-SORT) y demás claves de config de la fase.
