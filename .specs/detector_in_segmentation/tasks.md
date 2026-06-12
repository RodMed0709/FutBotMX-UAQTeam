# Tasks — Detector inyectable en segmentación (`detector_in_segmentation`)

- **Tarea atómica:** `detector_in_segmentation`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]` al
> completar.
>
> **Nota de ejecución:** las pruebas que invocan SAM3 corren **EXCLUSIVAMENTE en el
> pod (GPU)** (`run_pipeline` carga el modelo). La introspección de firma, el caso
> `ValueError` (que falla **antes** de cargar SAM3) y el lint estático
> (`ruff`/`black`) corren en **cualquier entorno**, sin GPU.

---

## Fase A — Detector inyectable en `run_pipeline` (`pipeline.py`)

- [x] **T1 — `run_pipeline` acepta `detector`**
  - Agregar `detector: str | None = None` **al final** de la firma (no romper
    llamadas posicionales).
  - **Verificación:** introspección de firma (`inspect.signature`) muestra `detector`
    con default `None`. (Sin GPU.)
  - *Origen:* plan §3.2 / criterio #1.

- [x] **T2 — Resolver `detector_fn` antes del loop y de `load_sam3`**
  - Tras `cfg_classes, outputs_dir, config_fps, config = _load_pipeline_config()`:
    ```python
    if detector is None:
        detector = config.get("detector", "sam3_text")
    detector_fn = get_detector(detector)
    ```
  - Ubicarlo **antes** de la llamada a `load_sam3()` para garantizar el fail-cheap.
  - **Verificación:** un `detector` inválido levanta `ValueError` **sin** cargar SAM3
    (test sin GPU). (Sin GPU.)
  - *Origen:* plan §3.3 / criterios #1, #4.

- [x] **T3 — Invocar la estrategia en el loop por-frame**
  - Reemplazar `detect_classes_in_frame(frame, classes=classes, bundle=bundle)` por
    `detector_fn(frame, classes=classes, bundle=bundle)` (hoy `pipeline.py:174`).
  - **Verificación:** el resto del cuerpo (overlay, `frame_record`) no cambia; lectura
    de diff confirma que solo se sustituye la llamada de detección.
  - *Origen:* plan §3.4 / criterio #2.

- [x] **T4 — Ajustar imports**
  - Agregar `from src.core.detectors import get_detector` (nivel de módulo).
  - Eliminar el import ahora sin uso `from src.core.segmentation import
    detect_classes_in_frame` **solo** si no queda otra referencia en `pipeline.py`.
  - **Verificación:** `ruff check .` sin F401. (Sin GPU.)
  - *Origen:* plan §3.4, §7.

## Fase B — Propagación en la fachada (`inference.py`)

- [x] **T5 — `run_inference` pasa `detector` a `run_pipeline`**
  - En la rama de segmentación (hoy `inference.py:69-71`), añadir `detector=detector`
    a la llamada `run_pipeline(...)`.
  - **Verificación:** lectura de diff; `run_inference(mode="segmentation",
    detector="yolo_sam3")` ya no descarta el argumento.
  - *Origen:* plan §3.5 / criterio #3.

- [x] **T6 — Corregir el docstring del parámetro `detector`**
  - Reescribir la parte (hoy ~`inference.py:49-55`) que dice "En `mode="segmentation"`
    se ignora": describir que `detector` es **ortogonal al modo** y que en
    segmentación selecciona la estrategia por-frame sin asociación temporal.
  - **Verificación:** el docstring ya no afirma que se ignora en segmentación.
  - *Origen:* plan §3.5 / criterio #3.

## Fase C — Anti-alcance (verificación de no-regresión)

- [x] **T7 — Confirmar que NO se tocó lo fuera de alcance**
  - Sin cambios en `inference_schema.py`, rutas de salida/`skip-done`,
    `segmentation.py`, estrategia `yolo_sam3`, `configs/`, ni `batch.py`.
  - **Verificación:** `git diff --name-only` lista solo `pipeline.py`, `inference.py`,
    `testing/test_pipeline.py`, `CLAUDE.md` (y los `.specs/` de esta tarea).
  - *Origen:* plan §3.6 / criterio #7.

## Fase D — Test (`testing/test_pipeline.py`)

- [x] **T8 — Smoke local: `ValueError` sin GPU**
  - Caso que pasa un `detector` inexistente a `run_pipeline` y espera `ValueError`
    **antes** de cualquier carga de modelo.
  - **Verificación:** corre y pasa en entorno sin GPU.
  - *Origen:* plan §5.1 / criterio #4.

- [ ] **T9 — Smoke pod/GPU: `yolo_sam3` en segmentación**
  - Sobre un clip corto: `mode`/`per_frame` con `detector="yolo_sam3"` corre sin error
    y produce un JSON con las **mismas claves** de header y `frames` que `sam3_text`.
  - **Verificación:** comparar estructura (claves) del JSON `yolo_sam3` vs `sam3_text`
    sobre el mismo clip. (Pod/GPU.)
  - *Origen:* plan §5.2 / criterios #5, #6.

- [ ] **T10 — Smoke pod/GPU: retrocompatibilidad**
  - `detector=None` (o `"sam3_text"`) sobre el mismo clip produce salida **equivalente**
    a la de hoy.
  - **Verificación:** estructura de JSON idéntica a la corrida sin `detector`.
    (Pod/GPU.)
  - *Origen:* plan §5.2 / criterio #6.

## Fase E — Documentación y calidad

- [x] **T11 — Actualizar `CLAUDE.md`**
  - En la descripción de `pipeline.py`, indicar que `run_pipeline` ahora acepta
    `detector` (además de `classes`/`bundle`), seleccionando la estrategia por-frame.
  - **Verificación:** la sección "Code architecture" lo refleja.
  - *Origen:* plan §3.1 / criterio #8.

- [x] **T12 — Lint y formato**
  - `ruff check .` y `black .` limpios.
  - **Verificación:** ambos sin hallazgos. (Sin GPU.)
  - *Origen:* plan §5.3.

---

## Trazabilidad resumida

| Criterio (spec §5) | Tareas |
|---|---|
| 1. `run_pipeline` acepta `detector`, default config | T1, T2 |
| 2. Detección vía estrategia resuelta | T3 |
| 3. Fachada propaga `detector`; docstring corregido | T5, T6 |
| 4. `ValueError` antes de cargar SAM3 | T2, T8 |
| 5. `yolo_sam3` en segmentación → mismo JSON | T9 |
| 6. Retrocompatibilidad sin `detector`/`sam3_text` | T9, T10 |
| 7. No toca schema/rutas/configs/`yolo_sam3` | T7 |
| 8. `CLAUDE.md` actualizado | T11 |

---

## Trabajo futuro (fuera de esta tarea)

- **`config_aware_output_paths` (tarea 2):** namespacing de salidas por config
  (`outputs/inference/<config>/<stem>/…`, subcarpetas) y `skip-done` por config, para
  correr varias configs sin pisarse.
- **Notebooks de benchmark:** Fase 1 (eficiencia de detectores, sin tracker) y Fase 2
  (trackers en **2×2**: ambos detectores × bytetrack/botsort, consistencia).
