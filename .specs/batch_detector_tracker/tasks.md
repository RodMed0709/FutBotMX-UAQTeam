# Tasks — Paridad de batch para detector/tracker (`batch_detector_tracker`)

- **Tarea atómica:** `batch_detector_tracker`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Validación temprana

- [x] **T1 — Helper `_validate_detector_tracker` en `src/core/batch.py`**
  - Función privada `_validate_detector_tracker(detector, tracker) -> None` con
    imports **perezosos** de `KNOWN_TRACKERS` (`src.core.trackers`) y `get_detector`
    (`src.core.detectors`).
  - `tracker is not None and tracker not in KNOWN_TRACKERS` ⇒ `ValueError` con el
    mensaje canónico (lista de válidos). `detector is not None` ⇒ `get_detector(detector)`
    (delega el `ValueError`). `None` se acepta en ambos.
  - **Verificación:** `import src.core.batch` **no** arrastra `trackers`/`detectors`
    (siguen perezosos); llamar la helper con `tracker="x"` o `detector="x"` lanza
    `ValueError`; con `None`/nombres válidos no lanza.
  - **Plan:** §3.1. **Spec:** AC-4.

---

## Fase B — Firma y propagación

- [x] **T2 — Ampliar la firma de `run_batch`**
  - Añadir `detector: str | None = None` y `tracker: str | None = None` **al final**
    de la firma (después de `overwrite`).
  - **Verificación:** inspección de firma muestra ambos params con default `None`;
    las llamadas posicionales previas siguen resolviendo.
  - **Plan:** §3.2. **Spec:** AC-1.

- [x] **T3 — Validación temprana en el cuerpo (antes de `load_sam3()`)**
  - Insertar `_validate_detector_tracker(detector, tracker)` como **primera** línea
    del cuerpo de `run_batch`, antes de `_load_outputs_dir`, `_select_videos` y
    `load_sam3()`.
  - **Verificación:** con un nombre inválido, `run_batch` aborta **sin** leer config,
    sin tocar el manifiesto y sin cargar SAM3.
  - **Plan:** §3.3. **Spec:** AC-4.

- [x] **T4 — Propagar a `run_inference` en el bucle**
  - Añadir `detector=detector, tracker=tracker` a la llamada `run_inference(...)`
    existente, sin reordenar el resto de argumentos.
  - **Verificación:** los dos argumentos llegan a `run_inference` en cada video; con
    `detector=None`/`tracker=None` el comportamiento es idéntico al actual.
  - **Plan:** §3.4. **Spec:** AC-2, AC-3.

- [x] **T5 — Docstring de `run_batch`**
  - Documentar `detector` y `tracker`: semántica "solo tracking" (ignorados en
    segmentación), `None` ⇒ default del config, inválido ⇒ `ValueError` antes de
    cargar SAM3. Añadir ese `ValueError` a la sección `Raises:`.
  - **Verificación:** el docstring describe ambos params y la condición de error;
    coherente con el de `run_inference`.
  - **Plan:** §3.5. **Spec:** AC-5, AC-6.

---

## Fase C — Test

- [x] **T6 — Ampliar `testing/test_batch_inference.py` (validación temprana)**
  - Caso(s): `run_batch(tracker="inexistente")` ⇒ `ValueError` sin cargar SAM3;
    `run_batch(detector="inexistente")` ⇒ `ValueError` sin cargar SAM3. Usar un
    `split`/`videos` que seleccione ≥1 video del manifiesto.
  - **Verificación (local, sin GPU):** el script corre y ambos casos lanzan
    `ValueError` antes de cualquier carga de modelo; `ruff check .` / `black .`
    limpios.
  - **Plan:** §5. **Spec:** AC-7, AC-8.

---

## Notas

- **Orden sugerido:** T1 → T2 → T3 → T4 → T5 → T6. T3 depende de T1 (la helper) y de
  T2 (los params en la firma).
- **Único archivo de código de producción:** `src/core/batch.py`. Único archivo de
  test: `testing/test_batch_inference.py`. Nada más se toca (ni config, ni
  `run_inference`, ni trackers/detectors, ni schema, ni `src/data`).
- **Cierre del proceso:** esta es la quinta y última tarea de la integración
  YOLO + SAM3 a `src/`.
