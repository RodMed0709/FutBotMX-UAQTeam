# Tasks — Instrumentación de tiempo y memoria en el batch (`batch_timing`)

- **Tarea atómica:** `batch_timing`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Helpers de medición

- [x] **T1 — Helpers + constante en `src/core/batch.py`**
  - `import time` a nivel de módulo. Añadir `_TIMING_NULL = {"elapsed_s": None,
    "peak_vram_mb": None, "fps": None}`.
  - `_reset_peak_vram()` y `_read_peak_vram_mb()` (import perezoso de `torch`, guardados
    por `torch.cuda.is_available()`; sin CUDA: reset no-op, lectura `None`; MB con
    `1e6`).
  - `_read_num_frames(json_path) -> int | None` (`json.load`, `None` ante
    `OSError/KeyError/ValueError/TypeError`).
  - **Verificación:** sin CUDA, `_read_peak_vram_mb()` devuelve `None` y
    `_reset_peak_vram()` no lanza; `_read_num_frames` lee un JSON con `num_frames` y
    devuelve `None` ante uno sin la llave o ruta inexistente; `import src.core.batch`
    no arrastra `torch` (sigue perezoso).
  - **Plan:** §3.1, §3.2. **Spec:** AC-3, AC-7.

---

## Fase B — Integración en `run_batch`

- [x] **T2 — Medir alrededor de `run_inference` (rama `done`)**
  - Envolver la llamada: `_reset_peak_vram()` → `t0 = time.perf_counter()` →
    `run_inference(...)` → `elapsed = perf_counter()-t0` → `peak_vram =
    _read_peak_vram_mb()` → `num_frames = _read_num_frames(res["json"])` →
    `fps = num_frames/elapsed if (num_frames is not None and elapsed > 0) else None`.
    Añadir `elapsed_s`, `peak_vram_mb`, `fps` al `entry` de `done`.
  - **Verificación:** en `done`, `elapsed_s` es float > 0 y `fps` = `num_frames/elapsed_s`
    (o `None` si no hay `num_frames`); la medición envuelve solo `run_inference`.
  - **Plan:** §3.3. **Spec:** AC-1, AC-2, AC-5.

- [x] **T3 — `skipped` y `failed` con `**_TIMING_NULL`**
  - Añadir `**_TIMING_NULL` al `entry` de la rama skip-done y al de la rama `failed`
    (except).
  - **Verificación:** entradas `skipped`/`failed` llevan las 3 llaves en `None`; un
    fallo en `run_inference` no produce métricas y no rompe el lote (aislamiento
    intacto).
  - **Plan:** §3.3. **Spec:** AC-4, AC-6.

- [x] **T4 — Docstring `Returns` de `run_batch`**
  - Documentar las 3 llaves nuevas: `elapsed_s` (s), `peak_vram_mb` (MB; `None` sin
    CUDA), `fps` (`num_frames/elapsed_s`; `None` si no se lee `num_frames`); `None` en
    `skipped`/`failed`.
  - **Verificación:** el docstring describe las 3 llaves y sus condiciones de `None`.
  - **Plan:** §3.4. **Spec:** AC-1, AC-3, AC-4, AC-7.

---

## Fase C — Test

- [x] **T5 — Ampliar `testing/test_batch_inference.py`**
  - Parte A: el `run_inference` fake escribe un JSON con `num_frames`; asserts de que
    `done` trae `elapsed_s` (float > 0), `fps` (= `num_frames/elapsed_s`) y
    `peak_vram_mb is None` (sin CUDA); `skipped`/`failed` con las 3 en `None`. Extender
    el set de llaves esperadas del resumen.
  - (Opcional) Parte B pod: un assert laxo de que `peak_vram_mb` > 0 y `fps` > 0 en una
    corrida real.
  - **Verificación (local, sin GPU):** el script corre y los asserts pasan;
    `ruff check .` / `black .` limpios.
  - **Plan:** §5. **Spec:** AC-1..AC-7.

---

## Notas

- **Orden sugerido:** T1 → T2 → T3 → T4 → T5. T2/T3 dependen de T1 (helpers).
- **Único archivo de código de producción:** `src/core/batch.py`. Único archivo de
  test: `testing/test_batch_inference.py`. Nada más se toca (ni `run_inference`, ni el
  esquema, ni trackers/detectors, ni config).
- **Proceso:** primera de dos tareas del benchmark; habilita `benchmark_metrics`.
