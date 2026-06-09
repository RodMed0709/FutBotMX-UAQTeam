# Tasks — Orquestación de inferencia por lotes (`batch_inference`)

- **Tarea atómica:** `batch_inference`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** la **Parte B del test corre EXCLUSIVAMENTE en el pod (GPU)**
> (carga SAM3 y procesa un lote real). La **Parte A** (firma, selección, skip-done,
> aislamiento con `run_inference`/`load_sam3` monkeypatcheados) y el lint estático
> (`ruff`/`black`) corren en **cualquier entorno**, sin GPU.

---

## Fase A — Módulo y firma

- [x] **T1 — Crear `src/core/batch.py` con la firma de `run_batch`**
  - Nuevo módulo con `run_batch(mode="segmentation", split=2, videos=None,
    sampling="auto", max_frames=None, include_masks=False, render_video=False,
    overwrite=False) -> list[dict]`. Importar `run_inference` a nivel de módulo;
    `pandas`/`load_sam3`/loaders de metadata se importan **dentro** de la función.
  - **Verificación:** `from src.core.batch import run_batch`; firma con los defaults
    del plan.
  - **Plan:** §3.1, §3.2. **Spec:** AC-1, AC-5.

- [x] **T2 — Lectura de config: `metadata_csv` y `outputs_dir`**
  - Obtener la ruta del manifiesto vía `_load_metadata_config()` y `outputs_dir` de la
    config activa (`CONFIG_FILENAME` del `.env`, parseo `strip()`), una sola vez.
  - **Verificación:** revisión de código; rutas resueltas con `get_abs_path`.
  - **Plan:** §3.3, §3.4. **Spec:** AC-2.

---

## Fase B — Cuerpo del orquestador

- [x] **T3 — Selección de videos desde el manifiesto**
  - `pandas` sobre `db_metadata.csv`, orden determinista por `id`. Si `videos` se
    pasa (rutas **o** ids) acota y **tiene prioridad** sobre `split`; id/ruta
    inexistente → `ValueError`. Si no, filtra por `df["split"] == split`. Devuelve
    `(id, ruta)` por fila.
  - **Verificación:** Parte A (filtro split, lista explícita, error de inexistente).
  - **Plan:** §3.3. **Spec:** AC-2, AC-3, AC-10.

- [x] **T4 — Skip-done por JSON canónico**
  - Por video, antes de invocar inferencia: `json_path =
    inference_paths(Path(ruta).stem, outputs_dir)[0]`; si existe y no `overwrite` →
    `"skipped"`, sin llamar a `run_inference`.
  - **Verificación:** Parte A (JSON falso → skipped; `overwrite=True` → reprocesa).
  - **Plan:** §3.4. **Spec:** AC-6.

- [x] **T5 — Carga única del modelo y bucle secuencial**
  - `bundle = load_sam3()` **una vez** antes del bucle; pasar `bundle` a cada
    `run_inference(ruta, mode, sampling, max_frames, include_masks, render_video,
    bundle)`. Iteración secuencial, sin paralelismo.
  - **Verificación:** revisión de código; Parte B (una sola carga reusada en 3 videos).
  - **Plan:** §3.5. **Spec:** AC-4, AC-9, AC-10.

- [x] **T6 — Aislamiento de errores**
  - `try/except Exception` por video → `"failed"` con `error=repr(exc)` y el bucle
    continúa; `KeyboardInterrupt` se **re-levanta** (abortable).
  - **Verificación:** Parte A (un video falla → failed, los demás done, no se detiene).
  - **Plan:** §3.5. **Spec:** AC-7.

- [x] **T7 — Resumen estructurado y logging**
  - Retorno `list[dict]` por video (`id, ruta, status, json, video, error`) según
    estado; print `[i/N] <ruta> -> <status>` y un resumen final con conteos
    (`done`/`skipped`/`failed`).
  - **Verificación:** Parte A (forma del retorno + conteos); revisión del print.
  - **Plan:** §3.6. **Spec:** AC-8.

- [x] **T8 — Docstring de `run_batch` (español)**
  - Documentar parámetros, fuente de videos, skip-done/`overwrite`, aislamiento de
    errores, render OFF por defecto y la forma del resumen.
  - **Verificación:** docstring presente y coherente con el plan.
  - **Plan:** §3.2, §3.6. **Spec:** AC-5, AC-8.

---

## Fase C — Anti-alcance (verificación de no-regresión)

- [x] **T9 — Confirmar que no cambia nada colateral**
  - Sin tocar `inference.py`, `pipeline.py`, `tracking.py`, `inference_schema.py`
    (incl. `SCHEMA_VERSION`), overlay/escritura/extracción ni `src/data/`
    (`db_metadata.csv` solo se lee). Sin paralelismo. `requirements.txt` sin cambios.
  - **Verificación:** `git diff` limitado a `src/core/batch.py` (nuevo) y el test
    nuevo; `SCHEMA_VERSION` intacto.
  - **Plan:** §3.7, §4. **Spec:** AC-11.

---

## Fase D — Test

- [x] **T10 — Crear `testing/test_batch_inference.py`**
  - **Parte A (local, sin GPU):** firma; selección sobre un `db_metadata.csv`
    **temporal** (filtro split + lista explícita ruta/id + `ValueError` por
    inexistente); skip-done con JSON falso en la ruta canónica (y `overwrite`);
    aislamiento monkeypatcheando `run_inference` para fallar en un video; todo con
    `load_sam3`/`run_inference` monkeypatcheados (sin SAM3).
  - **Parte B (pod/GPU):** **3 primeros videos del split reservado** (`split=0`):
    `run_batch(mode="segmentation", sampling="quota", include_masks=True,
    render_video=True)` y `run_batch(mode="tracking", max_frames=300,
    include_masks=True, render_video=True)`; ambos → 3 × `done` con mp4 + JSON (`rle`;
    tracking además `frames`+`tracks`); segunda corrida → 3 × `skipped`.
  - **Verificación:** el script existe; la Parte A es ejecutable localmente.
  - **Plan:** §5.1, §5.2. **Spec:** AC-12.

---

## Fase E — Ejecución y calidad

- [x] **T11 — Ejecutar la Parte A en local**
  - Correr la Parte A de `test_batch_inference.py` **sin GPU**.
  - **Verificación:** selección, skip-done, aislamiento y resumen pasan en local.
  - **Plan:** §5.1. **Spec:** AC-2, AC-3, AC-6, AC-7, AC-8, AC-12.

- [ ] **T12 — Ejecutar la Parte B en el pod (GPU)**
  - Correr la Parte B **en el pod** (SAM3 + GPU). **No se corre en local.**
  - **Verificación:** lote de 3 reservados en ambos modos; video + JSON extendido;
    carga única; segunda corrida `skipped`; resumen correcto.
  - **Plan:** §5.2. **Spec:** AC-1, AC-4, AC-5, AC-6, AC-8, AC-9.

- [x] **T13 — Calidad e importabilidad**
  - `ruff check .` y `black .` sin hallazgos; `from src.core.batch import run_batch` OK.
  - **Verificación:** lint limpio; import correcto.
  - **Plan:** §5.3. **Spec:** AC-11.

- [x] **T14 — Documentación de cierre (`CLAUDE.md` + docstring)**
  - Añadir la capa batch (`src/core/batch.py::run_batch`) a la arquitectura de
    `CLAUDE.md`, sobre la fachada `run_inference`.
  - **Verificación:** `CLAUDE.md` refleja la capa de lotes.
  - **Plan:** §4. **Spec:** AC-1.

- [ ] **T15 — Commit (requiere confirmación)**
  - Commitear `src/core/batch.py`, el test y `CLAUDE.md`. **El agente NO commitea por
    iniciativa propia:** pregunta y espera confirmación (constitución §11).
    Conventional Commits en inglés, scope `batch_inference`.
  - **Verificación:** tras tu confirmación, el commit existe.
  - **Plan:** —. **Spec:** —

---

## Trazabilidad resumida

| Tarea                              | Plan       | Spec (AC)                          |
| ---------------------------------- | ---------- | ---------------------------------- |
| T1 módulo + firma                  | §3.1, §3.2 | AC-1, AC-5                         |
| T2 lectura config (csv/outputs)    | §3.3, §3.4 | AC-2                               |
| T3 selección de videos             | §3.3       | AC-2, AC-3, AC-10                  |
| T4 skip-done                       | §3.4       | AC-6                               |
| T5 carga única + bucle secuencial  | §3.5       | AC-4, AC-9, AC-10                  |
| T6 aislamiento de errores          | §3.5       | AC-7                               |
| T7 resumen + logging               | §3.6       | AC-8                               |
| T8 docstring                       | §3.2, §3.6 | AC-5, AC-8                         |
| T9 anti-alcance (no-regresión)     | §3.7, §4   | AC-11                              |
| T10 crear test (A + B)             | §5.1, §5.2 | AC-12                              |
| T11 ejecutar Parte A (local)       | §5.1       | AC-2, AC-3, AC-6, AC-7, AC-8, AC-12|
| T12 ejecutar Parte B (pod)         | §5.2       | AC-1, AC-4, AC-5, AC-6, AC-8, AC-9 |
| T13 calidad/import                 | §5.3       | AC-11                              |
| T14 documentación de cierre        | §4         | AC-1                               |
| T15 commit (confirmación)          | —          | —                                  |

---

> **Fuera de esta tarea (futuro):** paralelismo del lote y el **ejemplo end-to-end
> sobre un video real completo** (cierre del roadmap del pipeline unificado).
