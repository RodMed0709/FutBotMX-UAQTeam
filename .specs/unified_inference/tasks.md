# Tasks — Fachada única de inferencia (`unified_inference`)

- **Tarea atómica:** `unified_inference`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** las pruebas que invocan SAM3 corren **EXCLUSIVAMENTE en el
> pod (GPU)** (`run_pipeline`/`track_video` cargan el modelo). La introspección de
> firma, los casos `ValueError` (que fallan **antes** de cargar SAM3) y el lint
> estático (`ruff`/`black`) corren en **cualquier entorno**, sin GPU.

---

## Fase A — Ampliación aditiva de `run_pipeline` (`pipeline.py`)

- [x] **T1 — `run_pipeline` acepta `bundle` y `classes`**
  - Firma `+ bundle: Sam3Bundle | None = None` y `+ classes: list[dict] | None = None`
    (defaults `None` = comportamiento actual). Tras `_load_pipeline_config()`:
    `classes = classes if classes is not None else cfg_classes`. Sustituir la carga
    incondicional por `bundle = bundle or load_sam3()`. No tocar el bucle, el
    muestreo, el esquema ni el render.
  - **Verificación:** revisión de código + lint; funcional en Fase E (pod, reuso de
    bundle). `testing/test_pipeline.py` sigue pasando sin cambios.
  - **Plan:** §3.5. **Spec:** AC-10.

- [x] **T2 — Documentar los nuevos parámetros de `run_pipeline`**
  - Docstring: qué hacen `bundle` (modelo precargado; `None` → `load_sam3()`) y
    `classes` (override; `None` → clases de config).
  - **Verificación:** presente en el docstring de `run_pipeline`.
  - **Plan:** §3.5. **Spec:** AC-10.

---

## Fase B — Fachada `run_inference` (`src/core/inference.py`)

- [x] **T3 — Crear el módulo y la firma de la fachada**
  - Nuevo `src/core/inference.py` con `run_inference(video_path, mode="segmentation",
    output_path=None, classes=None, sampling="auto", max_frames=None, bundle=None,
    include_masks=False, render_video=True) -> dict`. Importar `run_pipeline` y
    `track_video` a nivel de módulo (sin imports de torch en la fachada).
  - **Verificación:** `from src.core.inference import run_inference`; firma con los
    defaults del plan.
  - **Plan:** §3.1, §3.2. **Spec:** AC-1, AC-2, AC-7.

- [x] **T4 — Validación de `mode` y `sampling` (sin efectos colaterales)**
  - Al inicio de `run_inference`, **antes** de cargar SAM3 o tocar el video: `mode`
    desconocido → `ValueError`; resolver `sampling` según la tabla §3.3. Casos que
    levantan `ValueError`: `sampling="quota"`+`tracking` (AC-6),
    `sampling="contiguous"`+`segmentation`, `sampling` desconocido. Mensajes en
    español explícitos.
  - **Verificación:** los casos inválidos fallan sin invocar el modelo (Fase D,
    Parte A).
  - **Plan:** §3.3. **Spec:** AC-4, AC-5, AC-6.

- [x] **T5 — Enrutado a las implementaciones + retorno unificado**
  - `segmentation` → `run_pipeline(..., all_frames=<resuelto>, mode="per_frame",
    classes, bundle, include_masks, render_video)`, devolviendo
    `{"json", "video", "index": None}`. `tracking` → `track_video(...,
    max_frames=<resuelto>, classes, bundle, include_masks, render_video)` tal cual
    (ya trae `"index"`). `max_frames` no se pasa a `run_pipeline` (ignorado en seg,
    documentado).
  - **Verificación:** revisión de código; retorno con forma única
    `{"json": Path, "video": Path|None, "index": dict|None}`; funcional en Fase E.
  - **Plan:** §3.3, §3.4. **Spec:** AC-3, AC-8, AC-9, AC-10.

- [x] **T6 — Docstring de la fachada (español)**
  - Documentar `mode`, `sampling` (con la tabla de resolución), `max_frames` (cap
    contiguo; ignorado en seg), `bundle`/`classes`, `include_masks`/`render_video`
    (sobreescribibles, ortogonales) y el retorno `Path|None` / `dict|None`.
  - **Verificación:** docstring presente y coherente con §3.3.
  - **Plan:** §3.2, §7. **Spec:** AC-7, AC-8.

---

## Fase C — Anti-alcance (verificación de no-regresión)

- [x] **T7 — Confirmar que no cambia nada colateral**
  - Sin tocar `tracking.py`, `inference_schema.py`, `overlay.py`, `video_writer.py`,
    `frame_extraction.py`, `segmentation`, ByteTrack, el muestreo ni la config
    (`mode`/`sampling`/`max_frames` son parámetros, no claves de config;
    `requirements.txt` sin cambios). `pipeline.py` solo cambia de forma **aditiva**.
  - **Verificación:** `git diff` limitado a `src/core/inference.py` (nuevo),
    `pipeline.py` (aditivo) y el test nuevo; `SCHEMA_VERSION` sin cambios.
  - **Plan:** §3.6, §4. **Spec:** AC-11.

---

## Fase D — Test

- [x] **T8 — Crear `testing/test_unified_inference.py`**
  - **Parte A (local, sin GPU):** `inspect.signature(run_inference)` con los defaults
    esperados y `signature(run_pipeline)` con `bundle`/`classes` (default `None`).
    Casos `ValueError` sin cargar modelo: `mode="bad"`, `sampling="quota"`+`tracking`,
    `sampling="contiguous"`+`segmentation`, `sampling="rara"` (verificar que el error
    precede al acceso al modelo/video).
  - **Parte B (GPU/pod):** sobre clip corto — `segmentation` auto
    (`{"json","video","index": None}` + mp4); `segmentation` `sampling="all"`,
    `render_video=False` (`video` y `index` `None`, JSON presente); `tracking` auto
    con `max_frames` pequeño (`index` dict no vacío, JSON con `frames`+`tracks`);
    `tracking` `render_video=False`+`include_masks=True` (`video` `None`, JSON con
    `rle`); reuso de un `bundle` precargado en una llamada seg y una tracking sin
    recargar el modelo.
  - **Verificación:** el script existe; la Parte A es ejecutable localmente.
  - **Plan:** §5.1, §5.2. **Spec:** AC-12.

---

## Fase E — Ejecución y calidad

- [x] **T9 — Ejecutar la Parte A en local**
  - Correr la Parte A de `test_unified_inference.py` **sin GPU**.
  - **Verificación:** firma correcta y todos los casos `ValueError` pasan en local.
  - **Plan:** §5.1. **Spec:** AC-4, AC-5, AC-6, AC-12.

- [ ] **T10 — Ejecutar la Parte B en el pod (GPU)**
  - Correr la Parte B **en el pod** (SAM3 + GPU). **No se corre en local.**
  - **Verificación:** ambos modos; retorno unificado correcto; muestreo por modo;
    reuso de bundle sin recarga; caso OFF+masks produce `rle` sin video.
  - **Plan:** §5.2. **Spec:** AC-1, AC-2, AC-3, AC-7, AC-8, AC-10.

- [x] **T11 — Calidad e importabilidad**
  - `ruff check .` y `black .` sin hallazgos; `from src.core.inference import
    run_inference` y `from src.core.pipeline import run_pipeline` (firma ampliada) OK.
  - **Verificación:** lint limpio; imports correctos.
  - **Plan:** §5.3. **Spec:** AC-11.

- [x] **T12 — Documentación de cierre (`CLAUDE.md` + docstrings)**
  - Actualizar `CLAUDE.md`: existe la fachada `run_inference`; `run_pipeline` y
    `track_video` pasan a ser implementaciones internas; retirar la nota de que
    `mode="tracking"` es un stub / vive fuera del pipeline.
  - **Verificación:** `CLAUDE.md` refleja la fachada y el estado del modo tracking.
  - **Plan:** §4. **Spec:** AC-3.

- [ ] **T13 — Commit (requiere confirmación)**
  - Commitear `src/core/inference.py`, `pipeline.py`, el test y `CLAUDE.md`. **El
    agente NO commitea por iniciativa propia:** pregunta y espera confirmación
    (constitución §11). Conventional Commits en inglés, scope `unified_inference`.
  - **Verificación:** tras tu confirmación, el commit existe.
  - **Plan:** —. **Spec:** —

---

## Trazabilidad resumida

| Tarea                                  | Plan       | Spec (AC)                          |
| -------------------------------------- | ---------- | ---------------------------------- |
| T1 `run_pipeline` + `bundle`/`classes` | §3.5       | AC-10                              |
| T2 documentar params `run_pipeline`    | §3.5       | AC-10                              |
| T3 módulo + firma de la fachada        | §3.1, §3.2 | AC-1, AC-2, AC-7                   |
| T4 validación `mode`/`sampling`        | §3.3       | AC-4, AC-5, AC-6                   |
| T5 enrutado + retorno unificado        | §3.3, §3.4 | AC-3, AC-8, AC-9, AC-10            |
| T6 docstring de la fachada             | §3.2, §7   | AC-7, AC-8                         |
| T7 anti-alcance (no-regresión)         | §3.6, §4   | AC-11                              |
| T8 crear test (A + B)                  | §5.1, §5.2 | AC-12                              |
| T9 ejecutar Parte A (local)            | §5.1       | AC-4, AC-5, AC-6, AC-12            |
| T10 ejecutar Parte B (pod)             | §5.2       | AC-1, AC-2, AC-3, AC-7, AC-8, AC-10|
| T11 calidad/import                     | §5.3       | AC-11                              |
| T12 documentación de cierre            | §4         | AC-3                               |
| T13 commit (confirmación)              | —          | —                                  |

---

> **Fuera de esta tarea (siguiente en el roadmap):** `batch_inference` (tarea 4,
> orquestación de lotes sobre `run_inference`) y los follow-ups de tracking (overlay
> por `obj_id`, clases trackeables configurables, tuning de ByteTrack).
