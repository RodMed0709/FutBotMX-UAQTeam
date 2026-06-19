# Tasks — Render de mp4 opcional vía flag (`optional_render`)

- **Tarea atómica:** `optional_render`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** la **Parte B de las pruebas (Fase D) corre EXCLUSIVAMENTE
> en el pod (GPU)** — `run_pipeline`/`track_video` invocan SAM3. La **Parte A**
> (introspección de firma) y el lint estático (`ruff`/`black`) corren en **cualquier
> entorno**, sin GPU.

---

## Fase A — `run_pipeline` (seg-only)

- [x] **T1 — Añadir `render_video` a `run_pipeline`**
  - Firma `+ render_video=True` (default de un solo video). Mover la composición del
    overlay y el `composed.append(...)` dentro de `if render_video:`; mantener el
    `frame_record(...)` **fuera** del `if` (el entregable se construye siempre).
  - **Verificación:** revisión de código + lint; funcional en la Parte B (pod).
  - **Plan:** §3.2, §3.3. **Spec:** AC-1, AC-2, AC-3, AC-6.

- [x] **T2 — Escritura condicional del mp4 y retorno estable (seg-only)**
  - `write_video(...)` solo si `render_video`; con OFF, `mp4_out = None` (sin
    `np.stack` sobre lista vacía). Seguir derivando `mp4_path` para nombrar el JSON.
    Retorno `{"json": json_path, "video": mp4_out}` con `"video"` siempre presente.
  - **Verificación:** revisión de código; con OFF no se llama `write_video` ni se crea
    mp4; el JSON se escribe igual.
  - **Plan:** §3.3, §3.5. **Spec:** AC-4, AC-5, AC-7.

---

## Fase B — `track_video` (tracking)

- [x] **T3 — Añadir `render_video` a `track_video` con `nullcontext`**
  - Firma `+ render_video=True`. Abrir el escritor condicionalmente:
    `open_video_writer(...)` si `render_video` else `contextlib.nullcontext(None)`,
    bajo el **mismo** `with ... as append:`. La composición del overlay y
    `append(composed)` quedan dentro de `if render_video:`; el resto del bucle
    (ByteTrack, `obj_id`, `tracks`, `frame_record`) **sin cambios**.
  - **Verificación:** revisión de código; con OFF no se abre `imageio` ni se crea mp4;
    `append` nunca se invoca cuando es `None`.
  - **Plan:** §3.4. **Spec:** AC-1, AC-3, AC-6.

- [x] **T4 — JSON siempre y retorno estable (tracking)**
  - Cabecera + JSON unificado (`frames` + `tracks`) se escriben siempre. Retorno
    `{"json": json_path, "video": (mp4_path if render_video else None), "index":
    tracks}`. Seguir derivando `mp4_path` para nombrar el JSON.
  - **Verificación:** revisión de código; con OFF `"video"` is `None` y el JSON
    (con `frames` y `tracks`) existe e idéntico en forma al de render ON.
  - **Plan:** §3.4, §3.5. **Spec:** AC-4, AC-5, AC-7.

- [x] **T5 — Documentar el flag y el retorno `Path | None`**
  - Docstrings de ambos orquestadores: qué hace `render_video`, su default, y que
    `"video"` vale `Path` (render ON) o `None` (render OFF).
  - **Verificación:** presente en los docstrings de `run_pipeline` y `track_video`.
  - **Plan:** §3.2, §7. **Spec:** AC-7.

---

## Fase C — Anti-alcance (verificación de no-regresión)

- [x] **T6 — Confirmar que no cambia nada colateral**
  - Sin tocar `inference_schema.py`, `overlay.py`, `video_writer.py`,
    `frame_extraction.py`, `segmentation`, ByteTrack, el muestreo ni la config
    (`render_video` es parámetro, no clave de config; `requirements.txt` sin cambios).
  - **Verificación:** `git diff` limitado a `pipeline.py`, `tracking.py` y el nuevo
    test; `SCHEMA_VERSION` sin cambios.
  - **Plan:** §3.6, §4. **Spec:** AC-9.

---

## Fase D — Test

- [x] **T7 — Crear `testing/test_optional_render.py`**
  - **Parte A (local, sin GPU):** `inspect.signature` de `run_pipeline` y
    `track_video` incluye `render_video` con default `True`.
  - **Parte B (GPU/pod):** seg-only ON/OFF y tracking ON/OFF sobre un clip corto:
    JSON siempre presente; mp4 existe solo con ON; `"video"` = `Path`/`None` según
    flag; caso `render_video=False, include_masks=True` → JSON con `rle` y **sin** mp4.
  - **Verificación:** el script existe; la Parte A es ejecutable **localmente**.
  - **Plan:** §5.1, §5.2. **Spec:** AC-8, AC-10.

---

## Fase E — Ejecución y calidad

- [x] **T8 — Ejecutar la Parte A en local**
  - Correr la Parte A de `test_optional_render.py` **sin GPU**.
  - **Verificación:** la introspección de firma pasa en local.
  - **Plan:** §5.1. **Spec:** AC-10.

- [x] **T9 — Ejecutar la Parte B en el pod (GPU)**
  - Correr la Parte B **en el pod** (modelo SAM3 + GPU). **No se corre en local.**
  - **Verificación:** ambos modos con ON/OFF; JSON siempre, mp4 condicional, forma del
    retorno correcta; caso OFF+masks produce `rle` sin video.
  - **Plan:** §5.2. **Spec:** AC-1, AC-4, AC-5, AC-6, AC-7, AC-8.

- [x] **T10 — Calidad e importabilidad**
  - `ruff check .` y `black .` sin hallazgos; `from src.core.pipeline import
    run_pipeline` y `from src.core.tracking import track_video` OK.
  - **Verificación:** lint limpio; imports correctos.
  - **Plan:** §5.3. **Spec:** AC-9.

- [x] **T11 — Commit (requiere confirmación)**
  - Commitear `pipeline.py`, `tracking.py` y el test. **El agente NO commitea por
    iniciativa propia:** pregunta y espera confirmación (constitución §11).
    Conventional Commits en inglés, scope `optional_render`.
  - **Verificación:** tras tu confirmación, el commit existe.
  - **Plan:** —. **Spec:** —

---

## Trazabilidad resumida

| Tarea                              | Plan       | Spec (AC)                          |
| ---------------------------------- | ---------- | ---------------------------------- |
| T1 `render_video` en `run_pipeline`| §3.2, §3.3 | AC-1, AC-2, AC-3, AC-6             |
| T2 mp4 condicional + retorno (seg) | §3.3, §3.5 | AC-4, AC-5, AC-7                   |
| T3 `render_video` en `track_video` | §3.4       | AC-1, AC-3, AC-6                   |
| T4 JSON siempre + retorno (track)  | §3.4, §3.5 | AC-4, AC-5, AC-7                   |
| T5 documentar flag/retorno         | §3.2, §7   | AC-7                               |
| T6 anti-alcance (no-regresión)     | §3.6, §4   | AC-9                               |
| T7 crear test (A + B)              | §5.1, §5.2 | AC-8, AC-10                        |
| T8 ejecutar Parte A (local)        | §5.1       | AC-10                              |
| T9 ejecutar Parte B (pod)          | §5.2       | AC-1, AC-4, AC-5, AC-6, AC-7, AC-8 |
| T10 calidad/import                 | §5.3       | AC-9                               |
| T11 commit (confirmación)          | —          | —                                  |
