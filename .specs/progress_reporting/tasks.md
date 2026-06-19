# Tasks — Barra de progreso en inferencia (`progress_reporting`)

- **Tarea atómica:** `progress_reporting`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]` al
> completar.
>
> **Nota de ejecución:** firmas y `get_frame_count` se verifican **sin GPU** (este
> último necesita un `.MOV` real). La **barra visual** solo se confirma corriendo
> inferencia en el **pod (GPU)**.

---

## Fase A — Helper de conteo (`frame_extraction.py`)

- [x] **T1 — `get_frame_count(video_path)`**
  - Espejar `get_video_fps`: `_resolve_video_path` + `len(decord.VideoReader(...))`.
    Acepta ruta relativa o absoluta. Docstring con `Raises` análogas.
  - **Verificación:** sobre un `.MOV` real devuelve `int > 0` y coincide con
    `len(get_frame_indices(..., all_frames=True))`. (Sin GPU.)
  - *Origen:* plan §3.2 / criterio #4.

## Fase B — Barra en segmentación (`pipeline.py`)

- [x] **T2 — `run_pipeline` gana `progress` y muestra barra**
  - Firma `+ progress: bool = True` (al final). Import perezoso `from tqdm.auto import
    tqdm`. Envolver `enumerate(frames)` en `tqdm(total=total, desc=f"seg {stem}",
    unit="frame", leave=False, disable=not progress)`. **Eliminar**
    `print(f"  frame {i + 1}/{total}")`. Documentar el parámetro.
  - **Verificación:** introspección de firma (`progress` default `True`); lectura de
    diff confirma que el print desaparece y aparece la barra. (Sin GPU.)
  - *Origen:* plan §3.3 / criterios #1, #2, #5, #7.

## Fase C — Barra en tracking (`tracking.py`)

- [x] **T3 — `track_video` gana `progress` y muestra barra**
  - Firma `+ progress: bool = True`. Import perezoso de `tqdm.auto` y de
    `get_frame_count` (desde `frame_extraction`). Antes del `with writer_cm`, derivar
    `n_total = get_frame_count(video_path)` y `min(max_frames, n_total)` si hay tope.
    Envolver `iter_frames(...)` en `tqdm(total=n_total, desc=f"track {stem}",
    unit="frame", leave=False, disable=not progress)`. Documentar el parámetro.
  - **Verificación:** introspección de firma; lectura de diff. (Sin GPU.)
  - *Origen:* plan §3.4 / criterios #1, #3, #5, #7.

## Fase D — Hilvanado del flag

- [x] **T4 — `run_inference` propaga `progress`**
  - Firma `+ progress: bool = True`; reenviar a `run_pipeline` (segmentación) y a
    `track_video` (tracking). Documentar.
  - **Verificación:** introspección de firma; lectura de diff. (Sin GPU.)
  - *Origen:* plan §3.5 / criterio #6.

- [x] **T5 — `run_batch` propaga `progress`**
  - Firma `+ progress: bool = True`; reenviar a `run_inference`. Conservar el print
    `[i/n] ruta -> status`. Documentar.
  - **Verificación:** introspección de firma; el resumen por video sigue imprimiéndose.
    (Sin GPU.)
  - *Origen:* plan §3.5 / criterio #6.

## Fase E — Anti-alcance (verificación de no-regresión)

- [x] **T6 — Confirmar que NO se tocó lo fuera de alcance**
  - Sin cambios en el esquema JSON, rutas (`inference_paths`/`run_label`), muestreo,
    lógica de detección/tracking, ni las firmas de `iter_frames`/`extract_frames`.
  - **Verificación:** `git diff --name-only` lista solo `frame_extraction.py`,
    `pipeline.py`, `tracking.py`, `inference.py`, `batch.py`,
    `testing/test_frame_extraction.py`, `CLAUDE.md` (y los `.specs/` de esta tarea).
  - *Origen:* plan §3.6 / criterio #8.

## Fase F — Test (`testing/test_frame_extraction.py`)

- [x] **T7 — Smoke: `get_frame_count` + firmas (sin GPU)**
  - `get_frame_count` sobre un `.MOV` real (`int > 0`, coincide con el total del
    módulo); introspección de `progress` (default `True`) en las 4 funciones.
  - **Verificación:** corre y pasa sin GPU (si hay videos locales; si no, en el pod).
  - *Origen:* plan §5.1 / criterios #4, #9.

## Fase G — Documentación y calidad

- [x] **T8 — Actualizar `CLAUDE.md`**
  - Mencionar las barras de progreso, el flag `progress` (default `True`,
    silenciable) y el helper `get_frame_count`.
  - **Verificación:** la sección lo refleja.
  - *Origen:* plan §3.1 / criterio #10.

- [x] **T9 — Lint y formato**
  - `ruff check` y `ruff format --check` limpios en los archivos tocados.
  - **Verificación:** sin hallazgos. (Sin GPU.)
  - *Origen:* plan §5.3.

## Fase H — Verificación visual (opcional, pod/GPU)

- [ ] **T10 — Barra visible en el pod**
  - Correr segmentación y tracking sobre un clip y **ver** la barra (ETA + frames/s);
    con `progress=False` no aparece nada.
  - **Verificación:** confirmación visual en el pod.
  - *Origen:* plan §5.2 / criterios #2, #3, #5.

---

## Trazabilidad resumida

| Criterio (spec §5) | Tareas |
|---|---|
| 1. `progress` en orquestadores | T2, T3 |
| 2. Barra en segmentación, sin print | T2, T10 |
| 3. Barra en tracking con total | T3, T10 |
| 4. `get_frame_count` | T1, T7 |
| 5. `progress=False` desactiva | T2, T3, T10 |
| 6. `progress` propagado | T4, T5 |
| 7. `tqdm.auto`/perezoso/`desc`/`leave=False` | T2, T3 |
| 8. No cambia esquema/rutas/lógica | T6 |
| 9. Smoke sin GPU | T7 |
| 10. `CLAUDE.md` | T8 |

---

## Trabajo futuro (fuera de esta tarea)

- **Streaming de `pipeline.py`** (cura de RAM de `all_frames`): tarea **condicional**,
  solo si la segmentación de video completo se vuelve un entregable. Puede no hacerse.
