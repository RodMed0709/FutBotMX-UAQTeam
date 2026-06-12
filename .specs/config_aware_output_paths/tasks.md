# Tasks — Salidas con namespace por config (`config_aware_output_paths`)

- **Tarea atómica:** `config_aware_output_paths`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]` al
> completar.
>
> **Nota de ejecución:** toda la derivación de rutas es **lógica pura** y se verifica
> **sin GPU**. No hace falta SAM3 para esta tarea; la corrida real en el pod es
> opcional.

---

## Fase A — `inference_paths` con `namespace` (`inference_schema.py`)

- [x] **T1 — `inference_paths` acepta `namespace`**
  - Agregar `namespace: str | None = None`; cuando tiene valor, insertarlo **antes**
    del `<stem>`: `.../inference/<namespace>/<stem>/<stem>.{json,mp4}`. `None`/vacío ⇒
    ruta actual. Actualizar el docstring.
  - **Verificación:** llamadas con y sin `namespace` devuelven las rutas esperadas;
    dos namespaces distintos sobre el mismo stem dan rutas distintas. (Sin GPU.)
  - *Origen:* plan §3.2 / criterios #1, #4.

## Fase B — Hilvanado de `run_label` en los orquestadores

- [x] **T2 — `run_pipeline` propaga `run_label`**
  - Firma `+ run_label: str | None = None` (al final). En el bloque de ruta por
    defecto, `inference_paths(stem, outputs_dir, namespace=run_label)`. **No** tocar
    el camino `output_path is not None`.
  - **Verificación:** introspección de firma; lectura de diff confirma el paso del
    namespace y la prioridad intacta de `output_path`. (Sin GPU.)
  - *Origen:* plan §3.3 / criterios #2, #5.

- [x] **T3 — `track_video` propaga `run_label`**
  - Igual que T2 en `tracking.py` (firma + `inference_paths(..., namespace=run_label)`
    en el bloque de ruta por defecto).
  - **Verificación:** introspección de firma; lectura de diff. (Sin GPU.)
  - *Origen:* plan §3.3 / criterios #2, #5.

- [x] **T4 — `run_inference` propaga `run_label`**
  - Firma `+ run_label`; reenviarlo a `run_pipeline` (rama segmentación) y a
    `track_video` (rama tracking). Documentar el parámetro.
  - **Verificación:** lectura de diff; ambas ramas reciben `run_label`. (Sin GPU.)
  - *Origen:* plan §3.3 / criterio #2.

## Fase C — `skip-done` por config (`batch.py`)

- [x] **T5 — `run_batch` usa `run_label` en skip-done y lo reenvía**
  - Firma `+ run_label`. En el `skip-done` (hoy `batch.py:234`):
    `inference_paths(Path(ruta).stem, outputs_dir, namespace=run_label)`. En la
    llamada a `run_inference`, pasar `run_label=run_label`. Documentar el parámetro.
  - **Verificación:** un JSON colocado bajo `outputs/inference/<run_label>/<stem>/`
    hace que ese video se reporte `skipped` con `run_label` puesto, y **no** con
    `run_label=None` u otro label. (Sin GPU.)
  - *Origen:* plan §3.3 / criterio #6.

## Fase D — Anti-alcance (verificación de no-regresión)

- [x] **T6 — Confirmar que NO se tocó lo fuera de alcance**
  - Sin cambios en el esquema JSON (`build_header`/`frame_record`/
    `write_inference_json`), en la semántica de `output_path`, en detector/tracker,
    configs, muestreo, `aggregate_config` ni los notebooks.
  - **Verificación:** `git diff --name-only` lista solo `inference_schema.py`,
    `pipeline.py`, `tracking.py`, `inference.py`, `batch.py`,
    `testing/test_batch_inference.py`, `CLAUDE.md` (y los `.specs/` de esta tarea).
  - *Origen:* plan §3.4 / criterio #7.

## Fase E — Test (`testing/test_batch_inference.py`)

- [x] **T7 — Smoke: derivación de rutas (sin GPU)**
  - Casos de §5.1: con/sin `namespace` y no-colisión entre dos namespaces.
  - **Verificación:** corre y pasa sin GPU.
  - *Origen:* plan §5.1 / criterios #1, #4, #8.

- [x] **T8 — Smoke: `skip-done` por config (sin GPU)**
  - Caso de §5.2: fabricar el JSON namespaced y comprobar el `skipped` por config sin
    invocar SAM3 (reutilizar el patrón existente de JSON falsos).
  - **Verificación:** corre y pasa sin GPU.
  - *Origen:* plan §5.2 / criterios #6, #8.

## Fase F — Documentación y calidad

- [x] **T9 — Actualizar `CLAUDE.md`**
  - En *output placement* / descripción de `batch`, mencionar el namespace opcional
    por config (`outputs/inference/<run_label>/<stem>/…`) y el `skip-done` por config.
  - **Verificación:** la sección lo refleja.
  - *Origen:* plan §3.1 / criterio #9.

- [x] **T10 — Lint y formato**
  - `ruff check .` y `black .` limpios.
  - **Verificación:** ambos sin hallazgos. (Sin GPU.)
  - *Origen:* plan §5.3.

---

## Trazabilidad resumida

| Criterio (spec §5) | Tareas |
|---|---|
| 1. `inference_paths` con/sin `namespace` | T1, T7 |
| 2. `run_label` propagado por la cadena | T2, T3, T4, T5 |
| 3. Retrocompatibilidad `run_label=None` | T1, T2, T3, T7 |
| 4. JSON+mp4 bajo `<run_label>/<stem>/` | T1, T2, T3, T7 |
| 5. `output_path` tiene prioridad | T2, T3 |
| 6. `skip-done` por config | T5, T8 |
| 7. No cambia el esquema JSON | T6 |
| 8. Smoke sin GPU | T7, T8 |
| 9. `CLAUDE.md` actualizado | T9 |

---

## Trabajo futuro (fuera de esta tarea)

- **Notebooks de benchmark por fases:** Fase 1 (eficiencia de detectores, sin
  tracker) y Fase 2 (trackers en **2×2**), que pasarán `run_label` por config y, al no
  colisionar, podrán reanudar y soltar el baile de "una config a la vez".
- **Saneo de `run_label`** como segmento de ruta, si en el futuro llegan caracteres
  problemáticos.
