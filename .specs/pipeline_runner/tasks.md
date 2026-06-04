# Tasks — Primer pipeline ejecutable (`pipeline_runner`)

- **Tarea atómica:** `pipeline_runner`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** el pipeline usa modelo/GPU; el agente **no lo ejecuta**.
> Solo verifica lo ligero (lint, importabilidad, y que un `mode` inválido lanza
> `NotImplementedError` sin cargar el modelo). La corrida real
> (`testing/test_pipeline.py`) la hace el usuario en **RunPod**.

---

## Fase A — Módulo y función

- [x] **T1 — Crear `src/core/pipeline.py` con `_load_pipeline_config`**
  - Helper que lee `CONFIG_FILENAME` del `.env` con `strip()` → `get_abs_path` →
    `json.load` y devuelve `(classes, outputs_dir, output_fps)` en una sola lectura.
  - **Verificación:** devuelve las tres cosas de la config; ausencias relevantes
    lanzan `ValueError`/`KeyError`/`FileNotFoundError`.
  - **Plan:** §3.1, §3.3. **Spec:** AC-7.

- [x] **T2 — `run_pipeline` (orquestación + rutas + modos)**
  - Firma `run_pipeline(video_path, output_path=None, all_frames=False,
mode="per_frame") -> dict[str, Path]`.
  - Validar `mode` (≠ `per_frame` → `NotImplementedError` **antes** de cargar el
    modelo); componer rutas (`outputs/<stem>_annotated.mp4` + `_detections.json`,
    override por `output_path`; `mkdir` del dir del JSON).
  - Cargar `load_sam3()` **una vez**; `extract_frames(video, all_frames=...)`; por
    frame: `detect_classes_in_frame(frame, classes=classes, bundle=bundle)` →
    `overlay_detections(frame, dets, classes=classes)`; acumular compuestos y
    registros; imprimir progreso `frame i/N`.
  - `write_video(np.stack(composed), mp4_path, fps=fps)`.
  - **Verificación:** con un video válido produce mp4 + dict de rutas; `mode`
    inválido → `NotImplementedError`; el modelo se carga una sola vez.
  - **Plan:** §3.2, §3.4, §3.5, §3.7, §3.8. **Spec:** AC-2, AC-4, AC-5, AC-6, AC-7, AC-8.

- [x] **T3 — Escritura del JSON de detecciones**
  - Construir el payload (`video`, `mode`, `all_frames`, `fps`, `num_frames`,
    `classes`, `frames` con `{obj_id, score}` por clase) y escribirlo con
    `json.dumps(..., indent=2)`.
  - **Verificación:** se genera un JSON parseable con metadatos y `frames`
    coherentes (sin máscaras).
  - **Plan:** §3.6. **Spec:** AC-3.

---

## Fase B — Exportación

- [x] **T4 — Exportar en `src/core/__init__.py`**
  - Añadir `from src.core.pipeline import run_pipeline` y sumarlo a `__all__`.
  - **Verificación:** `from src.core import run_pipeline` funciona; `ruff check .`
    y `black .` pasan sobre el código nuevo.
  - **Plan:** §3.1. **Spec:** AC-1.

---

## Fase C — Script de prueba

- [x] **T5 — Crear `testing/test_pipeline.py`**
  - Localiza un `.MOV` real (rglob sobre `dataset_dir`), corre `run_pipeline(video)`
    (cuota por defecto), verifica que se generan mp4 + JSON y que el JSON es
    parseable.
  - **No ejecutar aquí**; solo crearlo (lo corre el usuario en RunPod).
  - **Verificación:** el archivo existe, pasa lint y es importable/parseable; la
    corrida real queda para la Fase D.
  - **Plan:** §5.2. **Spec:** AC-9.

---

## Fase D — Validación manual (a cargo del usuario, en RunPod/GPU)

- [x] **T6 — Ejecutar el pipeline en RunPod**
  - Correr:
    ```bash
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_pipeline.py
    ```
  - Confirmar el mp4 anotado y el JSON bajo `outputs/`, y revisar visualmente el
    video.
  - **Verificación:** salida coherente; criterios AC-1 a AC-9 satisfechos.
  - **Responsable:** usuario (RunPod).

---

## Trazabilidad resumida

| Tarea                         | Plan                         | Spec (AC)                          |
| ----------------------------- | ---------------------------- | ---------------------------------- |
| T1 `_load_pipeline_config`    | §3.1, §3.3                   | AC-7                               |
| T2 `run_pipeline`             | §3.2, §3.4, §3.5, §3.7, §3.8 | AC-2, AC-4, AC-5, AC-6, AC-7, AC-8 |
| T3 JSON de detecciones        | §3.6                         | AC-3                               |
| T4 exportación                | §3.1                         | AC-1                               |
| T5 script de prueba (crear)   | §5.2                         | AC-9                               |
| T6 validación manual (RunPod) | §5                           | AC-9                               |
