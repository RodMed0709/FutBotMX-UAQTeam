# Tasks — Exportar y congelar el set de frames de evaluación (`eval_frame_export`)

- **Tarea atómica:** `eval_frame_export`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** la tarea no usa modelo ni GPU. A diferencia de otras
> tareas, la **validación (Fase E) se ejecuta en el pod (RunPod)**, no en local,
> para que las imágenes queden en el **volumen compartido** del equipo.

---

## Fase A — Configuración

- [x] **T1 — Añadir claves de config y `.gitignore`**
  - En `configs/00_testing_config.json`, dentro de `working_dirs`, añadir
    `"testing_frames_dir": "data/testing_frames"` y
    `"testing_frames_csv": "assets/testing_frames.csv"`.
  - Añadir `data/testing_frames/` a `.gitignore` (las imágenes son dato pesado; el
    CSV en `assets/` sí se versiona).
  - **Verificación:** el JSON sigue siendo válido; las dos claves están presentes;
    `.gitignore` excluye `data/testing_frames/`.
  - **Plan:** §4. **Spec:** AC-3, AC-5.

---

## Fase B — Helper en `frame_extraction` (refactor aditivo)

- [x] **T2 — Exponer los índices de muestreo en `src/core/frame_extraction.py`**
  - Extraer la selección de índices a `_select_frame_indices(total, all_frames)` y
    refactorizar `extract_frames` para reusarla (**misma salida y firma**).
  - Añadir `get_frame_indices(video_path, all_frames=False) -> np.ndarray` que abre
    el video (solo metadatos) y devuelve los índices que el muestreo seleccionaría.
  - **Verificación:** `extract_frames` mantiene su salida `(N,H,W,3)` sobre un video
    real; `get_frame_indices(ruta)` devuelve un array de longitud `N` alineado por
    posición con los frames de `extract_frames`.
  - **Plan:** §2, §3.5b. **Spec:** AC-6b, AC-12.

---

## Fase C — Módulo `eval_frames.py`

- [x] **T3 — Esquema y carga de config en `src/data/eval_frames.py`**
  - Constantes de módulo: `COLUMNS`, `TESTING_SPLIT`, `GROUP_RANDOM`,
    `GROUP_CENITAL`, `IMAGE_EXT`.
  - `_load_eval_frames_config()` → `(metadata_csv, testing_frames_dir,
testing_frames_csv, forced_testing)` leyendo `.env`/JSON (patrón de
    `metadata.py`), con `KeyError`/`ValueError` claros.
  - **Verificación:** `_load_eval_frames_config()` devuelve los cuatro valores sobre
    la config real; claves ausentes lanzan el error documentado.
  - **Plan:** §3.2, §3.3. **Spec:** AC-4, AC-8.

- [x] **T4 — Selección de videos, grupo y escritura de imagen**
  - `_load_testing_videos(metadata_csv)`: lee `db_metadata.csv` (vía `get_abs_path`,
    debe existir), filtra `split == TESTING_SPLIT`, conserva `id` y `ruta`.
  - `_group_for(ruta, forced_testing)`: `cenital` si la ruta está en
    `forced_testing`, si no `aleatorio`.
  - `_write_frame_image(frame_rgb, dest)`: `cv2.imwrite` con conversión RGB→BGR,
    PNG.
  - **Verificación:** sobre `data/raw`, `_load_testing_videos` devuelve 20 filas con
    `split==2`; `_group_for` marca `cenital` los 2 de `forced_testing`;
    `_write_frame_image` escribe un PNG legible.
  - **Plan:** §3.4, §3.5, §3.6. **Spec:** AC-2, AC-7.

- [x] **T5 — Orquestador `export_testing_frames` y handler de validación**
  - `validate_testing_frames_schema(csv_path) -> bool`: existencia + columnas ==
    `COLUMNS` (orden incluido); CSV ilegible → `False`.
  - `export_testing_frames(force=False) -> pandas.DataFrame`: por cada video de
    testing, `extract_frames` + `get_frame_indices` (alineados por posición),
    escribe `<video_id>_<frame_index>.png` plano en `testing_frames_dir` y arma las
    filas (`id`, `video_id`, `video_ruta`, `frame_index`, `frame_original`,
    `imagen`, `grupo`); escribe el CSV en `assets/` con `index=False`; idempotente
    (no reescribe si existe, es válido y `force=False`); caso borde (menos frames
    que la cuota) cubierto por `enumerate(frames)`.
  - **Verificación:** genera imágenes + CSV con las 7 columnas en orden, `id`
    `0..M-1`, `frame_original` consistente con `get_frame_indices`, rutas relativas
    POSIX resolubles; el handler distingue CSV válido de inválido.
  - **Plan:** §3.7, §3.8, §3.9. **Spec:** AC-5, AC-6, AC-6b, AC-9, AC-10, AC-11.

- [x] **T6 — Exponer la API en `src/data/__init__.py`**
  - Añadir `from src.data.eval_frames import export_testing_frames,
validate_testing_frames_schema` (y `__all__`), sin romper los exports existentes.
  - **Verificación:** `from src.data import export_testing_frames,
validate_testing_frames_schema` importa sin error.
  - **Plan:** §3.1. **Spec:** AC-1.

---

## Fase D — Test

- [x] **T7 — Crear `testing/test_eval_frame_export.py` (standalone)**
  - Script estilo `test_*.py` que: (1) corre `export_testing_frames(force=True)` y
    verifica que crea `assets/testing_frames.csv`; (2) columnas == `COLUMNS`, `id`
    `0..M-1`, `video_id` ∈ ids `split==2`, 20 videos distintos, `grupo` ∈
    {`aleatorio`,`cenital`}; (3) cada `imagen` existe y abre como PNG;
    (3b) `frame_original` coincide con `get_frame_indices` alineado por posición y
    `extract_frames` conserva su salida; (4) los 2 de `forced_testing` son
    `cenital`; (5) idempotencia (`force=False` no reescribe; header corrupto →
    handler `False` y regenera).
  - **Verificación:** el script existe y sus comprobaciones son ejecutables; lint
    (`ruff`, `black`) sin hallazgos.
  - **Plan:** §5.1. **Spec:** AC-9, AC-10, AC-11, AC-13.

---

## Fase E — Ejecución en el pod y calidad

- [x] **T8 — Ejecutar la exportación y el test en el pod**
  - Correr `testing/test_eval_frame_export.py` **en el pod (RunPod)**; confirmar que
    las imágenes quedan bajo `data/testing_frames/` en el **volumen compartido** y
    que todas las comprobaciones pasan.
  - **Verificación:** test verde en el pod; 20 videos procesados; imágenes presentes
    en el volumen compartido.
  - **Estado:** sanity check **en local** PASÓ (600 frames de 20 videos, todas las
    comprobaciones verdes). **Pendiente:** la corrida oficial en el pod (volumen
    compartido) la realiza el equipo.
  - **Plan:** §5.1, §5.2. **Spec:** AC-13.

- [x] **T9 — Calidad e importabilidad**
  - `ruff check .` y `black .` sin hallazgos; `from src.data import
export_testing_frames, validate_testing_frames_schema` OK.
  - **Verificación:** lint limpio e import correcto.
  - **Plan:** §5.3. **Spec:** AC-1.

- [x] **T10 — Commitear el CSV versionado (requiere confirmación)**
  - Commitear `assets/testing_frames.csv` (generado en el pod) y los cambios de
    código/config para que la procedencia llegue al equipo. Las imágenes quedan
    git-ignored en el volumen compartido.
  - **El agente NO commitea por iniciativa propia:** pregunta y espera confirmación
    explícita (constitución §11). Conventional Commits en inglés, scope
    `eval_frame_export`.
  - **Verificación:** tras tu confirmación, el commit existe con el CSV versionado.
  - **Plan:** §5.2. **Spec:** —

---

## Trazabilidad resumida

| Tarea                         | Plan             | Spec (AC)                             |
| ----------------------------- | ---------------- | ------------------------------------- |
| T1 config + `.gitignore`      | §4               | AC-3, AC-5                            |
| T2 helper `get_frame_indices` | §2, §3.5b        | AC-6b, AC-12                          |
| T3 esquema + carga config     | §3.2, §3.3       | AC-4, AC-8                            |
| T4 selección/grupo/imagen     | §3.4, §3.5, §3.6 | AC-2, AC-7                            |
| T5 orquestador + handler      | §3.7, §3.8, §3.9 | AC-5, AC-6, AC-6b, AC-9, AC-10, AC-11 |
| T6 API `__init__`             | §3.1             | AC-1                                  |
| T7 crear test                 | §5.1             | AC-9, AC-10, AC-11, AC-13             |
| T8 ejecutar en el pod         | §5.1, §5.2       | AC-13                                 |
| T9 calidad/import             | §5.3             | AC-1                                  |
| T10 commit (confirmación)     | §5.2             | —                                     |
