# Tasks — Esquema común del entregable de inferencia (`inference_schema`)

- **Tarea atómica:** `inference_schema`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** la **Parte B de las pruebas (Fase E) corre EXCLUSIVAMENTE
> en el pod (GPU)** — requiere el modelo SAM3. La **Parte A** (helpers del módulo con
> datos sintéticos) y el lint estático (`ruff`/`black`) corren en **cualquier
> entorno**, sin GPU.

---

## Fase A — Dependencia

- [x] **T1 — Añadir `pycocotools` a `requirements.txt`**
  - Declarar `pycocotools` (codificación COCO-RLE); instalarla en el entorno.
    Documentar que es de **import perezoso** (solo se usa con `include_masks=True`).
  - **Verificación:** aparece en `requirements.txt` e importa en el entorno.
  - **Plan:** §2, §4. **Spec:** AC-10.

---

## Fase B — Módulo `src/core/inference_schema.py` (nuevo)

- [x] **T2 — Geometría y RLE**
  - `mask_to_bbox_centroid(mask)` (boundingRect → `[x,y,w,h]` + `[cx,cy]`; `None` si
    vacía); `encode_rle(mask)` / `decode_rle(rle)` con `pycocotools` (import
    perezoso, `counts` ↔ str ascii, ida-vuelta sin pérdida).
  - **Verificación:** sobre máscaras sintéticas, `decode_rle(encode_rle(m)) == m`;
    `mask_to_bbox_centroid` da la caja/centroide esperados y `None` ante vacía.
  - **Plan:** §3.4, §3.5. **Spec:** AC-4, AC-12.

- [x] **T3 — Builders del registro y de la cabecera**
  - `detection_record(det, include_masks)` (→ `{obj_id, bbox, centroid, score,
rle?}`; `None` si máscara vacía); `frame_record(frame_index, dets_by_class,
include_masks)`; `build_header(...)` con `SCHEMA_VERSION="1.0"`, `model_version`
    (`_model_version`: `sam3_dir` + versiones vía `importlib.metadata`, best-effort),
    `timestamp` UTC ISO-8601, `fps`, `resolution`, `num_frames`, `classes`,
    `include_masks`, `config` (snapshot completo).
  - **Verificación:** un registro con `include_masks=True` trae `rle`; con `False`
    no; la cabecera contiene todas las claves y `model_version` no falla si un
    paquete no resuelve.
  - **Plan:** §3.2, §3.3. **Spec:** AC-1, AC-3, AC-5.

- [x] **T4 — Rutas y escritura**
  - `inference_paths(video_stem, outputs_dir)` → `(json_path, mp4_path)` bajo
    `outputs/inference/<stem>/`; `write_inference_json(header, frames, json_path,
tracks=None)` (compone `{**header, frames, tracks?}`, crea carpeta padre, escribe).
  - **Verificación:** las rutas siguen la convención; el JSON se escribe y recarga
    como dict válido con `frames` (y `tracks` si se pasa).
  - **Plan:** §3.8. **Spec:** AC-6, AC-7.

---

## Fase C — Integración en los orquestadores

- [x] **T5 — `run_pipeline` (seg-only) emite el esquema común**
  - Firma `+ include_masks=False`; usar `get_frame_indices` para el **`frame_index`
    real** (mapear posición `i` → índice fuente); construir registros con
    `frame_record`; `resolution` desde `frames.shape[1:3]`; cabecera + escritura vía
    el módulo; salidas en `outputs/inference/<stem>/`; embeber el `config` completo;
    retorno `{"json", "video"}`.
  - **Verificación (local, sin SAM3 no aplica):** revisión de código + lint; la
    verificación funcional va en la Parte B (pod).
  - **Plan:** §3.6, §3.8. **Spec:** AC-1, AC-2, AC-6, AC-7.

- [x] **T6 — `track_video` (tracking) funde `frames` + `tracks` en un JSON**
  - Firma `+ include_masks=False`; en el loop, tras el overlay y **antes de descartar
    máscaras**, acumular `frame_record(frame_index, per_frame, include_masks)`
    (incluye warm-up `obj_id=-1`); mantener el índice `tracks`; serializar `tracks`
    como sección del **mismo** JSON (eliminar `_write_tracks_json` y el archivo
    aparte); `resolution` del primer frame; retirar `_mask_to_xyxy` local en favor de
    `mask_to_bbox_centroid` (derivando `xyxy` solo donde ByteTrack lo exige); salidas
    en `outputs/inference/<stem>/`; retorno `{"json", "video", "index"}`.
  - **Verificación:** revisión de código + lint; funcional en la Parte B (pod).
  - **Plan:** §3.7, §3.8. **Spec:** AC-1, AC-6, AC-8.

- [x] **T7 — Documentar la semántica de `obj_id`**
  - Docstrings que dejen explícito: `obj_id` **inestable** en per-frame, **estable**
    en tracking (misma clave, distinta semántica por modo).
  - **Verificación:** presente en los docstrings de ambos orquestadores / del módulo.
  - **Plan:** §3.7. **Spec:** AC-8.

---

## Fase D — Test

- [x] **T8 — Crear `testing/test_inference_schema.py`**
  - **Parte A (local, sin GPU):** round-trip RLE; geometría (`mask_to_bbox_centroid`,
    incl. vacía → `None`); ensamblado + `write_inference_json` + relectura
    (`schema_version`, `config`, `resolution`, `fps`; `rle` presente/ausente según
    `include_masks`); **reconstrucción sin modelo** (decodificar `rle` + pintar con
    `overlay_detections` sobre frame dummy).
  - **Parte B (GPU/pod):** `run_pipeline(..., include_masks=True)` →
    `outputs/inference/<stem>/<stem>.json` (`mode="segmentation"`, geometría + `rle`,
    `frame_index` = `get_frame_indices`); `track_video(..., max_frames=pequeño,
include_masks=True)` → **un único** JSON con `frames` **y** `tracks` (sin
    `_tracks.json`); caso `include_masks=False` → sin `rle` y sin importar
    `pycocotools`.
  - **Verificación:** el script existe; la Parte A es ejecutable **localmente**.
  - **Plan:** §5.1, §5.2. **Spec:** AC-9, AC-11.

---

## Fase E — Ejecución y calidad

- [x] **T9 — Ejecutar la Parte A en local**
  - Correr la Parte A de `test_inference_schema.py` **sin GPU**.
  - **Verificación:** round-trip RLE, geometría, serialización y reconstrucción
    sin modelo pasan en local.
  - **Plan:** §5.1. **Spec:** AC-4, AC-9.

- [x] **T10 — Ejecutar la Parte B en el pod (GPU)**
  - Correr la Parte B **en el pod** (modelo SAM3 + GPU). **No se corre en local.**
  - **Verificación:** ambos orquestadores producen el JSON unificado en la carpeta
    por video; tracking lleva `frames` + `tracks` en un solo archivo; `include_masks`
    se comporta en ON/OFF.
  - **Plan:** §5.2. **Spec:** AC-1, AC-2, AC-6, AC-7, AC-8, AC-12.

- [x] **T11 — Calidad e importabilidad**
  - `ruff check .` y `black .` sin hallazgos; `from src.core.inference_schema import
write_inference_json, encode_rle, decode_rle` OK.
  - **Verificación:** lint limpio; import correcto.
  - **Plan:** §5.3. **Spec:** AC-1.

- [x] **T12 — Commit (requiere confirmación)**
  - Commitear módulo/orquestadores/test/requirements. **El agente NO commitea por
    iniciativa propia:** pregunta y espera confirmación (constitución §11).
    Conventional Commits en inglés, scope `inference_schema`.
  - **Verificación:** tras tu confirmación, el commit existe.
  - **Plan:** —. **Spec:** —

---

## Trazabilidad resumida

| Tarea                           | Plan       | Spec (AC)                           |
| ------------------------------- | ---------- | ----------------------------------- |
| T1 dependencia `pycocotools`    | §2, §4     | AC-10                               |
| T2 geometría + RLE              | §3.4, §3.5 | AC-4, AC-12                         |
| T3 builders registro/cabecera   | §3.2, §3.3 | AC-1, AC-3, AC-5                    |
| T4 rutas + escritura            | §3.8       | AC-6, AC-7                          |
| T5 `run_pipeline` esquema común | §3.6, §3.8 | AC-1, AC-2, AC-6, AC-7              |
| T6 `track_video` JSON unificado | §3.7, §3.8 | AC-1, AC-6, AC-8                    |
| T7 documentar `obj_id`          | §3.7       | AC-8                                |
| T8 crear test (A + B)           | §5.1, §5.2 | AC-9, AC-11                         |
| T9 ejecutar Parte A (local)     | §5.1       | AC-4, AC-9                          |
| T10 ejecutar Parte B (pod)      | §5.2       | AC-1, AC-2, AC-6, AC-7, AC-8, AC-12 |
| T11 calidad/import              | §5.3       | AC-1                                |
| T12 commit (confirmación)       | —          | —                                   |
