# Tasks — Métricas y tabla comparativa del benchmark (`benchmark_metrics`)

- **Tarea atómica:** `benchmark_metrics`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Paquete y selector

- [x] **T1 — Paquete `src/eval/` + selector seeded**
  - Crear `src/eval/__init__.py` (barato, reexporta la API) y `src/eval/benchmark.py`.
  - `benchmark_videos(n=5, seed=42) -> list[str]`: lee `db_metadata.csv` (vía
    `_load_metadata_config`), filtra `split==2`, `sample(n, random_state=seed)`,
    devuelve rutas project-relative ordenadas por `id`. `pandas` perezoso.
  - **Verificación:** `import src.eval` no arrastra numpy/pandas; `benchmark_videos()`
    es reproducible (misma semilla → misma lista) y respeta `n`.
  - **Plan:** §3.1, §3.2. **Spec:** AC-7.

---

## Fase B — Métricas por video

- [x] **T2 — Trayectoria en `video_metrics`**
  - `video_metrics(doc, *, frag_window=5, frag_radius_frac=0.05) -> dict` con las
    llaves `{tracklet_len, frag_rate, smoothness, mask_iou, com_jitter}`.
  - Trayectoria de `doc["tracks"]`: `tracklet_len` (media de nº observaciones),
    `frag_rate` (fragmento si otro `obj_id` de la misma clase inicia en
    `(f_a, f_a+frag_window]` a `dist < frag_radius_frac*width`), `smoothness`
    (var de la magnitud de la aceleración de centroides, tracks con ≥3 obs). Sin
    `tracks` ⇒ las 3 en `None`.
  - **Verificación:** casos sintéticos — un track largo da `tracklet_len` correcto;
    dos tracks cercanos en frames contiguos dan `frag_rate > 0`; trayectoria recta da
    `smoothness ≈ 0`; sin `tracks` ⇒ `None`.
  - **Plan:** §3.3. **Spec:** AC-2, AC-5.

- [x] **T3 — Máscara en `video_metrics`**
  - De `doc["frames"]`, agrupar por `(class, obj_id)`, decodificar `rle` por frame
    (`decode_rle` perezoso): `mask_iou` (media IoU entre frames consecutivos),
    `com_jitter` (media de `norm(Δcentro_de_masa)/width`). Si ninguna detección trae
    `rle` ⇒ ambas `None`. Helpers `_mask_iou`, `_centroid_of_mask`.
  - **Verificación:** máscara idéntica entre frames ⇒ `mask_iou == 1.0`,
    `com_jitter == 0.0`; sin `rle` ⇒ ambas `None`.
  - **Plan:** §3.3. **Spec:** AC-3.

---

## Fase C — Agregación y tabla

- [x] **T4 — `aggregate_config`**
  - `aggregate_config(label, entries, *, frag_window=5, frag_radius_frac=0.05) -> dict`:
    filtra entries `done`, carga cada `entry["json"]`, llama `video_metrics`, añade
    `fps`/`peak_vram_mb` del entry, y promedia por columna **ignorando `None`**
    (`_mean_ignore_none`). Devuelve `{"config": label, ...}`.
  - **Verificación:** con 2 entries `done` + 1 `skipped` (ignorado), promedia bien y
    funde el timing; una config sin tracking deja trayectoria/máscara en `None`.
  - **Plan:** §3.4. **Spec:** AC-4, AC-5, AC-6.

- [x] **T5 — `comparison_table` + `write_table`**
  - `comparison_table(rows) -> pandas.DataFrame` con columnas ordenadas
    `[config, fps, peak_vram_mb, tracklet_len, frag_rate, smoothness, mask_iou,
    com_jitter]`. `write_table(df, path=None) -> Path` (default
    `outputs/benchmark/comparison.csv`, crea carpeta).
  - **Verificación:** DataFrame con las columnas esperadas; el CSV se escribe y se
    relee igual.
  - **Plan:** §3.5. **Spec:** AC-6.

---

## Fase D — Test y driver

- [x] **T6 — `testing/test_benchmark_metrics.py` (smoke sin GPU)**
  - JSON sintéticos: caso **tracking** (2-3 frames, 1-2 `obj_id`, `rle` fabricado) que
    verifica trayectoria + máscara; caso **segmentación** (sin `tracks`/`rle`) que
    verifica N/A; `aggregate_config` + `comparison_table` + `write_table`.
  - **Verificación (local, sin GPU):** el script corre y los asserts pasan;
    `ruff check .` / `black .` limpios.
  - **Plan:** §5. **Spec:** AC-1, AC-8.

- [x] **T7 — Driver `notebooks/benchmark_models/01_run_benchmark.ipynb`**
  - Define las 6 configs `(label, mode, detector, tracker)`; `benchmark_videos()`;
    carga SAM3 una vez; por config corre `run_batch(..., include_masks=True,
    overwrite=True)` → `aggregate_config` con los JSON frescos → acumula fila;
    `comparison_table` + `write_table`; muestra el `df`.
  - **Verificación:** notebook válido (`nbformat`); corre en el pod (no parte del
    código de `src/`).
  - **Plan:** §3.6. **Spec:** §3.1 (driver).

---

## Notas

- **Orden sugerido:** T1 → T2 → T3 → T4 → T5 → T6 → T7. T4 depende de T2/T3; T6 de
  T1–T5.
- **Archivos nuevos:** `src/eval/{__init__,benchmark}.py`,
  `testing/test_benchmark_metrics.py`, `notebooks/benchmark_models/01_run_benchmark.ipynb`.
  Nada de `src/core`/`src/data`/`configs` se toca.
- **Cierre del proceso:** segunda y última tarea del benchmark sin-GT.
- **Commits:** mostrar el mensaje sugerido y **no** ejecutar hasta que el usuario lo
  indique.
