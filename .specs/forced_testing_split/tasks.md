# Tasks — Videos fijados al split de testing (`forced_testing_split`)

- **Tarea atómica:** `forced_testing_split`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. Marcar `- [x]` al completar.

> **Nota de ejecución:** la tarea no usa modelo ni GPU; el agente valida en local
> sobre los videos reales de `data/raw`.

---

- [x] **T1 — Config: `splits.forced_testing`**
  - Añadir la sección `splits.forced_testing` a `configs/00_testing_config.json` con
    los dos videos de `data/raw/18abril/Camara_superior`.
  - **Verificación:** JSON válido; las dos rutas presentes.
  - **Plan:** §3.1. **Spec:** AC-1.

- [x] **T2 — `_load_metadata_config` devuelve 4 valores**
  - Añadir lectura de `splits.forced_testing` (lista, vacía si ausente); devolver
    `(dataset_dir, metadata_csv, split_seed, forced_testing)`.
  - **Verificación:** devuelve la lista esperada; ausencia → lista vacía.
  - **Plan:** §3.2. **Spec:** AC-1, AC-5.

- [x] **T3 — `_assign_splits` con fijación a testing**
  - Nueva firma `(_assign_splits(n, seed, forced_testing_idx))`: fija esos índices a
    testing, reparte el resto al azar; valida `> 20` fijados.
  - **Verificación:** fijados en testing; conteos 23/20/resto; reproducible con la
    seed; lista vacía equivale a la lógica previa.
  - **Plan:** §3.3. **Spec:** AC-2, AC-3, AC-4, AC-5, AC-6.

- [x] **T4 — `build_metadata_csv` resuelve rutas fijadas → índices**
  - Mapear `ruta -> idx` tras descubrir; resolver `forced_testing_idx`; `ValueError`
    si una ruta fijada no existe en el dataset; pasar a `_assign_splits`.
  - **Verificación:** rutas válidas se fijan; ruta inexistente lanza `ValueError`.
  - **Plan:** §3.4, §3.5. **Spec:** AC-2, AC-6.

- [x] **T5 — Regenerar `assets/db_metadata.csv`**
  - Borrar el CSV existente y regenerarlo con la nueva lógica.
  - **Verificación:** el CSV nuevo tiene los dos videos de Camara_superior en
    `split = 2` y conteos 23/20/resto.
  - **Plan:** §5. **Spec:** AC-7.

- [x] **T6 — Actualizar y ejecutar `testing/test_metadata.py`**
  - Ajustar desempaquetado a 4 valores; añadir comprobación de que los fijados están
    en testing; conservar el resto de comprobaciones.
  - **El agente lo ejecuta** en local.
  - **Verificación:** todas las comprobaciones pasan; `ruff`/`black` limpios.
  - **Plan:** §5. **Spec:** AC-8.
