# Tasks — Gestión y organización de metadatos del dataset (`csv_dataset_metadata`)

- **Tarea atómica:** `csv_dataset_metadata`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** la tarea no usa modelo ni GPU; el **agente ejecuta** toda
> la validación en local sobre los videos reales de `data/raw`.

---

## Fase A — Configuración

- [x] **T1 — Añadir claves de config en `configs/00_testing_config.json`**
  - Añadir `working_dirs.metadata_csv = "assets/db_metadata.csv"` y la sección
    `"seeds": {"split": 42}`.
  - **Verificación:** el JSON sigue siendo válido; las claves nuevas están presentes.
  - **Plan:** §4. **Spec:** AC-8.

---

## Fase B — Submódulo y esquema

- [x] **T2 — Crear el paquete `src/data/`**
  - Crear `src/data/__init__.py` que exporte `build_metadata_csv` y
    `validate_metadata_schema`, con `__all__`.
  - **Verificación:** `from src.data import build_metadata_csv, validate_metadata_schema`
    importa sin error (tras implementar T3–T4).
  - **Plan:** §3.1. **Spec:** AC-1.

- [x] **T3 — Definir esquema y carga de config en `src/data/metadata.py`**
  - Constantes de módulo: `COLUMNS`, `VIDEO_EXTENSIONS`, `SPLIT_*`, `SPLIT_SIZES`.
  - `decord.bridge.set_bridge("native")` al importar.
  - `_load_metadata_config()` → `(dataset_dir, metadata_csv, split_seed)` leyendo
    `.env`/JSON (patrón de `pipeline.py`), con `KeyError`/`ValueError` claros.
  - **Verificación:** `_load_metadata_config()` devuelve los tres valores sobre la
    config real; claves ausentes lanzan el error documentado.
  - **Plan:** §3.2, §3.3. **Spec:** AC-8.

---

## Fase C — Lógica de generación

- [x] **T4 — Descubrimiento, extracción y splits en `src/data/metadata.py`**
  - `_discover_videos(dataset_dir)`: `rglob` de `.MOV` (case-insensitive), orden
    alfabético por ruta POSIX relativa.
  - `_extract_video_metadata(abs_path)`: `duracion`, `ancho`, `alto`, `fps_average`
    vía `decord` (fps + `len(reader)` + `reader[0]`).
  - `_assign_splits(n, seed)`: permutación reproducible (`default_rng`), cortes
    contiguos 23/20/resto; `ValueError` si `n < 43`.
  - **Verificación:** sobre `data/raw`, `_discover_videos` lista los videos en orden
    estable; `_extract_video_metadata` devuelve valores plausibles (enteros/floats
    > 0); `_assign_splits` produce conteos 23/20/resto disjuntos.
  - **Plan:** §3.4, §3.5, §3.6. **Spec:** AC-3, AC-4, AC-6.

- [x] **T5 — Handler `validate_metadata_schema` y orquestador `build_metadata_csv`**
  - `validate_metadata_schema(csv_path) -> bool`: existencia + columnas == `COLUMNS`
    (orden incluido); CSV ilegible → `False`.
  - `build_metadata_csv(force=False) -> pandas.DataFrame`: descubrir → extraer →
    asignar splits → escribir `assets/db_metadata.csv` con `index=False`;
    idempotente (no reescribe si existe, es válido y `force=False`); `csv_path =
    PROJECT_ROOT / metadata_csv` + `mkdir(parents=True, exist_ok=True)`.
  - **Verificación:** genera el CSV con las 8 columnas en orden, `id` `0..N-1`, una
    fila por video, `ruta` relativa resoluble con `get_abs_path`; el handler
    distingue CSV válido de inválido.
  - **Plan:** §3.7, §3.8, §3.9. **Spec:** AC-1, AC-2, AC-5, AC-9, AC-10.

---

## Fase D — Validación

- [x] **T6 — Crear y ejecutar `testing/test_metadata.py` (agente, local)**
  - Script standalone que: (1) corre `build_metadata_csv(force=True)` y verifica que
    crea el CSV; (2) comprueba columnas/orden, `id` secuencial, una fila por video,
    tipos y rangos (`ancho/alto` int > 0, `fps_average/duracion` float > 0), `ruta`
    resoluble; (3) conteos de splits 23/20/resto disjuntos y cubrientes;
    (4) reproducibilidad (dos corridas `force=True` → misma columna `split`);
    (5) idempotencia (`force=False` no reescribe; header corrupto → handler `False`
    y regenera).
  - **El agente lo ejecuta** en local sobre `data/raw`.
  - **Verificación:** el script corre y todas las comprobaciones pasan; lint
    (`ruff`, `black`) sin hallazgos; import desde `src.data` OK.
  - **Plan:** §5.1, §5.2. **Spec:** AC-7, AC-11.

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 claves de config | §4 | AC-8 |
| T2 paquete `src/data/` | §3.1 | AC-1 |
| T3 esquema + carga config | §3.2, §3.3 | AC-8 |
| T4 descubrir/extraer/splits | §3.4, §3.5, §3.6 | AC-3, AC-4, AC-6 |
| T5 handler + orquestador | §3.7, §3.8, §3.9 | AC-1, AC-2, AC-5, AC-9, AC-10 |
| T6 test local (crear + correr) | §5.1, §5.2 | AC-7, AC-11 |
