# Fase 01 — Datos / dataset

> Preparación del dataset, **no** inferencia. Construye el manifiesto reproducible de
> los 123 videos (con splits seeded) y congela el set de evaluación. Es la base honesta
> para el [benchmark](07_benchmark.md) y la futura evaluación con ground-truth.

- **Tareas SDD:** [`csv_dataset_metadata`](../.specs/csv_dataset_metadata/),
  [`forced_testing_split`](../.specs/forced_testing_split/),
  [`eval_frame_export`](../.specs/eval_frame_export/), [`gt_annotation`](../.specs/gt_annotation/)
- **Salidas versionadas:** `assets/db_metadata.csv`, `assets/testing_frames.csv`

---

## `src/data/metadata.py` — el manifiesto

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `build_metadata_csv(force=False)` | [`metadata.py:235`](../src/data/metadata.py#L235) | Descubre videos (`rglob`) → extrae metadatos → asigna `split` → escribe `assets/db_metadata.csv`. **Idempotente** (si existe y valida, lo reutiliza salvo `force=True`). |
| `validate_metadata_schema(csv_path)` | [`metadata.py:212`](../src/data/metadata.py#L212) | Valida el esquema del CSV. |
| `_assign_splits(...)` | [`metadata.py:153`](../src/data/metadata.py#L153) | Asigna splits **seeded**: `0`=reserva, `1`=fine-tuning [23], `2`=testing [20]. `splits.forced_testing` fija videos concretos a testing. |

La semilla y la ruta salen del config (`seeds.split`, `working_dirs.metadata_csv`).

## `src/data/eval_frames.py` — congelar la evaluación

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `export_testing_frames(force=False)` | [`eval_frames.py:164`](../src/data/eval_frames.py#L164) | Extrae los frames de **cuota** de los videos de testing a `data/testing_frames/` (git-ignored) + escribe el control versionado `assets/testing_frames.csv`. Idempotente. |
| `validate_testing_frames_schema(csv_path)` | [`eval_frames.py:141`](../src/data/eval_frames.py#L141) | Valida el CSV de control. |

---

### Cómo encaja con el resto

El `split` del manifiesto es lo que hace **honesto** el [benchmark](07_benchmark.md): el
YOLO de [`yolo_sam3`](03_deteccion.md) se afinó solo con los videos NO-testing, así que
los 5 videos de testing del benchmark están intocados para ambos detectores. Los
`testing_frames` congelados serán la base de la evaluación mIoU/Dice cuando llegue el
ground-truth manual (proceso **pausado**, espera anotaciones del equipo).

## `src/bootstrap_data.py` — provisión y reproducibilidad

- **Tarea SDD:** [`bootstrap_data`](../.specs/bootstrap_data/)
- **Salida versionada:** `assets/bootstrap_manifest.json` (fuente de verdad de qué
  bajar) + `.env.example`.

Script interactivo (`python -m src.bootstrap_data`) que provee los insumos no
versionados desde Google Drive (`gdown`), idempotente y no-destructivo, y genera el
`.env` si falta. Dos paquetes declarados en el manifiesto:

- **`demo`** — paquete **autocontenido y reproducible**: clips demo + **sus JSON de
  tracking con `rle`** + pesos SAM 3/YOLO. Los JSON permiten correr la **Capa B en local
  sin GPU** ([capa métrica](09_capa_metrica.md) / [eventos](10_eventos.md)) al instante.
- **`all`** — dataset completo + pesos. El dataset (Drive público de la convocatoria,
  `17Abril`/`18abril`) excede el tope de carpetas de `gdown`, así que es **descarga
  manual** (`manual: true`): el bootstrap solo verifica presencia e imprime el enlace.

**Reproducibilidad de extremo a extremo:** sobre un demo, `main.py … --overwrite` ignora
lo descargado y rehace **todo de cero** (inferencia SAM 3 → eventos → broadcast); que el
resultado coincida con el del paquete demo valida la reproducibilidad. El esquema del
manifiesto lo comparte el flag [`--demo` del hub](06_pipeline_principal.md)
([`main_demo_flag`](../.specs/main_demo_flag/)).
