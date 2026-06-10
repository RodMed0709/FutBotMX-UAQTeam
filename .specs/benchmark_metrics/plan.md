# Plan técnico — Métricas y tabla comparativa del benchmark (`benchmark_metrics`)

- **Tarea atómica:** `benchmark_metrics`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso:** segunda y última tarea del benchmark sin-GT; lee los JSON + el timing
  de `run_batch` y emite la tabla comparativa de las 6 configs.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo (a) seleccionar de forma reproducible los 5 videos de testing;
(b) calcular, por video, las métricas de **trayectoria** (de `tracks`) y **máscara**
(de `frames`+`rle`); (c) **agregar por config** fundiendo el timing de `run_batch`; y
(d) emitir la **tabla** (DataFrame + CSV). Todo en un paquete nuevo `src/eval/`, como
módulo **lector/agregador puro** (no corre inferencia). Un **driver** (notebook)
orquesta las 6 corridas en el pod.

---

## 2. Stack técnico

- **Python:** 3.11.
- **`numpy`** (varianzas, normas, `nanmean`), **`pandas`** (manifiesto + tabla),
  **`decode_rle`** (de `src.core.inference_schema`) — todos **import perezoso** dentro
  de las funciones; `src/eval/__init__.py` barato.
- **Sin dependencias nuevas** (numpy/pandas ya están; `decode_rle` ya existe).
- **Sin GPU ni SAM3.**

---

## 3. Diseño

### 3.1 Paquete y API

```
src/eval/
  __init__.py        # exporta la API pública
  benchmark.py       # implementación
```

API pública (en `benchmark.py`, reexportada por `__init__.py`):

- `benchmark_videos(n=5, seed=42) -> list[str]`
- `video_metrics(doc, *, frag_window=5, frag_radius_frac=0.05) -> dict`
- `aggregate_config(label, entries, **kw) -> dict`
- `comparison_table(rows) -> pandas.DataFrame`
- `write_table(df, path=None) -> Path`

### 3.2 Selector seeded — `benchmark_videos`

```python
def benchmark_videos(n=5, seed=42) -> list[str]:
    import pandas as pd
    from src.data.metadata import _load_metadata_config
    from src.utils import get_abs_path

    _, metadata_csv, _, _ = _load_metadata_config()
    df = pd.read_csv(get_abs_path(metadata_csv))
    testing = df[df["split"] == 2]
    sel = testing.sample(n=min(n, len(testing)), random_state=seed)
    return [str(r) for r in sel.sort_values("id")["ruta"]]
```

Reproducible (mismo `seed` → mismos videos). Idéntico al criterio del smoke.

### 3.3 Métricas por video — `video_metrics`

Recibe el **JSON ya cargado** (`doc`). Devuelve un dict con 5 métricas; `None` donde
no aplica:

```python
{"tracklet_len", "frag_rate", "smoothness", "mask_iou", "com_jitter"}
```

**Trayectoria (de `doc["tracks"]`; si no hay `tracks` ⇒ las 3 en `None`):**

- **`tracklet_len`**: media de `len(t["observations"])` sobre los tracks.
- **`frag_rate`**: con `width = doc["resolution"]["width"]` y
  `radius = frag_radius_frac * width`. Para cada track `a` (fin en frame `f_a`,
  centroide `c_a` de su última observación), cuenta un fragmento si **existe** otro
  track `b` de la **misma clase** cuya **primera** observación cae en
  `f_a < f_b <= f_a + frag_window` y `dist(c_a, c_b) < radius`. Métrica =
  `nº fragmentos / nº tracks`.
- **`smoothness`**: por track con ≥3 observaciones, tomar la secuencia de centroides
  (en orden de `frame_index`), `vel = diff(centroides, axis=0)`,
  `acc = diff(vel, axis=0)`, `mag = norm(acc, axis=1)`; métrica del track =
  `var(mag)`; se promedia sobre tracks. (Frames faltantes se tratan como pasos
  consecutivos, sin interpolar — documentado como limitación.)

**Máscara (de `doc["frames"]`; requiere `rle`; si ninguna detección trae `rle` ⇒ las
2 en `None`):**

- Agrupar por `(class, obj_id)` recorriendo `frames` en orden; para cada par,
  decodificar `rle` por frame.
- **`mask_iou`**: media del IoU de máscara entre frames **consecutivos** del mismo
  `(class, obj_id)`.
- **`com_jitter`**: centro de masa de los píxeles `True` por frame; media de
  `norm(com_t - com_{t-1}) / width` entre frames consecutivos.

Helpers internos: `_mask_iou(a, b)`, `_centroid_of_mask(m)`, `_safe_mean(xs)`
(ignora `None`/`nan`).

### 3.4 Agregación por config — `aggregate_config`

```python
def aggregate_config(label, entries, *, frag_window=5, frag_radius_frac=0.05) -> dict:
    # entries: lista-resumen de run_batch de UNA config.
    done = [e for e in entries if e["status"] == "done"]
    rows = []
    for e in done:
        doc = json.loads(Path(e["json"]).read_text("utf-8"))
        m = video_metrics(doc, frag_window=frag_window,
                          frag_radius_frac=frag_radius_frac)
        m["fps"] = e.get("fps")
        m["peak_vram_mb"] = e.get("peak_vram_mb")
        rows.append(m)
    # promedio por columna ignorando None/N/A (np.nanmean sobre arrays con nan)
    return {"config": label, **_mean_ignore_none(rows)}
```

- Salta entries no-`done` (`skipped`/`failed`).
- Funde el timing (`fps`, `peak_vram_mb`) que viene del resumen de `run_batch`.
- `_mean_ignore_none`: por cada métrica, promedia las muestras no-`None`; si todas son
  `None`, la celda queda `None` (configs sin tracking ⇒ trayectoria/máscara `None`).

### 3.5 Tabla — `comparison_table` / `write_table`

- `comparison_table(rows)`: `pandas.DataFrame(rows)` con columnas ordenadas
  `[config, fps, peak_vram_mb, tracklet_len, frag_rate, smoothness, mask_iou,
  com_jitter]`.
- `write_table(df, path=None)`: escribe CSV; `path` default
  `outputs/benchmark/comparison.csv` (vía `get_abs_path`/`PROJECT_ROOT`), creando la
  carpeta. Devuelve la ruta. (`outputs/` es git-ignored.)

### 3.6 Driver (notebook, exploración — NO `src/`)

`notebooks/benchmark_models/01_run_benchmark.ipynb`:

1. `videos = benchmark_videos()`.
2. `CONFIGS = [(label, mode, detector, tracker), ...]` — 6 filas (2 seg sin tracker +
   4 tracking).
3. `bundle = load_sam3()` una vez; por cada config: `run_batch(mode, videos=videos,
   detector=..., tracker=..., include_masks=True, overwrite=True)` → `aggregate_config`
   con los JSON frescos (antes de la siguiente config) → acumula la fila.
4. `df = comparison_table(rows); write_table(df)`; mostrar `df`.

Procesar config por config evita la colisión de outputs (todas escriben en
`outputs/inference/<stem>/`).

---

## 4. Archivos afectados

| Archivo | Cambio |
|---|---|
| `src/eval/__init__.py` | **nuevo**: exporta la API pública. |
| `src/eval/benchmark.py` | **nuevo**: selector, métricas, agregación, tabla. |
| `testing/test_benchmark_metrics.py` | **nuevo**: smoke con JSON sintéticos (sin GPU). |
| `notebooks/benchmark_models/01_run_benchmark.ipynb` | **nuevo**: driver (pod). |

**No se tocan:** `run_batch`/`run_inference`/el esquema/`src/core/*`/`src/data/*`/
`configs/*`.

---

## 5. Verificación

- **Sin GPU (local):** `testing/test_benchmark_metrics.py` con `doc` sintéticos:
  - **Tracking** (2-3 frames, 1-2 `obj_id`, `rle` fabricado): `video_metrics` devuelve
    floats en las 5 métricas; valores esperados en casos triviales (p. ej. máscara
    idéntica ⇒ `mask_iou == 1.0`, `com_jitter == 0.0`; dos tracks cercanos en frames
    contiguos ⇒ `frag_rate > 0`).
  - **Segmentación** (sin `tracks`, sin `rle`): trayectoria y máscara `None`.
  - `aggregate_config` con un par de entries `done` (+ uno `skipped` que se ignora)
    funde `fps`/`peak_vram_mb` y promedia ignorando `None`.
  - `comparison_table` + `write_table` producen el DataFrame y el CSV.
  - `ruff check .` / `black .` limpios.
- **Con GPU (pod):** el driver corre las 6 configs sobre los 5 videos y emite la tabla
  (uso del benchmark, no parte del código de esta tarea).

---

## 6. Riesgos y mitigaciones

- **Colisión de outputs entre configs:** mitigada procesando **una config a la vez** en
  el driver (lee JSON frescos antes de sobrescribir). El módulo es agnóstico.
- **`frag_rate` y `smoothness` sensibles a frames faltantes:** se documenta que no se
  interpola; suficiente como proxy comparativo entre configs sobre los mismos videos.
- **Promedios con N/A:** mitigado con `_mean_ignore_none` (configs sin tracking dejan
  celdas `None`, no rompen ni sesgan).
- **Tamaño de los JSON con `rle`:** el módulo decodifica por frame en streaming (no
  retiene todas las máscaras); el costo es el medido en el smoke (~19 s/video).
- **Métricas sin GT no son accuracy:** documentado en el spec; la tabla informa, no
  dictamina un "ganador" automático.
