# Plan técnico — Orquestación de inferencia por lotes (`batch_inference`)

- **Tarea atómica:** `batch_inference`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso de referencia:** roadmap del pipeline de inferencia unificado + batch (tarea 4)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo construir `run_batch(...)` (en un módulo nuevo `src/core/batch.py`): una
capa **delgada y secuencial** sobre la fachada `run_inference` (tarea 3) que itera N
videos seleccionados del manifiesto `db_metadata.csv`, **carga SAM3 una sola vez**,
**salta lo ya procesado** (skip-done por JSON existente), **aísla errores** (un video
malo no tumba el lote) y devuelve un **resumen estructurado** por video. No reimplementa
inferencia ni toca `run_inference`/esquema/`src/data/`.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Sin dependencias nuevas** (`requirements.txt` no cambia). `pandas` ya es
  dependencia (lo usan tests y `src/data/`).
- **Módulo nuevo `src/core/batch.py`**: importa `run_inference` a nivel de módulo
  (barato, no arrastra torch); `pandas`, `load_sam3` y los loaders de metadata se
  importan **dentro** de `run_batch` (imports perezosos).
- **Reuso de bloques existentes:** `inference_paths` (`inference_schema`) para derivar
  la ruta canónica del JSON (skip-done); `_load_metadata_config` (`src/data/metadata`)
  para la ruta del CSV; `load_sam3` para la carga única del modelo.
- **Sin cambios** en `inference.py`, `pipeline.py`, `tracking.py`,
  `inference_schema.py`, overlay/escritura/extracción ni `src/data/`.

---

## 3. Diseño

### 3.1 Estructura de archivos

```
src/core/batch.py                  # NUEVO: run_batch (orquestador de lotes)
testing/test_batch_inference.py    # NUEVO: Parte A local (sin SAM3) + Parte B pod
```

### 3.2 Firma de `run_batch`

```python
def run_batch(
    mode: str = "segmentation",
    split: int = 2,
    videos: list[str | int] | None = None,
    sampling: str = "auto",
    max_frames: int | None = None,
    include_masks: bool = False,
    render_video: bool = False,
    overwrite: bool = False,
) -> list[dict]:
```

- **`mode`**: único para todo el lote (`"segmentation"` o `"tracking"`); se valida vía
  `run_inference` por video (un `mode` inválido falla en el primer video).
- **`split`**: filtro del manifiesto (0=reserva, 1=fine-tuning, 2=testing). Default
  testing.
- **`videos`**: lista explícita (rutas project-relative **o** ids); si se pasa, **tiene
  prioridad** sobre `split`.
- **`sampling`/`max_frames`/`include_masks`/`render_video`**: se pasan tal cual a
  `run_inference` y aplican a todo el lote. `render_video=False` es el default de lote.
- **`overwrite`**: si `True`, reprocesa aunque el JSON exista (desactiva skip-done).

### 3.3 Selección de videos (manifiesto)

```python
import pandas as pd
from src.data.metadata import _load_metadata_config
from src.utils import get_abs_path

_, metadata_csv, _, _ = _load_metadata_config()
df = pd.read_csv(get_abs_path(metadata_csv)).sort_values("id")  # orden determinista

if videos is not None:
    # Acepta rutas o ids; error explícito si alguno no está en el CSV.
    by_ruta = set(str(v) for v in videos)
    by_id = set(int(v) for v in videos if _is_int(v))
    sel = df[df["ruta"].isin(by_ruta) | df["id"].isin(by_id)]
    faltan = (by_ruta | {str(i) for i in by_id}) - (
        set(sel["ruta"]) | {str(i) for i in sel["id"]}
    )
    if faltan:
        raise ValueError(f"videos no encontrados en {metadata_csv}: {sorted(faltan)}")
else:
    sel = df[df["split"] == split]

rows = list(sel[["id", "ruta"]].itertuples(index=False))  # orden por id
```

- La columna **`ruta`** (project-relative) es lo que se pasa a `run_inference`.
- Orden **determinista por `id`** → corridas reproducibles.

### 3.4 `outputs_dir` y skip-done

- `run_batch` lee la config activa **una vez** (vía `CONFIG_FILENAME` del `.env`, con
  el mismo parseo `strip()` que el resto del código) para obtener
  `working_dirs.outputs_dir`. (No lee clases ni nada más; solo lo necesario para
  derivar rutas.)
- **Skip-done** por video, **antes** de invocar `run_inference`:
  ```python
  from src.core.inference_schema import inference_paths
  json_path, _ = inference_paths(Path(ruta).stem, outputs_dir)
  if json_path.exists() and not overwrite:
      status = "skipped"; ...; continue
  ```

### 3.5 Carga única del modelo y bucle

```python
from src.core.sam3_loader import load_sam3
from src.core.inference import run_inference  # (a nivel de módulo)

bundle = load_sam3()                 # UNA sola vez para todo el lote
results: list[dict] = []
n = len(rows)
for i, (vid, ruta) in enumerate(rows, start=1):
    # skip-done (§3.4) ...
    try:
        res = run_inference(
            ruta, mode=mode, sampling=sampling, max_frames=max_frames,
            include_masks=include_masks, render_video=render_video, bundle=bundle,
        )
        entry = {"id": int(vid), "ruta": ruta, "status": "done",
                 "json": str(res["json"]),
                 "video": str(res["video"]) if res["video"] else None,
                 "error": None}
    except KeyboardInterrupt:
        raise                         # abortable: no se traga
    except Exception as exc:          # aislamiento: registra y continúa
        entry = {"id": int(vid), "ruta": ruta, "status": "failed",
                 "json": None, "video": None, "error": repr(exc)}
    print(f"[{i}/{n}] {ruta} -> {entry['status']}")
    results.append(entry)
```

### 3.6 Estructura del resumen (retorno) y logging

- **Retorno:** `list[dict]`, una entrada por video con la forma:
  ```python
  {"id": int, "ruta": str, "status": "done"|"skipped"|"failed",
   "json": str | None, "video": str | None, "error": str | None}
  ```
  - `done`: `json` y `video` del retorno de `run_inference` (`video` puede ser `None`
    con render OFF).
  - `skipped`: `json` = la ruta existente; `video`/`error` `None`.
  - `failed`: `error` poblado (`repr(exc)`); `json`/`video` `None`.
- **Logging a stdout:** por video `[i/N] <ruta> -> <status>`; al final, un resumen con
  conteos:
  ```python
  done = sum(r["status"] == "done" for r in results)
  skipped = sum(r["status"] == "skipped" for r in results)
  failed = sum(r["status"] == "failed" for r in results)
  print(f"== batch: {done} done, {skipped} skipped, {failed} failed (de {n}) ==")
  ```
  Sin dependencia de `logging`.

### 3.7 Lo que NO cambia (anti-alcance técnico)

- `inference.py` (`run_inference`), `pipeline.py`, `tracking.py`,
  `inference_schema.py` (incluido `SCHEMA_VERSION`), overlay/escritura/extracción.
- `src/data/` y `db_metadata.csv`: el lote solo **lee** el manifiesto (vía
  `_load_metadata_config` + `pandas`), no recalcula metadatos ni splits.
- **Sin paralelismo:** iteración secuencial (un proceso satura la GPU con SAM3).

---

## 4. Cambios de configuración y dependencias

- **`requirements.txt`:** sin cambios.
- **Config:** sin cambios. `mode`/`split`/`videos`/flags son **parámetros de función**,
  no claves de config. El lote **lee** `working_dirs.metadata_csv` y
  `working_dirs.outputs_dir` (ya existentes).
- **`CLAUDE.md`:** al implementar, añadir la capa batch (`src/core/batch.py::run_batch`)
  a la sección de arquitectura, sobre la fachada `run_inference`.

---

## 5. Validación (`testing/test_batch_inference.py`)

> Filosofía del repo: smoke funcional; lo que invoca SAM3 corre en **pod/GPU**. La
> lógica de orquestación (selección, skip-done, aislamiento, resumen) es testeable
> **localmente sin modelo** monkeypatcheando `run_inference`/`load_sam3`.

### 5.1 Parte A — local, **sin GPU**

- **Firma:** `inspect.signature(run_batch)` con los defaults esperados
  (`mode="segmentation"`, `split=2`, `render_video=False`, `overwrite=False`, …).
- **Selección** (sobre un `db_metadata.csv` **temporal** de prueba, apuntando el config
  a él o monkeypatcheando `_load_metadata_config`):
  - filtro por `split` devuelve las filas correctas y en orden por `id`;
  - lista explícita por `ruta` y por `id` acota el lote y tiene prioridad;
  - id/ruta inexistente → `ValueError`.
- **Skip-done:** creando un JSON falso en la ruta canónica
  (`inference_paths(stem, outputs_dir)[0]`), ese video sale `"skipped"` y
  `run_inference` (monkeypatcheado) **no** se invoca para él; con `overwrite=True` sí.
- **Aislamiento de errores:** monkeypatchear `run_inference` para que lance en un video
  concreto → ese sale `"failed"` con `error`, los demás `"done"`, el bucle no se
  detiene y el resumen tiene los conteos correctos.
- Todo **sin** cargar SAM3 (`load_sam3` y `run_inference` monkeypatcheados).

### 5.2 Parte B — **pod/GPU**, lote de 3 videos

Selección sencilla: **los primeros 3 videos del split reservado** (`split=0`), por `id`
(helper `_pick_reserved_videos(n=3)`). Ambas corridas con **video + JSON extendido**:

- **Segmentación con cuota:**
  ```python
  run_batch(mode="segmentation", videos=<3 reservados>, sampling="quota",
            include_masks=True, render_video=True)
  ```
  → 3 × `"done"`; cada uno con mp4 + JSON existentes; JSON con `rle` en las
  detecciones (`include_masks`).
- **Tracking acotado a 300 frames:**
  ```python
  run_batch(mode="tracking", videos=<3 reservados>, max_frames=300,
            include_masks=True, render_video=True)
  ```
  → 3 × `"done"`; mp4 + JSON (con `frames`+`tracks` y `rle`) existentes.
- **Skip-done:** una **segunda** corrida idéntica (sin `overwrite`) marca los 3
  `"skipped"` y no recomputa.
- **Carga única:** el log muestra una sola carga de SAM3 por corrida (el `bundle` se
  reutiliza en los 3 videos).
- **Resumen:** el retorno y el print de conteos reflejan `done`/`skipped` correctos.

> Nota: 300 frames × clases con `include_masks=True` produce JSONs grandes y mp4s
> reales; es deliberado (prueba cercana al uso end-to-end), no un smoke mínimo.

### 5.3 Calidad

- `ruff check .` y `black .` sin hallazgos.
- Importabilidad: `from src.core.batch import run_batch`.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Orquestador único | §3.2, §3.5 | `run_batch` itera `run_inference` por video |
| AC-2 Fuente desde manifiesto | §3.3 | filtro por `split`, columna `ruta` |
| AC-3 Lista explícita prioritaria | §3.3 | `videos` (ruta/id) gana al `split` |
| AC-4 Carga única del modelo | §3.5 | `load_sam3()` una vez, `bundle` reusado |
| AC-5 Render OFF por defecto | §3.2 | `render_video=False` default, sobreescribible |
| AC-6 Skip-done | §3.4 | JSON canónico existente → `skipped`; `overwrite` lo fuerza |
| AC-7 Aislamiento de errores | §3.5 | `try/except Exception`, continúa; `KeyboardInterrupt` propaga |
| AC-8 Resumen estructurado | §3.6 | `list[dict]` por video + conteos a stdout |
| AC-9 Herencia de flags | §3.2, §3.5 | `mode`/`sampling`/`max_frames`/`include_masks` → `run_inference` |
| AC-10 Secuencial | §3.5, §3.7 | bucle secuencial, orden por `id` |
| AC-11 Sin cambios colaterales | §3.7, §4 | solo lee el CSV; nada de `run_inference`/esquema/`src/data` |
| AC-12 Verificación | §5.1, §5.2 | local (selección/skip/aislamiento) + pod (lote de 3) |

---

## 7. Riesgos y consideraciones

- **Lectura de config duplicada:** `run_batch` vuelve a parsear `.env`/config para
  `outputs_dir` (igual que hacen `pipeline.py`/`tracking.py`). Un helper compartido de
  config sigue siendo **trabajo futuro** (ya anotado en `unified_inference`); aquí se
  acepta la duplicación mínima para no ampliar alcance.
- **`videos` con ids vs. rutas:** se aceptan ambos; el matching es por pertenencia al
  CSV y cualquier valor no encontrado es `ValueError` (falla rápido, antes de cargar
  el modelo).
- **Skip-done por existencia del JSON:** si una corrida previa quedó a medias, el JSON
  podría no existir (no se escribió) → se reprocesa; si existe, se asume completo. No
  se valida el contenido del JSON (fuera de alcance).
- **Coste de la Parte B:** tracking a 300 frames + `include_masks` + render en 3 videos
  es pesado; es la prueba cercana al end-to-end pedida, no un smoke. Correr solo en el
  pod.
- **Alcance:** esta tarea entrega la orquestación secuencial; el paralelismo y el
  ejemplo end-to-end de cierre del roadmap quedan fuera.
