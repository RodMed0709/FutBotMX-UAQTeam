# Plan técnico — Barra de progreso en inferencia (`progress_reporting`)

- **Tarea atómica:** `progress_reporting`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Reemplazar el verbose por-frame de segmentación por una **barra `tqdm`** y darle a
tracking (hoy mudo) la **misma barra**, ambas con ETA y frames/s, silenciables vía un
flag `progress` que se hilvana por la fachada y los lotes. Para que tracking tenga ETA
real se añade un helper de conteo barato. Cambio **aditivo y retrocompatible**: solo
afecta el reporte de progreso, no la inferencia.

---

## 2. Stack técnico

- **`tqdm.auto`** (ya en `requirements.txt`, `tqdm>=4.66`): elige automáticamente la
  barra de notebook o de terminal. **Import perezoso** dentro de cada función (estilo
  del repo).
- **`decord`** para el conteo de frames (metadata, no decodifica), reutilizando el
  patrón de `get_video_fps`.

---

## 3. Diseño

### 3.1 Archivos a modificar

| Archivo | Cambio |
|---|---|
| `src/core/frame_extraction.py` | Nuevo helper `get_frame_count(video_path)`. |
| `src/core/pipeline.py` | `run_pipeline` gana `progress`; barra en el loop, sin el `print` por-frame. |
| `src/core/tracking.py` | `track_video` gana `progress`; barra envolviendo `iter_frames`. |
| `src/core/inference.py` | `run_inference` gana `progress`; lo reenvía a ambas ramas. |
| `src/core/batch.py` | `run_batch` gana `progress`; lo reenvía a `run_inference`. |
| `testing/test_frame_extraction.py` | Smoke (sin GPU) de `get_frame_count`. |
| `CLAUDE.md` | Mención de las barras, `progress` y `get_frame_count`. |

### 3.2 `get_frame_count` (`frame_extraction.py`)

Espejando `get_video_fps` (abre solo metadata):

```python
def get_frame_count(video_path: Path) -> int:
    """Nº total de frames del video (decord ``len``; metadata, no decodifica)."""
    abs_path = _resolve_video_path(video_path)
    reader = decord.VideoReader(str(abs_path))
    return len(reader)
```

- Acepta ruta relativa a `PROJECT_ROOT` o absoluta (vía `_resolve_video_path`), como
  el resto del módulo.
- Docstring con las mismas `Raises` (`ValueError`/`FileNotFoundError`).

### 3.3 Segmentación (`pipeline.py`)

En el loop (hoy `pipeline.py:191-192`):

```python
# antes
for i, frame in enumerate(frames):
    print(f"  frame {i + 1}/{total}")
    dets = detector_fn(frame, classes=classes, bundle=bundle)
    ...
# después
from tqdm.auto import tqdm  # import perezoso, junto al loop
bar = tqdm(
    enumerate(frames), total=total, desc=f"seg {stem}",
    unit="frame", leave=False, disable=not progress,
)
for i, frame in bar:
    dets = detector_fn(frame, classes=classes, bundle=bundle)
    ...
```

- Se **elimina** el `print(f"  frame {i + 1}/{total}")`.
- `total` ya existe (`total = len(frames)`), `stem` ya existe.

### 3.4 Tracking (`tracking.py`)

Antes del `with writer_cm as append:`, derivar el total y envolver el iterador:

```python
from tqdm.auto import tqdm  # import perezoso
n_total = get_frame_count(video_path)
if max_frames is not None:
    n_total = min(int(max_frames), n_total)

with writer_cm as append:
    for frame_index, frame in tqdm(
        iter_frames(video_path, max_frames),
        total=n_total, desc=f"track {stem}",
        unit="frame", leave=False, disable=not progress,
    ):
        ...
```

- `stem` ya se computa en `track_video` (para `inference_paths`).
- `get_frame_count` se importa de `frame_extraction` (donde ya se importa
  `get_video_fps`, `iter_frames`).

### 3.5 Hilvanado de `progress`

- **`run_pipeline` / `track_video`:** `progress: bool = True` al final de la firma;
  documentar.
- **`run_inference`:** `progress: bool = True`; reenviar a `run_pipeline` (rama
  segmentación) y a `track_video` (rama tracking).
- **`run_batch`:** `progress: bool = True`; reenviar a `run_inference`. El print
  `[i/n] ruta -> status` por video **se conserva**; la barra del video en curso aparece
  debajo y desaparece al terminar (`leave=False`).

### 3.6 Lo que NO cambia (anti-alcance técnico)

- El esquema/contenido del JSON, las rutas (`inference_paths`/`run_label`), el muestreo
  y la lógica de detección/tracking.
- La firma de `iter_frames` y `extract_frames`.
- El streaming de `pipeline.py` (tarea condicional aparte).

---

## 4. Cambios de configuración y dependencias

- **Ninguno.** `tqdm` ya está en `requirements.txt`.

---

## 5. Validación

### 5.1 Smoke sin GPU (`testing/test_frame_extraction.py`)

- `get_frame_count` sobre un `.MOV` real devuelve un entero `> 0` y **coincide** con la
  longitud de `get_frame_indices(..., all_frames=True)` (mismo total que ve el módulo).
- Introspección de firmas: `run_pipeline`, `track_video`, `run_inference`, `run_batch`
  exponen `progress` con default `True`.

### 5.2 Verificación visual (pod/GPU)

- Correr segmentación y tracking sobre un clip y **ver** la barra (ETA + frames/s);
  con `progress=False` no aparece nada.

### 5.3 Calidad

- `ruff check` y `ruff format --check` limpios en los archivos tocados.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec §5) | Cubierto por |
|---|---|
| 1. `progress` en `run_pipeline`/`track_video` | §3.3, §3.4, §3.5 |
| 2. Barra en segmentación, sin print | §3.3 |
| 3. Barra en tracking con total | §3.4 |
| 4. `get_frame_count` | §3.2, §5.1 |
| 5. `progress=False` desactiva | §3.3, §3.4 |
| 6. `progress` propagado | §3.5 |
| 7. `tqdm.auto`/perezoso/`desc`/`leave=False` | §3.3, §3.4 |
| 8. No cambia esquema/rutas/lógica | §3.6 |
| 9. Smoke sin GPU | §5.1 |
| 10. `CLAUDE.md` | §3.1 |

---

## 7. Riesgos y consideraciones

- **Doble apertura del video en tracking.** `get_video_fps` + `get_frame_count` abren
  el video dos veces (solo metadata, barato). Aceptable; no se optimiza para no
  complicar la firma.
- **`tqdm.auto` en terminal vs notebook.** `auto` resuelve el entorno; si en el pod se
  corre por `python testing/...`, cae a la barra de terminal sin problema.
- **`leave=False` en lotes.** Mantiene la salida limpia (las barras terminadas no se
  acumulan), conservando el resumen `[i/n]` de `run_batch`.
- **Orden de parámetros.** `progress` se agrega **al final** de cada firma para no
  romper llamadas posicionales.
