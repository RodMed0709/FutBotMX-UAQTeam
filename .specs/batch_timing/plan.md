# Plan técnico — Instrumentación de tiempo y memoria en el batch (`batch_timing`)

- **Tarea atómica:** `batch_timing`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso:** primera tarea del benchmark sin-GT; instrumenta `run_batch` con tiempo
  → FPS y VRAM pico por video.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo (a) **medir** wall-time y VRAM pico alrededor de la llamada
`run_inference` de cada video; (b) **derivar `fps`** leyendo `num_frames` del JSON de
salida; y (c) **inyectar** los 3 campos (`elapsed_s`, `peak_vram_mb`, `fps`) en cada
entrada del resumen, con `None` en `skipped`/`failed`. Un único archivo cambia:
`src/core/batch.py`. El resto del batch (selección, skip-done, aislamiento,
validación temprana de detector/tracker) se conserva.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Wall-time:** `time.perf_counter()` (monotónico), `import time` a nivel de módulo
  (barato).
- **VRAM:** `torch.cuda.reset_peak_memory_stats()` / `torch.cuda.max_memory_allocated()`,
  con `torch` import **perezoso** y guardado por `torch.cuda.is_available()`.
- **`num_frames`:** `json.load` del JSON de salida (la llave del header que escribe el
  esquema). `json` ya está importado en el módulo.
- **Sin dependencias nuevas.**

---

## 3. Diseño

### 3.1 Helpers de medición (privadas en `batch.py`)

```python
import time

_TIMING_NULL = {"elapsed_s": None, "peak_vram_mb": None, "fps": None}


def _reset_peak_vram() -> None:
    """Resetea el contador de pico de VRAM (no-op sin CUDA)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _read_peak_vram_mb() -> float | None:
    """Pico de VRAM en MB desde el ultimo reset, o None sin CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e6


def _read_num_frames(json_path) -> int | None:
    """num_frames del header del JSON de salida; None si falta o falla la lectura."""
    try:
        doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return int(doc["num_frames"])
    except (OSError, KeyError, ValueError, TypeError):
        return None
```

Notas:
- El reset es por video (en 3.3, antes de cada `run_inference`) ⇒ el pico queda
  aislado, no acumulativo.
- El pico incluye el modelo SAM3 residente (carga única) + activaciones del video: es
  el **footprint real** de correr esa config, que es lo que el benchmark reporta.
- Conversión decimal `1e6` (MB), consistente con el smoke.

### 3.2 Cálculo de `fps`

`fps = num_frames / elapsed_s` cuando ambos existen y `elapsed_s > 0`; si
`num_frames is None` (o `elapsed_s` no positivo), `fps = None`. Se encapsula en línea
en el bloque de medición (3.3), no necesita helper propia.

### 3.3 Integración en el bucle de `run_batch`

Hoy el bucle hace (resumido):

```python
try:
    res = run_inference(ruta, ..., bundle=bundle, detector=..., tracker=...)
    entry = {... "status": "done" ...}
except KeyboardInterrupt:
    raise
except Exception as exc:
    entry = {... "status": "failed" ...}
```

Se envuelve la llamada con la medición y se añaden los campos:

```python
try:
    _reset_peak_vram()
    t0 = time.perf_counter()
    res = run_inference(ruta, ..., bundle=bundle, detector=..., tracker=...)
    elapsed = time.perf_counter() - t0
    peak_vram = _read_peak_vram_mb()
    num_frames = _read_num_frames(res["json"])
    fps = num_frames / elapsed if (num_frames is not None and elapsed > 0) else None
    entry = {
        "id": vid, "ruta": ruta, "status": "done",
        "json": str(res["json"]),
        "video": str(res["video"]) if res["video"] else None,
        "error": None,
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram,
        "fps": fps,
    }
except KeyboardInterrupt:
    raise
except Exception as exc:
    entry = {
        "id": vid, "ruta": ruta, "status": "failed",
        "json": None, "video": None, "error": repr(exc),
        **_TIMING_NULL,
    }
```

Y la rama **skip-done** (antes del `try`) añade `**_TIMING_NULL` a su `entry`.

Así:
- La medición envuelve **solo** `run_inference` (no `load_sam3` ni la selección).
- Una excepción en `run_inference` corta antes de calcular métricas ⇒ `failed` lleva
  `None` (vía `_TIMING_NULL`), cumpliendo AC-4.
- `done` siempre lleva `elapsed_s` (float) y `fps` (float o `None` si no hubo
  `num_frames`).

### 3.4 Docstring de `run_batch`

Se actualiza la sección `Returns:` para documentar las 3 llaves nuevas y su semántica
(`None` en `skipped`/`failed`, `peak_vram_mb=None` sin CUDA, `fps=None` si no se pudo
leer `num_frames`).

---

## 4. Archivos afectados

| Archivo | Cambio |
|---|---|
| `src/core/batch.py` | + `import time`; + `_TIMING_NULL`, `_reset_peak_vram`, `_read_peak_vram_mb`, `_read_num_frames`; medición alrededor de `run_inference`; 3 llaves en las 3 ramas (`done`/`skipped`/`failed`); docstring `Returns`. |
| `testing/test_batch_inference.py` | + asserts de los campos de timing (Parte A) + (opcional) chequeo en Parte B pod. |

**No se tocan:** `run_inference`, `track_video`, `run_pipeline`, el esquema,
`src/core/trackers/*`, `src/core/detectors/*`, `src/data/*`, ni los JSON de `configs/`.

---

## 5. Verificación

- **Sin GPU (local), con `run_inference` monkeypatcheado** (el fake escribe un JSON con
  `num_frames`):
  - Cada entrada (`done`/`skipped`/`failed`) tiene `elapsed_s`, `peak_vram_mb`, `fps`.
  - `done`: `elapsed_s` float > 0; `fps` float ≥ 0 = `num_frames/elapsed_s`;
    `peak_vram_mb is None` (sin CUDA).
  - `skipped` y `failed`: las 3 en `None`.
  - No-regresión: skip-done, aislamiento, selección y validación temprana intactos.
  - `ruff check .` / `black .` limpios.
- **Con GPU (pod):** una corrida real reporta `peak_vram_mb` > 0 coherente (≈ lo que
  vio el smoke, ~2 GB) y `fps` plausible; comparar entre configs es el uso posterior
  (`benchmark_metrics`).

---

## 6. Riesgos y mitigaciones

- **Leer el JSON para `num_frames` añade IO por video:** aceptable (un archivo
  pequeño-mediano, una vez por video); aislado en `_read_num_frames` y tolerante a
  fallos (`None`).
- **`max_memory_allocated` no captura memoria de otros procesos / fragmentación:** es
  una aproximación del footprint, suficiente para comparar configs en el mismo pod;
  documentado como "VRAM pico del proceso".
- **El pico incluye el modelo residente:** intencional (footprint de correr la config),
  no un bug; documentado.
- **Romper consumidores del resumen:** mitigado: las llaves nuevas son aditivas y las
  existentes no cambian.
