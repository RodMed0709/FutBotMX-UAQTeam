# Spec — Instrumentación de tiempo y memoria en el batch (`batch_timing`)

- **Tarea atómica:** `batch_timing`
- **Paso de la metodología:** 2 (Especificación)
- **Proceso:** primera tarea del mini-proceso de **benchmark sin-GT** de las 6
  configuraciones detector × tracker. Instrumenta `run_batch` para que el resumen por
  video reporte **costo de inferencia** (tiempo → FPS y VRAM pico).
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que construye el benchmark del pipeline,
> **quiero** que `run_batch` mida y reporte **cuánto tarda** y **cuánta VRAM pico**
> consume la inferencia de cada video,
> **para** poder comparar la **eficiencia** de las 6 configuraciones (el "filtro de la
> realidad" que descarta combinaciones inviables antes de mirar la calidad), sin
> cronometrar a mano ni envolver cada llamada en un notebook.

---

## 2. Motivación (por qué)

- El benchmark sin-GT decide entre 6 configs (detector × tracker). Uno de sus ejes es
  **eficiencia** (FPS, VRAM): SAM3-only es pesado, YOLO→SAM3 filtra antes; BoT-SORT
  añade GMC frente a ByteTrack. Saber el costo es tan decisivo como la calidad,
  especialmente con la deadline de mediados de junio.
- Hoy el resumen de `run_batch` (`list[dict]` por video) trae solo
  `id`/`ruta`/`status`/`json`/`video`/`error`: **no hay tiempo ni memoria**. Medirlo a
  mano (como hizo el smoke `00_phase2_cost_smoke.ipynb`) no escala a 6 configs × 5
  videos.
- `run_batch` es el punto natural para medir: ya itera video por video y ya envuelve
  la llamada `run_inference`. Instrumentar ahí da la métrica **por video** de forma
  homogénea para toda la matriz del benchmark, que después `benchmark_metrics`
  consume para la tabla comparativa.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Tres campos nuevos por entrada del resumen:**
  - `elapsed_s`: wall-time (segundos) de la llamada `run_inference` de ese video.
  - `peak_vram_mb`: VRAM pico (MB) durante esa llamada.
  - `fps`: throughput = `num_frames / elapsed_s`.
- **Medición acotada a la inferencia por video:** envuelve **solo** la llamada
  `run_inference`, no la carga única de SAM3 ni la selección de videos.
- **VRAM por `torch.cuda`:** se resetea el pico **antes** de cada video y se lee
  **después**, aislando el pico por video (incluye el modelo residente + activaciones,
  que es el footprint real de correr esa config). Sin CUDA ⇒ `peak_vram_mb = None`.
- **`fps` a partir de `num_frames`** leído del header del JSON de salida del video.
- **Forma uniforme:** las 3 llaves existen en **todas** las entradas; en `skipped` y
  `failed` van como `None` (no se midió una inferencia completa).
- **Aditivo:** las llaves actuales del resumen no cambian.

### 3.2 Fuera de alcance

- **Cambiar `run_inference`, `track_video`, `run_pipeline` o el esquema JSON:** la
  medición vive **solo** en `run_batch`. (No se añade `num_frames` al valor de retorno
  de `run_inference`; el batch lo lee del JSON ya escrito.)
- **Las métricas del benchmark** (trayectoria, máscara, tabla comparativa): son la
  tarea siguiente `benchmark_metrics`.
- **Desglose fino del tiempo** (por etapa: detección vs tracking vs render): solo el
  total end-to-end por video.
- **Persistir las métricas a disco** (CSV/JSON de resumen): el resumen sigue siendo el
  valor de retorno en memoria; persistir es de `benchmark_metrics`.
- **Medición de RAM de CPU**, energía, o perfiles de GPU detallados.
- El **cómo técnico** (helper de medición, orden exacto): es del `plan.md`.

---

## 4. Comportamiento esperado (criterios de aceptación)

1. **Campos presentes:** cada entrada del resumen incluye `elapsed_s`,
   `peak_vram_mb` y `fps`, además de las llaves actuales.
2. **`done` con valores:** en una entrada `done`, `elapsed_s` es un float > 0 y `fps`
   es un float ≥ 0 (= `num_frames / elapsed_s`).
3. **VRAM condicional:** con CUDA, `peak_vram_mb` es un float > 0 aislado por video
   (reset previo); sin CUDA, es `None`.
4. **`skipped`/`failed` con `None`:** esas entradas llevan las 3 llaves en `None` (no
   se completó una inferencia medible).
5. **Medición acotada:** el tiempo reportado corresponde a la llamada `run_inference`
   del video, no a la carga de SAM3 ni a la selección.
6. **No-regresión:** las llaves existentes y el comportamiento previo
   (skip-done, aislamiento de errores, carga única, validación temprana de
   detector/tracker) se conservan idénticos.
7. **`fps` robusto:** si `num_frames` no se puede leer del JSON, `fps = None` sin
   romper el lote.

---

## 5. Dependencias y relación con otras tareas

- **Depende de:** `batch_detector_tracker` (la versión actual de `run_batch`, que se
  amplía) y del esquema de salida (campo `num_frames` en el header del JSON).
- **Habilita:** `benchmark_metrics` (la tabla comparativa de las 6 configs consume
  `elapsed_s`/`fps`/`peak_vram_mb` como el eje de eficiencia).
- **Prototipo de referencia:** el smoke `notebooks/benchmark_models/00_phase2_cost_smoke.ipynb`
  ya midió tiempo + VRAM por video a mano; esta tarea lo lleva dentro de `run_batch`.
- **No** depende de GT ni del detector/tracker concretos: mide cualquier config.
