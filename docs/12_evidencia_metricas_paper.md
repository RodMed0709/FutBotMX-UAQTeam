# 12 — Evidencia de métricas para el paper (MICAI)

Esta fase documenta **cómo se obtuvo cada número que aparece en el paper**. A
diferencia del resto de `docs/`, no describe código de `src/`: describe los
**scripts de medición** (todos bajo `notebooks/`) que consumen el pipeline ya
construido para producir métricas reproducibles.

> Contexto: ronda de evidencia MICAI (deadline 30 jun 2026). El objetivo fue
> respaldar con datos reproducibles las afirmaciones del draft que estaban sin
> fuente, parciales o rotas, **sin reabrir el pipeline**.

## Restricción rectora: *measurement-only*

Toda la ronda se hizo bajo una sola regla:

- **No se tocó `src/`.** Verificable: `git status --porcelain src/` está vacío en
  todos los commits de esta ronda. Los scripts solo **importan y leen** módulos de
  `src/` y los artefactos ya generados (tracking JSON, CSVs de benchmark, clips).
- **No se tocaron los notebooks de exploración** (`notebooks/fase_0/`, ni los
  notebooks de pipeline ya existentes). Cada script nuevo es un archivo aparte.
- **No se reentrenó ni reconfiguró nada.** Los scripts cronometran / puntúan /
  comparan sobre los mismos modelos, splits y artefactos que ya existían, para que
  los números sean **directamente comparables** con los resultados previos.
- Cambios fuera de los scripts nuevos, mínimos y justificados (ver
  [§ Cambios de soporte](#cambios-de-soporte)).

Cada script repite esta nota de alcance en su propio docstring (honestidad del
alcance: qué es autoridad cm vs. proxy px, qué corre en pod vs. local).

## Qué produjo cada número

### 1. Throughput del detector — FPS sin fuente → medido

- **Script:** `notebooks/fase_3_benchmark_models/detector_only_timing.py` (pod)
- **Qué cierra:** la fila ausente de la tabla de detectores; reemplaza el
  «11.3 FPS» sin fuente por un FPS de **YOLO solo-cajas** (box-only, sin SAM3).
- **Cómo, para que sea comparable:** reusa los *building blocks* del benchmark
  (`benchmark_videos(5, 42)` → los **mismos** 5 clips de testing; `iter_frames` /
  `get_frame_count` como denominador; `detect_boxes` de `yolo_boxes`, que es la
  primera mitad de `yolo_sam3.detect`, para aislar el costo del detector). Misma
  definición FPS = frames/wall-time y mismo idiom de VRAM pico que `src/core/batch.py`
  (`reset_peak_memory_stats` + `perf_counter` + `max_memory_allocated`).
- **Honestidad:** excluye warm-up (carga de modelo + init CUDA); cronometra solo la
  llamada por frame.
- **Salida:** `outputs/benchmark/detector_only.csv` (por clip + fila `ALL`).

### 2. Validación de tracking sin GT — ByteTrack vs BoT-SORT

- **Script:** `notebooks/fase_3_benchmark_models/tracking_validation.py` (local)
- **Qué cierra:** la tabla de tracking con **proxies objetivos** (no hay GT de
  tracking para este dataset, y se declara así en el paper).
- **Métricas (sobre los `tracks` del JSON ya generado):**
  - *Oclusión recuperada (intra-track):* hueco de ≥ `k` frames que reaparece con el
    **mismo `obj_id`** → identidad mantenida.
  - *ID switch (hand-off):* un track termina y otro de la **misma clase** arranca
    poco después y cerca → identidad perdida.
  - *Occlusion-recovery rate* = recuperadas / (recuperadas + switches).
  - *ID consistency* = cobertura intra-track media (`n_obs / span`).
  - Auxiliares: nº de tracks y longitud media de tracklet (fragmentación).
- **Alcance:** solo clases dinámicas (`robot`, `orange_ball`); zonas/`green_floor`
  excluidas. Detector fijo para aislar el efecto del tracker.
- **Salida:** `outputs/benchmark/tracking_validation.csv`.

### 3. Eventos discretos — Precision/Recall vs GT manual

Tres scripts encadenados que cubren el set completo de eventos
(`gol`, `tiro`, `fuera`, `lack_of_progress`, `pushing`):

- **Anotador** `eventos_discretos_annotator.py` (local): reproductor OpenCV donde el
  humano marca cada evento real (frame + calificador: portería `yellow|blue`, o
  causa `salida_campo|area_chica`). Solo lectura del `.mp4`; no toca pipeline ni
  modelos. Produce un CSV ligero de GT.
- **Conversor** `gt_tabla_to_csv.py` (local): extrae la tabla de GT del libro mayor
  (markdown, entre marcadores `<!-- GT_EVENTOS_START/END -->`) y la normaliza a CSV
  en segundos (acepta `mm:ss` o `s.s`; mapea sinónimos ES/EN de tipo y calificador).
- **Evaluador** `eventos_discretos_eval.py` (local): corre los detectores
  **desplegados** (`event_shot_goal`, `event_field_violations` sobre la capa métrica
  `compute_metric_positions`) y los empareja con el GT **por solape temporal** (un
  evento es una duración, no un frame) con holgura `--tol`. Calcula TP/FP/FN → P, R,
  F1 por tipo y agregado.
- **Honestidad de alcance (clave):** el evaluador **espeja exactamente** al detector
  desplegado en `event_broadcast_overlay`: ruta **cm** (autoridad) en clips de cámara
  superior con homografía fiable, y caída a ruta **px** (proxy universal, subdetecta)
  cuando la homografía no es fiable. `fuera` es cm-only; `lack_of_progress`/`pushing`
  son px. No infla métricas usando una ruta que el sistema desplegado no usaría.
- **Salida:** `outputs/eventos_gt/events_gt.csv` (GT) + `outputs/eventos_gt/eventos_pr.csv`
  (detalle TP/FP/FN).

### 4. (ya en commits previos de la ronda) YOLO vs GT y error de homografía

Para que esta doc cubra la ronda completa, los scripts ya commiteados antes:

- **`notebooks/fase_5_lora/02_yolo_eval_vs_gt.py`** + `03_diagnostico_factores.py`
  (commit `d9e79a8`): evaluación del YOLO fine-tuneado contra GT y diagnóstico de
  factores — respalda el mAP del detector.
- **`notebooks/fase_4_homografia/held_out_*.py`** (commit `f27d8eb`):
  `held_out_clicker.py` (anotación de puntos de control held-out),
  `held_out_error.py` (error de reproyección en cm), `held_out_figure.py` (figura) —
  respalda el error de homografía (~9–23 cm) sobre puntos **no** usados al ajustar.

## Cambios de soporte

Únicos cambios fuera de scripts nuevos, ambos necesarios para reproducir las
medidas localmente:

- **`notebooks/fase_6_kalman/06_demo_kalman_minimap.py`:**
  - Ruta del repo portable (`Path(__file__).resolve().parents[2]` en vez de la ruta
    fija del pod `/workspace/...`) → corre en local **y** pod sin editar.
  - Etiquetas de paneles en **inglés** (Segmentation / Tracking / Minimap
    (homography+Kalman) / Heatmap / «GOAL») para las figuras del paper.
- **`requirements.txt`:** pin `trackers==2.4.0`. La 2.5.0 publica un wheel roto (solo
  `dist-info`, sin el paquete); la 2.4.0 trae `ByteTrackTracker`. Sin esto el entorno
  de medición no instala.

## Cómo reproducir

```bash
# 1. Throughput detector (pod, necesita best.pt + GPU)
python notebooks/fase_3_benchmark_models/detector_only_timing.py \
    --n 5 --warmup 10 --out outputs/benchmark/detector_only.csv

# 2. Validación de tracking (local, lee JSON del benchmark)
python notebooks/fase_3_benchmark_models/tracking_validation.py \
    --root outputs/inference/trackers --detector yolo_sam3 \
    --out outputs/benchmark/tracking_validation.csv

# 3. Eventos discretos (local)
#  3a. anotar GT (o convertir la tabla del libro mayor)
python notebooks/fase_5_event_analysis/gt_tabla_to_csv.py \
    --md <libro_mayor>.md --out outputs/eventos_gt/events_gt.csv
#  3b. P/R contra los detectores desplegados
python notebooks/fase_5_event_analysis/eventos_discretos_eval.py \
    --gt outputs/eventos_gt/events_gt.csv \
    --clips IMG_9933_5m30=outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json \
    --out outputs/eventos_gt/eventos_pr.csv
```

> Las salidas (`outputs/...`) son pesadas y están git-ignored; lo versionado son los
> **scripts** que las generan y esta documentación.
