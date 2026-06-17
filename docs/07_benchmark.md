# Fase 07 — Benchmark (sin ground-truth)

> Como aún no hay anotación manual, no se mide exactitud (mAP/MOTA/mIoU) sino
> **eficiencia y consistencia**. Diseño de **dos fases**: detectores por eficiencia,
> trackers 2×2 por consistencia. Honesto: el YOLO se entrenó solo con videos NO-testing.

- **Notebooks:** [`fase_3_benchmark_models/`](../notebooks/fase_3_benchmark_models/)
  (`01_benchmark_detectors`, `02_benchmark_trackers`, `03_benchmark_analysis`)
- **Tareas SDD:** [`batch_timing`](../.specs/batch_timing/), [`benchmark_metrics`](../.specs/benchmark_metrics/)
- **Resultados:** figuras en `assets/benchmark/`, resumen en el [README](../README.md)

---

## `src/eval/benchmark.py` — métricas sin-GT

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `benchmark_videos(n=5, seed=42)` | [`benchmark.py:45`](../src/eval/benchmark.py#L45) | Selección **reproducible** de N videos de testing (semilla fija). |
| `video_metrics(...)` | [`benchmark.py:192`](../src/eval/benchmark.py#L192) | Métricas de un video (eficiencia + trayectoria + máscara). |
| `aggregate_config(...)` | [`benchmark.py:215`](../src/eval/benchmark.py#L215) | Agrega por configuración detector×tracker. |
| `comparison_table(rows)` / `write_table(...)` | [`benchmark.py:253`](../src/eval/benchmark.py#L253) | Tabla comparativa. |
| `_trajectory_metrics(...)` | [`benchmark.py:83`](../src/eval/benchmark.py#L83) | `frag_rate`, `tracklet_len`, suavidad. |
| `_mask_metrics(...)` | [`benchmark.py:158`](../src/eval/benchmark.py#L158) | `mask_iou` (suplementaria, apenas discrimina). |

## Diseño de dos fases

- **Fase 1 — detectores** ([`01_benchmark_detectors.ipynb`](../notebooks/fase_3_benchmark_models/01_benchmark_detectors.ipynb)):
  `sam3_text` vs `yolo_sam3`, eficiencia (FPS, VRAM).
- **Fase 2 — trackers 2×2** ([`02_benchmark_trackers.ipynb`](../notebooks/fase_3_benchmark_models/02_benchmark_trackers.ipynb)):
  detector × tracker, consistencia (`frag_rate`, `tracklet_len`).
- **Fase 3 — análisis** ([`03_benchmark_analysis.ipynb`](../notebooks/fase_3_benchmark_models/03_benchmark_analysis.ipynb)):
  gráficas comparativas.

**Por qué 2×2:** BoT-SORT solo ayuda emparejado con `yolo_sam3`; con `sam3_text` empeora.
La interacción no es separable por ejes, de ahí el cruce completo.

---

### Cómo encaja con el resto

El benchmark consume la salida de [06 Pipeline principal](06_pipeline_principal.md) sobre
los videos de testing de [01 Datos](01_datos.md). Mide **consistencia, no exactitud**: la
exactitud llegará con el ground-truth (proceso pausado). Sus figuras son material para el
[README](../README.md) / entregable.
