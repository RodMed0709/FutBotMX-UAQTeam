# Tasks — `metric_heatmap` (T5)

> Tercera métrica de la Capa B. Consume T3 (`metric_positions`). Código solo a partir de aquí.
> Insumo de referencia: `outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json`.

## Implementación

- [x] `src/core/metric_heatmap.py`: `_histogram(result, bin_cm)` — separa `xy_cm` en
      `ball`/`robot`, clip a la cancha, rejilla `(rows=y, cols=x)`.
- [x] `metric_heatmap.py`: `_smooth_normalize(grid, sigma_cm, bin_cm)` (cv2.GaussianBlur + norm 0..1).
- [x] `metric_heatmap.py`: `compute_heatmaps(source, *, bin_cm=5.0, sigma_cm=10.0) ->
      HeatmapResult`; `source` = ruta tracks_json (llama a T3) o `MetricResult`.
- [x] `metric_heatmap.py`: `render_heatmap(grid, bin_cm, *, sigma_cm, scale, margin_cm)` — colormap
      JET + blend con alpha por densidad sobre `render_field`; `write_heatmap_png`.
- [x] `metric_heatmap.py`: dataclass `HeatmapResult` (grids + bin_cm + resumen).
- [x] `testing/test_metric_heatmap.py`: resumen (muestras/cat, celda pico cm, % ocupación);
      invariantes; casos borde (categoría vacía); viz PNG balón + robots.
- [x] `ruff check` limpio.

## Verificación

- [x] Corre en **local sin GPU** sobre `IMG_9933_5m30.json`.
- [x] Heatmap del **balón**: foco caliente en el **centro** (saques/reinicios tras cada gol) +
      concentración en el **lado de la portería azul** (las jugadas de gol). 1254 muestras.
- [x] Heatmap de **robots**: ocupación con focos en la parte inferior / lado azul (5132 muestras,
      24.5% de celdas ocupadas) — donde se concentró el juego.
- [x] Casos borde no rompen (grid de ceros → render válido).

## Fuera de alcance (recordatorio)

Heatmap por obj_id/equipo, zonas del campo (T6), overlay/demo (T7). Reusa `render_field`.
