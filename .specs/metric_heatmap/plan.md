# Plan — `metric_heatmap` (T5)

## Enfoque

Módulo nuevo `src/core/metric_heatmap.py` que consume la salida de T3 y produce el histograma 2D
+ render del heatmap sobre la cancha. Reusa:

- `metric_positions.compute_metric_positions` / `MetricResult` (insumo: posiciones cm).
- `field_template` (`LENGTH_CM`, `WIDTH_CM`, `render_field` para el fondo de cancha).
- `events_core.BALL_CLASSES` (clasificar balón vs robot).
- numpy (histograma + suavizado simple); cv2 (colormap + blend); matplotlib NO necesario.

## Pasos

1. **Histograma por categoría** (`_histogram`): de `result.posiciones`, separar `xy_cm` válidas en
   `ball` (clase en `BALL_CLASSES`) y `robot` (clase `robot`); clip a `[0,LENGTH]×[0,WIDTH]`;
   acumular en una rejilla `(rows, cols)` con `bin_cm` (cols = ceil(LENGTH/bin), rows =
   ceil(WIDTH/bin)). Devuelve `{cat: grid float}`.
2. **Suavizado** (`_smooth_grid`): gaussiano (sigma en celdas = `sigma_cm/bin_cm`); implementar con
   `cv2.GaussianBlur` (kernel impar derivado de sigma) para no depender de scipy.
3. **Normalización**: dividir por el máximo (si >0) → `[0,1]`.
4. **Render** (`render_heatmap`): tomar el canvas de `field_template.render_field(scale,...)`,
   redimensionar la rejilla normalizada al tamaño del canvas (interp), aplicar un colormap
   (`cv2.applyColorMap`, p. ej. `COLORMAP_JET`), y **mezclar** sobre la cancha con alpha
   proporcional a la densidad (celdas vacías → transparentes, no tapan el campo). Devuelve la
   imagen BGR.
5. **API pública** `compute_heatmaps(source, *, bin_cm=5.0, sigma_cm=10.0) -> HeatmapResult`
   (`grids` por categoría + `resumen`); `render_heatmap(grid, *, scale=2.6, margin_cm=10.0)` →
   imagen; `write_heatmap_png(grid, path, ...)`.
6. **Test/harness** `testing/test_metric_heatmap.py`:
   - corre sobre `IMG_9933_5m30.json` (CPU local);
   - imprime resumen (muestras por cat, celda pico en cm, % ocupación);
   - **invariantes** (grids no negativos; máximo del balón cae hacia la portería azul x alto);
   - **casos borde** (categoría sin muestras → grid de ceros, render no rompe);
   - **viz**: PNG del heatmap de balón y de robots sobre la cancha.

## Decisiones técnicas

- **Suavizado con cv2** (no scipy): `GaussianBlur` con `ksize` impar ≈ `2*ceil(3*sigma)+1`.
  Mantiene dependencias del repo (cv2 ya es estándar).
- **Orientación de la rejilla**: fila = eje y (WIDTH), columna = eje x (LENGTH); `to_px`/escala
  del minimap mapea cm→px, así que el render se alinea con `render_field`.
- **Alpha por densidad**: `alpha = norm_grid` (clip 0..~0.8) para que el campo se siga viendo y
  las zonas frías queden transparentes.
- **Celda pico en cm**: convertir el `argmax` de la rejilla a centro de celda en cm para el
  resumen (interpretable).

## Riesgos / validación

- Pocas muestras del balón (fragmentado por tracking) → heatmap del balón disperso; el suavizado
  ayuda. Se reporta nº de muestras.
- `bin_cm` muy chico → ruido; muy grande → poca resolución. Default 5 cm es un compromiso.

## Estructura de archivos

- `src/core/metric_heatmap.py` (nuevo).
- `testing/test_metric_heatmap.py` (nuevo).
