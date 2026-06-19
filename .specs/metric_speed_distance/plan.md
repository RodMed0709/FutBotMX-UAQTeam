# Plan — `metric_speed_distance` (T4)

## Enfoque

Módulo nuevo `src/core/metric_kinematics.py` que consume la salida de T3 y emite
velocidad/distancia por `obj_id`. Reusa T3 sin duplicar:

- `metric_positions.compute_metric_positions` (insumo) / `MetricResult` (`posiciones`, `resumen`).
- `events_core` solo si hace falta clasificar robot/balón (las clases ya vienen en cada
  `MetricPosition.cls`).
- Numpy para los cálculos; matplotlib (lazy) solo en el test.

## Pasos

1. **Agrupar por `obj_id`** (`_series_by_obj`): de `result.posiciones`, filtrar muestras con
   `xy_cm` no nulo, ordenar por `frame_index` → `{obj_id: (class, [(frame, xy_cm)])}`.
2. **Cinemática por objeto** (`_kinematics`): recorrer muestras consecutivas; por par calcular
   `dt = (f2 - f1)/fps`, `d = ‖xy2 - xy1‖`, `v = d/dt`.
   - **Rechazo de outliers**: si `v > max_speed_cms`, descartar el segmento (no suma a distancia
     ni a la serie de velocidad); contar el descarte.
   - acumular `dist_cm += d` solo de segmentos aceptados.
   - construir la serie de velocidad (por frame medio del par) para suavizar.
3. **Suavizado** (`_smooth`): media móvil (ventana `smooth_win`) sobre la serie de velocidad
   aceptada → `v_media_cms` (media de la serie suavizada) y `v_max_cms` (máx de la suavizada).
   `dur_s = (último_frame - primer_frame)/fps`.
4. **Agregar** (`compute_kinematics(...) -> KinematicsResult`): por `obj_id`
   `{class, n_muestras, dur_s, dist_cm, v_media_cms, v_max_cms}` (+ serie opcional) + `resumen`
   (totales por clase, fps, params, nº outliers).
5. **API pública**: `compute_kinematics(source, *, fps=None, max_speed_cms=..., smooth_win=...,
   with_series=False)` donde `source` = ruta a tracks_json **o** un `MetricResult` (si es ruta,
   llama a `compute_metric_positions`). `write_kinematics_json(result, path)`.
6. **Test/harness** `testing/test_metric_speed_distance.py`:
   - corre sobre `IMG_9933_5m30.json` (CPU local);
   - imprime resumen + tabla de distancia/velocidad por `obj_id`;
   - **invariantes** (distancias ≥0; v_max ≥ v_media; outliers descartados ≥0; el balón con
     v_max ≥ que la mediana de robots);
   - **casos borde** (objeto con <2 muestras → dist 0/v None; todos los segmentos outliers);
   - **viz**: curva de velocidad por `obj_id` + barras de distancia → `.png`.

## Decisiones técnicas

- **Umbral de outliers** `max_speed_cms`: elegir un default físico para estos robots de mesa
  (la cancha es 243×182 cm; un robot no cruza la cancha en <1 s). Punto de partida razonable:
  ~300 cm/s (ajustable); documentar que es heurístico. El balón puede ir más rápido en tiros,
  así que el umbral puede ser por clase (balón mayor) — decidir en implementación; por defecto
  un único umbral generoso y dejar nota.
- **Suavizado**: media móvil simple (ventana impar pequeña, p. ej. 5) sobre la serie de
  velocidad; evita picos de 1 frame por ruido de H. Mantener simple.
- **Huecos**: el `dt` real por Δframe ya los maneja; un hueco grande + salto = probable
  outlier → lo filtra el umbral.
- **fps**: tomar de `MetricResult.resumen["fps"]` o del JSON; si falta, error claro.

## Riesgos / validación

- ID-switches inflan distancia (un track salta de un robot a otro): el rechazo de outliers
  mitiga los saltos bruscos, pero no los switches "suaves"; se reporta como limitación
  (cifras indicativas). No se toca el tracking (decisión vigente).
- Sensibilidad al umbral: el test imprime nº de outliers descartados para poder calibrar.

## Estructura de archivos

- `src/core/metric_kinematics.py` (nuevo).
- `testing/test_metric_speed_distance.py` (nuevo).
