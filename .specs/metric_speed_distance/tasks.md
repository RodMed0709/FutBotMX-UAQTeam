# Tasks — `metric_speed_distance` (T4)

> Segunda tarea de la Capa B. Consume la salida de T3 (`metric_positions`). Código solo a
> partir de aquí. Insumo de referencia:
> `outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json`.

## Implementación

- [x] `src/core/metric_kinematics.py`: `_series_by_obj(result)` — agrupa `posiciones` con
      `xy_cm` válido por `obj_id`, ordenadas por `frame_index`.
- [x] `metric_kinematics.py`: `_kinematics(...)` — por par `dt=Δframe/fps`, `d=‖Δxy‖`, `v=d/dt`;
      **descarta** segmentos con `v>max_speed_cms` (cuenta outliers); acumula `dist_cm` + serie.
- [x] `metric_kinematics.py`: `_smooth(serie_v, win)` (media móvil) → `v_media_cms`/`v_max_cms`.
- [x] `metric_kinematics.py`: `compute_kinematics(source, *, fps=None, max_speed_cms=300,
      smooth_win=5, with_series=False) -> KinematicsResult`; `source` = ruta tracks_json (llama a
      `compute_metric_positions`) **o** un `MetricResult`.
- [x] `metric_kinematics.py`: `write_kinematics_json`; dataclasses `ObjKinematics`/`KinematicsResult`.
- [x] `testing/test_metric_speed_distance.py`: corre sobre `IMG_9933_5m30.json` (CPU local);
      tabla por `obj_id`; invariantes; casos borde; viz velocidad+distancia → `.png`.
- [x] `ruff check` limpio.

## Verificación

- [x] Corre en **local sin GPU** sobre `IMG_9933_5m30.json`.
- [x] Velocidades en rango físico (≤~190 cm/s) tras rechazo de outliers. Líderes de distancia
      #3 (936 cm) y #22 (778 cm) = **los mismos líderes de posesión de T1** (cross-check).
- [x] Balón con picos ≥ mediana de robots (invariante pasa).
- [x] Casos borde no rompen (<2 muestras → dist 0; salto enorme → outlier descartado).

## Limitación observada

El balón sale **fragmentado en muchos tracks cortos** (ID-switches / huecos de detección) → su
distancia/velocidad por track infravalora el real. Es la limitación de tracking ya conocida
(no se toca el tracking); las cifras se reportan como **indicativas**.

## Fuera de alcance (recordatorio)

Equipos, gol geométrico, heatmap (T5), zonas (T6), overlay (T7). No tocar el tracking
(los ID-switches se mitigan, no se corrigen; cifras indicativas).
