# Tasks — `event_goal_geometric` (gol geométrico)

> Refinamiento en cm de T2. Capa B (consume T3). Reusa el motor de estados de T2
> (`event_goals._events_from_series`). Código solo a partir de aquí. Insumo de referencia:
> `outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json`.

## Implementación

- [x] `src/core/event_goal_geometric.py`: `_in_goal` (región en cm) + `_series` — booleano por
      frame y portería (amarilla x≤12+m / azul x≥231−m, boca y∈[61,121]±m); `any` sobre muestras
      del balón del frame, sobre timeline contiguo [min,max].
- [x] `event_goal_geometric.py`: `compute_geometric_goals(source, *, margin_cm=8, min_frames=3,
      exit_frames=3, cooldown_frames=15, fps=None) -> GeometricGoalResult` reusando
      `event_goals._events_from_series`; `xy_cm` del balón en `frame_inicio` (`_entry_xy`).
- [x] `event_goal_geometric.py`: dataclasses `GoalEventGeo` + `GeometricGoalResult`;
      `write_geometric_goals_json`.
- [x] `event_goal_geometric.py`: `source` = ruta tracks_json (llama a `compute_metric_positions`)
      **o** `MetricResult`; `fps` de resumen o argumento.
- [x] `testing/test_event_goal_geometric.py`: resumen + eventos; invariantes; **comparación con
      T2**; casos borde (geometría de la región); viz timeline + `xy_cm` sobre `render_field`.
- [x] `ruff check` limpio.

## Verificación

- [x] Corre en **local sin GPU** sobre `IMG_9933_5m30.json`.
- [x] Detecta los goles en la **portería azul**: **3 eventos** (frames 116–255, 827–1003,
      1173–1233). Coincide con los 2 de T2 (lances ~840 y ~1210) y añade uno **temprano**
      (frame 185 ≈ 6.2s) **verificado visualmente**: balón en la boca azul, que T2 NO marcó
      (su bbox no se solapó). El geométrico es **más sensible/completo**.
- [x] Más preciso que T2 (línea real en cm, no bbox).
- [x] Casos borde OK (`_in_goal`: dentro/fuera de boca, ambas porterías).

## Observación

3 eventos geométricos vs 2 de T2 = el detector en cm capta lances que el bbox de la zona se
pierde (no es un falso positivo: verificado). Duraciones largas (4–6 s) = balón permaneciendo
en/junto a la portería tras el gol. El `margin_cm=8` es calibrable si se quiere más estricto.

## Fuera de alcance (recordatorio)

Atribución de equipo, arbitraje, heatmap (T5), zonas (T6), overlay (T7). Reusa el motor de T2,
no lo reimplementa.
