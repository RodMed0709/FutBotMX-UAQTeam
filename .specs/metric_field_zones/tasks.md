# Tasks — `metric_field_zones` (T6)

> Cuarta métrica de la Capa B. Combina T3 (posiciones cm) + T1 (posesión). Código solo a partir
> de aquí. Insumo: `outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json`.

## Implementación

- [x] `src/core/metric_field_zones.py`: `_SCHEMES` (mitades, tercios) como `(labels, fn(x,y))`.
- [x] `metric_field_zones.py`: `_presence(metric, labels, fn)` — % por zona de balón y robots.
- [x] `metric_field_zones.py`: `_ball_cm_by_frame(metric)` + `_possession_by_zone(...)` — %
      de posesión por zona (frames con owner≠None), reusando T1 `compute_possession`.
- [x] `metric_field_zones.py`: `compute_field_zones(tracks_json, *, schemes=("mitades","tercios"),
      fps=None, metric=None) -> FieldZonesResult`; `write_field_zones_json`.
- [x] `metric_field_zones.py`: `render_zones(...)` (fronteras + % sobre `render_field`) +
      `write_zones_png`; dataclass `FieldZonesResult`.
- [x] `testing/test_metric_field_zones.py`: resumen; invariantes (presencia ~100%; sesgo azul);
      casos borde (esquema desconocido → error); viz PNG por esquema.
- [x] `ruff check` limpio.

## Verificación

- [x] Corre en **local sin GPU** sobre `IMG_9933_5m30.json`.
- [x] Presencia suma ~100% por categoría y esquema.
- [x] Sesgo azul (clip de gol): mitades → balón **87% azul** / posesión **89% azul**; tercios →
      balón **0% amarillo / 32% medio / 68% azul**. Coherente con T5 (heatmap) y el gol.
- [x] Casos borde OK (esquema desconocido → ValueError).

## Fuera de alcance (recordatorio)

Equipos, heatmap (T5), velocidad (T4), overlay/demo (T7). Esquemas como datos (extensible).
