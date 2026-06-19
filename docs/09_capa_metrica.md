# Fase 09 — Capa métrica (en cm)

> "Capa B": CPU-local, lee el tracking JSON y, vía la [homografía](08_homografia.md),
> produce **posiciones en cm reales** y todo lo que se deriva de ellas (velocidad,
> distancia, zonas, heatmap). Es la **base compartida** de los [eventos](10_eventos.md).
> No usa SAM3 ni GPU.

- **Notebooks:** [`fase_5_event_analysis/01_zonas_conteo.ipynb`](../notebooks/fase_5_event_analysis/01_zonas_conteo.ipynb),
  [`02_homografía_cancha.ipynb`](../notebooks/fase_5_event_analysis/)
- **Tareas SDD:** [`metric_positions`](../.specs/metric_positions/),
  [`metric_speed_distance`](../.specs/metric_speed_distance/),
  [`metric_field_zones`](../.specs/metric_field_zones/), [`metric_heatmap`](../.specs/metric_heatmap/)

---

## `src/core/metric_positions.py` — la base de todo

Función principal configurable (elige la homografía):

```python
compute_metric_positions(
    tracks_json: str | Path,
    video: str | Path | None = None,   # None ⇒ clip junto al JSON
    *,
    smooth_beta: float = 0.4,          # continuidad temporal de la homografía
    homography: str = "lines",         # "lines" (default) | "masks" (legacy)
) -> MetricResult                       # xy_cm por frame/obj_id + H_por_frame
```

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `compute_metric_positions(...)` | [`metric_positions.py:182`](../src/core/metric_positions.py#L182) | Proyecta robots/balón a cm por frame y guarda la `H_por_frame`. **La base compartida** del post. |
| `MetricResult` / `MetricPosition` | [`metric_positions.py:57`](../src/core/metric_positions.py#L57) | Resultado (posiciones cm + `resumen` con fps) / una posición. |
| `write_metric_positions_json(...)` | [`metric_positions.py:244`](../src/core/metric_positions.py#L244) | Persiste el resultado. |

## `src/core/metric_kinematics.py` — velocidad y distancia

```python
compute_kinematics(
    source: str | Path | MetricResult,   # ruta a tracks_json o MetricResult ya hecho
    *,
    fps: float | None = None,
    max_speed_cms: float = ...,           # corte duro de outliers (cm/s)
    smooth_win: int = ...,                # ventana de suavizado
    with_series: bool = False,
) -> KinematicsResult
```

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `compute_kinematics(...)` | [`metric_kinematics.py:105`](../src/core/metric_kinematics.py#L105) | Velocidad/distancia por `obj_id` por **diferencias finitas** + corte duro de outliers. (La alternativa suave es [Kalman](11_kalman.md).) |

## `src/core/metric_field_zones.py` — zonas y presencia

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `compute_field_zones(...)` | [`metric_field_zones.py:94`](../src/core/metric_field_zones.py#L94) | Presencia/posesión por zonas (mitades, tercios) en cm. |
| `render_zones(...)` / `write_zones_png(...)` | [`metric_field_zones.py:130`](../src/core/metric_field_zones.py#L130) | Visualización de zonas. |

## `src/core/metric_heatmap.py` — mapa de calor

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `compute_heatmaps(...)` | [`metric_heatmap.py:78`](../src/core/metric_heatmap.py#L78) | Ocupación en cm por categoría (robot/balón), histograma + suavizado gaussiano. |
| `render_heatmap(...)` / `write_heatmap_png(...)` | [`metric_heatmap.py:107`](../src/core/metric_heatmap.py#L107) | Visualización del heatmap (adaptado al estilo del minimapa cenital). |

---

### Cómo encaja con el resto

`compute_metric_positions` es **el cuello** que comparte toda la mitad de análisis: los
[eventos](10_eventos.md) (goles, posesión, fueras) leen sus cm; [Kalman](11_kalman.md)
refina su `xy_cm` río abajo; el [overlay narrativo](10_eventos.md) embebe sus zonas y
heatmap.
