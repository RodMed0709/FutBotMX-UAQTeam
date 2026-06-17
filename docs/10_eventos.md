# Fase 10 — Eventos y overlay narrativo (el entregable)

> Análisis de partido **en cm reales** sobre la [capa métrica](09_capa_metrica.md):
> goles vs tiros, posesión vs control, fueras/área/pushing. El producto final es el
> **video de espectador** (`render_broadcast_overlay`): marcador, banner de gol,
> posesión, lista de eventos, minimapa cenital + heatmap + homografía embebida.

- **Notebook:** [`fase_5_event_analysis/03_broadcast_overlay_demo.ipynb`](../notebooks/fase_5_event_analysis/03_broadcast_overlay_demo.ipynb)
  (CPU-local; usa el clip **crudo**, no el segmentado). Clip de referencia: `IMG_9933_5m30`.
- **Tareas SDD:** [`events_output_paths`](../.specs/events_output_paths/),
  [`event_goal_zone`](../.specs/event_goal_zone/), [`event_goal_geometric`](../.specs/event_goal_geometric/),
  [`event_shot_vs_goal`](../.specs/event_shot_vs_goal/), [`event_possession`](../.specs/event_possession/),
  [`event_possession_refine`](../.specs/event_possession_refine/),
  [`event_field_violations`](../.specs/event_field_violations/),
  [`event_broadcast_overlay`](../.specs/event_broadcast_overlay/),
  [`event_overlay_narrative`](../.specs/event_overlay_narrative/)

---

## Base común de eventos

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `load_frame_objects(tracks_json)` | [`events_core.py:28`](../src/core/events_core.py#L28) | Lee el tracking JSON a `{frame: [FrameObject]}`. La entrada común de los detectores. |
| `FrameObject` | [`events_core.py:18`](../src/core/events_core.py#L18) | Un objeto en un frame (clase, caja, centroide). |
| `events_paths(...)` | [`events_schema.py:21`](../src/core/events_schema.py#L21) | Ubica las salidas de eventos. |

## Detectores de eventos

| Símbolo | Ubicación | Qué detecta |
|---|---|---|
| `compute_shot_vs_goal(...)` → `ShotGoalResult` | [`event_shot_goal.py:387`](../src/core/event_shot_goal.py#L387) | **Gol estricto vs tiro** (la fuente recomendada del overlay): exige cruzar la línea/boca, no rozar el borde del bbox. |
| `compute_geometric_goals(...)` | [`event_goal_geometric.py:85`](../src/core/event_goal_geometric.py#L85) | Gol **geométrico** (más laxo, más falsos positivos). Candidato alternativo. |
| `compute_goal_zone_events(...)` | [`event_goals.py:136`](../src/core/event_goals.py#L136) | Gol por **zona de portería** (bbox + margen). Versión inicial. |
| `compute_possession_refine(...)` | [`event_possession_refine.py:258`](../src/core/event_possession_refine.py#L258) | **Posesión vs control**: la posesión se actualiza ante cambios radicales (no engañosa). |
| `compute_possession(...)` | [`events.py:149`](../src/core/events.py#L149) | Posesión base (vecino más cercano + histéresis). |
| `compute_field_violations(...)` | [`event_field_violations.py:376`](../src/core/event_field_violations.py#L376) | **Fueras / área / pushing** (centroide cruza líneas de cancha/área chica). |

## `src/core/event_broadcast_overlay.py` — EL entregable

Función principal configurable (el showpiece de espectador):

```python
render_broadcast_overlay(
    tracks_json: str | Path,
    *,
    layout: int = 2,                  # 1 | 2 (disposición de paneles)
    goal_source: str = "strict",      # "strict" | "geometric"
    use_kalman: bool = False,         # refinar cinemática con Kalman (fase_6)
    banner_secs: float = ...,         # duración del banner "¡Gooool!"
    max_items: int = ...,             # tope de la lista dinámica de eventos
    margin_px: int = ...,             # margen alrededor del video
    trajectory_window: int = ...,     # ventana de estela
    bin_cm: float = ..., sigma_cm: float = ...,   # heatmap
    out_fps: float | None = None,
    start_frame: int = 0, max_frames: int | None = None,
    draw_field_on_video: bool = True, # homografía embebida sobre el video
    progress: bool = True,
) -> BroadcastResult
```

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `render_broadcast_overlay(...)` | [`event_broadcast_overlay.py:287`](../src/core/event_broadcast_overlay.py#L287) | Compone el video de espectador: marcador 0-0 con color de portería, banner de gol deslizante, panel de posesión/control, lista dinámica de eventos (tope `max_items`), minimapa cenital + heatmap a los lados, campo reproyectado sobre el video. Renderiza su **propio** minimapa desde la métrica por líneas (no usa `minimap_pipeline`). |

> **Flag `use_kalman`** (de [`kalman_position_source`](11_kalman.md)): conmuta la fuente
> de cinemática entre diferencias finitas y Kalman. Decisión pendiente: encenderlo por
> defecto.

## Overlay 1 vs Overlay 2

`overlay.py`/`track_overlay.py` ([04](04_segmentacion.md)/[05](05_tracking.md)) son el
overlay "técnico" (máscaras + `obj_id`); este broadcast es el overlay "narrativo" para el
espectador. El técnico queda para análisis; el narrativo es el entregable audiovisual.
`demo_overlay.compose_demo` ([`demo_overlay.py:162`](../src/core/demo_overlay.py#L162)) es
el panel de 5 vistas legacy.

---

### Cómo encaja con el resto

Es el **final de la cadena**: tracking JSON ([05](05_tracking.md)) → cm
([09](09_capa_metrica.md)) → eventos → este video. Es el entregable audiovisual de la
convocatoria y el `main` reproducible debe poder regenerarlo end-to-end.
