# Fase 11 — Kalman (estimación de estado)

> **No es otra homografía ni la mejora.** Es un **estimador de estado** que corre *río
> abajo*, sobre la misma `xy_cm` de [`compute_metric_positions`](09_capa_metrica.md). Un
> filtro de Kalman de velocidad constante (CV) 2D, desde cero, con **relleno de oclusión**
> (predict-only) y rechazo de outliers por gating de Mahalanobis.

- **Notebooks/scripts:** [`fase_6_kalman/`](../notebooks/fase_6_kalman/) (drivers `.py` +
  `00_fase6_kalman.ipynb`); ablations y tablas en `assets/fase6/`. Clip fiable: `IMG_9933`.
- **Tareas SDD:** [`kalman_state_estimation`](../.specs/kalman_state_estimation/),
  [`kalman_position_source`](../.specs/kalman_position_source/)

---

## `src/core/kalman_state.py` — el filtro (desde cero)

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `KalmanCV` | [`kalman_state.py:55`](../src/core/kalman_state.py#L55) | Filtro CV 2D (estado = posición + velocidad). Predict/update clásicos. |
| `KFParams` | [`kalman_state.py:31`](../src/core/kalman_state.py#L31) | Parámetros del filtro (ruido de proceso/medida, `max_gap_frames`, gating). |
| `KalmanState` | [`kalman_state.py:41`](../src/core/kalman_state.py#L41) | Estado estimado de un frame (posición + velocidad + covarianza). |
| `run_kalman_on_track(samples, cls, obj_id, fps, params)` | [`kalman_state.py:121`](../src/core/kalman_state.py#L121) | Corre el KF sobre el rango denso de frames de un `obj_id`; `xy_cm=None` = oclusión (predict-only). Si la oclusión supera `max_gap_frames`, corta y re-inicializa. |

Validado 3/3 en [`testing/test_kalman_state.py`](../testing/test_kalman_state.py).

## `src/core/kalman_kinematics.py` — driver sobre la capa métrica

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `compute_kalman_states(metric, fps=...)` | [`kalman_kinematics.py:99`](../src/core/kalman_kinematics.py#L99) | Corre el KF sobre un `MetricResult` (todas las trayectorias). Alternativa suave a [`compute_kinematics`](09_capa_metrica.md). |
| `apply_kalman_to_metric(metric, kres, ball_only=True)` | [`kalman_kinematics.py:155`](../src/core/kalman_kinematics.py#L155) | Sustituye `xy_cm` por la estimación del KF (rellena oclusiones). |
| `refine_with_kalman(metric, *, ball_only=True)` | [`kalman_kinematics.py:196`](../src/core/kalman_kinematics.py#L196) | Conveniencia: corre + aplica en un paso. |
| `load_metric_result_from_json(path)` | [`kalman_kinematics.py:54`](../src/core/kalman_kinematics.py#L54) | Reconstruye un `MetricResult` desde JSON. |

## Qué aporta / qué no

- **Aporta:** velocidad más suave/física, **relleno de oclusión** (predict-only), rechazo
  robusto de outliers (gating Mahalanobis en vez del corte duro de 300 cm/s).
- **No toca:** el sesgo absoluto de landmarks (~9–23 cm) sigue igual; modela ruido
  *temporal* frame-a-frame, no el sesgo absoluto.

## Integración al entregable

El [overlay narrativo](10_eventos.md) ya expone el flag `use_kalman` en
`render_broadcast_overlay` (de [`kalman_position_source`](../.specs/kalman_position_source/)):
conmuta la fuente de cinemática entre diferencias finitas (default actual) y Kalman.
**Decisión pendiente:** encenderlo por defecto.

---

### Cómo encaja con el resto

Patrón espejo de la homografía: así como `homography="lines"|"masks"` elige la proyección
([08](08_homografia.md)), `use_kalman` elige el estimador de estado sobre la
[capa métrica](09_capa_metrica.md). Kalman es la última pieza de la cadena de análisis y
una mejora demostrada (mejor que extrapolación lineal a través de oclusiones).
