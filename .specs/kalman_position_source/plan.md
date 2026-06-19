# Plan — `kalman_position_source` (SDD-B)

Implementación técnica del refinador de posiciones Kalman desmontable por flag. Fiel a las
estructuras actuales: `MetricResult(posiciones, resumen, H_por_frame)`,
`MetricPosition(obj_id, cls, frame_index, xy_cm, status_H)`,
`KalmanState(obj_id, cls, frame_index, xy_cm, vxy_cms, speed_cms, pos_sigma_cm, source, nis)`.

## Enfoque

Un **único punto de refinación** transforma un `MetricResult` (posiciones crudas en cm, salidas
de la homografía por líneas) en otro `MetricResult` con el **balón** reemplazado por estados
Kalman, conservando `H_por_frame` y `resumen`. Todos los consumidores de cm reciben el
`MetricResult` ya elegido (crudo o refinado) según el flag `use_kalman`. La homografía y el
tracking **no se tocan**.

## Cambios por módulo

### 1. `src/core/metric_positions.py` — marcar el origen Kalman

- Añadir a `MetricPosition` un campo **opcional y retrocompatible**:
  ```python
  source: str | None = None   # None=crudo; "measured"|"predicted"|"gated" si viene de Kalman
  ```
  Por defecto `None`, así ningún llamador existente se rompe.
- `write_metric_positions_json`: incluir `source` solo si no es `None` (serialización opcional;
  no cambia los JSON actuales). `H_por_frame` sigue sin serializarse (en memoria).

### 2. `src/core/kalman_kinematics.py` — refinador

Dos funciones (separadas para no correr el KF dos veces en el broadcast):

```python
def apply_kalman_to_metric(
    metric: MetricResult, kres: KalmanResult, *, ball_only: bool = True
) -> MetricResult:
    """Devuelve un MetricResult con el balón reemplazado por estados Kalman.
    Conserva H_por_frame y resumen. Robots/zonas intactos."""
```
Lógica:
- Construye `kf_by_key = {(obj_id, frame_index): KalmanState}` a partir de `kres.por_obj[*].estados`,
  **solo** para clases de balón (`BALL_CLASSES`) cuando `ball_only=True`.
- Recorre `metric.posiciones`:
  - Si `(p.obj_id, p.frame_index)` está en `kf_by_key`: emite
    `MetricPosition(obj_id, cls, frame_index, xy_cm=s.xy_cm, status_H=p.status_H, source=s.source)`.
  - Si no, copia `p` tal cual.
- **Frames rellenados por oclusión**: los estados `predicted` del KF cubren frames donde el
  original tenía `xy_cm=None` (o no existía la posición). Para esos `(obj_id, frame)` se **añade**
  una `MetricPosition` nueva con `xy_cm` del KF y `source="predicted"` (así el minimap/heatmap los
  ven). Se itera sobre los estados Kalman del balón para no perder los rellenos.
- Devuelve `MetricResult(posiciones=nuevas, resumen=metric.resumen, H_por_frame=metric.H_por_frame)`.

```python
def refine_with_kalman(metric: MetricResult, *, ball_only: bool = True) -> MetricResult:
    """Conveniencia: corre el KF y aplica. Para tests/llamadores fuera del broadcast."""
    kres = compute_kalman_states(metric, fps=metric.resumen.get("fps"))
    return apply_kalman_to_metric(metric, kres, ball_only=ball_only)
```

### 3. `src/core/event_shot_goal.py` — conservador en `predicted`

En la ruta cm, al decidir el cruce de la línea de gol del balón:
- Una detección de **gol** debe estar **soportada por al menos un frame `source != "predicted"`**
  (measured/gated/None) dentro de la ventana del evento. Si el cruce se apoya **solo** en frames
  `predicted`, **no se declara gol** (se puede degradar a "tiro" o descartar, según la lógica
  existente del cruce).
- Implementación: donde se leen las posiciones del balón por frame (vía `_ball_by_frame` /
  `MetricResult`), considerar `p.source` (campo nuevo, `None` cuando el flag está apagado → sin
  cambio de comportamiento). Con el flag apagado **todas** las posiciones tienen `source=None`,
  por lo que el detector se comporta **exactamente** como hoy.

### 4. `src/core/event_broadcast_overlay.py` — flag + panel de velocidad

**Firma:** `render_broadcast_overlay(..., use_kalman: bool = False)`.

**Precómputo (tras `metric = compute_metric_positions(tracks_json)`):**
```python
kres = None
if use_kalman and metric is not None and not degradado:
    kres = compute_kalman_states(metric, fps=fps)
    metric = apply_kalman_to_metric(metric, kres)   # balón refinado; H_por_frame intacto
```
A partir de aquí, `shot/goal`, `metric_field_zones` (si aplica), `_mini_by_frame(metric)`,
`_live_heatmap_state(metric)` consumen el `metric` elegido **sin cambios en su código**. La
homografía embebida (`metric.H_por_frame`) sigue intacta.

**Panel de velocidad (solo si `use_kalman`):**
- Helper `_ball_speed_by_frame(kres) -> dict[int, float]` y `v_max` global del balón
  (`max ObjKalman.v_max_cms` para `cls in BALL_CLASSES`).
- Nuevo `_draw_velocity_panel(width, h, vdata) -> np.ndarray` (estilo de los paneles existentes:
  fondo `(28,28,28)`, título + `Ball speed: NNN cm/s` + `v_max: NNN cm/s`).
- Maquetación en `_compose_layout2`: la columna izquierda pasa de **2 zonas** (métricas/posesión
  arriba, feed abajo) a **3 zonas** cuando hay `vdata`: **posesión (arriba) · velocidad (medio) ·
  feed (abajo)** — el panel de velocidad queda **entre la posesión y la lista dinámica**, como se
  pidió. Sin `vdata` (flag off), se conserva el split de 2 zonas actual.
- `_compose_layout1`: análogo, el panel de velocidad se superpone como bloque adicional (opcional);
  prioridad visual en layout 2.
- El loop pasa `vdata = {"now": ball_speed_by_frame.get(fidx), "vmax": vmax}` (o `None` si flag off)
  a `compose(...)`; ambas firmas `_compose_layout{1,2}` ganan el parámetro `vdata`.

## Flujo de datos (resumen)

```
compute_metric_positions(homography="lines")  ──▶ metric (crudo)
        │
   use_kalman? ── no ──▶ metric  ─────────────────┐
        │ sí                                       │
   compute_kalman_states(metric)  ─▶ kres          │
   apply_kalman_to_metric(metric, kres) ─▶ metric  │ (balón refinado, H_por_frame intacto)
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
   shot/goal · zones · heatmap · _mini_by_frame · panel velocidad (kres)
                       ▼
              broadcast (layout 1|2)
```

## Validación

### Smoke test — `testing/test_kalman_position_source.py`
Sin GPU (CPU local, sobre un `MetricResult` de fixture o un JSON de T3 ya existente):
1. `refine_with_kalman(metric)` devuelve un `MetricResult` con `H_por_frame` y `resumen` **idénticos**.
2. Las posiciones de **robots/zonas** quedan **sin cambios**; las de **balón** cambian
   (xy_cm distinto y/o `source` no None).
3. Frames de oclusión del balón (original `xy_cm=None`) quedan **rellenados** con `source="predicted"`.
4. Con flag apagado (camino sin refinar), `event_shot_goal` produce **exactamente** el resultado de hoy
   (todas las posiciones `source=None`).

### Ablación — script en `notebooks/fase_6_kalman/`
Sobre `IMG_9933_5m30`: tabla/figura de **goles detectados** y **`v_max` del balón** con vs sin
Kalman (reusa `compute_shot_vs_goal` y `compute_kalman_states`). Salida a `assets/fase6/`.

### Notebook comparativo — `notebooks/fase_5_event_analysis/`
Renderiza y muestra: (a) `use_kalman=False`, (b) `use_kalman=True`, (c) `use_kalman=True` con
`layout=1` y `layout=2`. CPU local; consume el JSON de tracking + el clip **crudo**
(`00_prepare_clips.py`). Objetivo: que el equipo elija.

### Verificación en pod
Re-render del broadcast con `use_kalman=True` para revisión visual (estela de balón continua +
panel de velocidad) y confirmar que `use_kalman=False` es idéntico al actual.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Posición predicha dispara gol falso | Regla conservadora en `event_shot_goal` (gol exige ≥1 frame no-predicho) |
| Correr el KF dos veces (refinar + panel) | Una sola `compute_kalman_states`; `apply_kalman_to_metric` reusa `kres` |
| Romper llamadores de `MetricPosition` | Campo `source` opcional con default `None` (retrocompatible) |
| Robots degradados por CV | `ball_only=True` (robots intactos) |
| Cambiar el entregable validado | Flag **default-off**; con off, comportamiento bit-a-bit actual |

## Orden de implementación (se detalla en `tasks.md`)

1. Campo `source` en `MetricPosition` (+ serialización opcional).
2. `apply_kalman_to_metric` + `refine_with_kalman`.
3. Guarda conservadora en `event_shot_goal`.
4. Flag `use_kalman` + panel de velocidad en `event_broadcast_overlay` (layout 2 y 1).
5. Smoke test.
6. Ablación + notebook comparativo.
7. Re-render en pod.
