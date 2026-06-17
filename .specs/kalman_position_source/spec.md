# Spec — `kalman_position_source` (SDD-B) — Kalman como fuente de posiciones desmontable

## Contexto

Fase 6 (`kalman_state_estimation`) construyó un filtro de Kalman explícito en cm que corre
**sobre** los tracks ya asociados (T3 `metric_positions`, homografía por líneas) y produce
posiciones refinadas + velocidad física + relleno de oclusión. Está **validado** (test 3/3,
ablación NIS) pero hoy **no está integrado** al entregable: ni `event_broadcast_overlay` ni
`metric_positions` consumen Kalman. El video de espectador usa posiciones crudas y cinemática
por diferencias finitas (T4).

El hallazgo honesto de fase 6 acota dónde aporta Kalman: **el balón** (objeto suave/balístico)
mejora con el modelo de velocidad constante (KF vence a la extrapolación lineal a través de
oclusiones); el **robot** (maniobra) **no** se beneficia (ninguna extrapolación con velocidad
vence a "hold"). Por eso el refinamiento es **solo balón**.

Punto clave verificado: la homografía **no se toca** — Kalman vive río abajo, sobre las
posiciones en cm que ya salieron de la homografía por líneas. Cambiar la fuente de posiciones
no altera la homografía. Además los formatos ya calzan: los detectores de cm consumen un
`MetricResult`, y `kalman_kinematics` ya puede emitir uno.

## Objetivo

Hacer de Kalman una **fuente de posiciones desmontable** para la capa de análisis: un **flag
`use_kalman` (default `False`)** que, en un **punto único de refinación**, sustituye las
posiciones del balón por las estimadas por Kalman, de modo que **todos los consumidores de cm**
(detección de gol/tiro, zonas, heatmap y el minimap del broadcast) puedan usar —o no— las
posiciones refinadas **sin cambiar su código**. Con el flag apagado, el entregable es **idéntico**
al actual validado; encenderlo es opt-in y reversible.

> No-overclaim: esto **no** mete Kalman al tracking (el tracker ya tiene el suyo en px) ni mejora
> la homografía. Es un **refinador de estado del balón, desacoplado y reversible por flag**, que
> alimenta la capa de eventos/visualización en cm.

## Qué se construye

### 1. Refinador único a nivel `MetricResult`

`refine_with_kalman(metric: MetricResult, *, ball_only: bool = True) -> MetricResult` (en
`src/core/kalman_kinematics.py`):

- Corre `compute_kalman_states` sobre el `metric` de entrada.
- **Reemplaza solo las posiciones del balón** (`orange_ball`/`ball`) por los estados Kalman
  (frames `measured` + `predicted`); robots/zonas pasan **sin tocar**.
- **Conserva `H_por_frame` y `resumen`** intactos (la homografía embebida y los metadatos siguen
  funcionando).
- Marca el origen de cada posición de balón (`measured` | `predicted` | `gated`) para que los
  consumidores estrictos puedan ser conservadores.
- Reutiliza `CLASS_PARAMS` calibrado de fase 6 (sin recalibrar).

### 2. Flag `use_kalman` en el broadcast y en los detectores de cm

- `render_broadcast_overlay(..., use_kalman: bool = False)`: cuando es `True`, el `metric` que se
  pasa a los consumidores es `refine_with_kalman(metric)`; cuando es `False`, comportamiento
  idéntico al actual.
- Consumidores cableados (todos reciben el `metric` ya elegido): minimap del broadcast
  (`CenitalMinimapRenderer`, **se conserva el render pulido**, solo cambia la fuente),
  `metric_field_zones`, `metric_heatmap`, `compute_shot_vs_goal` (route cm) y
  `compute_geometric_goals`.
- **Manejo de `predicted`:** minimap y heatmap **sí** usan posiciones predichas (estela continua);
  el detector estricto de gol **no** declara gol sobre un balón solo-predicho (conservador).

### 3. Panel de velocidad (Ball Speed | v_max)

Nuevo elemento de UI en `_compose_layout2`, **entre la lista dinámica (feed) y el panel de
posesión**, alimentado por `compute_kalman_states`: muestra `speed_cms` del balón y `v_max`.
**Solo aparece con `use_kalman=True`.**

### 4. Entregables de validación

- **Ablación** (tabla/figura): goles detectados y `v_max` del balón **con vs sin Kalman** sobre
  `IMG_9933_5m30`.
- **Notebook comparativo** en `notebooks/fase_5_event_analysis/`: renderiza (a) sin Kalman,
  (b) con Kalman, y (c) Kalman con los **dos layouts del broadcast** (layout 1 y 2), para que el
  equipo elija cuál prefiere.

## Alcance

**Dentro:** refinador único `refine_with_kalman`; flag `use_kalman` default-off en el broadcast y
los detectores de cm centrados en balón; panel de velocidad; ablación; notebook comparativo.

**Fuera (explícito):**
- `possession_refine` (trabaja en px, no en cm) y `field_violations` ("fuera" es de robots): **no**
  se alimentan de Kalman.
- **Tracking sin Kalman** (trayectorias = centroides crudos).
- **Overlays de segmentación/tracking más limpios** = SDD-A (otra tarea).
- **Elipse de incertidumbre** en el `CenitalMinimapRenderer` = SDD-C / post-entrega.
- El flag **no** entra al JSON de config (parámetro de función).

## Comportamiento esperado

- `use_kalman=False` (default): video **idéntico** al actual validado; sin panel de velocidad.
- `use_kalman=True`: estela del balón continua y suave en el minimap (pulido, mismo renderer);
  panel Ball Speed | v_max visible; gol/tiro/zonas/heatmap calculados sobre posiciones refinadas
  del balón, con el detector estricto conservador en frames predichos.

## Criterios de éxito

1. El flag on/off funciona sin romper el render ni la homografía embebida.
2. on/off produce videos distintos **solo** en lo esperado (balón refinado + panel de velocidad).
3. Ablación corrida y reportada sobre `IMG_9933_5m30`.
4. Notebook comparativo reproducible (sin/con Kalman + dos layouts).
5. Verificación: smoke test local del flag + re-render en el pod para revisión visual.

## Prioridad

Para la entrega del 19 el flag queda **default-off** (cero riesgo al entregable). Encender Kalman
por default queda condicionado al resultado de la ablación y a la preferencia del equipo en el
notebook comparativo.

## Relación con otras tareas

- Consume directamente `kalman_state_estimation` (fase 6): `kalman_state` + `kalman_kinematics`.
- Se apoya en la homografía por líneas consolidada (`metric_positions`, `homography="lines"`).
- Complementa, no incluye: SDD-A (overlays seg/tracking) y SDD-C (elipse de incertidumbre).
