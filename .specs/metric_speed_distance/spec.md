# Spec — `metric_speed_distance` (T4, fase_5 · Capa B)

## Contexto

Cuarta tarea de fase_5 y segunda de la **Capa B (métrica, cm)**. Consume las **posiciones en
cm** producidas por T3 (`metric_positions`) y deriva **velocidad (cm/s)** y **distancia
recorrida (cm)** por objeto. Solo aplica a video de **cámara superior** (donde T3 tiene
posiciones fiables). Es una de las **métricas cuantitativas** que pide la convocatoria 3.7
(Profesional).

## Objetivo

Dado el JSON de tracking extendido de un clip de cámara superior (o un `MetricResult` de T3),
calcular por `obj_id`: distancia total recorrida, velocidad media y máxima, y (opcional) la
serie de velocidad por frame. Con **suavizado** y **rechazo de saltos imposibles** para no
inflar las cifras por ID-switches del tracking o ruido de la homografía. Corre en **CPU local**.

## Requisitos funcionales

1. **Insumo**: la ruta a un JSON de tracking extendido (T4 invoca `compute_metric_positions`
   internamente) **o** un `MetricResult` ya calculado de T3. No re-resuelve homografía ni
   re-infiere.
2. **Serie por `obj_id`**: se ordenan las muestras con `xy_cm` válido por `frame_index`. Los
   frames sin cm (`status_H="init"` u objeto ausente) son **huecos** (no se interpolan a 0).
3. **dt entre muestras consecutivas** = `(Δframe) / fps`, usando el `fps` del JSON y el Δframe
   real (respeta huecos; no asume frames contiguos).
4. **Velocidad instantánea** = `‖Δxy_cm‖ / dt` (cm/s) entre muestras consecutivas.
   **Distancia recorrida** = suma de los segmentos `‖Δxy_cm‖` aceptados.
5. **Suavizado + rechazo de outliers**:
   - se descartan segmentos cuya velocidad supere un umbral físico configurable
     (`max_speed_cms`, default razonable para estos robots) — son teleports por ID-switch o
     ruido de H, no movimiento real;
   - la serie de velocidad se suaviza (media móvil/EMA, ventana configurable) para reportar
     `v_media`/`v_max` estables.
6. **Salida (JSON nuevo, no toca el de T3)**: por `obj_id`
   `{class, n_muestras, dur_s, dist_cm, v_media_cms, v_max_cms}` + (opcional) serie de
   velocidad por frame; + resumen (totales por clase, fps, parámetros usados, nº de segmentos
   descartados por outlier).

## Visualización (en el test)

- Curva de **velocidad cm/s por `obj_id`** a lo largo del tiempo (matplotlib).
- **Distancias totales** por `obj_id` (barras o tabla impresa).
- Resumen impreso (totales, parámetros, outliers descartados).

## Fuera de alcance

- Clasificación de equipos (aliado/rival) y atribución de eventos.
- Gol geométrico (otra tarea, refina T2 en cm).
- Heatmap (T5), zonas del campo (T6), overlay/narrativa de video (T7).
- Re-resolver homografía o re-inferir modelos (lo hace T3 aguas arriba).

## Criterios de aceptación

- Sobre `IMG_9933_5m30.json` produce el JSON de métricas + resumen en CPU local, sin GPU.
- Las velocidades quedan en rango físico tras el rechazo de outliers (sin valores absurdos por
  ID-switch); la distancia total por robot es del orden esperable para ~1 min de juego.
- El balón presenta picos de velocidad mayores que los robots (tiros) — coherencia cualitativa.
- Casos borde manejados: `obj_id` con <2 muestras (distancia 0, velocidad no definida → se
  reporta 0/None sin romper); todos los segmentos de un objeto descartados como outliers;
  `fps` ausente (se exige o se documenta el fallo claro).
- Las cifras se rotulan como **indicativas** (limitadas por tracking/H), no absolutas.
