# Spec — `event_goal_geometric` (gol geométrico, fase_5 · Capa B)

## Contexto

Refinamiento en cm del detector de gol. T2 (`event_goal_zone`, Capa A) marca "candidato a gol"
en píxeles cuando el balón entra al **bbox** de una zona de portería — robusto pero impreciso
(el bbox de la zona no es la línea de gol). Con las **posiciones en cm** de T3
(`metric_positions`), el gol geométrico usa la **línea de gol real** y la **boca** sobre la
cancha canónica (`field_template`), dando un evento más preciso. Solo aplica a **cámara
superior** (donde T3 tiene posiciones fiables). Alimenta la convocatoria 3.7 / 3.5.3 (evento de
gol claro en el demo).

## Objetivo

Dado el JSON de tracking extendido de un clip de cámara superior (o un `MetricResult` de T3),
detectar los **goles geométricos**: el balón cruzando la línea de gol dentro de la boca, por
portería (amarilla / azul), con debounce/cooldown para no contar parpadeos. Corre en **CPU
local**.

## Requisitos funcionales

1. **Insumo**: ruta a un JSON de tracking extendido (se invoca `compute_metric_positions`
   internamente) **o** un `MetricResult` de T3. Solo se usa el **balón** (`orange_ball`/`ball`).
   No re-resuelve homografía ni re-infiere.
2. **Geometría** (de `field_template`, sin redefinir):
   - línea de gol **amarilla**: x ≤ `GOAL_LINE_X_LEFT_CM` (12 cm);
   - línea de gol **azul**: x ≥ `GOAL_LINE_X_RIGHT_CM` (231 cm);
   - **boca**: y ∈ [`_GOAL_TOP_Y_CM`, `_GOAL_BOTTOM_Y_CM`] = [61, 121] cm.
   - margen configurable (`margin_cm`) que ensancha la región (tolerancia al ruido de H).
3. **"Balón en gol" por frame**: el balón con x más allá de la línea (con margen) **y** y dentro
   de la boca (con margen). Si hay varias muestras del balón en el frame (ID-switch), **basta
   una** dentro de la región.
4. **Evento de gol**: secuencia sostenida del booleano por frame, vía la **máquina de estados
   `event_goals._events_from_series`** (debounce `min_frames`, cierre `exit_frames`,
   `cooldown_frames`). Una secuencia por portería.
5. **Salida (JSON nuevo, no toca T2/T3)**: lista de eventos
   `{zona: "yellow"|"blue", frame_inicio, frame_fin, t_s, xy_cm}` (xy_cm = posición del balón al
   inicio del evento) + resumen (`eventos_por_zona`, total, fps, params, `zonas_evaluadas`).

## Visualización (en el test)

- Línea de tiempo de eventos por portería (estilo T2).
- Marca de la posición de entrada del balón (`xy_cm`) sobre la cancha canónica
  (`field_template.render_field`).
- Resumen impreso (conteo por portería, params).

## Fuera de alcance

- Atribución de equipo (quién marcó): hoy los robots no tienen bando (futuro DINOv3).
- Validez reglamentaria / arbitraje: las cifras son indicativas.
- Heatmap (T5), zonas del campo (T6), overlay/narrativa (T7).
- Re-resolver homografía o tracking (T3 aguas arriba; tracking no se toca).

## Criterios de aceptación

- Sobre `IMG_9933_5m30.json` produce los eventos de gol en cm + resumen en CPU local, sin GPU.
- Detecta el/los gol(es) en la **portería azul** del clip (el balón cruza x≈231–237 con y en la
  boca, ≈ frames 840 y 1210) — coherente con T2 y con la verificación visual previa.
- Más preciso que T2: el evento se ancla a la línea real en cm, no al bbox de la zona.
- Casos borde manejados: balón nunca en la boca → 0 eventos; balón rozando la línea fuera de la
  boca (y fuera de [61,121]) → no cuenta; secuencia sostenida = 1 evento (no N parpadeos).
