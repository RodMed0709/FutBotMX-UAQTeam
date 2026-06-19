# Spec — `metric_heatmap` (T5, fase_5 · Capa B)

## Contexto

Quinta tarea de fase_5 y tercera métrica de la **Capa B (cm)**. Sobre las **posiciones en cm**
de T3 (`metric_positions`), acumula la ocupación del balón y de los robots en una rejilla de la
cancha canónica (`field_template`, 243×182 cm) y produce un **mapa de calor** superpuesto sobre
el campo. Refuerza la convocatoria 3.5.2 (visualización: ocupación / "flujo de juego"). Solo
aplica a **cámara superior** (donde T3 tiene posiciones fiables). En píxeles no sería útil (las
cámaras se mueven); el heatmap significativo es el **métrico**.

## Objetivo

Dado el JSON de tracking extendido de un clip de cámara superior (o un `MetricResult` de T3),
generar el heatmap de **balón** y de **robots (agregado)** sobre la cancha, en **CPU local**.

## Requisitos funcionales

1. **Insumo**: ruta a un JSON de tracking extendido (se invoca `compute_metric_positions`) **o**
   un `MetricResult`. Usa solo las posiciones con `xy_cm` válido. No re-resuelve homografía.
2. **Rejilla** sobre la cancha (0..`LENGTH_CM` × 0..`WIDTH_CM`) con tamaño de celda configurable
   (`bin_cm`, default ~5 cm). Histograma 2D de posiciones por categoría.
3. **Categorías agregadas**: un mapa para el **balón** (`orange_ball`/`ball`) y otro para los
   **robots** (todos los `robot` juntos, no por `obj_id`).
4. **Suavizado** gaussiano de la rejilla (`sigma_cm` configurable) + **normalización** (0..1 por
   el máximo) para colorear de forma estable.
5. **Recorte**: posiciones fuera de los límites de la cancha se **clip**an al rango de la rejilla
   (no crean celdas fuera del campo).
6. **Salida**:
   - PNG por categoría: heatmap coloreado (colormap con transparencia) **superpuesto sobre la
     cancha** (`render_field`);
   - opcional: la matriz de densidad por categoría (`.npz`) + un resumen JSON (nº muestras, celda
     más visitada en cm, % de celdas ocupadas, params). No toca T3/T4.

## Visualización (en el test)

- PNG(s) del heatmap sobre la cancha canónica (balón y robots).
- Resumen impreso (muestras por categoría, celda pico en cm, % ocupación, params).

## Fuera de alcance

- Heatmap por `obj_id` individual o por equipo (agregado por categoría; equipos = futuro).
- Zonas del campo / posesión por zona (T6).
- Velocidad/distancia (T4), gol (otra tarea), overlay/narrativa de video (T7).
- Heatmap en píxeles (no útil con cámara móvil) o re-resolver homografía/tracking.

## Criterios de aceptación

- Sobre `IMG_9933_5m30.json` produce los PNG de heatmap (balón y robots) + resumen, en CPU
  local sin GPU.
- El heatmap del **balón** concentra densidad hacia la **portería azul** (donde ocurrió la
  jugada de gol del clip) — coherente con T2/gol geométrico y la verificación visual.
- El heatmap de **robots** muestra la ocupación del lado donde se concentró el juego.
- Casos borde: categoría sin muestras → mapa vacío sin romper; `bin_cm`/`sigma_cm` configurables
  no rompen con valores razonables.
