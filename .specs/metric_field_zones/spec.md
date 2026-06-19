# Spec — `metric_field_zones` (T6, fase_5 · Capa B)

## Contexto

Sexta tarea de fase_5 y cuarta métrica de la **Capa B (cm)**. Divide la cancha canónica
(`field_template`) en **zonas en cm** y mide **presencia** (tiempo de cada categoría por zona) y
**posesión por zona** (combinando T1 con el balón en cm de T3). Resuelve el **"medio campo
real"** que en píxeles era impreciso (el viejo `W//2` partía la imagen a la mitad, sin
perspectiva). Solo aplica a **cámara superior**. Aporta a la convocatoria 3.7.

## Objetivo

Dado el JSON de tracking extendido de un clip de cámara superior, reportar, por esquema de zonas:
presencia del balón y de los robots por zona, y posesión por zona. CPU local.

## Requisitos funcionales

1. **Insumo**: ruta a un JSON de tracking extendido. Internamente usa T3
   (`compute_metric_positions`, posiciones cm) y T1 (`load_frame_objects` +
   `compute_possession`, poseedor por frame). No re-resuelve homografía.
2. **Esquemas de zonas** (en cm, de `field_template`):
   - **mitades**: amarilla (x < L/2) / azul (x ≥ L/2);
   - **tercios**: amarillo (x < L/3) / medio / azul (x ≥ 2L/3);
   - (opcional) **áreas** de penalti junto a cada portería (`PENALTY_*`).
   Por defecto: **mitades + tercios**. Posiciones fuera de cancha se clip-an.
3. **Asignación por frame**: cada posición (x,y) cae en exactamente una zona del esquema.
4. **Presencia** por zona = fracción de muestras (tiempo) de cada categoría (balón / robots) en
   esa zona.
5. **Posesión por zona** = de los frames con balón **controlado** (T1, `por_frame[f]` ≠ None), la
   zona del **balón** en cm (T3) en ese frame → % por zona.
6. **Salida (JSON nuevo)**: por esquema, `presencia` (ball/robot → {zona: %}) + `posesion`
   ({zona: %}) + resumen (esquemas, n_frames, frames con posesión, fps). No toca T1/T3.

## Visualización (en el test)

- Cancha con las **fronteras de zona** dibujadas + **% de presencia/posesión** rotulado por zona
  (sobre `field_template.render_field`).
- Resumen impreso.

## Fuera de alcance

- Equipos (presencia/posesión por categoría, no por bando).
- Heatmap (T5), velocidad (T4), gol (otra tarea), overlay/demo (T7).
- Re-resolver homografía/tracking.

## Criterios de aceptación

- Sobre `IMG_9933_5m30.json` produce el JSON de zonas + resumen + PNG, en CPU local sin GPU.
- Los porcentajes de presencia suman ~100% por categoría y esquema.
- En el clip del gol, presencia y posesión se inclinan hacia la **mitad/tercio azul** (donde
  ocurrió la jugada) — coherente con T5 (heatmap) y el gol geométrico.
- Casos borde: categoría/posesión sin muestras → 0% sin romper; esquema desconocido → error claro.
