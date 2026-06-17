# spec.md — Consolidación de homografía actual en `src` + overlay de eventos + heatmap

> Paso 2 de la metodología SDD. Describe **qué** se quiere construir y **por qué**.
> No se modifica código del proyecto en este paso.

---

## 1. Contexto y problema

La fase de homografía evolucionó en varios caminos. El camino que hoy da los mejores
resultados es la **homografía por líneas** (`src.core.homography_multifeature.VideoHomographyLines`,
demostrada en el notebook `notebooks/fase_4_homografia/v2_07_minimap_polish_cenital.ipynb`):
detecta las líneas blancas internas de la cancha, normaliza la orientación por color de
portería y produce posiciones métricas con error de **~9–23 cm** en el clip cenital fiable
(IMG_9933). Sobre ella, v2_07 dibuja un **minimapa cenital pulido** (máscara `green_floor`
translúcida, cajas YOLO por clase, IDs del tracker, overlay de rectángulo + círculo, y un
minimapa vertical con porterías anchas, robot en rojo / balón en naranja y estelas).

Sin embargo, el **overlay de eventos** entregado en la ronda anterior
(`src.core.event_broadcast_overlay`) **no usa esta homografía**:

- Obtiene su métrica de `src.core.metric_positions.compute_metric_positions`, que internamente
  usa el camino **viejo por máscaras** (`src.core.auto_homography.VideoHomography`) — el mismo
  que la bitácora de fase_6 documenta como inestable (size-mismatch a distintas resoluciones)
  y de peor calidad.
- Su minimapa embebido usa el `MinimapRenderer` genérico, **no** el estilo cenital pulido de v2_07.
- El **heatmap** (`src.core.metric_heatmap.render_heatmap`) cuelga de esa misma métrica vieja y
  no comparte el estilo visual del minimapa nuevo.

Resultado: el entregable visual final está desincronizado de la mejor homografía disponible, y la
lógica que produce el demo bueno vive solo en un notebook (no reutilizable, no aplicable a otros
videos desde `src`).

---

## 2. Objetivo

Lograr que el **código de `src`** sea capaz de generar, **sobre cualquier video**, el overlay de
eventos adoptando la **homografía actual** (por líneas, estilo cenital de v2_07), incluyendo su
**minimapa embebido** y su **heatmap** coherentes con ese estilo. El demo de v2_07 (clip IMG_9933)
es el **punto de referencia de inspección visual**, no el fin en sí mismo.

El **entregable final** es un **notebook nuevo** en `notebooks/fase_5_event_analysis/` que, llamando
únicamente a funciones de `src`, reproduce el video demo de referencia y produce el overlay de
eventos completo según los requisitos del draft.

---

## 3. Historias de usuario

### HU-1 — La métrica del overlay usa la homografía actual
**Como** responsable del análisis de partidos,
**quiero** que el overlay de eventos calcule las posiciones métricas con la homografía por líneas
(`VideoHomographyLines`) en vez del camino viejo por máscaras,
**para** que el minimapa y las métricas reflejen la mejor precisión disponible (~9–23 cm) y no se
rompan por size-mismatch de resolución.

**Criterios de aceptación**
- El cálculo métrico que alimenta el overlay usa `VideoHomographyLines`, reconstruida en **CPU**
  desde el `tracks_json` (carpet RLE + cajas/centroides) + el `.mp4`.
- Se aplica el **resize del frame a la resolución de la máscara de alfombra** antes de estimar la
  homografía, de modo que foot points (en esa resolución) y H queden alineados.
- En frames sin homografía fiable se **propaga la H previa** (`kept`); los objetos sin H se reportan
  sin coordenadas en cm (comportamiento degradado actual, conservado).
- El camino viejo por máscaras se conserva como opción/legacy, sin ser el predeterminado del overlay.

### HU-2 — El minimapa embebido adopta el estilo cenital pulido
**Como** espectador del overlay,
**quiero** ver el minimapa con el estilo cenital de v2_07 (cancha vertical, porterías anchas por
color, robot en rojo / balón en naranja, con estelas),
**para** entender el flujo de juego con la misma calidad visual del demo.

**Criterios de aceptación**
- El estilo de render cenital de v2_07 vive en `src` como función(es) **general(es) y parametrizada(s)**
  (no atadas al demo), reutilizable(s) por el overlay y por el notebook entregable.
- El minimapa embebido en el overlay de eventos usa ese estilo.

### HU-3 — El heatmap se adecúa al nuevo estilo del minimapa
**Como** espectador del overlay,
**quiero** que el heatmap se muestre con la misma cancha y orientación que el minimapa cenital,
**para** que ambas visualizaciones se lean como un conjunto coherente.

**Criterios de aceptación**
- `render_heatmap` (o su sucesor) usa la **misma cancha canónica, orientación/rotación, `scale` y
  `margin_cm`** que el minimapa cenital nuevo.
- En el overlay de eventos, minimapa y heatmap comparten estilo y se ubican juntos en el **margen
  derecho** (minimapa arriba, heatmap abajo), en el lado **opuesto a las métricas/eventos** del
  margen izquierdo — se conserva el layout de la ronda anterior, no se rediseña.

### HU-4 — `src` reproduce el demo y generaliza a cualquier video
**Como** desarrollador,
**quiero** funciones en `src` que produzcan este overlay sobre **cualquier** video parametrizando
las entradas (video + `tracks_json`),
**para** no depender del código del notebook y poder analizar nuevos partidos.

**Criterios de aceptación**
- Las funciones de `src` reciben el video y el `tracks_json` como parámetros (rutas vía config /
  `get_abs_path`), sin rutas hardcodeadas.
- Ejecutadas sobre el clip de referencia (IMG_9933), producen un overlay **visualmente equivalente**
  al demo de v2_07 (no se exige igualdad bit-for-bit).

### HU-5 — Notebook entregable de inspección visual
**Como** equipo,
**quiero** un notebook nuevo en `fase_5_event_analysis` que, llamando a `src`, reproduzca el video
demo de referencia y muestre el overlay de eventos completo,
**para** validar visualmente el resultado e iterar.

**Criterios de aceptación** — el notebook produce un overlay que incluye:
- **Márgenes** alrededor del video para que salga completo.
- **Marcador 0-0** inicial (cada lado con el color de su portería) que se actualiza en cada gol.
- **Banner “¡Goool! Portería {color}”** que se desliza de izquierda a derecha al ocurrir un gol.
- **Métricas de posesión/control** legibles en el **margen izquierdo**, con etiquetas claras.
- **Lista dinámica de eventos** (tiro a gol, fuera, etc.) con **tope de elementos**; los nuevos
  desplazan a los anteriores.
- **Minimapa** (estilo cenital nuevo) y **heatmap** en **lados opuestos**.
- El notebook corre el **render en CPU local** consumiendo `tracks_json` + `.mp4` (la detección
  pesada SAM3/YOLO que genera el `tracks_json` es trabajo de pod previo).

---

## 4. Fuera de alcance

- **No** se modifican los notebooks existentes (incluido `v2_07`, que queda como referencia) salvo
  petición explícita.
- **No** se reescribe el overlay base previo (`demo_overlay` / `track_overlay`); el trabajo se
  concentra en el broadcast/events overlay, el minimapa y el heatmap.
- **No** entran los puntos 1 (notebook genérico comparativo de overlays) ni 2 (estrategia pod/local)
  de “Novedades” como tareas separadas: aquí solo el notebook entregable de este overlay.
- **No** se cambia la estrategia de detección/tracking ni el fine-tuning de YOLO.
- Métricas cuantitativas adicionales y `cv2.undistort` (barril) quedan fuera.

---

## 5. Restricciones (constitución)

- Rutas relativas vía archivo de configuración y `src/utils.py::get_abs_path`; nada hardcodeado.
- Lazy imports de `cv2` / `torch` / etc. dentro de funciones (estilo del repo).
- Trabajo y documentos en **español**; commits en inglés siguiendo Conventional Commits.
- No se commitea ni pushea sin confirmación explícita.
- Esta tarea atómica vive en `.specs/homography_consolidation/` con sus `spec.md`, `plan.md`,
  `tasks.md`.

---

## 6. Criterios de aceptación globales

1. El overlay de eventos generado desde `src` usa la homografía por líneas para métrica, minimapa
   embebido y heatmap.
2. El minimapa y el heatmap comparten el estilo cenital nuevo (margen derecho, opuesto a
   métricas/eventos); se conserva el layout de la ronda anterior.
3. Existe un notebook nuevo en `fase_5_event_analysis` que reproduce el demo de referencia con todos
   los elementos del overlay listados en HU-5, corriendo en CPU local desde `tracks_json` + `.mp4`.
4. El código de `src` es general (aplica a cualquier video) y respeta las restricciones de la §5.
5. La documentación de estado de fase_4 (`context.md`) se actualiza marcando la consolidación, sin
   citar borradores de `.specs/drafts/`.
