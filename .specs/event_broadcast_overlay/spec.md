# Spec — Overlay de espectador (`event_broadcast_overlay`)

- **Tarea atómica:** `event_broadcast_overlay`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Ronda de entregable de eventos (refinar detección +
  overlay de espectador) sobre la fase_5 ya completa. Es el **cierre visual** de la ronda.
- **Depende de:** `event_shot_goal` (gol/tiro estricto), `event_goal_geometric` (gol
  geométrico, alterno), `event_possession_refine` (posesión/control),
  `event_field_violations` (fuera/lack/pushing), `metric_positions`/`metric_heatmap`/`minimap`
  (homografía, heatmap, minimapa) y `events_output_paths`.
- **Habilita:** la entrega final (convocatoria) — un video vistoso y autoexplicativo del
  partido.

---

## 1. Requisito (historia de usuario)

> **Como** equipo que presenta el proyecto,
> **quiero** un **video de espectador** que muestre el partido con marcador, banner de gol,
> minimapa, heatmap, métricas de posesión/control y una lista de eventos,
> **para** que cualquiera entienda lo que pasa sin conocer el pipeline.

---

## 2. Motivación (por qué)

- **Lo detectado no se ve.** Ya hay goles/tiros, posesión/control y violaciones de campo, pero
  viven en JSONs y vizs de validación. Falta un **producto visual** único que los integre.
- **El overlay viejo es de depuración.** `demo_overlay` (mosaico T7) y `track_overlay`
  (cajas+id) sirven para inspección; no son un entregable de espectador. Este overlay es el
  **showpiece** y los deja intactos para sus usos.
- **Configurable por contexto.** Se necesita poder elegir la **disposición** (dos layouts) y la
  **fuente del gol** (estricto vs geométrico) sin reescribir nada.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Módulo nuevo** `src/core/event_broadcast_overlay.py` que renderiza un **mp4 de espectador**
  componiendo el video del partido dentro de un lienzo con **márgenes** más paneles.
- **Parámetros clave:**
  - `layout ∈ {1, 2}` (default **2** = paneles laterales). Layout 1 = variante con paneles
    superpuestos en esquinas sobre el video. Mismas piezas, distinta disposición.
  - `goal_source ∈ {"strict", "geometric"}` (default **strict** = `event_shot_goal`; geometric
    = `event_goal_geometric`).
- **Marcador**: dos contadores, uno por portería, en el **color de cada portería**; arranca
  `0-0`; un gol en `zona` incrementa el contador de esa portería; acumulado por frame.
- **Banner de gol**: al ocurrir un gol, "¡Goool! Portería {color}" se **desliza de izquierda a
  derecha** sobre el video durante ~`banner_secs`.
- **Minimapa** (cenital con estela de `minimap.MinimapRenderer`) y **heatmap** (acumulado,
  `metric_heatmap`) en **lados opuestos** dentro de los márgenes.
- **Panel de métricas** (margen izquierdo): posesión y control (`event_possession_refine`) con
  etiquetas legibles.
- **Lista dinámica de eventos**: tiros, fueras, lack-of-progress, pushing conforme ocurren;
  tope `max_items`; los nuevos desplazan a los viejos.
- **Render incremental** frame a frame (sin cargar todo en RAM), con barra `tqdm`.
- **Salida** vía `events_paths(stem, "broadcast", "mp4")` + un PNG de muestra (frame con gol).

### 3.2 Fuera de alcance

- **No** modifica `demo_overlay` ni `track_overlay` (se conservan para mosaico/depuración).
- **No** introduce equipos/bandos (marcador por portería, no por equipo).
- **No** re-infiere (SAM3/YOLO): lee el `tracks_json` y recalcula homografía como
  `metric_positions`.
- **No** corre en GPU.
- **Sin homografía fiable** ⇒ modo **degradado**: omite minimapa/heatmap y conserva
  marcador/lista/posesión (en px); se anota en el resultado.

---

## 4. Comportamiento esperado

- Con `goal_source="strict"` sobre `IMG_9933_5m30`, el marcador termina **azul 1** (el único
  gol real); con `goal_source="geometric"`, **azul 3** (los 3 del gol laxo).
- Cuando ocurre el gol, el banner "¡Goool! Portería azul" cruza el video y el marcador azul
  sube en ese momento.
- El minimapa muestra robots/balón proyectados con estela; el heatmap se va acumulando; ambos
  en lados opuestos.
- El panel de métricas muestra posesión y control actuales/acumulados; la lista de eventos va
  mostrando tiros/fueras/etc. con tope `max_items`.
- `layout=1` y `layout=2` producen el mismo contenido en distinta disposición.

---

## 5. Criterios de aceptación

1. Existe `src/core/event_broadcast_overlay.py` con una función que genera el mp4 de espectador
   a partir de un `tracks_json`, con `layout` y `goal_source` configurables (defaults 2 /
   strict).
2. El video incluye: video del partido con márgenes, marcador por portería (color), banner de
   gol deslizante, minimapa con estela, heatmap acumulado en lado opuesto, panel de
   posesión/control y lista dinámica de eventos con tope.
3. El marcador sube solo con goles de la `goal_source` elegida (strict ⇒ 1 azul; geometric ⇒
   3 azul en el clip de referencia).
4. Render incremental (sin OOM) con barra de progreso; salida vía `events_paths`
   (`kind="broadcast"`), mp4 + PNG de muestra.
5. Modo degradado sin homografía (omite minimapa/heatmap, conserva el resto).
6. No se tocan `demo_overlay`/`track_overlay`.
7. Test manual sin GPU sobre `IMG_9933_5m30` (capado a N frames) que genera ambos layouts y
   exporta un PNG de muestra.

---

## 6. Notas / decisiones

- **Parámetros configurables** (defaults a fijar en el plan): `layout` (2), `goal_source`
  (strict), `banner_secs`, `max_items` (lista), `margin_px`, `out_fps`, `trajectory_window`
  (estela), `bin_cm`/`sigma_cm` (heatmap), cap de frames para test.
- **Marcador por portería** (no por equipo): un gol en `zona` incrementa esa portería; es la
  lectura directa de los eventos sin inventar bandos.
- **Cambio en el overlay viejo:** ninguno funcional; se documenta que `event_broadcast_overlay`
  es el overlay **de espectador** y `demo_overlay`/`track_overlay` quedan para mosaico y
  depuración (decisión pedida en el draft: "definir qué cambiará en el antiguo overlay" ⇒
  nada, solo el rol).
- **Pesado:** el test capa frames; el render completo se corre aparte (puede tardar).
- **Compatibilidad:** consume los módulos de eventos/minimapa/heatmap; no los modifica.
