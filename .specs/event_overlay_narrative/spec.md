# Spec — `event_overlay_narrative` (T7, fase_5) — video demo

## Contexto

Tarea **final** de fase_5: ensamblar el **video demo** de la convocatoria (3.5.3) a partir de
piezas que **ya existen** (segmentación, tracking, minimap) + las métricas en cm de la Capa B
(T1/T4/T5/T6 + gol geométrico). No introduce capacidades nuevas de visión: **compone y rotula**.
Refuerza 3.5.2 (visualización) y 3.7 (métricas Profesional). Corre en **CPU local** (todo viene
del JSON; no re-infiere).

## Objetivo

Producir un **mp4 ≤ 2 min** que muestre, sobre el clip de cámara superior (`IMG_9933_5m30`):
original + segmentación + tracking + minimap, con un **panel de métricas** (posesión, velocidad,
distancia, zona) y los **goles anotados**. El código vive en `src/core/` (empaquetado,
reusable); el harness en `testing/`.

## Requisitos funcionales

1. **Insumos**: el clip de cámara superior + su **JSON de tracking extendido** (mode=tracking,
   con `rle` y `frames[]`) + las salidas de T1/T4/T6/gol geométrico (se computan en local desde
   el JSON). No re-infiere modelos ni usa GPU.
2. **Frame combinado** (layout por defecto): en cada frame del clip se compone una imagen con
   - **original** y **segmentación** (máscaras por clase) lado a lado (cumple el OBLIGATORIO de
     3.5.3: original junto a segmentado);
   - **tracking** visible (caja + `nombre #id` + estela) — sobre el segmentado o como tercer panel;
   - **minimap** (trails sobre la cancha en cm);
   - **heatmap en vivo** (T5): densidad de ocupación que **se acumula frame a frame** sobre la
     **misma cancha canónica** que el minimap (reusa `metric_heatmap.render_heatmap`), de modo
     que minimap (trails) y heatmap (densidad) quedan como dos vistas de cancha contiguas;
   - **panel de métricas** (texto): posesión por `obj_id`, velocidad cm/s y distancia (T4), zona
     dominante (T6).
3. **Reusa** los overlays existentes (no los reimplementa): `overlay.overlay_detections`
   (segmentación), `track_overlay.render_obj_id_overlay` (tracking), `minimap`/
   `render_minimap_video` (minimap). El módulo de T7 los **compone**.
4. **Eventos de gol**: cuando el frame cae en un evento del gol geométrico, mostrar un **banner/
   flash** ("GOL · portería azul") durante el evento.
5. **Rótulos**: etiqueta por panel ("Original", "Segmentación", "Tracking", "Minimap",
   "Métricas") + título del demo. La explicación por **texto en pantalla** la pone el código.
6. **Salida**: mp4 ≤ 2 min al fps del clip, escrito con el video writer del repo
   (`open_video_writer`/`write_video`). Ubicación bajo `outputs/`.

## Visualización

- ES la visualización: el test produce el mp4 y un **frame de muestra** (`.png`) para revisión
  rápida sin reproducir el video.

## Fuera de alcance (PRODUCCIÓN, no código)

- Grabar/editar la **explicación o narración por voz**.
- Montar el **reel de Instagram ≥ 30 s** y poner el **link en el README**.
- Selección/edición fina de cortes para la versión final de jurado.
Estas quedan como **tareas manuales** listadas en `tasks.md`, separadas del código.

## Fuera de alcance (código)

- Capacidades nuevas de visión/segmentación/tracking (todo reusa lo existente).
- Re-inferir o re-resolver homografía (viene del JSON / T3).
- Asignación de equipos.

## Criterios de aceptación

- Sobre `IMG_9933_5m30` produce un mp4 ≤ 2 min, en CPU local sin GPU, que muestra original +
  segmentación + tracking + minimap + panel de métricas, con rótulos legibles.
- Los **goles** del gol geométrico aparecen anotados en sus frames.
- Las métricas del panel coinciden con las de T1/T4/T6 (mismos números).
- El test escribe el mp4 + un frame de muestra `.png` y no rompe.
