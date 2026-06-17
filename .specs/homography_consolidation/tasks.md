# tasks.md — Consolidación de homografía actual en `src` + overlay de eventos + heatmap

> Paso 4 de la metodología SDD. Lista de tareas ejecutables derivadas del `plan.md`.
> La implementación (paso 5) recién empieza tras aprobar este archivo.

Orden por dependencias: T1 → (T2, T3 en paralelo) → T4 → T5 → T6.

---

## T1 — `metric_positions.py`: backend de homografía por líneas + H por frame

- [ ] T1.1 Importar `VideoHomographyLines` de `homography_multifeature`.
- [ ] T1.2 Añadir `homography: str = "lines"` (kw-only) a `compute_metric_positions` y propagarlo a
  `_solve_homographies`. Validar `homography in {"lines", "masks"}` y `raise ValueError` antes de
  abrir el video.
- [ ] T1.3 En `_solve_homographies`, rama `"lines"`: por frame con `carpet_rle`, decodificar
  (`decode_rle`) + `_largest_component`, **redimensionar el frame a `(wm, hm)`** de la máscara y
  llamar `vh.update(frame_bgr, carpet, yc, bc)`; sin alfombra → conservar `vh.H` con status
  `"kept"`/`"none"`. Rama `"masks"`: comportamiento actual (`VideoHomography.update_masks`).
- [ ] T1.4 Añadir el campo `H_por_frame: dict[int, np.ndarray | None]` a `MetricResult` y poblarlo
  desde el solver (default seguro `None` cuando no hay H).
- [ ] T1.5 `resumen`: añadir `"homography"` y, en modo líneas, `"lines_stats": vh.stats()`;
  conservar claves existentes que apliquen.
- [ ] T1.6 Verificar que `write_metric_positions_json` **no** serializa matrices (salida pública
  intacta).
- [ ] T1.7 Actualizar el docstring del módulo (hoy dice "camino C, auto_homography.VideoHomography").

## T2 — `minimap.py`: renderer cenital + reproyección sobre frame

- [ ] T2.1 `class CenitalMinimapRenderer` con `__init__(scale=2.2, margin_cm=10.0, trail_len=40,
  robot_color=(255,0,0), ball_color=(255,140,0), goals=True, rotate="cw")`; base con
  `field_template.render_field`.
- [ ] T2.2 `update(projected: list[(obj_id, cls, x_cm, y_cm)])` → trails por `obj_id` (deque).
- [ ] T2.3 `render()` → porterías anchas por color (geometría desde `field_landmarks`/
  `field_template`, sin hardcode disperso), trails (líneas), posición actual (círculo lleno
  robot/balón con borde blanco), y rotación vertical final (`ROTATE_90_CLOCKWISE`). Devuelve RGB.
- [ ] T2.4 `draw_field_overlay_on_frame(frame, H)`: reproyecta rectángulo interior + círculo central
  (vía `field_landmarks.LANDMARK_POINTS` / `CENTER_CIRCLE`, `H` inversa). No-op si `H is None`.
- [ ] T2.5 No modificar `MinimapRenderer` ni `draw_field_overlay` existentes.

## T3 — `metric_heatmap.py`: adecuar al estilo cenital

- [ ] T3.1 `render_heatmap`: defaults `scale=2.2`, `margin_cm=10.0`; aplicar rotación vertical
  (`ROTATE_90_CLOCKWISE`) tras componer la densidad, para igualar orientación/proporción del
  minimapa cenital.
- [ ] T3.2 Mantener `COLORMAP_JET` + mezcla alpha; no tocar `_histogram`/`_smooth_normalize`/
  `compute_heatmaps`.

## T4 — `event_broadcast_overlay.py`: cableado de B1+B2+B3

- [ ] T4.1 Sustituir `MinimapRenderer(trail_len=trajectory_window)` por `CenitalMinimapRenderer(...)`.
- [ ] T4.2 Pasar `scale`/`margin_cm` coherentes a `render_heatmap` (igual que el minimapa cenital).
- [ ] T4.3 Añadir `draw_field_on_video: bool = True` a `render_broadcast_overlay`; por frame (no
  degradado) llamar `draw_field_overlay_on_frame(vid, metric.H_por_frame.get(fidx))`.
- [ ] T4.4 Ajustar `_fit_box`/tamaños de panel si la orientación vertical del minimapa/heatmap lo
  requiere, sin rediseñar `_compose_layout1|2`.
- [ ] T4.5 Confirmar que el modo degradado ignora el flag y no rompe.

## T5 — Notebook entregable `notebooks/fase_5_event_analysis/03_broadcast_overlay_demo.ipynb`

- [ ] T5.1 Crear notebook **nuevo** (no tocar los existentes) que resuelva rutas vía `get_abs_path`
  y tome el `tracks_json` + `.mp4` del clip de referencia (IMG_9933).
- [ ] T5.2 Llamar `render_broadcast_overlay(...)` de `src` (CPU local) y mostrar PNG de muestra y/o
  el mp4 resultante.
- [ ] T5.3 Celda markdown: requisitos cubiertos (marcador por color + actualización en gol, banner
  deslizante, métricas en margen izquierdo, feed con tope, minimapa + heatmap en lados opuestos,
  márgenes del video) y nota pod/local (tracks_json de pod, render en local).

## T6 — Documentación y validación

- [ ] T6.1 Actualizar `notebooks/fase_4_homografia/context.md`: métrica/overlay ya usan la
  homografía por líneas consolidada en `src` (sin citar `.specs/drafts/`).
- [ ] T6.2 Smoke: `render_broadcast_overlay` sobre el clip de referencia produce mp4 + PNG sin
  excepción, `overlay_degradado=False`, `n_con_cm>0`.
- [ ] T6.3 Inspección visual desde el notebook B5 contra el demo de v2_07 (minimapa cenital +
  heatmap coherentes y en lados opuestos; reproyección alineada con la cancha real).
- [ ] T6.4 No-regresión: `compute_metric_positions(..., homography="masks")` sigue funcionando.

---

## Notas de ejecución

- Lazy imports (`cv2`, etc.); rutas por config/`get_abs_path`; nada hardcodeado.
- No commitear sin confirmación (mensaje en inglés, Conventional Commits).
- Lo que necesita SAM3/GPU (generar el `tracks_json`) es trabajo previo de pod; T2–T6 corren en
  **CPU local** desde `tracks_json` + `.mp4`.
