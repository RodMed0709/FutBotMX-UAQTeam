# plan.md — Consolidación de homografía actual en `src` + overlay de eventos + heatmap

> Paso 3 de la metodología SDD. Redacción técnica de **cómo** se implementa el `spec.md`.
> No se modifica código del proyecto en este paso.

---

## 1. Estado actual (punto de partida)

- **Homografía buena (por líneas):** `src.core.homography_multifeature.VideoHomographyLines`
  (`update(frame_bgr, carpet_mask, yc, bc) -> (H, status, overlap)`, con `stats()`). Demostrada en
  `notebooks/fase_4_homografia/v2_07_minimap_polish_cenital.ipynb` (`smooth_beta=0.7`).
- **Métrica T3:** `src.core.metric_positions.compute_metric_positions` proyecta foot points a cm,
  pero internamente `_solve_homographies` usa `auto_homography.VideoHomography.update_masks`
  (**camino viejo por máscaras**). Devuelve `MetricResult(posiciones, resumen)`.
- **Driver de referencia (notebook, intacto):** `notebooks/fase_6_kalman/cm_positions_lines.py`
  ya hace el reemplazo correcto: `VideoHomographyLines` + **resize del frame a la resolución de la
  carpet-mask** antes de `update`, leyendo carpet RLE + foot points del `tracks_json`.
- **Minimapa genérico:** `src.core.minimap.MinimapRenderer` (cancha 2D, trails, robot cuadro gris,
  `scale=2.6`, sin rotación).
- **Estilo cenital pulido (notebook, intacto):** en v2_07, `make_minimap` usa
  `field_template.render_field(scale=2.2, margin_cm=10.0)`, dibuja porterías anchas por color
  (amarilla x≈0–18, azul x≈225–243, boca y≈55–127), trails como líneas, posición actual como
  círculo lleno (robot 9px rojo, balón 7px naranja con borde blanco) y **rota 90° CW** para vista
  vertical. Además `draw_overlay` reproyecta rectángulo interior (verde) + círculo central (naranja)
  sobre el frame principal.
- **Heatmap:** `src.core.metric_heatmap.render_heatmap` sobre `field_template.render_field`
  (`scale=2.6`, `margin_cm=10`, `COLORMAP_JET`, **sin rotación**).
- **Overlay de eventos:** `src.core.event_broadcast_overlay.render_broadcast_overlay`
  - `metric = compute_metric_positions(tracks_json)` (→ camino viejo).
  - minimapa vía `MinimapRenderer` (genérico).
  - heatmap vía `render_heatmap` + estado incremental `_live_heatmap_state`/`_accumulate`
    (de `demo_overlay`).
  - compone con `_compose_layout1|2`, marcador, banner, feed, métricas.

---

## 2. Arquitectura de la solución

Cinco bloques, de menor a mayor dependencia:

```
(B1) metric_positions  ──► homografía por líneas (VideoHomographyLines + resize a carpet-res)
(B2) minimap           ──► renderer cenital pulido (estilo v2_07) reutilizable
(B3) metric_heatmap    ──► adecuar estilo (orientación/scale/margin = minimapa cenital)
(B4) event_broadcast_overlay ─► usa B1 + B2 + B3 (homografía embebida + heatmap coherente)
(B5) notebook fase_5   ──► entregable: reproduce el demo llamando a src (CPU local)
```

### B1 — `src/core/metric_positions.py`: backend de homografía por líneas

- Sustituir el solver por el **camino por líneas**, replicando la lógica probada de
  `cm_positions_lines.compute_cm_positions_lines`:
  - importar `VideoHomographyLines` de `homography_multifeature`;
  - por frame: si hay `carpet_rle`, decodificar, tomar `_largest_component`, **redimensionar el
    frame a `(wm, hm)`** de la máscara antes de `vh.update(frame_bgr, carpet, yc, bc)`; si no hay
    alfombra, conservar `vh.H` con status `"kept"`/`"none"`.
- **Compatibilidad:** mantener el camino viejo por máscaras como opción **no predeterminada**.
  Parámetro nuevo en `compute_metric_positions` y `_solve_homographies`:
  `homography: str = "lines"` con valores `"lines" | "masks"`. Validar y `raise ValueError` ante
  valor desconocido (antes de abrir el video).
- `resumen` añade `"homography": "lines" | "masks"` y, en modo líneas, `"lines_stats": vh.stats()`.
  Conservar las claves existentes que apliquen (`fps`, `n_frames`, `n_con_cm`, etc.).
- **Exponer la H por frame (decisión confirmada):** `MetricResult` gana un campo nuevo
  `H_por_frame: dict[int, np.ndarray | None]` (la homografía px→cm usada en cada frame), para que el
  overlay pueda reproyectar el campo sobre el video. `write_metric_positions_json` **no** serializa
  las matrices (se mantiene su salida actual; el campo es solo en memoria).
- `MetricPosition` y la salida de `write_metric_positions_json` no cambian su forma pública.

### B2 — `src/core/minimap.py`: renderer cenital pulido

- Añadir un renderer **nuevo** (sin romper `MinimapRenderer`), p. ej.
  `class CenitalMinimapRenderer` (o `MinimapRenderer(style="cenital")` si resulta más limpio sin
  duplicar). Parametrizado y **general** (cualquier video):
  - `__init__(scale=2.2, margin_cm=10.0, trail_len=40, robot_color=(255,0,0),
    ball_color=(255,140,0), goals=True, rotate="cw")`.
  - base con `field_template.render_field(scale, margin_cm)`.
  - `update(projected)` (misma firma que `MinimapRenderer.update`:
    `list[(obj_id, cls, x_cm, y_cm)]`), trails por `obj_id`.
  - `render()`: dibuja porterías anchas por color (parámetros de geometría tomados de
    `field_landmarks`/`field_template`, no hardcode disperso), trails (líneas) y posición actual
    (círculo lleno robot/balón con borde blanco), y aplica la **rotación vertical** (`ROTATE_90_CW`)
    al final. Devuelve RGB (como `MinimapRenderer.render`).
- Promover también la **reproyección del campo sobre el frame principal** (rect interior + círculo
  central) como helper reutilizable, p. ej. `draw_field_overlay_on_frame(frame, H)` en `minimap.py`
  apoyándose en `field_landmarks` (`LANDMARK_POINTS`, `CENTER_CIRCLE`). Es la "homografía embebida"
  que pide el spec; será **opcional** vía flag del overlay.

### B3 — `src/core/metric_heatmap.py`: adecuar al estilo cenital

- `render_heatmap` adopta los mismos parámetros visuales que B2: `scale=2.2`, `margin_cm=10.0` y la
  **misma rotación vertical** (`ROTATE_90_CW`) tras componer la densidad sobre el campo, para que
  minimapa y heatmap queden en la misma orientación y proporción.
- Mantener `COLORMAP_JET` y la mezcla alpha actuales (el cambio es de encuadre/orientación, no de
  semántica del mapa). Firma pública estable; añadir solo defaults coherentes con el minimapa
  (`scale`, `margin_cm`), de modo que el overlay pase los mismos valores a ambos.
- `_histogram`/`_smooth_normalize`/`compute_heatmaps` no cambian (siguen en cm sobre la cancha
  canónica).

### B4 — `src/core/event_broadcast_overlay.py`: cableado

- `compute_metric_positions(tracks_json)` → ahora usa líneas por defecto (sin cambios de llamada,
  el default `homography="lines"` lo cubre). Conservar el `try/except` → modo degradado.
- Reemplazar `MinimapRenderer(trail_len=trajectory_window)` por `CenitalMinimapRenderer(...)`.
- `render_heatmap(grid, bin_cm, sigma_cm=...)` → pasar `scale`/`margin_cm` coherentes con el
  minimapa cenital (para que B3 alinee).
- **Homografía embebida sobre el video (decisión confirmada: ON por defecto):** flag
  `draw_field_on_video: bool = True`. Por frame, reproyecta rectángulo interior + círculo central
  sobre el video principal con `minimap.draw_field_overlay_on_frame(vid, H)` usando
  `metric.H_por_frame.get(fidx)` (si la H es `None`, se omite ese frame, sin romper). En modo
  degradado el flag se ignora.
- `_compose_layout1|2` siguen colocando minimapa y heatmap en lados opuestos; verificar que el
  cambio de orientación (vertical) del minimapa/heatmap encaja en los paneles (ajustar `_fit_box`
  si hace falta, sin rediseñar el layout).

### B5 — Notebook entregable `notebooks/fase_5_event_analysis/`

- **Notebook nuevo** (no se tocan los existentes), p. ej. `03_broadcast_overlay_demo.ipynb`:
  - resuelve rutas vía `src.utils.get_abs_path` (config), tomando el `tracks_json` + `.mp4` del clip
    de referencia (IMG_9933, insumo `outputs/inference/.../IMG_9933_*.json`).
  - llama a `render_broadcast_overlay(...)` de `src` (CPU local).
  - muestra el PNG de muestra y/o el mp4 resultante para inspección visual.
  - documenta en una celda markdown los requisitos cubiertos (marcador, banner, métricas margen
    izq., feed con tope, minimapa+heatmap opuestos, márgenes) y que el render corre en local
    consumiendo el `tracks_json` generado en pod.

### Documentación

- Actualizar `notebooks/fase_4_homografia/context.md`: marcar que la métrica/overlay ya usan la
  homografía por líneas consolidada en `src`, sin citar `.specs/drafts/`.

---

## 3. Contratos y firmas (resumen)

| Símbolo | Cambio |
|---|---|
| `metric_positions.compute_metric_positions(..., homography="lines")` | nuevo kw-only, default `"lines"`; `"masks"` = legacy |
| `metric_positions._solve_homographies(..., homography)` | rama líneas (resize a carpet-res) / máscaras |
| `metric_positions.MetricResult.H_por_frame` | campo nuevo: `dict[int, np.ndarray\|None]` (solo en memoria) |
| `minimap.CenitalMinimapRenderer` | clase nueva; `update`/`render` compatibles con el flujo del overlay |
| `minimap.draw_field_overlay_on_frame(frame, H)` | nuevo helper (reproyección rect+círculo) |
| `metric_heatmap.render_heatmap(..., scale=2.2, margin_cm=10.0)` | defaults + rotación vertical |
| `event_broadcast_overlay.render_broadcast_overlay(..., draw_field_on_video=True)` | usa los anteriores; reproyección ON por defecto |

---

## 4. Convenciones y restricciones

- Lazy imports de `cv2`/`torch` dentro de funciones (estilo del repo).
- Rutas solo vía config/`get_abs_path`; nada hardcodeado (las constantes geométricas de porterías
  salen de `field_landmarks`/`field_template`).
- No tocar notebooks existentes ni el overlay base (`demo_overlay`/`track_overlay`).
- Sin GPU para el camino del overlay/notebook (CPU desde `tracks_json` + `.mp4`).
- Commits en inglés (Conventional Commits), solo tras confirmación.

---

## 5. Riesgos y mitigaciones

- **Orientación de paneles:** rotar minimapa/heatmap a vertical puede desbalancear `_compose_layout`.
  Mitigación: ajustar solo `_fit_box`/tamaños de panel, sin rediseñar el layout.
- **Resolución carpet vs `.mp4`:** ya resuelto por el resize a la resolución de la máscara (probado
  en `cm_positions_lines`); replicar exactamente esa secuencia.
- **Regresión del camino viejo:** se conserva `homography="masks"` para no romper consumidores
  existentes de `compute_metric_positions` (p. ej. otros eventos de Capa B).
- **Homografía embebida sobre el video:** si exponer H por frame complica el overlay, se entrega con
  el minimapa cenital como homografía embebida y el flag de reproyección sobre video OFF.

---

## 6. Estrategia de validación

- Smoke funcional (filosofía de tests del repo): correr `render_broadcast_overlay` sobre el clip de
  referencia en CPU local y verificar que produce mp4 + PNG sin excepción, en modo **no** degradado
  (`overlay_degradado=False`, `n_con_cm>0`).
- Inspección visual desde el notebook B5: minimapa cenital + heatmap coherentes y en lados opuestos;
  comparar contra el demo de v2_07.
- Verificar `homography="masks"` sigue funcionando (no regresión).
