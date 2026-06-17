# Tasks — Overlay de espectador (`event_broadcast_overlay`)

- **Tarea atómica:** `event_broadcast_overlay`
- **Paso de la metodología:** 4 (Descomposición en tareas) → habilita el paso 5
  (implementación).
- **Spec/plan de referencia:** `.specs/event_broadcast_overlay/{spec,plan}.md`.

> A partir de aquí (y **solo** aquí) se autoriza escribir/modificar código.

---

## T1 · Estructuras y precómputo de eventos

- [x] Crear `src/core/event_broadcast_overlay.py` con `BroadcastResult` (video, sample_png,
      resumen).
- [x] Precómputo: goles según `goal_source` (`compute_shot_vs_goal` strict / `compute_geometric_goals`),
      tiros, `compute_possession_refine`, `compute_field_violations`.
- [x] Normalizar goles a `[(frame, zona)]`; reunir eventos para la lista (tiros + violaciones).
- [x] Docstring de módulo en español (overlay de espectador; no toca demo/track overlay).

**Verificación:** `python -c "import src.core.event_broadcast_overlay"` sin error.

---

## T2 · Series por frame (marcador, banner, feed)

- [x] `marcador[f] = (yellow_n, blue_n)` acumulado de goles con `frame ≤ f`.
- [x] `banner[f] = (texto, x_pos)` si `f ∈ [gol, gol+banner_secs·fps)`, x interpolada 0→ancho.
- [x] `event_feed[f]` = últimos `max_items` eventos con `frame_inicio ≤ f` (por recencia).

**Verificación:** el marcador final coincide con el conteo de goles de la `goal_source`
(strict ⇒ azul=1; geometric ⇒ azul=3 en `IMG_9933_5m30`).

---

## T3 · cm: minimapa y heatmap (con modo degradado)

- [x] `compute_metric_positions(tracks_json)` (una vez); `MinimapRenderer` (estela) + heatmap
      acumulado (`metric_heatmap`).
- [x] Si falla / sin cm fiable ⇒ `overlay_degradado=True`, omitir minimapa/heatmap.

**Verificación:** con cm, el minimapa y el heatmap se generan; sin cm, el overlay corre igual
sin esos paneles.

---

## T4 · Componentes de dibujo (cv2)

- [x] `_draw_scoreboard(panel, yellow_n, blue_n)` con colores de portería.
- [x] `_draw_goal_banner(frame, texto, x_pos)` deslizante sobre el video.
- [x] `_draw_metrics_panel(panel, posesion, control)` legible (estilo `demo_overlay._metrics_bar`).
- [x] `_draw_event_feed(panel, items)` con tope `max_items` (color/icono por tipo).
- [x] Reusar `_label`/`_fit` de `demo_overlay` para encajar minimapa/heatmap.

**Verificación:** cada componente devuelve una imagen válida en una prueba unitaria rápida.

---

## T5 · Layouts (1 y 2)

- [x] `_compose_layout2(...)` (default): márgenes anchos; marcador arriba; métricas a la
      izquierda; minimapa y heatmap en lados **opuestos**; lista de eventos; video al centro;
      banner sobre el video.
- [x] `_compose_layout1(...)`: video casi a pantalla completa; marcador arriba; minimapa/heatmap
      en esquinas opuestas; métricas y lista en esquinas restantes; banner sobre el video.
- [x] Ambos consumen los **mismos** paneles dibujados.

**Verificación:** ambos layouts producen un lienzo del tamaño esperado con todas las piezas.

---

## T6 · API pública y render incremental

- [x] `render_broadcast_overlay(tracks_json, *, layout, goal_source, banner_secs, max_items,
      margin_px, trajectory_window, bin_cm, sigma_cm, out_fps, max_frames, progress)` →
      `BroadcastResult` (firma del plan §6).
- [x] Render incremental con `iter_frames` + `open_video_writer`; `tqdm` si `progress`.
- [x] Salida: mp4 vía `events_paths(stem, "broadcast", "mp4")` + PNG de muestra (frame con gol).
- [x] `resumen`: marcador final, layout, goal_source, degradado, conteos. Imports perezosos.

**Verificación:** `render_broadcast_overlay(<json>, max_frames=…)` genera el mp4 y el PNG.

---

## T7 · Test manual

- [x] Crear `testing/test_event_broadcast_overlay.py` (script directo, sin pytest, sin GPU),
      sobre `IMG_9933_5m30` **capado** (`max_frames`).
- [x] Genera `layout=2, goal_source="strict"` y `layout=1`; verifica que el mp4 existe y abre.
- [x] Verifica el marcador final del resumen (strict ⇒ azul=1).
- [x] Exporta y verifica el PNG de muestra (frame con gol/banner).

**Verificación:** `python testing/test_event_broadcast_overlay.py` termina OK (local).

---

## T8 · Cierre

- [x] `ruff check` limpio en los archivos nuevos.
- [x] Confirmar que `demo_overlay.py`/`track_overlay.py` y los módulos de eventos/minimapa/
      heatmap quedaron intactos (solo importados).
- [x] Confirmar con el usuario antes de cualquier commit (constitución §7.1/§11).

---

## Orden sugerido

T1 → T2 (series) → T3 (cm/minimapa/heatmap) → T4 (componentes) → T5 (layouts) → T6
(API/render) → T7 (test) → T8.

---

## Fuera de alcance (recordatorio del spec)

- No modifica `demo_overlay`/`track_overlay` (rol = el viejo queda para mosaico/depuración).
- No introduce equipos/bandos (marcador por portería).
- No re-infiere ni corre en GPU.
