# Tasks — `event_overlay_narrative` (T7)

> Tarea final de fase_5: ensamble del video demo (3.5.3). Compone componentes ya renderizados +
> panel de métricas. Código en `src/core/`, harness en `testing/`. Insumo:
> `outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json` (+ `_seg.mp4`, `_minimap.mp4`).

## Implementación (código)

- [x] `src/core/demo_overlay.py`: `_resolve_components` — clip + `_seg.mp4` + `_minimap.mp4`;
      genera `_obj_id.mp4` con `render_obj_id_overlay` si falta; degrada con aviso (omite panel).
- [x] `demo_overlay.py`: `_collect_metrics` — posesión (T1), velocidad/distancia (T4), zona (T6),
      goles (gol geométrico); líneas del panel + intervalos de gol para el banner; T3 se computa
      una sola vez y se comparte.
- [x] `demo_overlay.py`: `compose_demo(tracks_json, *, output_path=None, max_seconds=120,
      components=None) -> Path` — mosaico en fila (Original·Segmentación·Tracking·Minimap·**Heatmap
      vivo**) + barra de métricas + rótulos + banner de gol; `open_video_writer` al fps (BGR→RGB).
- [x] `demo_overlay.py`: **heatmap en vivo** — `_live_heatmap_state`/`_accumulate` acumulan las
      posiciones de T3 por frame y se renderizan con `metric_heatmap.render_heatmap` (misma cancha
      que el minimap), panel que se "llena" con el tiempo.
- [x] `demo_overlay.py`: tope `max_seconds`; recorte al mínimo común de frames.
- [x] `testing/test_demo_overlay.py`: escribe mp4 + frame de muestra `.png`; invariantes
      (mp4 >0 bytes; ≤2 min); degradación si falta un componente.
- [x] `ruff check` limpio.

## Verificación (código)

- [x] Corre en **local sin GPU** sobre `IMG_9933_5m30`; genera `_obj_id.mp4` en local.
- [x] mp4 demo **60 s** con 5 paneles (Original·Segmentación·Tracking·Minimap·Heatmap vivo) + barra
      de métricas (posesión #22 16s/68.7%, dist máx #3 936cm, v_max balón 92cm/s, balón azul 87%,
      goles 3), rótulos legibles (frame de muestra verificado).
- [x] **Gol anotado** (banner rojo "GOL - portería blue") en sus frames (muestra en frame 850).
- [x] Métricas del panel == T1/T4/T6.

## Tareas de PRODUCCIÓN (NO código — manuales)

- [ ] Grabar/editar la **explicación o narración por voz** del enfoque.
- [ ] Corte final ≤2 min para jurado (selección de momentos, intro/outro).
- [ ] **Reel de Instagram ≥30 s** + **link en el README**.

## Fuera de alcance

Capacidades nuevas de visión, re-inferencia, homografía (viene del JSON/T3), equipos.
