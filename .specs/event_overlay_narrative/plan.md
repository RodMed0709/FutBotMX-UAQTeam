# Plan — `event_overlay_narrative` (T7)

## Enfoque

Módulo nuevo `src/core/demo_overlay.py` (empaquetado, reusable) que **compone** un video demo a
partir de **componentes ya renderizados** + un panel de métricas. Clave: no reimplementa
overlays; **tilea videos** frame-sincronizados y superpone texto/banner.

Componentes por panel:
- **Original**: el clip (`IMG_9933_5m30.mp4`).
- **Segmentación**: `<stem>_seg.mp4` (ya generado en el pod; máscaras por clase).
- **Tracking**: `<stem>_obj_id.mp4` — T7 lo **genera en local** con
  `track_overlay.render_obj_id_overlay` (es dibujo puro desde el JSON, sin modelo).
- **Minimap**: `<stem>_minimap.mp4` (ya generado en el pod; trails en cm).
- **Panel de métricas**: texto compuesto por T7 desde T1/T4/T6/gol (CPU local).

## Pasos

1. **Resolver componentes** (`_resolve_components`): junto al JSON, ubicar `clip`, `_seg.mp4`,
   `_minimap.mp4`; generar `_obj_id.mp4` con `render_obj_id_overlay` si falta. Permitir overrides
   por argumento. Avisar claro si falta un componente que no se puede generar en local (seg/minimap
   vienen del pod).
2. **Métricas** (`_collect_metrics`): computar agregados del clip — posesión (T1
   `compute_possession` vía `load_frame_objects`), velocidad/distancia (T4 `compute_kinematics`),
   zona dominante (T6 `compute_field_zones`), y los eventos del **gol geométrico**
   (`compute_geometric_goals`). Guardar texto del panel + intervalos de gol (para el banner).
3. **Composición por frame** (`compose_demo`): abrir los videos componentes; por frame `i`
   - leer un frame de cada componente (manejar longitudes desiguales: parar en el mínimo);
   - **redimensionar** cada panel a su celda y **tilear** en un lienzo (layout por defecto:
     fila sup [Original | Segmentación], fila inf [Tracking | Minimap]) + **barra de métricas**;
   - rotular cada panel; si `i` cae en un evento de gol, dibujar **banner "GOL · azul"**;
   - escribir el frame al mp4 (`open_video_writer`), al fps del clip.
4. **Tope de duración**: `max_seconds` (default 120) → `max_frames = fps*max_seconds`; el clip ya
   es 60 s, así que entra completo.
5. **API**: `compose_demo(tracks_json, *, output_path=None, layout="grid", max_seconds=120,
   components=None) -> Path`. `write` interno con el video writer del repo.
6. **Test/harness** `testing/test_demo_overlay.py`:
   - corre sobre `IMG_9933_5m30.json` (CPU local), generando `_obj_id.mp4` si falta;
   - produce el mp4 demo + un **frame de muestra** `.png`;
   - **invariantes** (mp4 existe y >0 bytes; nº de frames ≈ del clip o ≤ max_frames; las métricas
     del panel coinciden con T1/T4/T6);
   - si faltan `_seg.mp4`/`_minimap.mp4`, **degradar** el layout (omitir ese panel) con aviso, sin
     romper (para poder probar en local aunque falte un componente del pod).

## Decisiones técnicas

- **Componer videos, no reimplementar overlays**: cada overlay pesado ya tiene su función; T7 es
  un compositor. Mantiene CPU-local y evita duplicar lógica de dibujo.
- **Frames portrait (1080×1920)**: cada celda se redimensiona a un alto fijo; el lienzo final es
  landscape (mosaico). Layout configurable; default `grid` 2×2 + barra de métricas.
- **Sincronía**: todos los componentes derivan del mismo clip (mismos frames), así que el frame
  `i` corresponde 1:1. Si un mp4 fue re-encodeado a otra longitud, se recorta al mínimo común.
- **Degradación**: si un componente no está, se deja su celda con el original o se omite (aviso),
  para que el test corra en local aunque seg/minimap no estén descargados.

## Riesgos / validación

- `_seg.mp4`/`_minimap.mp4` se generan en el pod (seg = máscaras SAM3; minimap = detección de
  campo por frame). En local pueden faltar → degradación + aviso. El test no debe romper.
- Tamaño/legibilidad del texto en el mosaico: usar celdas suficientemente grandes; el frame de
  muestra `.png` permite validar legibilidad sin reproducir.

## Tareas de PRODUCCIÓN (no código, en tasks.md)

Narración/voz, reel IG ≥30 s, link en README, corte final para jurado.

## Estructura de archivos

- `src/core/demo_overlay.py` (nuevo).
- `testing/test_demo_overlay.py` (nuevo).
