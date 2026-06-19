# Tasks — fase_4 homografía + minimap

## Hecho

- [x] Explorar APIs reutilizables (segmentation, tracking schema, frame_extraction, video_writer, config)
- [x] `field_template.py` (geometría oficial Reglas §7 + `render_field` + `world_to_px`)
- [x] `homography.py` — **camino A (SAM3 puro)**: quad de `green_floor` + orientación por
      `yellow_zone` + continuidad temporal + RANSAC + EMA + propagación
- [x] `minimap.py` (`MinimapRenderer`: trails por `obj_id` + composite arriba-derecha)
- [x] `minimap_pipeline.py` (driver SAM3: anclas + objetos por texto → H → render)
- [x] config: añadir clase `blue_zone`
- [x] Smoke 1 frame
- [x] Scan visibilidad porterías (Meta-Glasses portrait → solo una portería; pivote a **cámara superior**)
- [x] Probe prompts azul → SAM3 no segmenta `blue_zone` con ningún prompt; orientación solo por amarilla
- [x] **Camino B (color auto, `auto_homography.py`)** — experimento local sin GPU:
      ancla = 4 esquinas del rectángulo interior (líneas blancas), orientación dura por color de portería
- [x] **Revisión adversarial del camino B** (fixes C1/C3/M2/M3/M5) — eliminó homografías basura (430–1017 cm)
- [x] **Camino C (SAM3+YOLO integrado, `pod_minimap_sam3.py`)** — el pedido para categoría Profesional:
      YOLO `best.pt` (robot/balón/porterías) + SAM3 `green_floor` → `solve_masks` → H sobre anclas SAM3/YOLO
- [x] Bug `green_floor` (líneas blancas como huecos) resuelto (`_white_in_carpet`, MORPH_CLOSE 25×25)
- [x] Gate de consistencia temporal en `VideoHomography` (rechaza saltos >70 px; 1ª H solo con err <22 cm)
- [x] **5 videos demo** del camino C (`IMG_9933_a/b/c`, `IMG_9938_a/b`) + `batch.log`
- [x] Mover módulos a `src/core/` (homography, minimap, minimap_pipeline, field_template)

## Consolidación al repo (rama `feat/consolidate-homography-path-c`, 2026-06-15)

- [x] **Solver camino C en `src/core/auto_homography.py`** — copiado del script de notebook,
      import a `from src.core import field_template as ft`, lint limpio (ruff)
- [x] **Swap en `src/core/minimap_pipeline.py`**: `HomographyState`/`estimate_homography`
      (camino A) → `VideoHomography.update_masks` (camino C); centroides de portería vía
      `mask_centroid`; `bc=None` cuando SAM3 no segmenta la azul; sin `orient_once` (la H ya
      orienta); dict de salida con `rejected`
- [x] **Smoke local sin GPU** sobre `IMG_9933.MOV` (ruta color, no requiere SAM3):
      `solve_masks` ok, error portería 11.9 cm; `VideoHomography` 40/40 frames con H
      (37 estimados, 3 rechazados por el gate temporal) — coincide con lo medido por el equipo
- [x] Notebook `01_homografia_minimap.ipynb` ya importa `render_minimap_video` (firma sin cambios)
- [x] **Match del render de la demo** en `src/core/minimap.py`: robots = cuadro gris (marcador
      uniforme), balón = círculo naranja, trail por paleta/naranja; defaults al de la demo
      (`scale=2.6`, `margin_cm=3.0`, `panel_width_frac=0.34`); nueva `draw_field_overlay`
      (rectángulo interior + círculo central + esquinas reproyectados sobre el video, bajo
      `draw_overlay=True`). El swap de overlay vive en `minimap_pipeline` (reemplaza el
      relleno de máscaras SAM3 del camino A)
- [x] **`detector` pluggable en `render_minimap_video`** (`sam3_text` | `yolo_sam3` | `yolo`),
      vía `get_detector` + `detect_boxes`; valida el nombre (`ValueError`). Anclas/objetos salen
      del detector elegido, no de `segment_with_text` hardcodeado. Permite comparar la homografía
      entre pipelines flipeando un parámetro. Objetos también vía `tracks_json` (cualquier 2×2).
      - `yolo` = cajas YOLO + `green_floor` SAM3-texto → **1 SAM3/frame**, reproduce
        `pod_minimap_sam3` a su misma velocidad (el minimap solo usa cajas, no máscaras de objetos).
      - `yolo_sam3` (YOLO→SAM3 box-prompt, máscaras finas) se conserva para **fase_5** (análisis).
- [x] **`start_frame`/`frame_step`** en `iter_frames`/`render_minimap_video` (clip de cualquier
      tramo; submuestreo). Capacidad general → su propia SDD `frame_window_sampling`.
- [x] **Fixes de paridad con la demo**: fps de salida = `fps/frame_step` (duración real);
      `green_floor` reducido a su mayor componente conexo (descarta muñequeras/reflejos que
      ensanchaban la alfombra); `conf` de YOLO expuesto como parámetro.
- [x] **Comparativa notebook vs src** (`testing/test_homografia_comparativa.py`): corre ambos
      sobre el mismo clip `IMG_9933_c` (start=15000, every=2). **Verificado en pod (2026-06-15):
      `notebook == demo`; `src` con `detector="yolo"` se ve igual y va a la misma velocidad.**
      → consolidación CERRADA: `render_minimap_video` reemplaza a `pod_minimap_sam3.py`.

## Pendiente
- [ ] (limpieza) `notebooks/fase_4_homografia/pod_minimap_sam3.py` queda **OBSOLETO** (referencia):
      reemplazado por `render_minimap_video(detector="yolo")`. Marcado en su docstring; eventual
      retiro junto con los demás scripts sueltos de fase_4 (`auto_homography`/`minimap_auto` ya en `src`).
- [ ] (limpieza) decidir destino del camino A en `src/core/homography.py` (hoy sin uso por el
      pipeline; `mask_centroid`/`project_points` sí se siguen reusando)
- [ ] (opcional) `cv2.undistort` para la distorsión de barril (~10 cm de error central)
- [ ] (opcional) NB12 depth (DepthAnything-V2) en pod — secundario: campo plano, la H ya da posición métrica

## Notas de estado

- **Tres caminos de homografía**: A (SAM3 puro, `src/core/homography.py`, limitación medida en el
  borde de alfombra — hoy **sin uso** por el pipeline), B (color auto, local, 85% ok / ~12 cm),
  C (SAM3+YOLO, el elegido para Profesional, err 9–23 cm). **El camino C está consolidado en
  `src/core/`** (`auto_homography.py` + `minimap_pipeline.py`); B sigue como script de notebook.
- **Canónico de aquí en adelante:** `src.core.minimap_pipeline.render_minimap_video`. El script
  `notebooks/fase_4_homografia/pod_minimap_sam3.py` queda como **referencia obsoleta**
  (`render_minimap_video(detector="yolo")` lo reemplaza 1:1, verificado en pod).
- **`detector` define la fuente de anclas/objetos** (eje detector del 2×2); el eje **tracker** no es
  parámetro del minimap: entra solo si se pasa un `tracks_json` (cualquier config 2×2). Para la
  homografía el tracker es irrelevante (depende solo de anclas).
- El recorte por tramo/submuestreo del video (`start_frame`/`frame_step` en `iter_frames` y
  `render_minimap_video`) se añadió para reproducir clips concretos y abaratar el costo, pero es una
  capacidad **general de `frame_extraction`** → se documenta en su **propia tarea SDD**
  (`frame_window_sampling`), no aquí.
- Métricas cuantitativas sobre la H (velocidad cm/s, posesión, zonas, heatmap) **NO** son de esta
  tarea: pasan a **fase_5 (análisis de eventos)**, que se construye sobre la salida métrica de aquí.
