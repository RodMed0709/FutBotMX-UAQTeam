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

## Pendiente

- [ ] **Integrar `VideoHomography` (camino C, el que funciona) al `minimap_pipeline.py` del repo**,
      reemplazando `estimate_homography` (camino A, refine probado y falla) y conservando los
      objetos vía `tracks_json` de fase_2 (`yolo_sam3`)
- [ ] Notebook `01_homografia_minimap.ipynb` ejecutable como recipe de la fase
- [ ] (opcional) `cv2.undistort` para la distorsión de barril (~10 cm de error central)
- [ ] (opcional) NB12 depth (DepthAnything-V2) en pod — secundario: campo plano, la H ya da posición métrica

## Notas de estado

- **Tres caminos coexisten**: A (SAM3 puro, en `src/core/homography.py`, con limitación medida
  en el borde de alfombra), B (color auto, local, 85% ok / ~12 cm), C (SAM3+YOLO en pod, el elegido
  para Profesional, err 9–23 cm). B y C viven en `notebooks/fase_4_homografia/` (scripts), aún no
  consolidados en `src/core/`.
- Métricas cuantitativas sobre la H (velocidad cm/s, posesión, zonas, heatmap) **NO** son de esta
  tarea: pasan a **fase_5 (análisis de eventos)**, que se construye sobre la salida métrica de aquí.
