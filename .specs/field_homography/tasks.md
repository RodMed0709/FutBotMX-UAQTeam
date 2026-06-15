# Tasks — fase_4 homografía + minimap

- [x] Explorar APIs reutilizables (segmentation, tracking schema, frame_extraction, video_writer, config)
- [x] `field_template.py` (geometría + render_field)
- [x] `homography.py` (quad + orientación + continuidad + RANSAC + propagación)
- [x] `minimap.py` (trails + composite)
- [x] `minimap_pipeline.py` (driver)
- [x] config: añadir `blue_zone`
- [x] Smoke 1 frame
- [x] Scan visibilidad porterías (Meta-Glasses → solo yellow)
- [x] Probe prompts azul + ver frame → pivote a cámara superior
- [ ] Verificación visual homografía en clip (cámara superior)
- [ ] Ajustes según inspección (escala/flip/suavizado)
- [ ] Pipeline completo con `track_video` → mp4 en notebook outputs
- [ ] Notebook `01_homografia_minimap.ipynb`
- [ ] Revisión por agente adversarial + integrar fixes
- [ ] Mover módulos a repo (src/core) y limpiar scripts de prueba
