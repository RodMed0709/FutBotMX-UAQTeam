# Plan — fase_4 homografía + minimap

## Módulos (src/core/)
1. `field_template.py` — geometría métrica (Reglas §7) + `render_field` → cancha 2D + `world_to_px`.
2. `homography.py` — `field_quad` (4 esquinas forzadas), `_oriented_candidates`,
   orientación por portería + continuidad temporal, `_refine` (RANSAC), `estimate_homography`,
   `project_points`, `HomographyState`.
3. `minimap.py` — `MinimapRenderer`: trails por `obj_id`, dibujo, `composite` arriba-derecha.
4. `minimap_pipeline.py` — `render_minimap_video`: tracks (JSON o `track_video`) + anclas SAM3 → H → proyección → mp4.

## Decisiones validadas en datos
- Input correcto = **cámara superior** (campo completo + márgenes; `data/raw/18abril/Camara_superior/`).
  Las tomas Meta-Glasses portrait (close-up, 1 portería) NO sirven para homografía.
- **Ancla primaria = cuadrilátero de `green_floor`** (4 esquinas, ~68% del frame, márgenes visibles).
- **Orientación = `yellow_zone`** (la azul no la segmenta SAM3 con ningún prompt; no se depende de ella).
- **Flip vertical** (yellow en eje de simetría) → resuelto por **continuidad temporal** con la H previa.

## Pasos de ejecución
- F0: explorar APIs reutilizables (hecho).
- F1: módulos + config `blue_zone` (hecho).
- F2: smoke 1 frame, scan visibilidad, probe prompts, **ver frame** (hecho → pivote a cámara superior).
- F3: verificación visual de homografía en clip (en curso).
- F4: pipeline completo con `track_video` → mp4 en `notebooks/fase_4_homografia/outputs/`.
- F5: notebook `01_homografia_minimap.ipynb`.
- F6: revisión por agente adversarial; integrar fixes.

## Verificación
Ver `spec.md` §Verificación. Gate visual: trails plausibles sobre la cancha, sin saltos de flip.
