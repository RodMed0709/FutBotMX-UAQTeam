# Plan — fase_4 homografía + minimap

## Módulos (src/core/)
1. `field_template.py` — geometría métrica (Reglas §7) + `render_field` → cancha 2D + `world_to_px`.
2. `homography.py` — **camino A (SAM3 puro)**: `field_quad` (4 esquinas de `green_floor`),
   `_oriented_candidates`, orientación por portería amarilla + continuidad temporal, `_refine`
   (RANSAC), `estimate_homography`, `project_points`, `HomographyState`.
3. `minimap.py` — `MinimapRenderer`: trails por `obj_id`, dibujo, `composite` arriba-derecha.
4. `minimap_pipeline.py` — `render_minimap_video`: tracks (JSON o `track_video`) + anclas SAM3 → H → proyección → mp4.

## Decisiones validadas en datos
- Input correcto = **cámara superior** (`data/raw/18abril/Camara_superior/`, campo completo + márgenes).
  Las tomas Meta-Glasses portrait (close-up, 1 portería) NO sirven para homografía.
- La cámara superior **se mueve** (panea/rota): una H fija no cubre el clip → estimación por-frame
  + propagación + suavizado EMA.
- **Orientación = `yellow_zone`** (la azul no la segmenta SAM3 con ningún prompt; no se depende de ella).
- **Flip vertical** (yellow en el eje de simetría) → resuelto por **continuidad temporal** con la H previa
  y, en el camino color, por condición dura de color de portería (x<L/2 vs x>L/2).

## Tres caminos de homografía (realidad de la ejecución)
- **A — SAM3 puro** (`src/core/homography.py`): ancla = cuadrilátero de `green_floor`. **Limitación medida:**
  el borde superior de la alfombra lo corrompe la portería que sobresale y el lado derecho lo recorta el
  frame → fit del borde de alfombra poco fiable (`_refine` **probado y falla, no repetir**).
- **B — color automático** (`notebooks/fase_4_homografia/auto_homography.py`, local sin GPU): ancla = 4 esquinas
  del **rectángulo interior** (219×158, líneas blancas, visible aunque se corte el borde) vía `fitLine` +
  intersección; orientación dura por color de portería. **Medido: 85% ok, error reproyección ~12 cm (~5% campo);
  clip 250 frames 99.6% H.** Pasó revisión adversarial (fixes C1/C3/M2/M3/M5).
- **C — SAM3+YOLO integrado** (`notebooks/fase_4_homografia/pod_minimap_sam3.py`, pod GPU): **el elegido para
  categoría Profesional.** YOLO `best.pt` (robot/balón/porterías) + SAM3 `green_floor` (text_mask) →
  `auto_homography.solve_masks` → H sobre anclas SAM3/YOLO (homografía construida sobre SAM3 = innovación 3.7.3).
  Gate de consistencia temporal + EMA + propagación. **5 videos demo, error 9–23 cm.**

## Camino elegido y trabajo de consolidación pendiente
- **Elegido: C** (SAM3+YOLO). El `minimap_pipeline.py` del repo aún usa el camino A (`estimate_homography`);
  el paso de consolidación es **integrar `VideoHomography`/`solve_masks` (camino C) al pipeline del repo**,
  conservando los objetos vía `tracks_json` de fase_2 (`yolo_sam3`). Hoy B y C viven como scripts en
  `notebooks/fase_4_homografia/`, no en `src/core/`.

## Verificación
Ver `spec.md` §Verificación. Gate visual: trails plausibles sobre la cancha, sin saltos de flip.
Cumplido en el camino C (5 videos demo + revisión adversarial del solver de color subyacente).
