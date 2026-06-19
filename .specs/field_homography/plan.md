# Plan — fase_4 homografía + minimap

## Módulos (src/core/)
1. `field_template.py` — geometría métrica (Reglas §7) + `render_field` → cancha 2D + `world_to_px`.
2. `homography.py` — **camino A (SAM3 puro)**: `field_quad` (4 esquinas de `green_floor`),
   `_oriented_candidates`, orientación por portería amarilla + continuidad temporal, `_refine`
   (RANSAC), `estimate_homography`, `project_points`, `HomographyState`.
3. `minimap.py` — `MinimapRenderer` (trails por `obj_id`, `composite`) + `draw_field_overlay`
   (cancha/círculo reproyectados sobre el video).
4. `auto_homography.py` — solver camino C consolidado (`VideoHomography`/`solve_masks`).
5. `minimap_pipeline.py` — `render_minimap_video`: detector pluggable
   (`sam3_text` | `yolo_sam3` | `yolo`) para anclas/objetos (u objetos de `tracks_json`) →
   `VideoHomography` (camino C) → proyección → mp4. `start_frame`/`frame_step` para recortar tramo.

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

## Camino elegido y consolidación (HECHA, rama `feat/consolidate-homography-path-c`)
- **Elegido: C** (SAM3+YOLO). Ya **consolidado en `src/core/`**:
  - `src/core/auto_homography.py` — `VideoHomography`/`solve_masks` (copiado del script, import al
    `field_template` del repo, lint limpio).
  - `src/core/minimap_pipeline.py` — usa `VideoHomography.update_masks` (antes camino A
    `estimate_homography`), conserva objetos vía `tracks_json` de fase_2 (`yolo_sam3`).
  - `src/core/minimap.py` — render alineado con la demo (cuadro gris robot, balón naranja) +
    `draw_field_overlay` (cancha reproyectada sobre el video).
- El camino A (`src/core/homography.py`) queda **sin uso** por el pipeline (solo se reusan sus helpers
  `mask_centroid`/`project_points`); decidir su destino es limpieza pendiente.
- B sigue como script de notebook (driver color local, sin SAM3).

## Detector pluggable (anclas + objetos)
`render_minimap_video(detector=...)` toma las detecciones del detector del repo, no de SAM3-texto
hardcodeado, para poder comparar la homografía entre pipelines:
- `sam3_text` — todo por SAM3-texto (la azul suele faltar → orientación más floja).
- `yolo_sam3` — YOLO→SAM3 box-prompt (máscaras finas; más lento). Reservado para **fase_5** (análisis).
- `yolo` — cajas YOLO (`detect_boxes`) + `green_floor` por SAM3-texto: **1 SAM3/frame**, rápido,
  reproduce `pod_minimap_sam3`. El minimap solo usa cajas (foot-points/centroides), no las máscaras
  de objetos, así que `yolo` da el mismo resultado que `yolo_sam3` mucho más rápido.

El eje **tracker** (bytetrack/botsort) no es parámetro del minimap: los `obj_id` estables llegan vía
`tracks_json` (cualquier 2×2) o del `_GreedyTracker` interno. La homografía no depende del tracker.

## Canónico
`render_minimap_video` **reemplaza** a `notebooks/fase_4_homografia/pod_minimap_sam3.py` (marcado
obsoleto/referencia). Verificado en pod: `detector="yolo"` iguala a la demo en velocidad y resultado.

## Verificación
Ver `spec.md` §Verificación. Gate visual: trails plausibles sobre la cancha, sin saltos de flip.
Cumplido en el camino C: 5 videos demo + revisión adversarial; comparativa pod-vs-src en el mismo
clip `IMG_9933_c` (`testing/test_homografia_comparativa.py`) confirmada en pod (2026-06-15).
