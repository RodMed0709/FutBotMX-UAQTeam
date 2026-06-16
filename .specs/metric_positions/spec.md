# Spec — `metric_positions` (T3, fase_5 · Capa B)

## Contexto

Tercera tarea de fase_5 (análisis de eventos) y **primera de la Capa B (métrica, en cm)**.
La Capa A (T1 posesión, T2 gol-por-zona) ya opera en píxeles sobre el JSON de tracking.
La Capa B proyecta las posiciones a **centímetros** sobre la cancha canónica usando la
homografía ya consolidada (camino C, `VideoHomography`), y solo aplica a video de
**cámara superior** (los únicos con homografía fiable: `IMG_9933`, `IMG_9938`).

`metric_positions` es la **base de toda la Capa B**: T4 (velocidad/distancia), T5 (heatmap),
T6 (zonas del campo) y el **gol geométrico** (línea de gol en cm) consumen su salida.

## Objetivo

Dado el **JSON de tracking extendido** (`include_masks=True`) de un clip de cámara superior,
producir las **posiciones en cm** de robots y balón por `obj_id` y `frame_index`, más un
**reporte de calidad** de la homografía. Sin re-inferir modelos: todo se reconstruye del JSON
y corre en **CPU local**.

## Requisitos funcionales

1. **Insumo**: la ruta a un JSON de tracking generado con `include_masks=True` (p. ej.
   `outputs/inference/fase5_clips/IMG_9933_5m30/IMG_9933_5m30.json`). El JSON debe contener:
   - `tracks[].observations[]` con `obj_id` estable, `class`, `bbox`, `centroid` (objetos);
   - `frames[].detections[clase][]` con `rle` (máscaras) y `centroid` para `green_floor`,
     `yellow_zone`, `blue_zone` (necesarios para resolver la homografía por frame).
2. **Homografía por frame**: reusar `auto_homography.VideoHomography.update_masks(img=None,
   carpet_mask, yc, bc)` — alfombra `green_floor` (máscara binaria decodificada del `rle`,
   limpiada a su componente conexa mayor) + centroides de `yellow_zone`/`blue_zone`.
   - El camino HSV (`update`) NO se usa.
   - En frames sin H propia (`init`/`propagated`/`rejected`) se usa la **última H buena**
     (propagación), y la posición resultante se etiqueta con el `status` de H del frame.
3. **Punto a proyectar** (imagen → cm, vía `cv2.perspectiveTransform` con la `H` del frame):
   - **robots**: foot-point = centro-inferior del bbox `(x + w/2, y + h)`;
   - **balón**: `centroid`.
4. **Salida (JSON nuevo, no modifica el de tracking)**:
   - posiciones por objeto: `{obj_id, class, frame_index, xy_cm: [x, y], status_H}`;
   - resumen de calidad: `n_estimated`, `n_propagated`, `n_rejected`, `% frames con H válida`,
     error de ancla (`init_max_err_cm` efectivo / `goal_err_cm` del ancla si se expone),
     `n_frames`, `n_objetos_proyectados`, `fps`.
5. **Geometría**: las coordenadas cm usan el sistema de `field_template.py` (cancha
   243×182 cm, origen y ejes ya definidos); NO se redefine geometría.
6. **Clases proyectadas**: solo **robots y balón**. Las zonas (`yellow/blue/green_floor`)
   se usan únicamente para resolver H; NO se emiten como objetos de salida.

## Visualización (en el test, no overlay definitivo)

- Trayectorias en cm sobre la **cancha canónica** (reusar `field_template.render_field`),
  una polilínea por `obj_id` (color por clase), sin tocar el video.
- Resumen de calidad de H impreso (estimadas/propagadas/rechazadas, % válido).

## Fuera de alcance

- Velocidad, distancia, heatmap, zonas del campo (T4/T5/T6).
- Gol geométrico (refinamiento posterior de T2 en cm; consume esta salida).
- Overlay/narrativa de video (T7).
- Re-inferir modelos o resolver la homografía con un detector en vivo (el minimap ya hace
  eso; T3 lee la alfombra/zonas del JSON para correr sin GPU).
- Gate automático de elegibilidad Capa B por error de reproyección: en T3 el gate es
  **simple** (cámara superior = elegible); se deja como nota para el futuro.

## Criterios de aceptación

- Sobre `IMG_9933_5m30.json` produce el JSON de posiciones en cm + resumen, en CPU local,
  sin cargar SAM3/YOLO.
- El resumen de calidad de H es coherente con el del minimap del mismo clip (ratio de
  estimadas/propagadas/rechazadas del mismo orden).
- Las trayectorias en cm caen **dentro** del rectángulo de la cancha (0..243 × 0..182) salvo
  ruido puntual; el balón en los frames de los eventos de gol (≈840 y ≈1210) cae cerca de la
  línea de gol azul (x ≈ 231–237 cm).
- Casos borde manejados: frame sin objetos; objeto sin H válida aún (antes del ancla) →
  se omite o se marca `status_H="init"` sin romper; clip sin `green_floor` en un frame →
  propaga H previa.
