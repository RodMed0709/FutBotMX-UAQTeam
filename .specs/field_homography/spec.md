# Spec — Homografía de campo + minimap de trayectorias (fase_4)

## Objetivo

Proyectar las trayectorias de robots y balón a una **vista cenital métrica** del
campo Copa FutBotMX y superponerla como **minimap** en el video, con cámara móvil
y orientación/resolución variable.

Cumple entregables de la convocatoria:
- **3.5.2 Visualización/Narrativa**: visualización del flujo del juego (trails).
- **3.7.3 Innovación SAM3 → Post-procesamiento**: análisis geométrico (homografía).
- Refuerza **3.5.1** (rastreo de trayectorias) y alimenta el video demo (3.5.3).

## Entrada

- Video de partido (cualquier orientación/resolución).
- Trayectorias de robots/balón: JSON de tracking de fase_2/benchmark
  (`outputs/inference/.../<stem>.json`) **o** se generan en el momento con
  `track_video`.

## Salida

- `notebooks/fase_4_homografia/outputs/<stem>_minimap.mp4`: video con minimap
  arriba-derecha, trails acumulados por `obj_id` (robots por paleta, balón naranja).
- Frame de muestra `.jpg` para inspección.
- Estadística de homografía (frames resueltos vs propagados).

## Método

### Modelo del campo (`field_template.py`)
Geometría oficial (Reglas §7, Fig.1) en cm. Origen esquina sup-izq de la alfombra
(243×182). Provee puntos-ancla (endpoints/centros de portería, esquinas de alfombra)
y `render_field` → cancha 2D + `world_to_px`.

### Homografía por frame (`homography.py`)
Anclas desde máscaras SAM3 (el pipeline ya las produce):
- **Primaria**: 4 endpoints de portería (2 por portería, `minAreaRect` de
  `yellow_zone`/`blue_zone`). Color fija izquierda/derecha → orientación resuelta.
- **Soporte**: esquinas del cuadrilátero de `green_floor` (RANSAC).
- `cv2.findHomography(RANSAC)` + validación (reproyección de centroides de portería)
  + **suavizado EMA** (anti-jitter) + **propagación** de la H previa cuando faltan
  anclas (paneo/oclusión).

### Render (`minimap.py` + `minimap_pipeline.py`)
Punto de contacto con el piso (centro-inferior bbox robot; centroide balón) →
`project_points(H)` → cm → trail acumulado → `composite` arriba-derecha.

## No-objetivos (scope acotado, deadline 19 jun)
- Registro de cancha por detección completa de líneas (queda como trabajo futuro/paper).
- Métricas derivadas (velocidad, heatmap, posesión) — posibles extensiones, no aquí.
- Asignación de equipo (aliado/rival) — se colorea por `obj_id`, no por bando.

## Riesgos / mitigaciones
- `blue_zone` no calado en SAM3 → smoke test dedicado; si falla, ajustar prompt.
- Esquinas de `green_floor` ruidosas → entran solo vía RANSAC, no se depende de ellas.
- Cámara panea y sale una portería → propagación de H previa.
- Flip vertical global del minimap (ambigüedad top/bottom) → constante de config si aparece.

## Verificación
- Smoke 1 frame: áreas de máscara > 0, H válida, centroides reproyectan cerca del target.
- Clip corto: inspección visual de trails plausibles sobre la cancha.
- Video completo: ratio propagated/estimated razonable (<~40% propagado).
- Revisión final por agente revisor de código adversarial.

## Adenda — realidad de la ejecución (reconciliación)

El método de §Método describe el **camino A (SAM3 puro)**, la intención inicial. En la
ejecución, el ancla "borde de alfombra `green_floor`" resultó **poco fiable** (la portería
que sobresale corrompe el borde superior y el frame recorta el lado derecho; `_refine`
quedó *probado y falla*). El trabajo pivotó a dos caminos que sí miden bien:

- **B — color automático** (`notebooks/fase_4_homografia/auto_homography.py`, local sin GPU):
  ancla = 4 esquinas del **rectángulo interior** de líneas blancas (visible aunque se corte
  el borde de alfombra). **Medido: 85% ok, error ~12 cm (~5% campo).** Pasó revisión adversarial.
- **C — SAM3+YOLO integrado** (`notebooks/fase_4_homografia/pod_minimap_sam3.py`, pod GPU):
  **el elegido para categoría Profesional.** YOLO (robot/balón/porterías) + SAM3 `green_floor`
  → `solve_masks` → H sobre anclas SAM3/YOLO. Gate de consistencia temporal + EMA + propagación.
  **5 videos demo, error 9–23 cm.** Cumple el gate visual y de reproyección de §Verificación.

**Consolidación HECHA y CERRADA** (rama `feat/consolidate-homography-path-c`, 2026-06-15): el
camino C vive en `src/core/` (`auto_homography.py` + `minimap_pipeline.py`), con render alineado
a la demo (`minimap.py`: cuadro gris robot, balón naranja, `draw_field_overlay`). El driver
`render_minimap_video` es **agnóstico al pipeline**: parámetro `detector` ∈
{`sam3_text`, `yolo_sam3`, `yolo`} para las anclas/objetos (u objetos de `tracks_json` de cualquier
2×2); `start_frame`/`frame_step` para recortar tramo; fps de salida real. El camino `yolo` (cajas
YOLO + `green_floor` SAM3, 1 SAM3/frame) **reemplaza a `pod_minimap_sam3.py`** (marcado obsoleto):
verificado en pod sobre el mismo clip `IMG_9933_c` que iguala a la demo en velocidad y resultado
(`testing/test_homografia_comparativa.py`). `yolo_sam3` (máscaras finas) se conserva para fase_5.
Las **métricas cuantitativas** (velocidad cm/s, posesión, zonas, heatmap) que la H métrica habilita
**no** son de esta tarea: pasan a **fase_5 (análisis de eventos)**.
