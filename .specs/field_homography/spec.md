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
