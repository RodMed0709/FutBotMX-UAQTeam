# Homografía robusta multi-feature (Fase 4 v2)

**Fecha:** 2026-06-16
**Estado:** aprobado, en ejecución autónoma
**Objetivo:** homografía campo→cenital que funcione en TODOS los videos (cenital + lateral + oclusión parcial), base para medir velocidad, distancia y posición del balón.

## Problema con el método actual (camino C)

- Usa **solo las 4 esquinas** del rectángulo interior de líneas → falla si un lado se corta/ocluye.
- **Solo cámara cenital**; no hay nada para vista lateral.
- Distorsión de barril → ~10 cm de error central (no corregido).
- Consistencia temporal = EMA + gate de salto, pero **sin métrica publicada** de jitter/exactitud → no se puede decir "cuál variante mejora".

## Modelo métrico oficial del campo (PDF Reglas CopaFutBotMX 2026, Fig. 1)

Todo en cm. Origen en esquina; x = largo (243), y = ancho (182).

- Cancha total: **243 × 182**; paredes negras ≥22 cm.
- Rectángulo interior de líneas blancas: **219 × 158** (inset 12 cm de cada pared).
- Línea central: x = 121.5 (divide en mitades idénticas).
- Círculo central: Ø **60** (r=30), centrado en (121.5, 91).
- Áreas de penalti: **25 (profundidad) × 80 (ancho)**, centradas frente a cada portería.
- Líneas de gol: ancho **60**, centradas.
- Porterías: 60 × 10 × 10; interior **amarillo** (un extremo) / **azul** (otro). Postes amarillo en x≈6, azul en x≈237.
- Grosor de líneas: 2 cm. Tolerancia dimensional ±5%.
- Robots (Abierta, balón golf naranja 42 mm): máx **18 cm** Ø y altura; marcador superior blanco ≥4 cm.

`src/core/field_template.py` ya coincide con estos números (243/182, 219/158, centro 121.5,91, r=30, boca 60). Se extiende, no se reemplaza.

## Diseño

### A. Núcleo geométrico — solver multi-feature
Solver que acumula **todas** las correspondencias visibles y resuelve por RANSAC:

- Landmarks del template (cm), con nombre: 4 esquinas interiores, endpoints línea central, círculo central (cónica), 8 esquinas de áreas de penalti, endpoints líneas de gol, postes de portería.
- Detección por frame: alfombra (`green_floor` SAM3) → líneas blancas (HSV dentro de alfombra) → primitivas semánticas (rectángulo exterior, línea central, círculo→elipse, áreas, líneas de gol). Postes amarillo/azul (SAM3/YOLO) fijan orientación y qué portería es cuál.
- `cv2.findHomography(pts_img, pts_cm, RANSAC, thr)` sobre N≥4 correspondencias redundantes. La **cónica del círculo** ancla centro+escala aunque falten esquinas → resuelve oclusión y "ver 3 esquinas, imaginar la 4ª" de forma natural.
- Funciona cenital y lateral por igual (más landmarks visibles desde el lado, perspectiva tolerada por DLT/RANSAC).

### B. Métrica de consistencia (criterio de comparación)
Sobre un set fijo de clips de evaluación (cenital + lateral):

1. **Error de reproyección (cm):** mediana de distancia entre landmarks detectados y su mapeo al template vía H. → exactitud.
2. **Jitter temporal (cm/frame):** desviación estándar del reproyectado de puntos estáticos del campo entre frames consecutivos. → estabilidad ("tiembla").
3. **Cobertura:** % de frames con H válida.

Cada variante de experimento se puntúa con (reproj↓, jitter↓, cobertura↑).

### C. Notebooks de fase progresiva
Harness que corre variantes sobre los clips y produce tabla comparativa + plots. Ejes:

- `00` template de landmarks + harness de métricas + set de clips eval
- `01` baseline (camino C actual) medido cenital y lateral
- `02` solver multi-feature (esquinas + líneas)
- `03` + restricción cónica del círculo central
- `04` + corrección de distorsión de barril (undistort + calibración)
- `05` + suavizado temporal (EMA vs Kalman sobre H / landmarks)
- `06` canales de color (HSV vs alternativas) para líneas/alfombra
- `07` depth (DepthAnything) — balón en el aire, altura de robot, cross-check
- `08` reconstrucción de terreno desde vista lateral → minimap cenital
- `final` combo ganador consolidado en `src/core/homography.py`

### Alcance del primer ciclo
Fundación: `00`–`03`. Da oclusión + lateral básico medibles. Ejes `04`–`08` son ciclos cortos posteriores sobre esa base.

## Componentes y archivos

- `src/core/field_landmarks.py` (nuevo): landmarks nombrados del template en cm + helpers de muestreo.
- `src/core/homography_metrics.py` (nuevo): reproj error, jitter temporal, cobertura; runner de variantes.
- `src/core/homography.py` (upgrade al final): solver multi-feature consolidado.
- `notebooks/fase_4_homografia/v2_robust/00..08_*.ipynb`: experimentos.
- Clips eval: cenital (IMG_9933, IMG_9938) + lateral (subset de 18abril/Cámaras, incl. IMG_9913 ya anotado).

## No-objetivos

- No re-anotar; no tocar Supervisely.
- No métricas de juego (velocidad/posesión) aquí — consumen la H, van después.
- No reescribir el minimap renderer; solo alimentarlo con mejor H.

## Riesgos

- Detección de elipse del círculo bajo perspectiva lateral fuerte puede ser inestable → RANSAC + fallback a solo-puntos.
- Pocos clips cenital (2) → eval cenital limitado; compensar con lateral abundante (~47).
- Calibración de lente Meta Glasses desconocida → undistort puede requerir auto-calibración por líneas rectas.
