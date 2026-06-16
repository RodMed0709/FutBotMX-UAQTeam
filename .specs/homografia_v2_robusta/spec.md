# Spec retroactivo — Homografía robusta cenital + minimap métrico (Fase 4 v2)

**Fecha:** 2026-06-16 (retroactivo)
**Estado:** cenital funcional y consolidado; lateral pendiente.
**Config activa:** `configs/01_yolo_sam3_config.json`
**Módulos:** `src/core/{field_template,field_landmarks,homography_metrics,homography_multifeature,homography_tracked,video_stabilize}.py`
**Notebooks:** `notebooks/fase_4_homografia/v2_robust/00..06_*.ipynb`
**Rama:** `feat/homografia-v2-robust`

---

## 1. Objetivo y alcance

Estimar por frame la homografía **imagen (px) → mundo (cm)** del campo Copa FutBotMX (alfombra 243 × 182 cm) para proyectar robots y balón a una **vista cenital métrica** del campo y renderizarla como **minimap** con trails. La H métrica habilita métricas cuantitativas (velocidad cm/s, distancia, posesión) que exige la categoría Profesional (convocatoria 3.7.3 / 3.5.2).

Alcance del ciclo actual:

- **Cenital (objetivo primario, LOGRADO):** cámara superior (`IMG_9933.MOV`, `IMG_9938.MOV`) que panea/rota entre frames. Solver por líneas blancas con gate de overlap + EMA, estable y sin jitter visible.
- **Lateral (PENDIENTE):** clips externos (`IMG_9913/9914/9915`). El solver actual degrada bajo perspectiva fuerte; queda como trabajo siguiente con un detector de líneas dedicado.

No-objetivos: re-anotar / tocar Supervisely; reescribir el renderer del minimap; calcular métricas de juego aquí (consumen la H, van después).

---

## 2. Clases/objetos: qué se usa y con qué modelo

Clases reales de `configs/01_yolo_sam3_config.json` (NO existen `robot_a`/`robot_b`):

| Clase | Modelo | Para qué | Usado |
|---|---|---|---|
| `robot` (yolo_id 0) | YOLO (`best.pt`) | Objeto minimap (foot-point del robot) | SÍ |
| `orange_ball` (yolo_id 1) | YOLO (`best.pt`) | Objeto minimap (posición del balón) | SÍ |
| `yellow_zone` (yolo_id 2) | YOLO (`best.pt`) | Orientación de H (centroide → portería izquierda, `x < centro`) | SÍ |
| `blue_zone` (yolo_id 3) | YOLO (`best.pt`) | Orientación de H (centroide → portería derecha, `x > centro`) | SÍ |
| `green_floor` (sin yolo_id) | SAM3 (texto: "green playing surface with lines" / "green floor") | Homografía: máscara de alfombra de la que se extraen las líneas blancas | SÍ |
| Líneas blancas | derivado de `green_floor` (HSV, no modelo) | Homografía: ajuste de esquinas + overlap | SÍ |
| Círculo central | derivado de líneas blancas (`detect_center_circle`, elipse) | Detector existe; landmark held-out / restricción | PARCIAL (no en el solver de líneas final) |

Qué NO se usa en el pipeline cenital consolidado (nb06):

- **Línea central en el overlay/template de overlap: QUITADA.** `_template_perimeter_cm` muestrea **solo el rectángulo interior** (no la línea central) porque el campo apenas la marca y metía ruido/temblor a la métrica y al overlay.
- **`video_stabilize.stabilize_frames`: DESACTIVADA** (ver §6).
- **`VideoHomographyTracked` (flujo óptico, `homography_tracked.py`): NO es el solver final.** Quedó como experimento (nb04); pierde tracking en lateral (14 frames `lost` en `IMG_9913`).
- **Restricción cónica del círculo:** `detect_center_circle` está implementada pero NO entra como correspondencia en el solver de líneas ganador (queda como held-out / ramo futuro).
- **DepthAnything, undistort de barril, robot_a/robot_b, equipos por color:** no implementados en este ciclo.

---

## 3. Módulos (src/core nuevos)

### `field_template.py` — fuente de verdad geométrica
Geometría oficial en cm (origen esquina sup-izq de la alfombra; x=largo 243, y=ancho 182). Constantes: `LENGTH_CM`, `WIDTH_CM`, `LINE_BORDER_CM=12`, `CENTER_CM=(121.5,91)`, `CIRCLE_RADIUS_CM=30`, `YELLOW_GOAL_X_CM=6`, `BLUE_GOAL_X_CM=237`, `CARPET_CORNERS`.
- `render_field(scale=2.2, margin_cm=10.0) -> (canvas, world_to_px)`: dibuja el minimap cenital base (alfombra, rectángulo interior, línea central, círculo, áreas en forma de D, porterías amarilla/azul) y devuelve la función `world_to_px(cm) -> (px, py)`.
- `_penalty_outline_cm(goal_x, inner_x, r, n)`: contorno del área chica con las dos esquinas internas redondeadas (forma D).

### `field_landmarks.py` — landmarks nombrados del template
Capa sobre `field_template`. 23 puntos-ancla en cm.
- `LANDMARK_POINTS: dict[str,(x,y)]` — esquinas interiores (`inner_tl/tr/br/bl`), línea central (`center_top/bot/center`), círculo (`circle_top/bot/left/right`), áreas (`penL/R_top/bot`), boca de portería (`goalL/R_top/bot`), postes de color (`postY/B_top/bot`).
- `LANDMARK_LINES: dict[str,((x1,y1),(x2,y2))]` — segmentos de borde, central y frentes de área.
- `CENTER_CIRCLE = (121.5, 91, 30)`.
- `points_array(names=None) -> (nombres, (N,2) cm)`.
- `static_world_points() -> (5,2)`: 4 esquinas interiores + centro, para medir jitter.
- `draw_landmarks(canvas, world_to_px, radius=3)`.

### `homography_metrics.py` — criterio de comparación
`H` mapea imagen(px)→mundo(cm).
- `@dataclass FrameResult`: `frame_idx`, `H`, `detections` (fit), `eval_points` (held-out). `measure_points()` devuelve held-out si existen (evita circularidad de medir el error sobre los puntos que definieron H).
- `project_img_to_world(H, pts_img) -> (N,2) cm`.
- `reproj_error_cm(fr) -> float|None`: mediana de la distancia landmark↔template.
- `_clip_jitter_cm(frames, min_samples=3)`: std temporal del cm reconstruido por landmark estático.
- `summarize(per_clip) -> {reproj_cm, jitter_cm, coverage, n_frames, per_clip}`.
- `run_variant(clips, solver_fn) -> dict`: corre un `solver_fn(frame_rgb, idx) -> FrameResult` intercambiable sobre clips y agrega métricas.

### `homography_multifeature.py` — solver ganador (líneas blancas + overlap)
- `field_white_lines(img_bgr, carpet_mask, close_ksize=25, white_v_min=140, white_s_max=90) -> mask`: cierra/dilata la máscara verde de SAM3 para abarcar la franja de líneas y borde, e intersecta con píxeles blanquecinos (alto V, baja S). Idea de Rodrigo: las líneas viven dentro del `green_floor`.
- `inner_corners_extrapolated(white, frac=0.80, min_side_px=15, max_oob_frac=1.5) -> (4,2)|None`: clasifica píxeles blancos en 4 lados por los ejes del `minAreaRect`, ajusta recta robusta por lado (`_fit_line_robust`) e **intersecta lados adyacentes extrapolando** → recupera esquinas ocluidas/cortadas fuera del frame.
- `field_quad_from_white(white) -> (4,2)|None`: 4 esquinas por convex-hull + `approxPolyDP` (robusto en trapecio lateral donde `minAreaRect` falla).
- `solve_lines_masks(img_bgr, carpet_mask, yc=None, bc=None) -> {H, corners, overlap, white, ok}`: combina 3 fuentes de esquinas (interiores extrapoladas, hull del blanco, hull del verde→`CARPET_CORNERS`), prueba las 4 rotaciones del etiquetado, descarta orientaciones inválidas (amarillo debe caer `x<centro`, azul `x>centro`) y elige la de **mayor `registration_overlap`**.
- `registration_overlap(white, H, band=7)` + `line_overlap_score(white, projected_pts, band=6)`: fracción de puntos del perímetro del template proyectado que caen sobre blanco dilatado (criterio de calidad).
- `_template_perimeter_cm(step=4.0)`: muestrea **solo el rectángulo interior** (línea central excluida deliberadamente).
- **`class VideoHomographyLines(min_overlap=0.40, smooth_beta=0.4)`** — solver por video ganador. `update(img_bgr, carpet_mask, yc, bc) -> (H, status, overlap)` con `status in {fit, kept, none}`: re-ajusta cada frame con `solve_lines_masks`, **acepta solo si `overlap >= min_overlap`** (nunca fija algo torcido), suaviza con EMA, y si el frame ajusta mal **conserva la última H buena** (`kept`). `stats() -> {fit, kept, none}`.
- `detect_center_circle(img_bgr, white=None, ...) -> (cx,cy)|None`: quita rectas largas (`_remove_long_lines`, HoughLinesP) y ajusta elipse al residual (el círculo). Landmark held-out.

### `homography_tracked.py` — experimento (no es el final)
- **`class VideoHomographyTracked(min_track=12, target_pts=120, correct_every=20, correct_beta=0.15, max_correct_px=60.0)`**: ancla con SAM3 (`auto_homography.solve_masks`), siembra puntos sobre la alfombra (`goodFeaturesToTrack`), propaga con flujo óptico Lucas-Kanade (`calcOpticalFlowPyrLK`) recalculando H por RANSAC, re-siembra al agotar puntos y corrige deriva con EMA suave / re-ancla si el salto supera `max_correct_px`. `update(...) -> (H, status)` con `status in {anchored, tracked, corrected, reanchored, lost}`.

### `video_stabilize.py` — preproceso (DESACTIVADO)
- `estimate_transforms(frames_gray) -> (N-1,3)`: afín por par de frames (Lucas-Kanade + `estimateAffinePartial2D`).
- `stabilize_frames(frames_rgb, smooth_radius=15, crop_ratio=0.04) -> list`: media móvil de la trayectoria + warp. No usado en el pipeline (§6).

---

## 4. Flujo de datos por frame (pipeline cenital — nb06)

1. **Segmentación de campo (SAM3):** `detect_classes_in_frame` → máscara `green_floor` (carpet_mask).
2. **Detección de objetos y orientación (YOLO `best.pt`):** `detect_boxes` → cajas de `robot`, `orange_ball` (objetos) y `yellow_zone`, `blue_zone` (centroides `yc`/`bc` para orientación).
3. **Homografía:** `VideoHomographyLines.update(bgr, carpet_mask, yc, bc)`:
   - `field_white_lines` extrae las líneas blancas dentro de la alfombra.
   - `solve_lines_masks` ajusta esquinas (interiores extrapoladas / hull blanco / hull verde), fija orientación con `yc`/`bc`, elige la H de mayor `overlap`.
   - gate `overlap >= min_overlap` + EMA (`smooth_beta` alto ≈ 0.7 en cenital) → `fit` / `kept` / `none`.
4. **Proyección:** `_objects_from_boxes` + `_box_centroid` → foot-points; `_GreedyTracker` asigna IDs estables; `project_points(H, ...)` mapea cada objeto a `(oid, clase, x_cm, y_cm)` en cancha.
5. **Minimap:** dibuja sobre `render_field` los objetos con trails y compone el minimap sobre el video. Overlay sobre el frame = rectángulo interior + círculo reproyectados (línea central excluida).

---

## 5. Estado actual

### Cenital — FUNCIONA
- `VideoHomographyLines` sobre `IMG_9933` (110 frames, step=3): **~108–109/110 frames `fit`**, overlay suave, círculo central bien colocado, 4–6 objetos/frame en el minimap.
- Overlap del template proyectado sobre la blanca real en régimen cenital: **~0.65–0.73** con el solver de líneas + `smooth_beta` alto (mejor que el ~0.55 del primer barrido de nb05).
- Extracción de líneas blancas por HSV dentro de `green_floor`: **perfecta**.
- Esquinas extrapoladas toleran oclusión / borde recortado sin romper el mapa.
- Baseline camino C (nb00): reproj ≈ 0 por construcción (H ajustada a las 4 esquinas) → la señal informativa es **cobertura** (cenital 1.00 en 9933) y **jitter**.

### Lateral — PENDIENTE
- Cobertura muy baja en clips laterales (4–46 % en el baseline; overlap ~0.31 con el solver de líneas).
- `VideoHomographyTracked` (flujo óptico) pierde tracking: 14 frames `lost` en `IMG_9913`.
- **Por qué fallan los atajos de esquinas:** `inner_corners_extrapolated` se apoya en los ejes ortogonales del `minAreaRect`; bajo perspectiva lateral fuerte el rectángulo es un **trapecio** y la clasificación en 4 lados + intersección se degrada. `field_quad_from_white` (hull) ayuda algo pero las **áreas en forma de D y la línea central contaminan** el hull. No hay agrupamiento real de segmentos de línea → faltan correspondencias punto-línea robustas para la perspectiva.

---

## 6. Decisiones y por qué

- **Overlap-gate vs jitter (camino ganador):** re-ajustar cada frame da una H dinámica que sigue a la cámara (resuelve "líneas chuecas que se quedan fijas"), pero re-detectar tiembla. Solución: aceptar el ajuste **solo si `overlap >= min_overlap`** (nunca fijar algo torcido) + EMA + conservar la última H buena (`kept`) cuando el frame ajusta mal. Combina exactitud y estabilidad sin congelar errores.
- **Línea central quitada del template/overlay:** el campo apenas la marca; incluirla en `_template_perimeter_cm` metía ruido a la métrica de overlap y temblor visible al overlay. Se muestrea solo el rectángulo interior.
- **Estabilización naive (`video_stabilize`) desactivada:** estima la afín global entre frames por features, pero en estos clips **los objetos en movimiento (robots, balón) dominan los features** y sesgan la trayectoria estimada → el warp "persigue" a los robots en vez de quitar el temblor de cámara. La estabilidad se resolvió mejor aguas abajo (gate + EMA sobre H), no en el preproceso.
- **Flujo óptico (`VideoHomographyTracked`) descartado como final:** funciona en cenital (1 `anchored`, 99 `tracked`, 0 `lost` en 9933) pero pierde tracking en lateral; el solver por líneas + overlap es más simple y robusto para cenital.
- **green_floor = SAM3, objetos/porterías = YOLO:** SAM3 segmenta la alfombra de forma fiable (base para las líneas); YOLO `best.pt` da cajas limpias de robot/balón y centroides de portería para orientación. Reparto explícito.

---

## 7. Pendiente / siguiente

1. **Detector de líneas dedicado para lateral:** agrupar segmentos con **Hough/LSD** → identificar las 4 líneas del campo → correspondencias **punto + línea** (estilo PnLCalib) que sí toleran perspectiva fuerte, en vez de los atajos de esquinas por `minAreaRect`.
2. **Incorporar la restricción cónica del círculo** (`detect_center_circle`) como correspondencia en el solver, no solo como held-out (ancla centro+escala bajo oclusión).
3. **Equipos por color en el minimap:** inferir color de equipo (los robots son clase única `robot`; hoy no se distinguen `robot_a`/`robot_b`). Mientras tanto: robots en **rojo**, balón en **naranja**.
4. **`cv2.undistort`** con calibración para la distorsión de barril de la lente (~10 cm de error central).
5. **DepthAnything-V2** (pod GPU) como cross-check de altura / balón en el aire.
6. **Reconstrucción cenital desde vista lateral** una vez que el lateral tenga H fiable, para alimentar el mismo minimap canónico.
