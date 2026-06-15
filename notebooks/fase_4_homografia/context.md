# Fase 4 — Homografía + Minimap · context.md

Estado de la fase de homografía (proyección de robots/balón a un minimap cenital
métrico del campo). Espejo local en `_fase4_stage/`; destino repo:
`notebooks/fase_4_homografia/` + `src/core/`.

---

## Objetivo

Proyectar por frame las posiciones de robots y balón a una **vista cenital
métrica** del campo (243×182 cm) vía homografía, y superponerla como **minimap**
con trails sobre el video. Cumple convocatoria **3.7.3** (post-procesamiento:
análisis geométrico = innovación sobre SAM3) + **3.5.2** (visualización del flujo
de juego) y habilita métricas cuantitativas (requisito categoría Profesional).

---

## Input correcto

- **Cámara superior** (`data/raw/18abril/Camara_superior/IMG_9933.MOV` 12 min,
  `IMG_9938.MOV` 9 min): campo completo con márgenes, plano cenital.
- Las tomas Meta-Glasses portrait NO sirven para homografía (close-up de un lado).
- **Hallazgo:** la cámara superior **SE MUEVE** (rota/panea entre frames) — NO es
  estática. Una sola H fija no cubre el clip → estimación por-frame + propagación.

---

## Geometría del campo (`field_template.py`)

Oficial (Reglas §7), en cm, origen esquina sup-izq de la alfombra:
- Alfombra 243 (largo, x) × 182 (ancho, y).
- Rectángulo interior de líneas: inset 12 cm → 219 × 158.
- Áreas chicas **redondeadas (forma D)**, no rectángulos de puntas.
- Círculo central r=30. Porterías: amarilla x≈6 (izq), azul x≈237 (der), boca 60.

---

## Dos caminos de homografía

### A. Camino SAM3 (`homography.py`) — el del repo
Anclas desde máscaras SAM3-texto: `green_floor` (cuadrilátero) + `yellow_zone`/
`blue_zone` (orientación). RANSAC + EMA + propagación.
**Limitación medida:** el borde superior de la alfombra lo corrompe la portería que
sobresale; el lado derecho lo recorta el frame → fit por-frame del borde de
alfombra poco fiable (refine PROBADO Y FALLA, no repetir).

### B. Camino color AUTOMÁTICO (`auto_homography.py`) — experimento local, sin GPU
- Aísla verde (alfombra) por HSV; dentro, las **líneas blancas**.
- Ancla = **4 esquinas del rectángulo interior** (219×158), que se ve completo
  aunque el borde de alfombra se corte. Esquinas vía **fit de 4 rectas-lado
  (`cv2.fitLine`) + intersección** (tolera perspectiva).
- Orientación por **color de portería** (amarilla→x<L/2, azul→x>L/2, condición
  dura → mata el flip 180°/espejo silencioso).
- `VideoHomography`: EMA + propagación de la última H buena.

**Resultados medidos:**
- Barrido 27 frames dispersos (12+9 min): **85% ok**, mediana error reproyección
  porterías **~12 cm (~5% campo)**, p90 ~20, max ~31.
- Clip contiguo 250 frames IMG_9933: **99.6% H estimada** (EMA ayuda).
- Modo de fallo único: un lado del rectángulo cortado por el borde → `ok=False` →
  propaga H vecina.
- **Distorsión de barril** de la lente arquea las líneas → ~10 cm de error central.
  Mejora futura: `cv2.undistort` con calibración.

---

## Revisión adversarial (agente) — fixes aplicados a `auto_homography.py`

- **C1:** ejes de `minAreaRect` derivados de `boxPoints` (la convención del ángulo
  cambió entre OpenCV <4.5 y ≥4.5).
- **C3:** orientación dura amarillo x<L/2 / azul x>L/2 (los centroides en y=91 eran
  invariantes al flip 180°/espejo → fallos silenciosos plausibles).
- **M2:** rechazo de outliers por MAD en `fitLine` (áreas-D y línea central
  contaminaban un lado).
- **M3:** umbral `det` 0.2 (rechaza lados casi paralelos = intersección degenerada).
- **M5:** validación de cuadrilátero convexo con área mínima.
- **Impacto:** eliminó homografías basura de 430–1017 cm (fallos silenciosos).

---

## Minimap + driver

- `minimap.py` (`MinimapRenderer`): trails por obj_id + composición arriba-derecha.
- `minimap_pipeline.py` (repo, SAM3): segmenta anclas + objetos por SAM3-texto,
  estima H, renderiza. Depende de todo el repo (`src.core.*`) + GPU.
- `minimap_auto.py` (autocontenido, local, sin SAM3/GPU): video→`VideoHomography`→
  minimap canónico→mp4. Objetos por color local (demo-grade). La H normaliza la
  orientación → minimap en coords canónicas (no rota).
- Demo: `minimap_auto_demo.mp4` (250 frames, 99.6% H).

---

## Notebooks de experimentación (`nb/`)

- `10_exp_canales_color.ipynb` — qué canal aísla qué (verde/blanco/amarillo/azul).
- `11_exp_homografia_auto.ipynb` — homografía auto + barrido + reproyección.
- `12_exp_depth.ipynb` — DepthAnything-V2 (POD GPU). Depth secundario: campo plano
  → la H ya da posición métrica; depth solo aporta altura/balón-en-aire.
- (`10` y `11` ejecutan limpio local; `12` requiere pod.)

---

## C. Camino INTEGRADO CON SAM3 (pod) — `pod_minimap_sam3.py`

El que pidió el equipo para Profesional (usa SAM3 de verdad). Corre en pod GPU.
- **YOLO `best.pt`** (clases: 0 robot, 1 orange_ball, 2 yellow_zone, 3 blue_zone) →
  cajas de robot/balón (foot-points) + centroides de porterías (orientación).
- **SAM3 `green_floor` (text_mask)** → máscara de alfombra.
- `auto_homography.solve_masks(rgb, carpet, yc, bc)` → H sobre esas anclas SAM3/YOLO
  (homografía explícitamente construida sobre SAM3 = innovación 3.7.3).
- Tracker greedy NN → IDs estables. Minimap canónico + composición. Escribe RGB.

**Bug resuelto:** la máscara `green_floor` deja las líneas blancas como HUECOS
(no son verdes) → `white∩carpet` las borraba (white ~45 px). Fix `_white_in_carpet`:
`MORPH_CLOSE 25×25` sobre la alfombra antes de intersectar. Post-fix white 45-59k,
ok=True, err 9-23 cm.

**Gate de consistencia temporal** (`VideoHomography`): la cámara casi no se mueve →
una H que salta >70 px (esquinas del campo) entre frames = falso positivo → se
RECHAZA y se mantiene la previa. La 1ª H (ancla) solo se fija con error <22 cm.
Más EMA + propagación. Render: margen negro fino, robot=cuadro gris, balón=naranja,
esquinas pintadas + rectángulo/círculo reproyectados sobre el video.

**Videos:** 5 en `outputs/` (IMG_9933_a/b/c + IMG_9938_a/b). Log `outputs/batch.log`.

## Confirmación visual de la convocatoria (3.5)

- **3.5.3 video ≤2 min:** original + segmentado (lado a lado/superpuesto) +
  indicadores claros de segmentación+tracking+visualización + explicación texto/voz.
  **+ Reel IG ≥30 s** (link en README).
- **3.5.2:** ≥1 visualización — trails (flujo de juego) cumple.
- **Pro (3.7):** sumar métricas cuantitativas (velocidad cm/s, posesión) que la
  homografía métrica habilita.

## Pendiente / siguiente

1. **Usar yolo_sam3 (fase_2) para los OBJETOS** (robots/balón) en vez del detector
   de color demo → detecciones limpias. Es el camino que el equipo quiere
   (innovación sobre SAM3). Requiere pod (GPU).
2. Decidir fuente de anclas de H: (a) método color robusto sobre el frame, o
   (b) SAM3 `green_floor` + extracción robusta de esquinas sobre las líneas blancas
   dentro de esa máscara. La opción (b) "usa SAM3" explícitamente.
3. Integrar `VideoHomography` al `minimap_pipeline.py` del repo (reemplazar
   `estimate_homography`, conservar objetos vía `tracks_json` de fase_2).
4. Correr NB12 depth en pod.
5. Métricas cuantitativas sobre la H: velocidad cm/s, distancia, heatmap, posesión.
6. `cv2.undistort` (barril). Mover módulos a repo `src/core`.

---

## Notas de entorno

- Local fuera de git en `_fase4_stage/` (cv2 4.13 en anaconda, numpy pineado <2
  para no romper matplotlib).
- Pod env: `/workspace/envs/futbot-cpu/bin/python` (torch+CUDA, transformers 5.9,
  ultralytics, kernels 0.12.3). SAM3 en `assets/sam3/`. SSH fresco cada reconexión.
</context>
