# Fase 0 — Segmentación con SAM3 + Ground Truth (context)

> Para quien llega nuevo (o con Claude Code): qué se hizo aquí y por qué.

## Objetivo
Establecer **SAM3 (Meta Segment Anything 3)** como el motor de segmentación del proyecto (fútbol robótico, RoboCup) y producir el **ground-truth de evaluación** anotado por humanos.

## Qué se hizo
1. **Inferencia SAM3 zero-shot** con text-prompts (notebooks `01`–`03`): segmentación multi-clase por frame y tracking en video.
   - Clases y prompts ganadores: `robot`→"robot", `orange_ball`→"orange ball", `green_floor`→"green playing surface with lines", `yellow_zone`→"yellow zone".
   - **Manejo correcto de máscara**: SAM3 saca logits ~288×288 → **upscale BILINEAR** → threshold (no NEAREST sobre binario, que pixela).
2. **Generación del set de evaluación**: 600 frames de 20 videos de testing (equiespaciados sobre todo el video) en `data/testing_frames/`, manifiesto `assets/testing_frames.csv`.
3. **TTA ×3** (identidad + flip + gamma) con promediado a nivel instancia (empareja por IoU, vota ≥2/3) → máscaras suaves.
4. **Subida a Supervisely** como instance segmentation (bitmaps): project `FutBot_Testing_600_smoothv3_tta` (id 378046), 600 imgs, 5 clases (`robot_a`, `robot_b` vacía para equipo B, `orange_ball`, `green_floor`, `yellow_zone`).
   - Imágenes renombradas legibles `<video>_f<frame_real>.png`. Traza en `assets/gt_sly_name_map.csv`.
   - 2 Labeling Jobs (300/300 por video) para repartir corrección humana.

## Archivos clave (en `notebooks/fase_0/`)
- `01_sam3_text_inference.ipynb`, `02_sam3_multi_class_per_frame.ipynb`, `03_sam3_video_tracking.ipynb` — inferencia SAM3 (la referencia del manejo de máscara correcto).
- `prueba_sam3_*.py` / `subir_smoothv3_tta.py` / `reformatear_proyecto.py` / `crear_jobs.py` — pipeline de GT a Supervisely.
- `outputs/` — COCO/datasets intermedios.

## Estado
GT en Supervisely listo para que 2 anotadores corrijan máscaras + asignen `robot_b` (equipo B). Estos 600 = **training de LoRA** + held-out para métricas (NO se usan como train de YOLO; ver fase_1).

## Decisión central
**SAM3 es el centro del proyecto.** Todo lo demás (YOLO en fase_1/2) existe para acelerarlo, no para reemplazarlo.
