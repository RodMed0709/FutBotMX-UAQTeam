# Fase 2 — Pipeline YOLO + SAM3 (SAM3-céntrico) (context)

> Para quien llega nuevo (o con Claude Code) y NO tocó fase_1: qué hace este pipeline.

## La arquitectura (SAM3 al centro)
```
1. YOLO (best.pt)  → CAJAS rápidas  (robot, orange_ball, yellow_zone)   ~130 FPS
2. cajas → SAM3 BOX-PROMPT (Sam3TrackerModel) → MÁSCARAS finas          ← SAM3 = estrella
3. green_floor → SAM3 text-prompt (Sam3VideoModel, región estática)
4. overlay de máscaras + cajas → video
```

**Por qué así:** SAM3 text-prompt busca en TODA la imagen = lento. YOLO dice "el objeto está aquí" (caja) y SAM3 **box-prompt** segmenta solo ahí = rápido y confiable. La caja no necesita ser perfecta — **SAM3 hace la máscara buena**. Así SAM3 produce todas las máscaras (sigue siendo el centro) y YOLO solo lo acelera. **NO se usa YOLO-seg** (reemplazaría a SAM3 y aprendería de pseudo-labels imperfectos).

## Hallazgo técnico clave
SAM3 en HuggingFace tiene 2 caras desde el MISMO checkpoint (`assets/sam3`):
- `Sam3VideoModel` + `add_text_prompt` → segmentación por **texto** (detector open-vocab). Para green_floor.
- `Sam3TrackerModel` + `input_boxes` → segmentación por **geometría (cajas/puntos)**, estilo SAM2. Para box-prompt. (Carga del mismo checkpoint; el warning de "sam3_video→sam3_tracker" es benigno, los pesos del tracker sí están — verificado, da máscaras precisas.)

## Archivos (en `notebooks/fase_2_YOLO_SAM3/`)
- `pipeline_yolo_sam3.py` — librería: `load_models`, `boxes_to_masks` (box-prompt), `text_mask` (green_floor), `render` (genera video; modo `yolo` o `yolo_sam3`).
- `01_inference_pipeline.ipynb` — notebook que corre todo en GPU del pod.
- `make_demos.py` — genera los 2 videos demo.
- `best.pt` — el detector YOLO11s entrenado (copia de fase_1).
- `demo_yolo_only.mp4` — inferencia solo YOLO (cajas).
- `demo_yolo_sam3.mp4` — pipeline completo (cajas → máscaras SAM3 + green_floor).

## Cómo correr (POD GPU)
```bash
cd notebooks/fase_2_YOLO_SAM3
python make_demos.py        # genera los 2 videos
# o abre 01_inference_pipeline.ipynb y corre las celdas
```
Cambia `VIDEO` por cualquier video de testing. SAM3 + YOLO viven en el pod; NO corre en la laptop.

## Estado / siguiente
Pipeline funcional end-to-end. Falta: tracking temporal (IDs estables entre frames), homografía (usando green_floor), eventos (gol con yellow_zone), y LoRA team-aware (robot_a/robot_b).
