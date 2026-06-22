# Spec — `oral_baselines` (Fase 8) — baselines comparativas para MICAI oral

## Contexto / motivación
El review adversarial MICAI marcó 2 huecos que limitan el paper a poster:
1. **Sin baseline comparativa** — todos los números (mIoU 0.922, mAP50 0.947) están aislados.
   La Intro invoca "classical robot-soccer vision based on hand-tuned color thresholds" como el
   foil pero NUNCA lo mide.
2. **Tesis de velocidad contradicha** por Table 4 (SAM3-text 1.82 vs YOLO+SAM3 1.71 FPS): el
   pipeline de máscaras NO es más rápido. El valor real = amortización de anotación + modo box-only.

## Objetivo
Producir 2 resultados, ambos desde assets que YA existen (sin anotar nada nuevo):

### E1 — Baseline HSV (umbrales de color) vs SAM3 zero-shot, contra los 600 GT
Detector clásico HSV por clase de color (orange_ball, yellow_zone, blue_zone, green_floor) +
morfología. Robots NO son un color único → se espera que HSV falle ahí (eso es el punto).
Métrica IDÉNTICA al eval de SAM3 (`01_seg_eval_vs_gt.py`): IoU/Dice/Boundary-IoU por clase,
mismos 600 frames GT, robots unidos (robot_a∪robot_b).
**Salida:** `outputs/seg_eval/seg_eval_hsv.csv` + `.json`.
**Claim que habilita:** "SAM3 zero-shot supera a umbrales HSV hechos a mano (esp. en robots),
SIN tuning por clase" → cierra el loop de la Intro.

### E2 — FPS de YOLO11 standalone (modo box-only)
Cronometrar inferencia de `assets/yolo/best.pt` SIN SAM3, sobre N frames de los videos test
(warm-up descartado), reportar FPS mean±std + ms/frame. Respalda el reframe "modo box-only en
tiempo real".
**Salida:** `outputs/benchmark/yolo_fps.json`.
**Claim que habilita:** "YOLO11 solo corre a XX FPS (real-time), vs el pipeline de máscaras
SAM3 a ~1.7 FPS" → distingue modo box-only (rápido) de modo máscaras (preciso).

## No-objetivos
- NO goal precision/recall (descartado por el equipo).
- NO re-anotar; NO tocar el pipeline SAM3 existente.
- HSV es CPU; YOLO-FPS usa GPU.

## Contratos
- E1 reutiliza el loader GT Supervisely y las métricas exactas de `01_seg_eval_vs_gt.py`.
- Mismas EVAL_CLASSES y mismo dataset `testing_600`.
- Resultados versionados junto a los de SAM3 para comparación directa en Table 4 del paper.
