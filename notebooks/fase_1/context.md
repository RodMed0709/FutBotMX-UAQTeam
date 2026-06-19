# Fase 1 — Detector YOLO11 (destilación de SAM3) (context)

> Para quien llega nuevo (o con Claude Code): qué se hizo aquí y por qué.

## Idea (destilación maestro→alumno)
SAM3 es preciso pero **lento** (~1–8 FPS, no deployable). YOLO es **rápido** (~130 FPS, deployable). Entonces: **SAM3 (maestro) auto-etiqueta** videos → **YOLO11 (alumno) aprende** a localizar rápido.

**YOLO NO reemplaza a SAM3.** En el pipeline final (fase_2) YOLO da cajas → SAM3 las convierte en máscaras (box-prompt). Aquí solo se entrena el detector.

## Anti-leakage (clave)
- Train: SAM3 auto-labels de los **103 videos NO-testing** (split 0 reserva + 1 finetuning).
- Eval: los 20 videos de testing (intocados) + held-out.
- Videos disjuntos train/test → cero fuga.

## Qué se hizo
1. **Auto-label SAM3→YOLO** (`sam3_yolo.py` + `run_autolabel.py`): 103 videos → `yolo_dataset/` (2460 train + 630 val imgs, formato YOLO detección, split por video). Clases: `robot`, `orange_ball`, `yellow_zone` (sin `green_floor`: como caja sería casi todo el frame, inútil — eso es segmentación).
2. **Entrenamiento YOLO11s** (`train_yolo.py`): 100 epochs, imgsz 960.

## Resultados (val = fidelidad de destilación vs SAM3, NO accuracy vs humano)
- **mAP50 0.947, mAP50-95 0.850.** robot 0.962, orange_ball 0.988, yellow_zone 0.891.
- Pesos: `runs/yolo11s_futbot/weights/best.pt`.

## Archivos (en `notebooks/fase_1/`)
- `sam3_yolo.py` — librería auto-label (revisada por agente experto: cajas no-degeneradas, escritura atómica, resumable, anti-OOM, score≥0.40).
- `01_sam3_to_yolo_autolabel.ipynb`, `run_autolabel.py` — generar dataset.
- `02_train_yolo.ipynb`, `train_yolo.py` — entrenar.
- `yolo_dataset/` (data.yaml, images/labels train/val, manifest.csv), `runs/yolo11s_futbot/` (best.pt).

## Pendiente
- Eval honesto de `best.pt` sobre held-out con GT manual (mAP real + FPS) → tabla SAM3 vs YOLO.

## Nota
`yolo11s` = YOLO v11 small (lo usado). `yolo26n.pt` que aparece = descarga del AMP-check de Ultralytics, basura, ignorar/borrar.
