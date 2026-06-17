# Fase 02 — Preliminares (exploración + innovación base)

> Trabajo **exploratorio** que precede al `src/` productivo: las primeras pruebas de
> SAM3 y, sobre todo, el **fine-tuning de YOLO con auto-etiquetas de SAM3** que habilita
> el detector [`yolo_sam3`](03_deteccion.md) — la innovación del proyecto sobre SAM3.
>
> **Aviso:** este código vive en `notebooks/`, **no** en `src/`. Es referencia
> reproducible del *cómo se llegó*, no parte del pipeline de producción.

- **Notebooks:** [`fase_0/`](../notebooks/fase_0/) (spikes SAM3),
  [`fase_1/`](../notebooks/fase_1/) (auto-label + entrenamiento YOLO)
- **Sin tareas SDD** (exploración previa a la formalización).

---

## `notebooks/fase_0/` — exploración SAM3

Spikes para entender SAM3 por texto, multi-clase per-frame y propagación en video. No
es código de pipeline; sirvió para fijar prompts y el esquema de `Detection`.

| Notebook | Qué exploró |
|---|---|
| `01_sam3_text_inference.ipynb` | SAM3 por prompt de texto sobre una imagen. |
| `02_sam3_multi_class_per_frame.ipynb` | Varias clases por frame (germen de `detect_classes_in_frame`). |
| `03_sam3_video_tracking.ipynb` | Propagación en video con SAM3. |
| `06`–`09` | Overlay, pipeline completo, evaluación de prompts de gol. |

> Contexto detallado: [`notebooks/fase_0/context.md`](../notebooks/fase_0/context.md).

## `notebooks/fase_1/` — la innovación: YOLO afinado con SAM3

El bucle **SAM3-assisted labeling**: SAM3 genera auto-etiquetas sobre los videos
NO-testing y un YOLO aprende a localizar cajas rápido. Esas cajas alimentan luego a SAM3
por box-prompt en producción ([`yolo_sam3`](03_deteccion.md)).

| Notebook / script | Qué hace |
|---|---|
| `01_sam3_to_yolo_autolabel.ipynb` | SAM3 → dataset YOLO (auto-etiquetas). |
| `02_train_yolo.ipynb` / `train_yolo.py` | Entrena/afina el YOLO sobre esas etiquetas. |
| `sam3_yolo.py`, `sam3_yoloseg.py`, `run_autolabel.py` | Drivers del auto-etiquetado. |

Pesos resultantes: `assets/yolo/best.pt` (git-ignored), cargados por
[`load_yolo`](../src/core/detectors/yolo_boxes.py#L129).

> Contexto detallado: [`notebooks/fase_1/context.md`](../notebooks/fase_1/context.md).

---

### Cómo encaja con el resto

Esta fase **produce el modelo** que consume [03 Detección](03_deteccion.md). La
estrategia de fine-tuning definitiva (Roboflow vs. SAM3-assisted) sigue **abierta**; lo
ya entregado es el camino SAM3-assisted, suficiente para `yolo_sam3` y el
[benchmark](07_benchmark.md).
