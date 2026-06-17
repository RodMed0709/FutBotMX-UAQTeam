# Fase 04 — Segmentación

> Convierte el frame en **máscaras por clase**. Aquí se carga SAM3 una sola vez, se
> corre por prompts de texto y se compone el overlay con color **por clase**. La unidad
> que produce —`Detection`— es la **moneda** que circula por detección, tracking y todo
> el post-proceso.

- **Notebooks:** [`fase_0/06_segmentation_overlay_check.ipynb`](../notebooks/fase_0/06_segmentation_overlay_check.ipynb),
  [`fase_0/07_pipeline_full_video_check.ipynb`](../notebooks/fase_0/07_pipeline_full_video_check.ipynb)
- **Tareas SDD:** [`sam3_loader`](../.specs/sam3_loader/), [`classes_config`](../.specs/classes_config/),
  [`text_segmentation`](../.specs/text_segmentation/), [`segmentation_overlay`](../.specs/segmentation_overlay/),
  [`pipeline_runner`](../.specs/pipeline_runner/)

---

## `src/core/sam3_loader.py` — cargar SAM3 una vez

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `Sam3Bundle` | [`sam3_loader.py:36`](../src/core/sam3_loader.py#L36) | `(processor, model, device)`: el modelo cargado, que se pasa entre llamadas para reusar (carga única en lotes). |
| `load_sam3(use_cache=True, device=None)` | [`sam3_loader.py:138`](../src/core/sam3_loader.py#L138) | Carga SAM3 (HF transformers, bf16, cuda si hay). Cacheado. Pesos desde `assets/sam3`. |
| `ensure_tracker_loaded(bundle)` | [`sam3_loader.py:168`](../src/core/sam3_loader.py#L168) | Garantiza el componente de tracking de SAM3 cargado. |

## `src/core/segmentation.py` — máscaras por clase

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `Detection` | [`segmentation.py:32`](../src/core/segmentation.py#L32) | **La moneda compartida**: `(obj_id, mask, score)`. `obj_id` inestable per-frame, estable en tracking. |
| `detect_classes_in_frame(frame, classes, bundle)` | [`segmentation.py:163`](../src/core/segmentation.py#L163) | `{nombre_clase: [Detection]}` corriendo una sesión SAM3 de **texto por clase**. Es lo que envuelve el detector [`sam3_text`](03_deteccion.md). |
| `segment_with_text(...)` | [`segmentation.py:110`](../src/core/segmentation.py#L110) | Una clase, un prompt → máscaras. |

**Las clases son config-data:** cada una tiene `name`, `sam3_prompts`, `color`,
`coco_id` (y `yolo_id` si va por YOLO) bajo la clave `classes` del config. Añadir una
clase es **solo config**, sin tocar código.

## `src/core/overlay.py` — pintar las máscaras

Función principal configurable (color **por clase**, transparencia ajustable):

```python
overlay_detections(
    frame: np.ndarray,                       # (H,W,3) RGB uint8
    detections_by_class: dict[str, list],    # {clase: [Detection]} (usa det.mask)
    classes: list[dict] | None = None,       # None ⇒ clases del config
    alpha: float | None = None,              # None ⇒ visualization.overlay_alpha
) -> np.ndarray                              # frame compuesto (no muta la entrada)
```

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `overlay_detections(...)` | [`overlay.py:96`](../src/core/overlay.py#L96) | Mezcla las máscaras sobre el frame con el color de cada clase. **No** dibuja `obj_id` (eso es [05](05_tracking.md)). |
| `show_overlay(...)` | [`overlay.py:143`](../src/core/overlay.py#L143) | Variante solo-display. |

---

### Cómo encaja con el resto

`detect_classes_in_frame` + `overlay_detections` son lo que orquesta `run_pipeline` en
[06 Pipeline principal](06_pipeline_principal.md) para el modo **segmentación**. El mismo
`Detection` (sin overlay) es lo que el [tracking](05_tracking.md) convierte en cajas para
asociar identidades.
