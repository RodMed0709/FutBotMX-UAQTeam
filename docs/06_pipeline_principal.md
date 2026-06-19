# Fase 06 — Pipeline principal (modular, config-driven)

> La **puerta de entrada única** del proyecto. `run_inference` enruta cada video a
> segmentación o tracking, resuelve el muestreo de frames y unifica la salida;
> `run_batch` lo corre sobre lotes reusando SAM3. Todo es **config-driven** y los ejes
> (detector × tracker) se eligen por nombre.

- **Notebook:** [`cookbook_pipeline.ipynb`](../notebooks/cookbook_pipeline.ipynb)
- **Tareas SDD:** [`inference_schema`](../.specs/inference_schema/), [`optional_render`](../.specs/optional_render/),
  [`unified_inference`](../.specs/unified_inference/), [`batch_inference`](../.specs/batch_inference/),
  [`batch_detector_tracker`](../.specs/batch_detector_tracker/),
  [`config_aware_output_paths`](../.specs/config_aware_output_paths/),
  [`progress_reporting`](../.specs/progress_reporting/)

---

## `src/core/inference.py` — la fachada

Punto único por video. Router fino: valida `mode`/`sampling` **antes** de cargar SAM3,
resuelve el muestreo según el modo y unifica el retorno a `{"json", "video", "index"}`.

```python
run_inference(
    video_path: Path | str,
    mode: str = "segmentation",          # "segmentation" | "tracking"
    output_path: Path | None = None,
    classes: list[dict] | None = None,   # None ⇒ clases del config
    sampling: str = "auto",              # auto | quota | all | contiguous
    max_frames: int | None = None,       # tope contiguo (solo tracking)
    bundle: Sam3Bundle | None = None,    # SAM3 precargado (reuso)
    include_masks: bool = False,         # COCO-RLE en el JSON
    render_video: bool = True,           # mp4 opcional
    detector: str | None = None,         # "sam3_text" | "yolo_sam3"
    tracker: str | None = None,          # "bytetrack" | "botsort"
    run_label: str | None = None,        # namespacing de salidas por config
    progress: bool = True,
) -> dict                                # {"json", "video", "index"}
```

Resolución de muestreo (`sampling="auto"`):

| sampling | segmentation | tracking |
|---|---|---|
| `auto` | cuota equiespaciada | prefijo contiguo (respeta `max_frames`) |
| `quota` | cuota | **`ValueError`** |
| `all` | completo | completo (`max_frames=None`) |
| `contiguous` | **`ValueError`** | contiguo |

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `run_inference(...)` | [`inference.py:87`](../src/core/inference.py#L87) | Valida, resuelve muestreo, enruta a `run_pipeline`/`track_video`, unifica salida. **No** reimplementa el loop. |

## `src/core/pipeline.py` — orquestador per-frame (segmentación)

```python
run_pipeline(
    video_path, output_path=None, all_frames=False, mode="per_frame",
    classes=None, bundle=None, include_masks=False, render_video=True,
    detector=None, run_label=None, progress=True,
) -> dict[str, Path | None]
```

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `run_pipeline(...)` | [`pipeline.py:90`](../src/core/pipeline.py#L90) | `video → extract_frames → detector por frame → overlay → mp4 + JSON`. `obj_id` **inestable**. Acepta `detector=` (`yolo_sam3` también segmenta standalone). |

(El orquestador de **tracking** es [`track_video`](05_tracking.md).)

## `src/core/batch.py` — lotes

```python
run_batch(
    mode="segmentation", split=2, videos=None, sampling="auto", max_frames=None,
    include_masks=False, render_video=False, overwrite=False,
    detector=None, tracker=None, run_label=None, progress=True,
) -> list[dict]
```

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `run_batch(...)` | [`batch.py:160`](../src/core/batch.py#L160) | N videos (por `split` o lista), SAM3 cargado **una vez**, **skip-done** (salta si el JSON existe), aísla errores por video, resumen con timing/VRAM. `render_video=False` por defecto (el dato es el producto). |

## `src/core/inference_schema.py` — el esquema de salida

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `inference_paths(stem, namespace=None)` | [`inference_schema.py:219`](../src/core/inference_schema.py#L219) | Ubica `outputs/inference/[<run_label>/]<stem>/<stem>.{json,mp4}`. El `run_label` evita que configs distintas se pisen (resume por config). |
| `build_header(...)` | [`inference_schema.py:175`](../src/core/inference_schema.py#L175) | Cabecera auto-descriptiva (config, versiones, modelo). |
| `encode_rle` / `decode_rle` | [`inference_schema.py:57`](../src/core/inference_schema.py#L57) | Máscaras COCO-RLE. |
| `write_inference_json(...)` | [`inference_schema.py:243`](../src/core/inference_schema.py#L243) | Escribe el JSON unificado. |

---

### Cómo encaja con el resto

Esta fase es **lo que el `main` reproducible llamará** para producir el tracking JSON. Su
salida alimenta el [benchmark](07_benchmark.md) y abre la mitad CPU-local
([homografía](08_homografia.md) → [métrica](09_capa_metrica.md) → [eventos](10_eventos.md)).
