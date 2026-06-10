# Plan técnico — Segmentación por caja con SAM3 (`sam3_box_prompt`)

- **Tarea atómica:** `sam3_box_prompt`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso:** esta tarea es la primera del proceso que integra el pipeline YOLO + SAM3 a `src/`.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo (a) **exponer la 2ª cara de SAM3**
(`Sam3TrackerModel` / `Sam3TrackerProcessor`) extendiendo `sam3_loader` con carga
**perezosa/opt-in**, y (b) implementar el building block **box-prompt**: dado un
frame y una lista de cajas xyxy, devolver una **máscara fina por caja** empaquetada
en la moneda común `Detection`, alineada 1:1 con las cajas. Sustituye el
`boxes_to_masks` suelto y ad-hoc del notebook `fase_2_YOLO_SAM3` por una pieza
config-driven, reutilizable y con las convenciones del repo. Además, definir el
script de validación manual (smoke) que corre en el pod.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Carga del modelo (cara tracker):** `transformers` — `Sam3TrackerModel` y
  `Sam3TrackerProcessor` con `from_pretrained` apuntando al **mismo directorio
  local** de pesos que ya resuelve `sam3_loader` (`working_dirs.sam3_dir`). Es la
  API validada por el notebook `fase_2_YOLO_SAM3`.
- **Inferencia / tensores / dtype:** `torch` (`@torch.no_grad()`,
  `torch.bfloat16`, `.to(device)`, `torch.is_floating_point`).
- **Imagen:** `PIL.Image` (el processor espera imágenes PIL); `numpy` para la
  máscara booleana de salida.
- **Moneda común:** `src.core.segmentation.Detection` (reutilizada, no se redefine).
- **Carga base / rutas:** `src.core.sam3_loader` (se extiende; reusa
  `_resolve_sam3_dir`, `get_abs_path`).
- **Frames para el test:** `src.core.frame_extraction.extract_frames` (no `decord`
  suelto).

> Nota: `torch`, `transformers` y `PIL` son pesadas; ver §3.6 (imports perezosos)
> para que `import src.core` no las arrastre.

---

## 3. Diseño

### 3.1 Ubicación y módulos

- **Subpaquete nuevo:** `src/core/detectors/` con `__init__.py`.
- **Archivo nuevo:** `src/core/detectors/box_prompt.py`.
- **Extensión:** `src/core/sam3_loader.py` (campos nuevos en `Sam3Bundle` + función
  de carga perezosa del tracker).
- **Exportación:** en `src/core/detectors/__init__.py`,
  `from src.core.detectors.box_prompt import boxes_to_masks` y `__all__ =
  ["boxes_to_masks"]`. (No se fuerza re-exportar desde `src/core/__init__.py` para
  no encarecer ese import; queda como decisión menor de la implementación.)

### 3.2 Extensión de `sam3_loader` — carga perezosa de la cara tracker

`Sam3Bundle` gana **dos campos opcionales** (default `None`), de modo que la carga
por defecto (`load_sam3()`) **no cambia** y no carga el tracker:

```python
@dataclass
class Sam3Bundle:
    processor: Any
    model: Any
    device: str
    tracker_processor: Any = None   # 2ª cara, perezosa
    tracker_model: Any = None       # 2ª cara, perezosa
```

Función nueva, idempotente, que rellena esos campos **bajo demanda** sobre el mismo
bundle (singleton incluido) y devuelve el bundle:

```python
def ensure_tracker_loaded(bundle: Sam3Bundle) -> Sam3Bundle:
    if bundle.tracker_model is not None:
        return bundle
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    sam3_dir = _resolve_sam3_dir()
    bundle.tracker_processor = Sam3TrackerProcessor.from_pretrained(str(sam3_dir))
    bundle.tracker_model = (
        Sam3TrackerModel.from_pretrained(
            str(sam3_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(bundle.device).eval()
    )
    return bundle
```

- Mismas convenciones que `_build_bundle`: `bfloat16`, `low_cpu_mem_usage=True`,
  `.eval()`, **device del bundle** (carga e inferencia comparten fuente).
- **Idempotente**: segunda llamada no recarga. Sobre el bundle cacheado, la cara
  tracker queda cargada una sola vez para toda la sesión.
- `torch`/`transformers` se importan **dentro** de la función (perezoso).

### 3.3 Building block `boxes_to_masks` — firma

```python
def boxes_to_masks(
    frame: np.ndarray,
    boxes: list[tuple[float, float, float, float]] | list[list[float]],
    bundle: Sam3Bundle | None = None,
    scores: list[float] | None = None,
) -> list[Detection]:
    ...
```

- `frame`: numpy `(H, W, 3)` RGB (convención del repo).
- `boxes`: lista de cajas **xyxy en píxeles absolutos**.
- `bundle`: SAM3 ya cargado; `None` ⇒ `load_sam3()`. En ambos casos se invoca
  `ensure_tracker_loaded(bundle)`.
- `scores`: score por caja (el del detector). `None` ⇒ default `1.0` por detección.
- **Retorno:** `list[Detection]` con **N elementos** (N = nº de cajas), 1:1 y en
  orden.

### 3.4 Flujo interno (`@torch.no_grad()`)

1. `bundle = bundle or load_sam3()`; `ensure_tracker_loaded(bundle)`.
2. **Caso vacío:** si `not boxes` → `return []` (no se invoca al modelo).
3. `img = PIL.Image.fromarray(frame)` (RGB).
4. `inp = bundle.tracker_processor(images=[img], input_boxes=[boxes],
   return_tensors="pt").to(bundle.device)`.
5. Castear a `bfloat16` **solo** los tensores flotantes:
   `inp2 = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
   for k, v in inp.items()}`.
6. `out = bundle.tracker_model(**inp2, multimask_output=False)`.
7. `masks = bundle.tracker_processor.post_process_masks(out.pred_masks.cpu(),
   inp["original_sizes"])` → máscaras a **resolución del frame** (este es el
   camino correcto; **no** se usa `_mask_from_logits`, que es del text-prompt).
8. `m = np.array(masks[0]); if m.ndim == 4: m = m[:, 0]` → `m` con forma
   `(N, H, W)`; `m = m.astype(bool)`.
9. Empaquetar: para `i, mask in enumerate(m)` →
   `Detection(obj_id=i, mask=mask, score=(scores[i] if scores else 1.0))`.
   `obj_id` = índice posicional (per-frame, **inestable** por diseño).
10. `return detections`.

> **Robustez (defensivo):** si por cualquier razón el modelo devuelve menos
> máscaras que cajas, se empaquetan las que haya **en orden** sin reventar; el
> contrato esperado es 1:1 (`multimask_output=False`).

### 3.5 Máscara vacía/degenerada

No se filtra: si una máscara sale vacía (`mask.any() == False`), su `Detection`
**se devuelve igual** (preserva el 1:1). El descarte de máscaras vacías es del
consumidor (p. ej. `inference_schema.detection_record`, que ya retorna `None` si la
caja envolvente es nula).

### 3.6 Imports perezosos

`box_prompt.py`: a nivel de módulo solo `numpy` y el `Detection` (dataclass barata).
`torch` y `PIL.Image` se importan **dentro** de `boxes_to_masks`. En `sam3_loader`,
`transformers`/`torch` ya se importan dentro de las funciones de carga; la nueva
función mantiene ese patrón.

### 3.7 Warning benigno

La carga de `Sam3TrackerModel` emite `sam3_video → sam3_tracker`. **Se documenta**
en el docstring de `ensure_tracker_loaded` como esperado (los pesos del tracker sí
están; verificado que produce máscaras precisas). **No** se silencia con código
(`warnings.filterwarnings`) para no ocultar otros avisos.

---

## 4. Script de validación (smoke)

Archivo: `testing/test_box_prompt.py` (script manual, **no** pytest; corre en el
**pod** por requerir SAM3 + GPU).

**Estrategia (sin depender de YOLO, que aún no existe):** usar el detector
SAM3-text ya existente para producir cajas reales y re-segmentarlas por box-prompt
(prueba cruzada del camino nuevo):

1. `extract_frames(<video real>, ...)` → tomar **1 frame** (un video que **no** sea
   de `splits.forced_testing`).
2. `bundle = load_sam3()`.
3. `dets = detect_classes_in_frame(frame, bundle=bundle)` (text-prompt) y derivar
   cajas con `mask_to_bbox_centroid` para una clase con detecciones (p. ej.
   `robot`). Alternativa: cajas hardcoded conocidas si se prefiere independencia.
4. `out = boxes_to_masks(frame, boxes, bundle=bundle, scores=...)`.
5. **Aserciones:** `len(out) == len(boxes)`; al menos una máscara no vacía
   (`mask.sum() > 0`); cada máscara con forma `(H, W)` y dtype bool.
6. **Inspección visual:** componer un overlay con `overlay_detections` (o guardar
   un PNG) bajo `outputs/` para revisar a ojo que las máscaras encajan con las
   cajas. (Coherente con la filosofía de tests: smoke funcional ahora; validación
   visual con caso real cuando se pida.)

---

## 5. Riesgos y consideraciones

- **Doble carga de pesos en VRAM:** la cara tracker es un segundo modelo en GPU
  además del video/text. Aceptado (es el diseño del notebook, corre en el pod). La
  carga perezosa evita pagarlo cuando no se usa box-prompt.
- **Forma de `pred_masks`:** puede variar (`(N,H,W)` vs `(N,1,H,W)`); el paso 8
  normaliza `ndim==4`. Si una versión de `transformers` cambia la forma, el smoke
  lo detecta.
- **dtype del processor:** algunos tensores enteros (índices/cajas) no deben
  castearse a bf16; por eso el casteo es **solo** sobre flotantes (`is_floating_point`).
- **Sin cambios de config:** esta tarea no añade claves; usa `working_dirs.sam3_dir`
  ya existente. (Las claves de YOLO/BoT-SORT entran en sus tareas.)
- **No regresión:** los campos nuevos de `Sam3Bundle` tienen default `None` y la
  ruta por defecto de `load_sam3()` no toca el tracker → los llamadores actuales no
  cambian.

---

## 6. Qué NO incluye este plan

- El detector YOLO que produce las cajas (tarea `yolo_detector`).
- El cableado a `track_video`/`run_pipeline`/`run_inference`/batch (tarea
  `detector_strategy`).
- `green_floor` por text-prompt (ya existe), propagación SAM3-video, e identidad
  estable entre frames.
- La descomposición en pasos accionables y su checklist: corresponde a `tasks.md`.
