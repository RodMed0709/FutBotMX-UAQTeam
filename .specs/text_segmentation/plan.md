# Plan técnico — Núcleo de segmentación por texto (`text_segmentation`)

- **Tarea atómica:** `text_segmentation`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** no hay draft previo para este plan.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar el núcleo de inferencia por-frame:
la dataclass `Detection`, `segment_with_text(frame, prompt)` y
`detect_classes_in_frame(frame, classes)`, devolviendo máscaras booleanas a
tamaño del frame, obteniendo el modelo de `load_sam3` y las clases de la
configuración. Además, definir el script de validación manual (en GPU).

---

## 2. Stack técnico

- **Python:** 3.11.
- **Modelo / inferencia:** SAM3 vía `transformers`, a través del `Sam3Bundle` que
  entrega `src.core.sam3_loader.load_sam3` (processor + model + device).
- **Tensores:** `torch` (`@torch.no_grad()`, `.detach().cpu().float().numpy()`).
- **Procesado de imagen:** `numpy` (frames), `PIL.Image` (entrada a SAM3),
  `cv2` (upscale bilinear de logits — `cv2.resize`).
- **Post-proceso de máscara:** lo aporta `kernels` en la inferencia (disponible en
  los 3 entornos); **no** se porta `refine_mask`.
- **Configuración:** `json` (estándar) + `src.utils.get_abs_path` para leer las
  clases (`config["classes"]`), con la misma convención que el resto del repo.

> `torch`, `cv2` y `PIL` son dependencias pesadas; se importan de forma **perezosa**
> dentro de las funciones (ver §3.6) para que `import src.core` no las arrastre.

---

## 3. Diseño

### 3.1 Ubicación y módulo

- Archivo nuevo: `src/core/segmentation.py`.
- Exportación en `src/core/__init__.py`:
  `from src.core.segmentation import Detection, segment_with_text, detect_classes_in_frame`
  y sumarlos a `__all__`.

### 3.2 Dataclass `Detection`

```python
@dataclass
class Detection:
    obj_id: int
    mask: np.ndarray   # booleana, forma (H, W) del frame de entrada
    score: float
```

### 3.3 Firmas

```python
def segment_with_text(
    frame: np.ndarray,
    prompt: str,
    bundle: Sam3Bundle | None = None,
) -> list[Detection]: ...

def detect_classes_in_frame(
    frame: np.ndarray,
    classes: list[dict] | None = None,
    bundle: Sam3Bundle | None = None,
) -> dict[str, list[Detection]]: ...
```

- `frame`: `np.ndarray (H, W, 3)` RGB (un frame de `extract_frames`).
- `bundle`: si es `None`, se obtiene con `load_sam3()`.
- `classes`: si es `None`, se leen de la configuración (§3.7).

### 3.4 `segment_with_text` — inferencia por prompt

1. **Validar** `frame` (§3.8) y obtener `H, W = frame.shape[:2]`.
2. **Resolver** el bundle: `bundle = bundle or load_sam3()`.
3. **Convertir** a PIL: `img = Image.fromarray(frame)`.
4. **Sesión SAM3** (port NB 01/02), bajo `@torch.no_grad()`:
   ```python
   session = bundle.processor.init_video_session(
       video=[img], inference_device=bundle.device, dtype=torch.bfloat16,
   )
   session = bundle.processor.add_text_prompt(session, text=prompt)
   out = bundle.model(inference_session=session, frame_idx=0)
   ```
5. **Recolectar detecciones:** por cada `oid in out.object_ids`:
   - `m = out.obj_id_to_mask[oid].detach().cpu().float().numpy()`; aplanar
     (`m[0,0]` si 4D, `m[0]` si 3D) a `(h, w)` de logits.
   - `mask = _mask_from_logits(m, W, H)` (§3.5) → booleana `(H, W)`.
   - `score = float(out.obj_id_to_score.get(oid, 0.0))`.
   - `Detection(obj_id=int(oid), mask=mask, score=score)`.
6. **Devolver** la lista (vacía si no hay objetos).

### 3.5 `_mask_from_logits` — upscale bilinear + umbral (port NB 01/02)

```python
def _mask_from_logits(logits: np.ndarray, W: int, H: int) -> np.ndarray:
    lo = logits.astype(np.float32)
    if lo.shape != (H, W):
        lo = cv2.resize(lo, (W, H), interpolation=cv2.INTER_LINEAR)
    return lo > 0.0
```

Upscale **bilinear** de los logits a `(H, W)` y luego umbral → borde suave
sub-pixel y máscara a tamaño del frame. El resize vive **aquí**.

### 3.6 Imports perezosos

`torch`, `cv2` y `PIL.Image` se importan **dentro** de las funciones (no a nivel
de módulo), igual que en `sam3_loader` y `show_frames`. `numpy` y `dataclass`
pueden ir a nivel de módulo (ligeros / ya presentes en `src.core`).

### 3.7 `detect_classes_in_frame` — todas las clases del frame

1. **Resolver** el bundle **una sola vez**: `bundle = bundle or load_sam3()`.
2. **Resolver** las clases: `classes = classes if classes is not None else _load_classes()`.
3. Por cada `cls` en `classes`:
   - `prompt = cls["sam3_prompts"][0]` (prompt activo).
   - `result[cls["name"]] = segment_with_text(frame, prompt, bundle)` (se reusa el
     bundle ya resuelto, sin recargar el modelo).
4. **Devolver** `{name: [Detection, ...]}`.

`_load_classes()` lee la config con la convención del repo (leer `CONFIG_FILENAME`
del `.env` con `strip()` → `get_abs_path(f"configs/{...}")` → `json.load` →
`config["classes"]`). Es código de **esta** tarea (la `classes_config` no añade
helpers).

### 3.8 Manejo de errores

| Situación | Excepción |
|---|---|
| `frame` no es `np.ndarray` o no es `(H, W, 3)` | `ValueError` |
| `CONFIG_FILENAME`/config/clases ausentes (al leer clases) | `ValueError`/`FileNotFoundError`/`KeyError` (propagados) |
| Falla de carga del modelo (vía `load_sam3`) | excepción propagada |
| Frame sin objetos para el prompt | **no es error**: lista vacía |

---

## 4. Cambios de configuración

- **Ninguno.** Las clases ya existen en `configs/00_testing_config.json`
  (tarea `classes_config`).

---

## 5. Script de validación manual

- Ubicación: `testing/test_segmentation.py` (ejecutable manual, no pytest).
- **Flujo:**
  1. Localizar un `.MOV` real (rglob sobre `dataset_dir`) y extraer un frame con
     `extract_frames` (tomar uno, p. ej. el central).
  2. `dets = segment_with_text(frame, "robot")` → imprimir nº de detecciones,
     forma de la máscara (debe ser `(H, W)` del frame) y scores.
  3. `by_class = detect_classes_in_frame(frame)` → imprimir, por clase, conteo y
     score medio.
  4. Manejar **sin abortar** la ausencia de pesos/datos.
- **Rendimiento:** en **CPU es inviable** (minutos por inferencia). El script se
  ejecuta donde haya **GPU**; opcionalmente acota a una clase/un prompt para no
  colgarse y avisa del costo.
- **Ejecución** (donde haya pesos + GPU, p. ej. contenedor/pod):
  ```bash
  docker compose --env-file .env -f docker/docker-compose.yml \
    exec futbotmx26 python testing/test_segmentation.py
  ```

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Módulo y piezas | §3.1, §3.2, §3.3 | `segmentation.py` + export |
| AC-2 `Detection` | §3.2 | obj_id, mask, score |
| AC-3 Segmentación por prompt | §3.4 | lista (vacía si no hay objetos) |
| AC-4 Máscara full-res | §3.5 | bilinear + umbral → `(H, W)` |
| AC-5 Detección por clases | §3.7 | dict `{name: [Detection]}` |
| AC-6 Clases desde config | §3.7 | `_load_classes()` |
| AC-7 Modelo vía `load_sam3` | §3.4, §3.7 | sin globals; bundle reusado |
| AC-8 Validación manual | §5 | `testing/test_segmentation.py` |

---

## 7. Riesgos y consideraciones

- **Costo en CPU:** una inferencia tarda minutos en CPU (medido); el modo
  por-frame multiplica por el nº de clases. Validación y corridas reales en **GPU**;
  acotar el clip de prueba.
- **Forma de los logits de SAM3:** la salida puede venir 4D/3D; se aplana a 2D
  antes del upscale (§3.4). Si una versión de SAM3 cambiara la forma, ajustar ahí.
- **`obj_id` no estable:** en modo por-frame el `obj_id` es por-sesión; no
  identifica objetos entre clases ni entre frames. La identidad estable es de la
  tarea `video_tracking` (5).
- **Dependencia de `kernels`:** si faltara (entorno mal provisionado), la calidad
  de máscara podría degradarse; está cubierto por el `requirements.txt` y el
  aprovisionamiento de los 3 entornos. `refine_mask` (cv2) queda como recurso
  opcional futuro, no parte de esta tarea.
