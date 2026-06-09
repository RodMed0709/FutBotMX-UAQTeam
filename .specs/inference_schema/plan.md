# Plan técnico — Esquema común del entregable de inferencia (`inference_schema`)

- **Tarea atómica:** `inference_schema`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso de referencia:** roadmap del pipeline de inferencia unificado + batch (tarea 1)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo introducir **un esquema JSON común** emitido por `run_pipeline`
(seg-only) y `track_video` (tracking). El núcleo es un **módulo nuevo**
(`src/core/inference_schema.py`) que centraliza: derivación de geometría
(máscara→bbox/centroide), codificación **COCO-RLE** opcional, construcción de la
cabecera con metadatos, ensamblado de los registros frame-indexed, y la
**ubicación de salidas por video** (`outputs/inference/<stem>/`). Los dos
orquestadores existentes se modifican para **construir y escribir** vía ese módulo,
sin cambiar su lógica de detección/tracking.

---

## 2. Stack técnico

- **Python:** 3.11.
- **RLE:** `pycocotools.mask` (`encode`/`decode`/`toBbox`), **import perezoso**;
  **nueva dependencia** en `requirements.txt`.
- **Geometría:** `cv2.boundingRect` (OpenCV, ya dependencia) sobre la máscara
  booleana.
- **Serialización:** `json` (stdlib).
- **Metadatos de versión:** `importlib.metadata.version` para capturar versiones de
  paquetes; ruta del checkpoint desde la config (`working_dirs.sam3_dir`).
- **Sin cambios** en `detect_classes_in_frame`, `load_sam3`, ByteTrack ni el muestreo
  de frames.

---

## 3. Diseño

### 3.1 Estructura de archivos

```
src/core/inference_schema.py   # NUEVO: builders del esquema, RLE, rutas, writer
src/core/pipeline.py           # MOD: run_pipeline construye/escribe vía el módulo
src/core/tracking.py           # MOD: track_video funde frames+tracks en un JSON
testing/test_inference_schema.py  # NUEVO: validación (local + GPU/pod)
requirements.txt               # + pycocotools
```

El módulo nuevo es **sin estado** (funciones puras + helpers de IO); no carga
modelo ni config global salvo donde ya se hace.

### 3.2 Forma del esquema (constante `SCHEMA_VERSION = "1.0"`)

```jsonc
{
  "schema_version": "1.0",
  "video": "data/raw/.../IMG_xxxx.MOV",
  "mode": "segmentation" | "tracking",
  "model_version": { "sam3_dir": "assets/sam3", "packages": {"sam3": "...", "transformers": "..."} },
  "timestamp": "2026-06-08T12:34:56Z",      // UTC ISO-8601
  "fps": 29.97,                              // fps real de la fuente
  "resolution": { "height": 1080, "width": 1920 },
  "num_frames": 30,
  "include_masks": false,
  "classes": ["robot", "orange_ball", "green_floor"],
  "config": { /* snapshot COMPLETO de la config activa */ },
  "frames": [
    {
      "frame_index": 0,                      // índice REAL en el video fuente
      "detections": {
        "robot": [
          { "obj_id": 3, "bbox": [x, y, w, h], "centroid": [cx, cy],
            "score": 0.87, "rle": {"size": [H, W], "counts": "..."} }
        ],
        "orange_ball": [ ... ],
        "green_floor": [ ... ]
      }
    }
  ],
  // SOLO en mode == "tracking":
  "tracks": [
    { "obj_id": 3, "class": "robot",
      "observations": [ {"frame_index": 0, "bbox": [...], "centroid": [...], "score": 0.87}, ... ] }
  ]
}
```

- `rle` presente **solo** si `include_masks=True`. `bbox` en píxeles absolutos
  `(x, y, w, h)`; `centroid` el centro de esa caja.
- `tracks` es el índice agnóstico actual (`Track`/`TrackObservation`), **fundido** en
  el mismo archivo (se elimina el `_tracks.json` aparte).

### 3.3 API del módulo `inference_schema.py`

```python
SCHEMA_VERSION = "1.0"

def mask_to_bbox_centroid(mask: np.ndarray) -> tuple[list[int], list[float]] | None:
    """boundingRect(mask) -> (bbox=[x,y,w,h], centroid=[cx,cy]); None si vacía."""

def encode_rle(mask: np.ndarray) -> dict:
    """Máscara booleana (H,W) -> COCO-RLE JSON-serializable.

    Usa pycocotools.mask.encode sobre uint8 Fortran-contiguo; 'counts' (bytes) se
    decodifica a str ascii para que sea serializable. Import perezoso de pycocotools.
    """

def decode_rle(rle: dict) -> np.ndarray:
    """Inverso de encode_rle -> máscara booleana (H,W) idéntica (sin pérdida)."""

def detection_record(det, include_masks: bool) -> dict | None:
    """Detection -> {obj_id, bbox, centroid, score, rle?}; None si la máscara es vacía.
    bbox/centroid via mask_to_bbox_centroid; rle solo si include_masks."""

def frame_record(frame_index: int, dets_by_class: dict[str, list], include_masks: bool) -> dict:
    """-> {frame_index, detections: {clase: [detection_record, ...]}}."""

def build_header(*, video, mode, fps, resolution, num_frames, classes,
                 include_masks, config, bundle) -> dict:
    """Ensambla la cabecera (schema_version, model_version, timestamp UTC, etc.)."""

def inference_paths(video_stem: str, outputs_dir: str) -> tuple[Path, Path]:
    """-> (json_path, mp4_path) bajo PROJECT_ROOT/outputs_dir/inference/<stem>/.
    json=<stem>.json, mp4=<stem>.mp4. NO crea carpetas (lo hace el writer)."""

def write_inference_json(header: dict, frames: list[dict], json_path: Path,
                         tracks: list[dict] | None = None) -> None:
    """Compone {**header, frames, tracks?} y lo escribe (crea carpeta padre)."""
```

- **`model_version`** (helper interno `_model_version(bundle, config)`): toma
  `config["working_dirs"]["sam3_dir"]` como puntero al checkpoint + versiones de
  `sam3`/`transformers` vía `importlib.metadata` (best-effort: si un paquete no
  resuelve, se omite esa clave, no falla).
- **`timestamp`**: `datetime.now(timezone.utc)` en ISO-8601.

### 3.4 Codificación RLE (detalle)

```python
def encode_rle(mask):
    from pycocotools import mask as mask_utils
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)                 # {'size':[H,W], 'counts': b'...'}
    rle["counts"] = rle["counts"].decode("ascii")  # JSON-serializable
    return rle

def decode_rle(rle):
    from pycocotools import mask as mask_utils
    r = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    return mask_utils.decode(r).astype(bool)
```

- COCO-RLE garantiza ida-vuelta **sin pérdida** y es exactamente el formato que
  `pycocotools`/`prediction_export` consumirán (AC-12).

### 3.5 Geometría (máscara → caja/centroide)

```python
def mask_to_bbox_centroid(mask):
    import cv2
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w == 0 or h == 0:
        return None
    return [int(x), int(y), int(w), int(h)], [x + w / 2.0, y + h / 2.0]
```

- Misma fuente de verdad para ambos modos. En tracking sustituye/centraliza el
  `_mask_to_xyxy` local (que devolvía `xyxy`); aquí se estandariza a `xywh`.

### 3.6 Integración en `run_pipeline` (seg-only)

Cambios mínimos, sin tocar la inferencia:

- **Firma:** `run_pipeline(video_path, output_path=None, all_frames=False,
  mode="per_frame", include_masks=False)`.
- **`frame_index` real:** obtener los índices fuente con
  `get_frame_indices(video_path, all_frames)` (alineados por posición con
  `extract_frames`); `frame_index = source_indices[i]` en vez del `i` posicional.
- **Registros:** por frame, `frame_record(source_indices[i], dets, include_masks)`
  (antes: `{index, detections:{clase:[{obj_id,score}]}}`).
- **Resolución:** `resolution` desde `frames.shape[1:3]` (H, W).
- **Cabecera + escritura:** `build_header(...)` + `write_inference_json(...)`.
- **Rutas:** `inference_paths(stem, outputs_dir)` → `outputs/inference/<stem>/`.
  El mp4 (que hoy se escribe siempre) va a `<stem>.mp4` en esa carpeta.
- **Snapshot de config:** `_load_pipeline_config` pasa a devolver también el `config`
  completo ya parseado (o se relee una vez) para embeberlo.
- **Retorno:** `{"json": json_path, "video": mp4_path}`.

### 3.7 Integración en `track_video` (tracking)

- **Firma:** `track_video(video_path, output_path=None, classes=None,
  max_frames=None, bundle=None, include_masks=False)`.
- **Registros por frame:** en el loop ya se arma `per_frame: {clase: [Detection]}`
  (las detecciones pintadas, con `det.obj_id`, `det.mask`, `det.score`). Tras
  componer el overlay, **antes de descartar las máscaras**, construir
  `frame_record(frame_index, per_frame, include_masks)` y acumularlo. Esto reusa las
  máscaras vivas en RAM → el RLE sale sin coste extra de inferencia.
  - Las detecciones en **warm-up** (`obj_id == -1`) se incluyen en el frame record
    (reflejan lo que se detectó/pintó); su geometría sale de la máscara igual.
- **Índice de tracks:** se mantiene `tracks: dict[int, Track]` como hoy; al final se
  serializa con la misma forma de `_write_tracks_json` pero **como sección `tracks`
  del JSON unificado**, no como archivo aparte.
- **Un solo JSON:** se **elimina** `_write_tracks_json` (archivo separado) y el
  `json_path = ..._tracks.json`. Se usa `inference_paths` + `write_inference_json`
  con `frames=` y `tracks=`.
- **Resolución:** del primer frame iterado (`frame.shape[:2]`).
- **Retorno:** `{"json": json_path, "video": mp4_path, "index": tracks}`.
- `_mask_to_xyxy` local se retira en favor de `mask_to_bbox_centroid` del módulo
  (la conversión a `sv.Detections` necesita `xyxy`: se deriva de `xywh` ahí mismo, o
  se conserva una conversión interna mínima solo para alimentar ByteTrack).

### 3.8 Ubicación de salidas

```
outputs/inference/<video_stem>/
├── <video_stem>.json     # siempre (el entregable)
└── <video_stem>.mp4      # mientras el render siga activo (hasta optional_render)
```

- Todo bajo `outputs/` (git-ignored). `inference_paths` centraliza la convención;
  `write_inference_json`/`open_video_writer` crean la carpeta.

### 3.9 Manejo de errores

| Situación | Excepción |
|---|---|
| `CONFIG_FILENAME` ausente / claves de config faltantes | `ValueError` / `KeyError` (comportamiento actual) |
| Video inexistente | `FileNotFoundError` (vía `_resolve_video_path`) |
| `pycocotools` no instalado **y** `include_masks=True` | `ImportError` (import perezoso) |
| Máscara vacía | la detección se omite del registro (`detection_record` → `None`) |

- Con `include_masks=False`, `pycocotools` **no** se importa → no es requisito para
  el caso por defecto.

---

## 4. Cambios de configuración y dependencias

- **`requirements.txt`:** **+ `pycocotools`** (única dependencia nueva). Import
  perezoso; solo se usa cuando `include_masks=True`.
- **`configs/00_testing_config.json`:** **sin cambios obligatorios**. El snapshot
  embebe la config completa tal cual; `include_masks` es parámetro, no config. (El
  `model_version` se deriva de `working_dirs.sam3_dir`, que ya existe.)

---

## 5. Validación (`testing/test_inference_schema.py`)

### 5.1 Parte A — local, **sin GPU** (helpers del módulo con datos sintéticos)

- **RLE ida-vuelta:** máscara booleana sintética (H,W) → `encode_rle` → `decode_rle`
  → **idéntica** (`np.array_equal`).
- **Geometría:** `mask_to_bbox_centroid` sobre una máscara rectangular conocida →
  bbox/centroid esperados; máscara vacía → `None`.
- **Ensamblado/serialización:** construir cabecera + `frames` sintéticos, escribir
  con `write_inference_json`, releer el JSON y verificar: `schema_version`, presencia
  de `config` (snapshot), `resolution`, `fps`, y que con `include_masks=True` cada
  detección trae `rle` y con `False` **no**.
- **Reconstrucción sin modelo (AC-9):** decodificar el `rle` del JSON y pintarlo con
  `overlay_detections` sobre un frame dummy → produce un array compuesto **sin
  invocar SAM3**.

### 5.2 Parte B — **GPU/pod** (orquestadores reales, clip corto)

- `run_pipeline(video, all_frames=False, include_masks=True)` → existe
  `outputs/inference/<stem>/<stem>.json` con `mode="segmentation"`, `frames` con
  geometría + `rle`, y `frame_index` reales (coinciden con
  `get_frame_indices`).
- `track_video(video, max_frames=<pequeño>, include_masks=True)` → un **único** JSON
  con **`frames` y `tracks`** en el mismo archivo (no hay `_tracks.json`), `mp4` y
  `json` en la carpeta del video.
- Caso por defecto `include_masks=False`: el JSON no contiene `rle` y **no** se
  importó `pycocotools`.

### 5.3 Calidad

- `ruff check .` y `black .` sin hallazgos.
- Importabilidad: `from src.core.inference_schema import write_inference_json,
  encode_rle, decode_rle`.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Esquema común | §3.2, §3.6, §3.7 | mismo builder para ambos modos |
| AC-2 Geometría en seg-only | §3.5, §3.6 | bbox/centroid + `frame_index` real |
| AC-3 Metadatos | §3.2, §3.3 | `build_header` (versión, fps, resolución, config) |
| AC-4 Máscaras opcionales RLE | §3.4 | `encode/decode_rle` sin pérdida |
| AC-5 `include_masks` por parámetro | §3.6, §3.7 | argumento de función, default `False` |
| AC-6 Un solo archivo / ubicación | §3.7, §3.8 | JSON unificado en `inference/<stem>/` |
| AC-7 mp4 reubicado | §3.6, §3.8 | `inference_paths`, junto al JSON |
| AC-8 `obj_id` documentado | §3.7 | docstring (estable en tracking, inestable per-frame) |
| AC-9 Reconstrucción sin modelo | §5.1 | decode RLE + overlay sobre frame dummy |
| AC-10 Dependencia declarada | §4 | `pycocotools`, import perezoso |
| AC-11 Verificación | §5.1, §5.2 | local (helpers) + GPU/pod (orquestadores) |
| AC-12 Contrato para evaluación | §3.4 | RLE = COCO-RLE, proyectable a COCO estándar |

---

## 7. Riesgos y consideraciones

- **`frame_index` real en seg-only:** el riesgo principal es desalinear posición↔
  índice fuente; se mitiga usando `get_frame_indices` (ya alineado por construcción
  con `extract_frames`). En `all_frames=True` los índices son `0..N-1`.
- **`pycocotools` en el entorno:** rueda binaria; en RunPod/Blackwell suele instalar
  sin problema, pero al ser import perezoso **no** bloquea el caso por defecto
  (`include_masks=False`). Documentar en `requirements.txt`.
- **Tamaño del JSON con `include_masks=True`:** RLE es compacto pero no nulo; por eso
  default `False` y uso acotado (eval/depuración), nunca 123 videos completos.
- **`tracks` vs `frames` (redundancia):** ambas vistas comparten geometría; es
  intencional (frames = vista canónica para viz/eval; tracks = identidad temporal).
  No se deduplica para mantener cada vista autocontenida.
- **Ruptura de formato:** se elimina `_tracks.json` y cambia el JSON de seg-only; son
  outputs git-ignored desechables (sin retrocompatibilidad, por spec).
- **`model_version` best-effort:** si `importlib.metadata` no resuelve un paquete, se
  omite esa clave sin fallar; el puntero al checkpoint (`sam3_dir`) siempre va.
- **Límite de alcance:** el mp4 se sigue escribiendo siempre (el flag `render_video`
  es la tarea `optional_render`); aquí solo se **reubica** a la carpeta por video.
