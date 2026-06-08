# Plan técnico — Tracking por detección per-frame + ByteTrack (`video_tracking`)

- **Tarea atómica:** `video_tracking`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** [`../drafts/mvp_sam3_only_roadmap.md`](../drafts/mvp_sam3_only_roadmap.md) (tarea 5)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo implementar `src/core/tracking.py`: un recorrido **en streaming** del
video que, por frame, detecta con `detect_classes_in_frame`, convierte máscaras a
cajas, asocia con **ByteTrack** (un tracker por clase) en `obj_id` estables y únicos,
escribe un mp4 incremental con overlay y retiene un **índice de tracks agnóstico** +
JSON. Reúsa los cimientos existentes; añade dos helpers aditivos (streaming reader y
escritor incremental) sin alterar las funciones actuales.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Detección/segmentación:** `src.core.segmentation.detect_classes_in_frame`
  (modo per-frame, sin tocarlo) + `src.core.sam3_loader.load_sam3`.
- **Tracking:** **`trackers.ByteTrackTracker`** (un tracker por clase) +
  **`supervision`** para `sv.Detections`. Nota: `supervision.ByteTrack` quedó
  **deprecado** en supervision 0.28 (se elimina en 0.30); su reemplazo vigente es el
  paquete `trackers`. **Ambos ya están en `requirements.txt`** (`supervision`,
  `trackers`) → no hace falta nueva dependencia.
- **Cajas / máscaras:** `cv2.boundingRect` (OpenCV, ya dependencia).
- **Lectura de frames:** `decord` (vía un helper de streaming en `frame_extraction`).
- **Escritura mp4:** `imageio`/ffmpeg (vía escritor incremental en `video_writer`).
- **Rutas / config / fps:** `get_abs_path`, `PROJECT_ROOT`, `get_video_fps`.

---

## 3. Diseño

### 3.1 Estructura de archivos

```
src/core/frame_extraction.py   # se AÑADE iter_frames (generador, aditivo)
src/core/video_writer.py       # se AÑADE un escritor incremental (aditivo)
src/core/tracking.py           # lógica de la tarea (nuevo)
testing/test_tracking.py       # script manual standalone (corre en GPU/pod)
configs/00_testing_config.json # nueva seccion "tracking"
requirements.txt               # + supervision
```

`tracking.py` expone la API pública: `track_video`, `get_trajectories` (+ las
dataclasses del índice).

### 3.2 Modelo de datos (índice de tracks agnóstico)

```python
@dataclass
class TrackObservation:
    frame_index: int
    bbox: tuple[int, int, int, int]   # (x, y, w, h)
    centroid: tuple[float, float]     # (cx, cy)
    score: float

@dataclass
class Track:
    obj_id: int          # global y único entre clases
    class_name: str
    observations: list[TrackObservation]
```

- El índice es `dict[int, Track]` (`obj_id -> Track`). **Sin máscaras** (memoria
  acotada para video completo).
- Serialización JSON: lista de tracks con sus observaciones (mismo espíritu que el
  JSON del pipeline per-frame, que también omite máscaras).

### 3.3 Carga de configuración

```python
def _load_tracking_config() -> tuple[dict, int | None, str]:
    """Devuelve (bytetrack_kwargs, max_frames, outputs_dir) desde la seccion
    'tracking' del config (con defaults). max_frames puede ser null = video completo.

    bytetrack_kwargs usa los nombres reales de trackers.ByteTrackTracker:
    track_activation_threshold, lost_track_buffer, minimum_consecutive_frames,
    minimum_iou_threshold. (frame_rate se pasa aparte = get_video_fps.)
    """
    # parseo .env -> CONFIG_FILENAME -> JSON (patron de los otros modulos)
```

### 3.4 Streaming reader (aditivo en `frame_extraction.py`)

```python
def iter_frames(video_path: Path, max_frames: int | None = None):
    """Generador: yield (frame_index, frame_rgb) frame a frame, sin cargar todo.

    Reusa _resolve_video_path; lee con decord (bridge nativo -> numpy RGB). Si
    max_frames es None, recorre todo el video; si es un entero, lo acota.
    """
    abs_path = _resolve_video_path(video_path)
    reader = decord.VideoReader(str(abs_path))
    total = len(reader)
    n = total if max_frames is None else min(max_frames, total)
    for i in range(n):
        yield i, reader[i].asnumpy()   # (H, W, 3) RGB
```

- No cambia `extract_frames`/`get_frame_indices`; solo añade el generador.

### 3.5 Escritor mp4 incremental (aditivo en `video_writer.py`)

`write_video` ya usa `imageio.get_writer` + `append_data`; se extrae esa apertura a
un context manager para escribir frame a frame:

```python
from contextlib import contextmanager

@contextmanager
def open_video_writer(output_path: Path | str, fps: float | None = None):
    """Context manager: abre un writer mp4 incremental y lo cierra al salir.

    yield una funcion append(frame_uint8_rgb). Reusa codec/fps de write_video.
    """
    import imageio
    fps = fps if fps is not None else _load_output_fps()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path), format="FFMPEG", mode="I", fps=fps,
        codec="libx264", pixelformat="yuv420p",
    )
    try:
        yield writer.append_data
    finally:
        writer.close()
```

- `write_video` (batch) queda **intacto**; ambos comparten el backend.

### 3.6 Máscara → caja

```python
def _mask_to_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """boundingRect de una mascara booleana -> (x1, y1, x2, y2); None si vacia."""
    import cv2
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w == 0 or h == 0:
        return None
    return x, y, x + w, y + h
```

### 3.7 ByteTrack por clase + `obj_id` global

```python
trackers = {c["name"]: ByteTrackTracker(frame_rate=fps, **bytetrack_kwargs)
            for c in classes}
global_id: dict[tuple[str, int], int] = {}   # (clase, tracker_id) -> obj_id
next_obj_id = 0
```

- Por frame y por clase se construye `sv.Detections(xyxy, confidence, data={"src": idx})`
  donde `src` indexa el `Detection` origen (para recuperar su máscara tras el
  tracking). `tracker.update(dets, frame)` devuelve `Detections` con `tracker_id` y
  **preserva `data["src"]`** (verificado en supervision 0.28 / trackers); se recupera
  el `Detection` por `data["src"]` y se mapea `(clase, tracker_id) -> obj_id`
  (asignando uno nuevo la primera vez).
- **Warm-up de ByteTrack:** en el frame de aparición el `tracker_id` es **`-1`**
  (aún sin confirmar); esas detecciones se pintan pero **no** se les asigna `obj_id`
  ni se registran en el índice hasta que se confirman (frame siguiente).
- **`obj_id` global y único**: el namespacing por `(clase, tracker_id)` evita
  colisiones entre los trackers por clase. La **clase** sale por construcción.

### 3.8 Orquestador `track_video` (streaming)

```python
def track_video(video_path, output_path=None, classes=None,
                max_frames=None, bundle=None) -> dict:
    classes = classes if classes is not None else _load_classes()
    bundle = bundle or load_sam3()
    cfg = _load_tracking_config()
    max_frames = max_frames if max_frames is not None else cfg["max_frames"]
    fps = get_video_fps(video_path)

    # rutas de salida (auto-naming bajo outputs/ si output_path es None)
    stem = Path(video_path).stem
    base = PROJECT_ROOT / outputs_dir
    mp4_path = Path(output_path) if output_path else base / f"{stem}_tracked.mp4"
    json_path = mp4_path.with_name(f"{mp4_path.stem}_tracks.json")

    trackers = {c["name"]: sv.ByteTrack(frame_rate=fps, ...) for c in classes}
    global_id, next_obj_id = {}, 0
    tracks: dict[int, Track] = {}

    with open_video_writer(mp4_path, fps=fps) as append:
        for frame_index, frame in iter_frames(video_path, max_frames):
            dets = detect_classes_in_frame(frame, classes=classes, bundle=bundle)
            per_frame: dict[str, list[Detection]] = {}
            for c in classes:
                name = c["name"]
                # construir sv.Detections de las cajas no vacias de esta clase
                # -> tracker.update -> recuperar Detection por data["src"]
                # -> asignar obj_id global, fijar det.obj_id, acumular en per_frame
                # -> registrar TrackObservation en tracks[obj_id]
                ...
            composed = overlay_detections(frame, per_frame, classes=classes)
            append(composed)

    _write_tracks_json(tracks, json_path)
    return {"video": mp4_path, "tracks": json_path, "index": tracks}
```

- **Memoria acotada:** por frame se usan máscaras/overlay y se descartan; solo se
  retiene el índice ligero (`tracks`).
- **Overlay:** `overlay_detections` colorea **por clase** (el coloreado por `obj_id`
  queda como mejora opcional fuera del MVP).

### 3.9 Trayectorias (utilidad derivada)

```python
def get_trajectories(tracks: dict[int, Track]) -> dict[int, list[tuple[int, float, float]]]:
    """obj_id -> [(frame_index, cx, cy), ...] a partir de las observaciones."""
    return {
        oid: [(o.frame_index, *o.centroid) for o in t.observations]
        for oid, t in tracks.items()
    }
```

### 3.10 Manejo de errores

| Situación | Excepción |
|---|---|
| `CONFIG_FILENAME` ausente | `ValueError` |
| Faltan claves de config (`classes`, `working_dirs.outputs_dir`, ...) | `KeyError` |
| Video inexistente | `FileNotFoundError` (vía `_resolve_video_path`) |
| `supervision` no instalado | `ImportError` (import perezoso) |

---

## 4. Cambios de configuración

En `configs/00_testing_config.json`, nueva sección (nombres = params reales de
`trackers.ByteTrackTracker`):

```jsonc
"tracking": {
  "track_activation_threshold": 0.4,
  "lost_track_buffer": 30,
  "minimum_consecutive_frames": 1,
  "minimum_iou_threshold": 0.2,
  "max_frames": null          // null = video completo; entero = clip
}
```

`requirements.txt`: **sin cambios** — `supervision` y `trackers` ya estaban
declarados.

---

## 5. Validación

Ambas pruebas viven en `testing/test_tracking.py` y se ejecutan **en GPU/pod**
(requieren el modelo SAM3; no corren en local sin modelo).

### 5.1 Prueba A — clip corto (sanity rápido)

- `track_video(video, max_frames=<pequeño, p. ej. 30>)`.
- Verifica: se crean mp4 + JSON; hay ≥ 1 `obj_id` que aparece en **varios frames**
  (identidad estable); cada `Track.class_name` ∈ clases del config; `obj_id` únicos
  entre clases; el índice no contiene máscaras; `get_trajectories` devuelve centroides.

### 5.2 Prueba B — video real completo (end-to-end / streaming)

- Selecciona, de forma **determinista**, un video real que **NO** esté en
  `splits.forced_testing` (p. ej. el de menor `id` en `db_metadata.csv` cuya `ruta`
  no sea forzada).
- `track_video(video, max_frames=None)` (video completo, streaming).
- Verifica: completa **sin OOM**; mp4 + JSON generados; nº de frames del mp4 acorde
  al video; hay tracks con observaciones a lo largo del tiempo; objetos que aparecen
  después del inicio reciben track.
- **Nota:** puede ser de **corrida larga** (SAM3 per-frame es lento); es el costo
  esperado, no un fallo.

### 5.3 Calidad

- `ruff check .` y `black .` sin hallazgos.
- Importabilidad: `from src.core.tracking import track_video, get_trajectories`.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Módulo presente | §3.1, §3.8 | `src/core/tracking.py` |
| AC-2 Reúso del per-frame | §3.8 | `detect_classes_in_frame`, sin tocarlo |
| AC-3 Identidad estable / única | §3.7 | ByteTrack + namespacing global |
| AC-4 Clase por construcción | §3.7 | tracker por clase |
| AC-5 Genérico sobre clases | §3.8 | `classes=None` → todas del config |
| AC-6 Streaming / sin OOM | §3.4, §3.5, §3.8 | iter_frames + writer incremental |
| AC-7 Objetos nuevos | §3.7 | ByteTrack abre track al aparecer |
| AC-8 Salidas | §3.8 | mp4 + JSON + índice |
| AC-9 Trayectorias | §3.9 | `get_trajectories` |
| AC-10 Config | §3.3, §4 | sección `tracking` |
| AC-11 Acople | §3.8 | `overlay_detections`/writer reusados |
| AC-12 Dependencia | §4 | `supervision` en requirements |
| AC-13 Verificación | §5.1, §5.2 | clip corto + video real no-forzado |

---

## 7. Riesgos y consideraciones

- **Mapeo salida de ByteTrack → `Detection` origen:** `update` puede
  filtrar/reordenar; se resuelve llevando un índice `data["src"]` en `sv.Detections`
  para recuperar la máscara correcta (preservación de `data` verificada). Detalle
  clave de la implementación.
- **`sv.ByteTrack` deprecado:** en supervision 0.28 `sv.ByteTrack` está deprecado
  (se elimina en 0.30); se usa **`trackers.ByteTrackTracker`** (ya en
  `requirements.txt`). Su firma: `ByteTrackTracker(frame_rate, track_activation_
  threshold, lost_track_buffer, minimum_consecutive_frames, minimum_iou_threshold,
  ...)` y `update(detections, frame) -> Detections`.
- **Asociación por caja en el balón:** objeto rápido y pequeño → posibles
  pérdidas/intercambios de ID; riesgo asumido (sin evaluación en este MVP).
- **Costo de cómputo:** SAM3 per-frame domina (varias sesiones por frame); el video
  completo es lento pero acotado en memoria. La prueba B puede tardar.
- **Overlay por clase, no por `obj_id`:** se reusa `overlay_detections` tal cual; el
  coloreado por instancia es mejora futura.
- **`obj_id` con semántica nueva:** estable en tracking (a diferencia del per-frame);
  documentar en el código.
