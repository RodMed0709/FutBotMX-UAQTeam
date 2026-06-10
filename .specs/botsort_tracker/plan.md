# Plan técnico — Tracker BoT-SORT intercambiable (`botsort_tracker`)

- **Tarea atómica:** `botsort_tracker`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso:** cuarta tarea de la secuencia que integra el pipeline YOLO + SAM3 a
  `src/`; añade BoT-SORT como tracker intercambiable junto a ByteTrack.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo (a) abstraer el tracker tras una **interfaz común**
(`.update(detections, frame) -> detections`); (b) implementar el subpaquete
`src/core/trackers/` con `bytetrack` (el actual, sin cambios), `botsort`
(**adaptador** de `ultralytics.trackers.BOTSORT`) y un **factory** `get_tracker`;
(c) **refactorizar `track_video`** para construir los trackers vía el factory y
aceptar `tracker` por parámetro/config; y (d) **crecer el config** con el selector y
la sección `botsort`. Reutiliza todo el bucle de tracking (schema, overlay,
`obj_id`). Además, definir el smoke A/B (pod).

---

## 2. Stack técnico

- **Python:** 3.11.
- **BoT-SORT:** `ultralytics.trackers.BOTSORT` (ya instalado; trae **GMC**). Import
  perezoso.
- **ByteTrack:** `trackers.ByteTrackTracker` (Roboflow), **sin cambios**.
- **Detecciones:** `supervision` (`sv.Detections`) — la moneda que `track_video` ya
  pasa al tracker (xyxy, confidence, `data["src"]`).
- **Args de BoT-SORT:** `types.SimpleNamespace` construido desde la config.
- **Imports pesados** (`ultralytics`, `supervision`, `trackers`) **perezosos**.

---

## 3. Diseño

### 3.1 Interfaz común de tracker

El contrato es el que `track_video` ya consume del `ByteTrackTracker`:

```python
tracker.update(detections: sv.Detections, frame: np.ndarray) -> sv.Detections
```

La salida debe traer `tracker_id` (int por fila), `xyxy` y `data["src"]`
**preservado** (índice de la detección de entrada, para recuperar la máscara).
ByteTrack ya lo cumple; el adaptador BoT-SORT debe replicarlo.

### 3.2 Subpaquete y factory — `src/core/trackers/`

- `__init__.py`:

```python
KNOWN_TRACKERS = ("bytetrack", "botsort")

def get_tracker(name, frame_rate, *, bytetrack_kwargs=None, botsort_config=None):
    if name == "bytetrack":
        from src.core.trackers.bytetrack import make_bytetrack
        return make_bytetrack(frame_rate, bytetrack_kwargs or {})
    if name == "botsort":
        from src.core.trackers.botsort import make_botsort
        return make_botsort(frame_rate, botsort_config or {})
    raise ValueError(
        f"tracker '{name}' no soportado (usa uno de {list(KNOWN_TRACKERS)})."
    )
```

- Importar `src.core.trackers` es **barato** (no arrastra ultralytics/supervision;
  esos viven dentro de `make_*`/`update`).

### 3.3 `bytetrack.py` — sin cambios de comportamiento

```python
def make_bytetrack(frame_rate, kwargs):
    from trackers import ByteTrackTracker
    return ByteTrackTracker(frame_rate=frame_rate, **kwargs)
```

Devuelve el **mismo** objeto que hoy, con los mismos `tracking.*` kwargs → ByteTrack
byte-idéntico (no-regresión).

### 3.4 `botsort.py` — adaptador de `ultralytics.trackers.BOTSORT`

Defaults espejo de `botsort.yaml` de ultralytics (GMC activo, ReID off):

```python
_BOTSORT_DEFAULTS = {
    "track_high_thresh": 0.25, "track_low_thresh": 0.1, "new_track_thresh": 0.25,
    "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True,
    "gmc_method": "sparseOptFlow", "proximity_thresh": 0.5,
    "appearance_thresh": 0.25, "with_reid": False,
}
```

```python
class BotSortTracker:
    def __init__(self, frame_rate, config):
        from types import SimpleNamespace
        from ultralytics.trackers import BOTSORT
        args = SimpleNamespace(**{**_BOTSORT_DEFAULTS, **config})
        self._t = BOTSORT(args, frame_rate=int(round(frame_rate)))

    def update(self, detections, frame):
        import numpy as np
        import supervision as sv

        n = len(detections)
        if n == 0:
            self._t.update(_Det(np.empty((0, 4)), np.empty(0), np.empty(0)), frame)
            return sv.Detections.empty()

        xyxy = np.asarray(detections.xyxy, dtype=np.float32)
        conf = np.asarray(detections.confidence, dtype=np.float32)
        cls = np.zeros(n, dtype=np.float32)  # un tracker por clase -> clase única
        out = self._t.update(_Det(xyxy, conf, cls), frame)
        # out: filas [x1,y1,x2,y2, track_id, score, cls, idx]
        if out is None or len(out) == 0:
            return sv.Detections.empty()
        out = np.asarray(out, dtype=np.float32)
        idx = out[:, 7].astype(int)
        src_in = detections.data.get("src") if detections.data else None
        src = src_in[idx] if src_in is not None else idx
        return sv.Detections(
            xyxy=out[:, :4],
            confidence=out[:, 5],
            tracker_id=out[:, 4].astype(int),
            data={"src": np.asarray(src)},
        )
```

- `_Det`: objeto ligero que expone lo que `BOTSORT.update` lee de un "results":
  `.conf`, `.cls`, `.xyxy` (y `.xywh` derivado por robustez entre versiones).
  ```python
  class _Det:
      def __init__(self, xyxy, conf, cls):
          self.xyxy = xyxy; self.conf = conf; self.cls = cls
          self.xywh = _xyxy_to_xywh(xyxy)
  ```
- **GMC**: BoT-SORT lo aplica usando `frame` (`img`) en `update`. Por eso se pasa el
  frame (no se omite). Un tracker por clase ⇒ GMC se calcula por clase (aceptado).
- **Formato de salida**: se asume `[x1,y1,x2,y2,track_id,score,cls,idx]` (8 cols, el
  `result` de `STrack` de ultralytics). El **smoke en el pod** confirma el shape; si
  una versión difiere, se ajusta el indexado.
- **Diferencia conocida vs ByteTrack (Roboflow)**: BoT-SORT devuelve **solo tracks
  confirmados** (no las detecciones en warm-up con `tracker_id=-1`). En consecuencia,
  en los primeros frames una detección aún no confirmada **no** aparece en el registro
  de ese frame (con ByteTrack sí aparecía con `obj_id=-1`). Es comportamiento normal
  del tracker; el resto del flujo (mapeo por `src`, `obj_id` estable) es idéntico.

### 3.5 Refactor de `track_video`

- **Nueva firma**: `def track_video(..., detector=None, tracker: str | None = None)`.
- **Resolución temprana** (junto a la del detector, antes de imports/carga pesada):
  ```python
  if tracker is None:
      tracker = config.get("tracking", {}).get("tracker", "bytetrack")
  if tracker not in KNOWN_TRACKERS:
      raise ValueError(f"tracker '{tracker}' no soportado (usa uno de {list(KNOWN_TRACKERS)}).")
  ```
  (Validación de nombre **antes** de `load_sam3()`; `KNOWN_TRACKERS` se importa de
  `src.core.trackers`, import barato.)
- **Construcción de trackers** (donde hoy se crea el dict, ya con `fps`):
  ```python
  trackers = {
      cls["name"]: get_tracker(
          tracker, frame_rate=fps,
          bytetrack_kwargs=bytetrack_kwargs,
          botsort_config=config.get("botsort", {}),
      )
      for cls in classes
  }
  ```
- El resto del bucle (`mask→bbox→ sv.Detections → tracker.update → tracker_id → src →
  obj_id estable`, schema, overlay) **no cambia**: la interfaz común garantiza que la
  salida de ambos trackers se consume igual.

### 3.6 Cableado en `run_inference`

- Añadir `tracker: str | None = None`; propagarlo a `track_video` en `mode="tracking"`
  (paralelo a `detector`). En `mode="segmentation"` se ignora (documentado).

### 3.7 Crecimiento del config de la fase

Editar `configs/01_yolo_sam3_config.json`:

- En `tracking`, añadir el selector `"tracker": "bytetrack"` (default ⇒ no-regresión).
- Añadir sección top-level `"botsort"` con los `_BOTSORT_DEFAULTS` (GMC
  `sparseOptFlow`, `with_reid` false). ByteTrack sigue leyendo sus claves actuales.

---

## 4. Script de validación A/B (smoke, pod)

Archivo: `testing/test_botsort_tracker.py` (manual, **pod**: requiere ultralytics +
SAM3 + GPU).

- **Mismo video** que el smoke anterior: `data/raw/17Abril/Cámaras/IMG_9871.MOV`
  (full frames), para A/B directo botsort vs bytetrack.
- **Validación temprana**: `get_tracker("inexistente", ...)` ⇒ `ValueError`.
- Correr `track_video(detector="yolo_sam3", tracker="botsort", render_video=True)`;
  aserciones (resultado `{"json","video","index"}`; `tracks` no vacío; `obj_id`
  único; `green_floor` presente).
- **Comparativa de estabilidad**: correr también `tracker="bytetrack"` (mismo video)
  y reportar nº de tracks y nº de tracks por clase para cada uno; con GMC se espera
  **≤** tracks (menos fragmentación). Es reporte, no aserción dura.
- **No-regresión rápida**: `tracker="bytetrack"` produce JSON+mp4 como antes.

---

## 5. Riesgos y consideraciones

- **Interfaz de `BOTSORT.update`**: el conjunto exacto de atributos que lee del
  "results" y el shape de salida pueden variar entre versiones de ultralytics; el
  adaptador provee `.conf/.cls/.xyxy/.xywh` y el smoke en el pod valida el indexado de
  salida (8 columnas asumidas).
- **GMC por clase**: óptico-flujo redundante (N clases ⇒ N cálculos/frame). Aceptado;
  optimización futura = tracker global multi-clase.
- **Warm-up**: BoT-SORT no emite detecciones no confirmadas (vs `obj_id=-1` de
  ByteTrack). Diferencia esperada, documentada (§3.4).
- **No-regresión**: `tracker="bytetrack"` reusa el objeto y kwargs actuales; el cambio
  acotado a la **construcción** del tracker garantiza que el camino actual no cambia.
- **Validación temprana**: nombre de tracker inválido falla antes de cargar modelos.

---

## 6. Qué NO incluye este plan

- ReID, tracker global multi-clase, tuneo fino de BoT-SORT.
- Paridad de batch (`detector`/`tracker` en `run_batch`): tarea siguiente.
- Cambios al esquema JSON, a `overlay`/`track_overlay` o a la lógica `mask→bbox→obj_id`.
- La descomposición en pasos accionables y su checklist: corresponde a `tasks.md`.
