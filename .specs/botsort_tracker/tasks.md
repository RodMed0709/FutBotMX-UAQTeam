# Tasks — Tracker BoT-SORT intercambiable (`botsort_tracker`)

- **Tarea atómica:** `botsort_tracker`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Config de la fase

- [x] **T1 — Crecer `configs/01_yolo_sam3_config.json`**
  - En `tracking`, añadir el selector `"tracker": "bytetrack"`.
  - Añadir sección top-level `"botsort"` con los defaults (espejo de `botsort.yaml`:
    `track_high_thresh` 0.25, `track_low_thresh` 0.1, `new_track_thresh` 0.25,
    `track_buffer` 30, `match_thresh` 0.8, `fuse_score` true,
    `gmc_method` "sparseOptFlow", `proximity_thresh` 0.5, `appearance_thresh` 0.25,
    `with_reid` false).
  - **Verificación:** JSON válido; claves existentes intactas; `tracking.tracker` =
    `"bytetrack"`.
  - **Plan:** §3.7. **Spec:** AC-6.

---

## Fase B — Subpaquete de trackers

- [x] **T2 — `src/core/trackers/bytetrack.py` (sin cambios de comportamiento)**
  - `make_bytetrack(frame_rate, kwargs)` → `ByteTrackTracker(frame_rate=frame_rate,
    **kwargs)` (import perezoso de `trackers`).
  - **Verificación:** devuelve un `ByteTrackTracker` equivalente al de hoy; mismos
    kwargs.
  - **Plan:** §3.3. **Spec:** AC-2.

- [x] **T3 — `src/core/trackers/botsort.py` (adaptador BoT-SORT)**
  - `_Det(xyxy, conf, cls)` con `.xyxy/.conf/.cls/.xywh`; `_xyxy_to_xywh`.
  - `BotSortTracker(frame_rate, config)`: construye `SimpleNamespace` desde
    `_BOTSORT_DEFAULTS | config` y `BOTSORT(args, frame_rate=...)` (imports perezosos
    de `ultralytics`/`supervision`). `.update(detections, frame)`: convierte a `_Det`,
    llama `BOTSORT.update(..., frame)`, parsea `[x1,y1,x2,y2,tid,score,cls,idx]`, mapea
    `src` por `idx`, devuelve `sv.Detections(xyxy, confidence, tracker_id, data)`.
    Caso vacío → `sv.Detections.empty()`.
  - `make_botsort(frame_rate, config)` → `BotSortTracker(...)`.
  - **Verificación (pod):** `.update` devuelve detecciones con `tracker_id` y `src`
    preservado; caso vacío no rompe; GMC se aplica (pasa el frame).
  - **Plan:** §3.1, §3.4. **Spec:** AC-3, AC-4.

- [x] **T4 — Factory `get_tracker` + `KNOWN_TRACKERS` en `src/core/trackers/__init__.py`**
  - `KNOWN_TRACKERS = ("bytetrack", "botsort")`; `get_tracker(name, frame_rate, *,
    bytetrack_kwargs=None, botsort_config=None)` que delega en `make_*`; nombre
    desconocido → `ValueError`. Exportar ambos.
  - **Verificación:** `get_tracker("bytetrack"/"botsort", ...)` resuelve; nombre
    desconocido ⇒ `ValueError`; `import src.core.trackers` no arrastra
    `ultralytics`/`supervision`/`trackers`; sin import circular.
  - **Plan:** §3.2. **Spec:** AC-5.

---

## Fase C — Refactor del tracking + fachada

- [x] **T5 — `track_video` recibe `tracker` (resolución temprana + factory)**
  - Añadir `tracker: str | None = None`. Resolver **antes** de cargar modelos: `None`
    → `config["tracking"].get("tracker", "bytetrack")`; validar contra
    `KNOWN_TRACKERS` (ValueError temprano). Construir los trackers por clase vía
    `get_tracker(tracker, frame_rate=fps, bytetrack_kwargs=..., botsort_config=
    config.get("botsort", {}))`. Nada más del bucle cambia.
  - **Verificación:** con `tracker="bytetrack"` (default) el resultado es idéntico al
    actual; un nombre inválido lanza `ValueError` sin cargar SAM3.
  - **Plan:** §3.5. **Spec:** AC-1, AC-2, AC-5.

- [x] **T6 — `run_inference` propaga `tracker`**
  - Añadir `tracker: str | None = None`; propagarlo a `track_video` en
    `mode="tracking"`; ignorado en segmentación (documentado).
  - **Verificación:** `run_inference(mode="tracking", tracker="botsort")` enruta con
    ese tracker; sin indicarlo conserva el comportamiento actual; combinable con
    cualquier `detector` (ortogonalidad).
  - **Plan:** §3.6. **Spec:** AC-3, AC-7.

---

## Fase D — Validación

- [x] **T7 — Script smoke A/B `testing/test_botsort_tracker.py` (pod, full frames)**
  - Pinear `data/raw/17Abril/Cámaras/IMG_9871.MOV` (mismo del smoke anterior). Validar
    `get_tracker("inexistente")` ⇒ `ValueError`. Correr
    `track_video(detector="yolo_sam3", tracker="botsort", render_video=True)`;
    aserciones (`{"json","video","index"}`; `tracks` no vacío; `obj_id` único;
    `green_floor` presente). Comparativa: correr también `tracker="bytetrack"` y
    reportar nº de tracks por clase (se espera ≤ con BoT-SORT). Guarda de
    no-regresión.
  - **Verificación (pod):** corre end-to-end; aserciones pasan; el reporte A/B es
    coherente (BoT-SORT no más fragmentado que ByteTrack).
  - **Plan:** §4. **Spec:** AC-8.

---

## Fase E — Cierre

- [x] **T8 — Lint, formato y no-regresión**
  - `ruff check .` y `black .` limpios sobre lo nuevo. Confirmar: `import src.core`
    no arrastra pesados; default `"bytetrack"` reproduce el camino actual; schema,
    overlay y lógica `mask→bbox→obj_id` sin cambios.
  - **Verificación:** linters limpios; import barato; no-regresión confirmada.
  - **Plan:** §3, §5. **Spec:** AC-2.

---

## Trabajo futuro (fuera de esta tarea)

- **Paridad de batch** (tarea siguiente): propagar `detector` y `tracker` en
  `run_batch`.
- ReID de BoT-SORT, tracker **global multi-clase** (vs per-clase, evita GMC
  redundante), y tuneo fino de parámetros de BoT-SORT.
