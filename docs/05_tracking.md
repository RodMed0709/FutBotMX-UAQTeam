# Fase 05 — Tracking

> El segundo eje intercambiable: **mantener la identidad** de cada objeto entre frames.
> Recorre el video en *streaming*, deriva cajas de las máscaras y las asocia con un
> tracker (ByteTrack o BoT-SORT) en `obj_id` **estables y únicos**. Su salida —el
> **tracking JSON**— es la frontera dura con todo el post-proceso.

- **Tareas SDD:** [`video_tracking`](../.specs/video_tracking/),
  [`botsort_tracker`](../.specs/botsort_tracker/), [`obj_id_overlay`](../.specs/obj_id_overlay/)

---

## `src/core/tracking.py` — asociar identidad

Función principal configurable (detector y tracker inyectables, streaming sin OOM):

```python
track_video(
    video_path: Path | str,
    output_path: Path | None = None,
    classes: list[dict] | None = None,
    max_frames: int | None = None,           # tope de frames contiguos
    bundle: Sam3Bundle | None = None,        # SAM3 precargado (reuso en lotes)
    include_masks: bool = False,             # embebe COCO-RLE en el JSON
    render_video: bool = True,               # mp4 anotado opcional
    detector: str | Callable | None = None,  # "sam3_text" | "yolo_sam3"
    tracker: str | None = None,              # "bytetrack" | "botsort"
    run_label: str | None = None,            # namespacing por config
    progress: bool = True,
) -> dict                                     # {"json", "video", "index"}
```

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `track_video(...)` | [`tracking.py:177`](../src/core/tracking.py#L177) | Streaming frame a frame ([`iter_frames`](../src/core/frame_extraction.py#L195)) + detector + un tracker **por clase** → `obj_id` estables. Escribe el JSON unificado (vista `frames` + índice `tracks`). El **JSON siempre se escribe**; el mp4 es opcional. |
| `Track` | [`tracking.py:76`](../src/core/tracking.py#L76) | Un objeto seguido: su `obj_id`, clase y lista de observaciones. |
| `TrackObservation` | [`tracking.py:59`](../src/core/tracking.py#L59) | Una observación (frame, caja, centroide, score). |
| `get_trajectories(...)` | [`tracking.py:139`](../src/core/tracking.py#L139) | Extrae las trayectorias por `obj_id`. |

## `src/core/trackers/` — trackers intercambiables

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `get_tracker(name, ...)` | [`trackers/__init__.py:19`](../src/core/trackers/__init__.py#L19) | Resuelve `"bytetrack"` o `"botsort"`. |
| `make_bytetrack(frame_rate, kwargs)` | [`trackers/bytetrack.py:16`](../src/core/trackers/bytetrack.py#L16) | ByteTrack (Kalman + IoU). El de menor fragmentación con `sam3_text`. |
| `BotSortTracker` / `make_botsort(...)` | [`trackers/botsort.py:87`](../src/core/trackers/botsort.py#L87) | BoT-SORT (añade compensación de movimiento de cámara). El mejor emparejado con `yolo_sam3`. |

Parámetros (`lost_track_buffer`, `minimum_consecutive_frames`…) salen de la sección
`tracking`/`botsort` del config; subirlos reduce fragmentación.

## `src/core/track_overlay.py` — hacer visible el tracking

Post-pase **desacoplado** (no re-infiere): lee un tracking JSON + el video y escribe un
mp4 nuevo con caja + `nombre #id` + estela por `obj_id`.

```python
render_obj_id_overlay(
    json_path: Path | str,                    # JSON de mode="tracking"
    video_path: Path | str | None = None,     # None ⇒ de la cabecera del JSON
    output_path: Path | None = None,          # None ⇒ <stem>_obj_id.mp4
    draw_masks: bool = False,                  # rellena máscara si el JSON trae rle
    trajectory_window: int | None = None,      # N frames de estela; None ⇒ config
    excluded_classes: list[str] | None = None, # None ⇒ config (def ["green_floor"])
) -> Path
```

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `render_obj_id_overlay(...)` | [`track_overlay.py:205`](../src/core/track_overlay.py#L205) | Dibuja caja + `nombre #id` + estela (color **por clase**). Excluye `green_floor` por config. **No** usa SAM3. |

> Diferencia con [`overlay_detections`](04_segmentacion.md): el overlay vivo colorea por
> clase y rellena máscara; este post-pase añade **identidad** (`#id`) y trayectoria.

---

### Cómo encaja con el resto

El **tracking JSON** (`Track`/`TrackObservation`, `obj_id` estable) es la entrada de
**todo** el post-proceso CPU-local: la [capa métrica](09_capa_metrica.md) lo proyecta a
cm, los [eventos](10_eventos.md) lo analizan y [Kalman](11_kalman.md) lo refina. El post
**no sabe** qué detector/tracker lo generó.
