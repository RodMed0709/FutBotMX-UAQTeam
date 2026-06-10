# Plan técnico — Paridad de batch para detector/tracker (`batch_detector_tracker`)

- **Tarea atómica:** `batch_detector_tracker`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso:** quinta y **última** tarea de la integración YOLO + SAM3 a `src/`;
  lleva las perillas `detector`/`tracker` de `run_inference` a `run_batch`.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo (a) ampliar la firma de `run_batch` con `detector`/`tracker`
(`str | None = None`, al final); (b) **validarlos temprano** —antes de la carga
única de SAM3— reutilizando los mecanismos existentes (`KNOWN_TRACKERS` y
`get_detector`); y (c) **propagarlos** a la llamada `run_inference(...)` del bucle.
Un único archivo cambia: `src/core/batch.py`. Todo lo demás del batch (selección de
videos, skip-done, aislamiento, resumen) se conserva.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Sin dependencias nuevas.** Solo se reutilizan piezas ya presentes:
  - `KNOWN_TRACKERS` de `src.core.trackers` (tupla de nombres válidos).
  - `get_detector` de `src.core.detectors` (levanta `ValueError` ante un nombre no
    registrado, sin cargar modelos).
- **Imports perezosos:** la validación importa `KNOWN_TRACKERS`/`get_detector`
  **dentro** de la helper, no a nivel de módulo, para que `import src.core.batch`
  siga sin arrastrar `trackers`/`detectors` ni sus dependencias pesadas.
- **Sin cambios de config:** los defaults de detector/tracker ya viven en los JSON de
  `configs/` de tareas previas.

---

## 3. Diseño

### 3.1 Helper de validación temprana

Nueva función privada en `batch.py`:

```python
def _validate_detector_tracker(detector: str | None, tracker: str | None) -> None:
    """Valida nombres de detector/tracker SIN cargar modelos.

    Se llama al inicio de run_batch, antes de load_sam3(), para fallar barato.
    None se acepta (se usará el default del config en run_inference).

    Raises:
        ValueError: si tracker no está en KNOWN_TRACKERS, o si detector no está
            registrado (delegado a get_detector).
    """
    from src.core.detectors import get_detector
    from src.core.trackers import KNOWN_TRACKERS

    if tracker is not None and tracker not in KNOWN_TRACKERS:
        raise ValueError(
            f"tracker '{tracker}' no soportado (usa uno de {list(KNOWN_TRACKERS)})."
        )
    if detector is not None:
        get_detector(detector)  # levanta ValueError con su mensaje canónico
```

Notas de diseño:
- **Tracker por membresía** (no se puede usar `get_tracker`, que construye el tracker
  y requiere `frame_rate`): basta con comparar contra `KNOWN_TRACKERS`.
- **Detector reusa `get_detector`**: no se duplica la lista de nombres válidos en
  `batch.py`; el mensaje de error queda en una sola fuente (el registro de detectors).
- `None` es válido en ambos: significa "usar el default del config", que
  `run_inference`/`track_video` resuelven más adelante.

### 3.2 Firma de `run_batch`

Se añaden los dos parámetros **al final** (después de `overwrite`) para no romper
llamadas posicionales:

```python
def run_batch(
    mode: str = "segmentation",
    split: int = 2,
    videos: list[str | int] | None = None,
    sampling: str = "auto",
    max_frames: int | None = None,
    include_masks: bool = False,
    render_video: bool = False,
    overwrite: bool = False,
    detector: str | None = None,   # nuevo
    tracker: str | None = None,    # nuevo
) -> list[dict]:
```

### 3.3 Orden de operaciones en el cuerpo

La validación va **antes** de `load_sam3()` (la carga única), para fallar barato:

```python
_validate_detector_tracker(detector, tracker)   # nuevo, primero
outputs_dir = _load_outputs_dir()
rows = _select_videos(split, videos)
...
bundle = load_sam3()
```

Colocarla la primera línea garantiza que un nombre inválido aborte **sin** leer
config, sin tocar el manifiesto y sin cargar SAM3 (cumple AC-4).

### 3.4 Propagación en el bucle

Se añaden los dos argumentos a la llamada `run_inference(...)` existente
(`batch.py:176-184`), sin reordenar el resto:

```python
res = run_inference(
    ruta,
    mode=mode,
    sampling=sampling,
    max_frames=max_frames,
    include_masks=include_masks,
    render_video=render_video,
    bundle=bundle,
    detector=detector,   # nuevo
    tracker=tracker,     # nuevo
)
```

`run_inference` ya ignora `detector`/`tracker` en `mode="segmentation"` y los aplica
en `mode="tracking"`; el batch no replica esa lógica.

### 3.5 Docstring

Se documentan los dos parámetros nuevos en el docstring de `run_batch`, con la
semántica de "solo tracking":

- `detector`: estrategia de detección por frame (`"sam3_text"` | `"yolo_sam3"`).
  `None` ⇒ default del config. Solo aplica en `mode="tracking"`; ignorado en
  segmentación. Inválido ⇒ `ValueError` antes de cargar SAM3.
- `tracker`: tracker (`"bytetrack"` | `"botsort"`). `None` ⇒ default del config.
  Solo aplica en `mode="tracking"`; ignorado en segmentación. Inválido ⇒ `ValueError`
  antes de cargar SAM3.

Se añade `ValueError` (detector/tracker inválido) a la sección `Raises:`.

---

## 4. Archivos afectados

| Archivo | Cambio |
|---|---|
| `src/core/batch.py` | + helper `_validate_detector_tracker`; firma de `run_batch` +2 params; llamada de validación temprana; +2 args en `run_inference(...)`; docstring. |
| `testing/test_batch_inference.py` | + caso(s) de validación temprana (tracker/detector inválido ⇒ `ValueError` sin cargar SAM3). |

**No se tocan:** `run_inference`, `track_video`, `src/core/trackers/*`,
`src/core/detectors/*`, el schema, `src/data/*`, ni los JSON de `configs/`.

---

## 5. Verificación

- **Sin GPU (local):**
  - `run_batch(tracker="inexistente")` ⇒ `ValueError`, sin cargar SAM3.
  - `run_batch(detector="inexistente")` ⇒ `ValueError`, sin cargar SAM3.
  - Para garantizar que la validación se alcanza antes de la carga, se usa un
    `split`/`videos` que seleccione ≥1 video del manifiesto (si la selección fuera
    vacía la validación igual corre primero, pero el caso con ≥1 fila es el
    representativo).
  - Inspección de firma: `detector`/`tracker` presentes con default `None`.
  - `ruff check .` / `black .` limpios.
- **Con GPU (pod):** una corrida real de `run_batch(mode="tracking",
  detector="yolo_sam3", tracker="botsort", videos=[...])` produce los JSON de
  tracking esperados para esa configuración (fuera del alcance de código de esta
  tarea; se deja como uso posterior).

---

## 6. Riesgos y mitigaciones

- **Romper llamadas posicionales existentes:** mitigado colocando los nuevos
  parámetros al final de la firma.
- **Encarecer `import src.core.batch`:** mitigado con imports perezosos dentro de la
  helper (no a nivel de módulo).
- **Divergencia de mensajes de error de detector:** mitigada delegando en
  `get_detector` en vez de duplicar la lista de nombres válidos.
- **Que la validación quede después de `load_sam3()`:** mitigado fijando la llamada
  como **primera** línea del cuerpo (AC-4 lo exige).
