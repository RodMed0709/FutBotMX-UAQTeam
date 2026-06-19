# Plan técnico — Detector inyectable en segmentación (`detector_in_segmentation`)

- **Tarea atómica:** `detector_in_segmentation`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo hacer que **`run_pipeline` (segmentación) acepte un `detector`
inyectable**, resuelto con el mismo factory `get_detector()` que ya usa
`track_video`, y que la fachada `run_inference` **propague** ese `detector` en su rama
de segmentación (hoy lo descarta). El resultado: `mode="segmentation"` +
`detector="yolo_sam3"` corre, habilitando la Fase 1 del benchmark (eficiencia de
detectores sin tracker). El cambio es **aditivo y retrocompatible**: sin `detector`
(o con `"sam3_text"`) la salida es idéntica a la de hoy.

---

## 2. Stack técnico

- **Reutiliza** `src.core.detectors.get_detector` (factory ya existente, import
  barato a nivel de módulo, igual que en `tracking.py:43`).
- **Reutiliza** la estrategia `yolo_sam3` tal cual (sin tocarla).
- La estrategia `sam3_text` resuelta por `get_detector("sam3_text")` es
  **funcionalmente equivalente** a la llamada directa actual a
  `detect_classes_in_frame(frame, classes=..., bundle=...)` — base de la
  retrocompatibilidad.
- Sin dependencias nuevas, sin cambios de config.

---

## 3. Diseño

### 3.1 Archivos a modificar

| Archivo | Cambio |
|---|---|
| `src/core/pipeline.py` | `run_pipeline` gana `detector`; resuelve `detector_fn`; el loop lo invoca; se ajusta el import. |
| `src/core/inference.py` | La rama de segmentación de `run_inference` pasa `detector` a `run_pipeline`; se corrige el docstring. |
| `testing/test_pipeline.py` | Smoke test de segmentación con `detector="yolo_sam3"` (Parte pod/GPU). |
| `CLAUDE.md` | La descripción de `pipeline.py` menciona que ahora acepta `detector`. |

### 3.2 Firma ampliada de `run_pipeline` (`pipeline.py`)

Se agrega **un** parámetro, **al final** de la firma para no romper llamadas
posicionales existentes:

```
run_pipeline(
    video_path, output_path=None, all_frames=False, mode="per_frame",
    classes=None, bundle=None, include_masks=False, render_video=True,
    detector=None,            # <-- NUEVO: str | None
)
```

- Tipo **`str | None`** (per supuesto técnico #1; no se acepta `Callable`, a
  diferencia de `track_video`, para mantener la firma simple).
- Default `None` ⇒ comportamiento actual.

### 3.3 Resolución del detector

Justo después de `cfg_classes, outputs_dir, config_fps, config = _load_pipeline_config()`
(hoy `pipeline.py:147`), espejando `tracking.py:243-245`:

```python
if detector is None:
    detector = config.get("detector", "sam3_text")   # default del config
detector_fn = get_detector(detector)                 # ValueError si nombre inválido
```

- La resolución ocurre **una sola vez, antes del loop de frames** y **antes** de
  `load_sam3()` (que en `run_pipeline` está más abajo), de modo que un nombre inválido
  **falla barato**, antes de cargar el modelo (criterio de aceptación #4).
- `config` ya está disponible (lo devuelve `_load_pipeline_config`); no hace falta
  releer nada.

### 3.4 Sustitución en el loop + ajuste de import

En el bucle por-frame (hoy `pipeline.py:174`):

```python
# antes
dets = detect_classes_in_frame(frame, classes=classes, bundle=bundle)
# después
dets = detector_fn(frame, classes=classes, bundle=bundle)
```

La firma de la estrategia (`detect(frame, classes=None, bundle=None)`) coincide con la
de `detect_classes_in_frame`, así que el resto del cuerpo (overlay, `frame_record`) no
cambia.

**Imports:** agregar `from src.core.detectors import get_detector` (nivel de módulo,
como en tracking). El import de `detect_classes_in_frame` (`pipeline.py:39`) queda
**sin uso** y se **elimina** para no disparar ruff (F401).

### 3.5 Propagación en la fachada (`inference.py`)

En la rama de segmentación de `run_inference` (hoy `inference.py:69-71`) se añade el
paso del detector:

```python
res = run_pipeline(
    ...,
    detector=detector,   # <-- NUEVO: hoy se descarta
)
```

Y se corrige el docstring del parámetro `detector` (hoy líneas ~49-55) que afirma "En
`mode="segmentation"` se ignora": pasa a describir que es **ortogonal al modo** y que
en segmentación selecciona la estrategia por-frame sin asociación temporal.

### 3.6 Lo que NO cambia (anti-alcance técnico)

- `src/core/inference_schema.py`, las rutas de salida y el `skip-done` (eso es la
  tarea 2, `config_aware_output_paths`).
- `src/core/segmentation.py` y la estrategia `yolo_sam3`.
- Los archivos de `configs/` (la clave `detector` ya existe).
- El esquema/formato del JSON de salida.
- `src/core/batch.py`: ya reenvía `detector` a `run_inference` en cualquier modo.

---

## 4. Cambios de configuración y dependencias

- **Ninguno.** No hay nuevas dependencias ni claves de config. El default de
  segmentación sigue saliendo de `config.get("detector", "sam3_text")`, idéntico a
  tracking.

---

## 5. Validación (`testing/test_pipeline.py`)

Smoke test (funcional), siguiendo la filosofía del repo (funcional ahora, visual
después):

### 5.1 Parte local — **sin GPU**

- Verificar que `run_pipeline` acepta `detector` y que un **nombre inválido** levanta
  `ValueError` **sin** cargar SAM3 (se puede comprobar pasando un detector basura y
  esperando la excepción antes de cualquier inferencia).

### 5.2 Parte **pod/GPU** — clip corto

- `mode="segmentation"`, `detector="yolo_sam3"` sobre un video corto: corre sin error
  y produce un JSON con la **misma estructura** que `sam3_text` (mismas claves de
  header y de `frames`).
- `detector="sam3_text"` (o `None`) sobre el mismo clip: salida **equivalente** a la
  de hoy (retrocompatibilidad).

### 5.3 Calidad

- `ruff check .` y `black .` limpios (en particular, sin F401 por el import retirado).

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec §5) | Cubierto por |
|---|---|
| 1. `run_pipeline` acepta `detector`, default config | §3.2, §3.3 |
| 2. Detección vía estrategia resuelta | §3.4 |
| 3. Fachada propaga `detector`; docstring corregido | §3.5 |
| 4. `ValueError` antes de cargar SAM3 | §3.3, §5.1 |
| 5. `yolo_sam3` en segmentación → mismo JSON | §3.4, §5.2 |
| 6. Retrocompatibilidad sin `detector`/`sam3_text` | §2, §3.2, §5.2 |
| 7. No toca schema/rutas/configs/`yolo_sam3` | §3.6 |
| 8. `CLAUDE.md` actualizado | §3.1 |

---

## 7. Riesgos y consideraciones

- **Equivalencia `sam3_text` ↔ `detect_classes_in_frame`.** La retrocompatibilidad
  depende de que `get_detector("sam3_text")` envuelva exactamente esa función con la
  misma firma. Es así por diseño de `detector_strategy`, pero el smoke test §5.2 lo
  confirma comparando estructura de JSON.
- **Orden de la resolución vs. `load_sam3`.** Hay que resolver `detector_fn` **antes**
  de `load_sam3()` dentro de `run_pipeline` para que el fail-cheap se cumpla; ubicar
  el bloque §3.3 inmediatamente tras `_load_pipeline_config()` lo garantiza.
- **Import retirado.** Quitar `detect_classes_in_frame` exige confirmar que no se usa
  en otro punto de `pipeline.py` (hoy solo en el loop); si quedara otra referencia, se
  conserva el import.
- **Alcance contenido.** El namespacing de salidas (necesario para correr varias
  configs sin pisarse) **no** entra aquí; hasta la tarea 2, el benchmark de Fase 1
  debe seguir corriendo "una config a la vez" o sobrescribirá el JSON por stem.
