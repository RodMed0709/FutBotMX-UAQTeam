# Plan técnico — Fachada única de inferencia (`unified_inference`)

- **Tarea atómica:** `unified_inference`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso de referencia:** roadmap del pipeline de inferencia unificado + batch (tarea 3)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo construir **una sola función fachada** `run_inference(...)` (en un módulo
nuevo `src/core/inference.py`) que sea la **única puerta de entrada por video**, con
`mode ∈ {"segmentation", "tracking"}`. La fachada:

- delega en las implementaciones existentes `run_pipeline` (per-frame) y `track_video`
  (tracking) **sin reimplementar** el bucle de inferencia;
- resuelve el **muestreo por modo** mediante un selector explícito `sampling`, y
  **valida** la combinación inválida (cuota + tracking) con `ValueError`;
- **unifica el retorno** a `{"json", "video", "index"}` (con `"index": None` en
  segmentación);
- hereda `include_masks` y `render_video` (tareas 1 y 2) como parámetros
  **sobreescribibles** con su default por uso;
- propaga un `bundle` SAM3 ya cargado hacia **ambos** modos.

Para que la propagación del modelo (y el filtrado de clases por llamada) funcione
también en segmentación, esta tarea **amplía `run_pipeline`** de forma aditiva para
aceptar `bundle` y `classes` (hoy solo los acepta `track_video`).

---

## 2. Stack técnico

- **Python:** 3.11.
- **Sin dependencias nuevas** (`requirements.txt` no cambia).
- **Módulo nuevo `src/core/inference.py`**: solo importa `run_pipeline` y
  `track_video` a nivel de módulo (baratos, no arrastran torch). Los imports
  perezosos (torch/cv2/supervision/trackers) se mantienen **dentro** de las
  implementaciones, no en la fachada.
- **Sin cambios** en `inference_schema.py`, `overlay.py`, `video_writer.py`,
  `frame_extraction.py`, `segmentation`, `load_sam3`, ByteTrack ni el muestreo de
  cada camino. El único módulo de implementación que se toca es `pipeline.py`
  (ampliación aditiva de la firma de `run_pipeline`).

---

## 3. Diseño

### 3.1 Estructura de archivos

```
src/core/inference.py               # NUEVO: run_inference (fachada única)
src/core/pipeline.py                # MOD: run_pipeline acepta bundle y classes (aditivo)
testing/test_unified_inference.py   # NUEVO: smoke ambos modos + casos inválidos (local + pod)
```

`tracking.py` **no se toca** (ya acepta `classes`, `bundle`, `max_frames`,
`include_masks`, `render_video` y devuelve `"index"`).

### 3.2 Firma de la fachada

```python
def run_inference(
    video_path: Path | str,
    mode: str = "segmentation",
    output_path: Path | None = None,
    classes: list[dict] | None = None,
    sampling: str = "auto",
    max_frames: int | None = None,
    bundle: "Sam3Bundle | None" = None,
    include_masks: bool = False,
    render_video: bool = True,
) -> dict:
```

- **`mode`**: `"segmentation"` (default) o `"tracking"`. Otro valor → `ValueError`.
  Traducción interna: `"segmentation"` → `run_pipeline(mode="per_frame")`;
  `"tracking"` → `track_video`.
- **`sampling`**: selector explícito de estrategia de muestreo (ver §3.3). Sustituye
  al `all_frames` booleano **a nivel de fachada**; la fachada lo traduce a los
  controles internos de cada implementación.
- **`max_frames`**: tope de frames **contiguos** (solo relevante en tracking; en
  segmentación se ignora de forma documentada).
- **`classes` / `bundle` / `include_masks` / `render_video`**: ortogonales al modo,
  sobreescribibles, con sus defaults heredados (`render_video=True` por ser uso de
  un solo video; `include_masks=False`).

### 3.3 Resolución y validación del muestreo (`sampling`)

`sampling ∈ {"auto", "quota", "all", "contiguous"}`, default `"auto"` (el **modo**
decide la estrategia). La fachada lo traduce a los controles internos y **rechaza**
las combinaciones sin sentido **antes** de cargar modelo o extraer frames:

| `sampling`     | `segmentation` → `run_pipeline`        | `tracking` → `track_video`              |
|----------------|----------------------------------------|-----------------------------------------|
| `"auto"`       | cuota: `all_frames=False`              | contiguo: `max_frames=max_frames`       |
| `"quota"`      | cuota: `all_frames=False`              | **`ValueError`** (AC-6)                 |
| `"all"`        | completo: `all_frames=True`            | completo: `max_frames=None`             |
| `"contiguous"` | **`ValueError`** (seg no hace prefijo) | contiguo: `max_frames=max_frames`       |
| otro valor     | `ValueError`                           | `ValueError`                            |

Notas:
- **AC-6 (caso inválido) = `sampling="quota"` + `mode="tracking"`** → `ValueError`
  con mensaje en español explícito, p. ej.:
  *"sampling='quota' no es compatible con mode='tracking' (ByteTrack requiere frames
  contiguos)."* Guard simétrico: `sampling="contiguous"` + `segmentation` también
  levanta `ValueError` (segmentación no tiene muestreo de prefijo contiguo).
- **`max_frames` en segmentación se ignora** (documentado): el conteo de cuota lo fija
  `preprocess.frame_quota` de la config y `all_frames` decide cuota/completo. La
  fachada no lo pasa a `run_pipeline`.
- **`max_frames` en tracking** es el tope contiguo (cap). `"auto"`/`"contiguous"`
  respetan el cap; `"all"` fuerza video completo (`max_frames=None`, documentado).
- La validación de `mode` y `sampling` ocurre **al inicio** de `run_inference`, sin
  efectos colaterales (no carga SAM3 ni abre el video) — así los casos inválidos son
  testeables localmente sin GPU.

### 3.4 Mapeo a las implementaciones y retorno unificado

```python
# pseudocódigo de la fachada (sin imports perezosos: torch vive en las implementaciones)
def run_inference(video_path, mode="segmentation", output_path=None, classes=None,
                  sampling="auto", max_frames=None, bundle=None,
                  include_masks=False, render_video=True):
    if mode == "segmentation":
        all_frames = _resolve_segmentation_sampling(sampling)   # raises on quota+? / contiguous
        res = run_pipeline(
            video_path, output_path=output_path, all_frames=all_frames,
            mode="per_frame", classes=classes, bundle=bundle,
            include_masks=include_masks, render_video=render_video,
        )
        return {"json": res["json"], "video": res["video"], "index": None}

    if mode == "tracking":
        eff_max = _resolve_tracking_sampling(sampling, max_frames)  # raises on quota
        return track_video(
            video_path, output_path=output_path, classes=classes,
            max_frames=eff_max, bundle=bundle,
            include_masks=include_masks, render_video=render_video,
        )

    raise ValueError(f"mode '{mode}' no soportado (usa 'segmentation' o 'tracking').")
```

- **Normalización del retorno:** `track_video` ya devuelve
  `{"json", "video", "index"}`; en segmentación la fachada toma el
  `{"json", "video"}` de `run_pipeline` y **añade `"index": None`**. Resultado: forma
  única `{"json": Path, "video": Path | None, "index": dict | None}` para que
  `batch_inference` (tarea 4) la agregue sin ramificar por modo.
- La fachada **no** reabre ni reescribe archivos; solo enruta y adapta el `dict`.

### 3.5 Ampliación aditiva de `run_pipeline` (`pipeline.py`)

Para lograr **simetría real** (la batch carga SAM3 una sola vez y puede filtrar
clases por llamada también en segmentación), se amplía `run_pipeline` igual que ya
hace `track_video`:

- **Firma:** se añaden `classes: list[dict] | None = None` y
  `bundle: Sam3Bundle | None = None` (parámetros nuevos, **default `None` = comportamiento
  actual**; no rompe `testing/test_pipeline.py` ni llamadas existentes).
- **Clases:** tras leer la config, se respeta el argumento si llega:
  ```python
  cfg_classes, outputs_dir, config_fps, config = _load_pipeline_config()
  classes = classes if classes is not None else cfg_classes
  ```
  Las `classes` ya se usan en `detect_classes_in_frame`, `overlay_detections` y
  `build_header`, así que el override fluye sin más cambios.
- **Modelo:** `bundle = bundle or load_sam3()` (idéntico patrón a `track_video`),
  sustituyendo la carga incondicional actual (`pipeline.py:156`).
- **Sin otros cambios**: el bucle per-frame, el muestreo (`all_frames`), el esquema y
  el render quedan igual.

> Esto resuelve la asimetría que el spec señaló en §6 ("lectura de config sin
> unificar"): no se unifica la *lectura* de config, pero sí se igualan las **entradas**
> (`bundle`/`classes`) para que la fachada y la batch traten ambos modos por igual.

### 3.6 Lo que NO cambia (anti-alcance técnico)

- **`tracking.py`**, `inference_schema.py`, `overlay.py`, `video_writer.py`,
  `frame_extraction.py`, `segmentation`, `load_sam3`, la asociación ByteTrack y el
  muestreo de cada camino.
- **Firma pública de `track_video`** y semántica de `run_pipeline` con sus defaults
  (la ampliación es puramente aditiva).
- **No** se construye `batch_inference` (tarea 4) ni se cambia ningún default a
  "render OFF" (eso lo hará la batch al pasar el flag).
- **No** se unifica la *lectura* de configuración entre caminos (helper compartido =
  trabajo futuro).
- **No** se añade muestreo disperso a tracking ni continuidad a segmentación: la
  asimetría se resuelve **exponiéndola** (`sampling`) y **validándola**.

---

## 4. Cambios de configuración y dependencias

- **`requirements.txt`:** sin cambios.
- **Config (`configs/00_testing_config.json`):** sin cambios. `mode`, `sampling`,
  `max_frames`, `include_masks`, `render_video` son **parámetros de función**, no
  claves de config (mismo criterio que `include_masks`/`render_video`).
- **`CLAUDE.md`:** al implementar, actualizar la sección de arquitectura (existe una
  fachada `run_inference`; `run_pipeline` y `track_video` pasan a ser
  implementaciones internas) y retirar la nota de que `mode="tracking"` está cableado
  vía stub / fuera del pipeline.

---

## 5. Validación (`testing/test_unified_inference.py`)

> Filosofía de tests del repo: smoke funcional; lo que invoca SAM3 corre en **pod/GPU**.
> Aquí, además, **buena parte de la lógica nueva (validación de `mode`/`sampling`,
> normalización del retorno) es testeable localmente sin modelo**, porque la
> validación ocurre antes de cargar SAM3.

### 5.1 Parte A — local, **sin GPU**

- **Firma:** `inspect.signature(run_inference)` incluye `mode`, `sampling`,
  `max_frames`, `classes`, `bundle`, `include_masks`, `render_video` con los defaults
  esperados; y `signature(run_pipeline)` ahora incluye `classes` y `bundle` (default
  `None`).
- **Validación sin efectos colaterales (sin cargar SAM3):**
  - `mode="bad"` → `ValueError`.
  - `sampling="quota"` + `mode="tracking"` → `ValueError` (AC-6).
  - `sampling="contiguous"` + `mode="segmentation"` → `ValueError`.
  - `sampling="rara"` → `ValueError`.
  - (Se comprueba que estos casos fallan **sin** invocar el modelo, p. ej.
    monkeypatcheando `load_sam3`/`run_pipeline`/`track_video` o usando una ruta de
    video inexistente y verificando que el `ValueError` precede al acceso al video.)

### 5.2 Parte B — **pod/GPU**, clip corto

- **segmentation (auto):** `run_inference(video, mode="segmentation")` → retorno
  `{"json": Path, "video": Path, "index": None}`; JSON + mp4 en
  `outputs/inference/<stem>/`.
- **segmentation completo:** `run_inference(video, mode="segmentation",
  sampling="all", render_video=False)` → `"video"` is `None`, `"index"` is `None`,
  JSON presente.
- **tracking (auto, cap):** `run_inference(video, mode="tracking",
  max_frames=<pequeño>)` → `{"json", "video": Path, "index": dict}` no vacío; JSON con
  `frames` y `tracks`.
- **tracking OFF + masks:** `run_inference(video, mode="tracking",
  max_frames=<pequeño>, render_video=False, include_masks=True)` → `"video"` is
  `None`, JSON con `rle`, `"index"` dict.
- **reuso de bundle en ambos modos:** cargar `bundle = load_sam3()` una vez y pasarlo
  a una llamada `segmentation` y a una `tracking`; ambas corren **sin** recargar el
  modelo (verifica la ampliación de `run_pipeline`).

### 5.3 Calidad

- `ruff check .` y `black .` sin hallazgos.
- Importabilidad: `from src.core.inference import run_inference`;
  `from src.core.pipeline import run_pipeline` (firma ampliada) sigue importando.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Puerta única | §3.1, §3.2 | `run_inference` en `src/core/inference.py` |
| AC-2 Modo por defecto | §3.2 | `mode="segmentation"` default |
| AC-3 Tracking cableado | §3.4 | `tracking` → `track_video` (sin `NotImplementedError`) |
| AC-4 Muestreo por modo | §3.3 | `sampling="auto"`: seg=cuota, tracking=contiguo |
| AC-5 Controles expuestos | §3.3 | `sampling="all"` (seg completo); `max_frames` (cap tracking) |
| AC-6 Caso inválido | §3.3 | `quota`+`tracking` → `ValueError` |
| AC-7 Herencia de flags | §3.2 | `render_video`/`include_masks` sobreescribibles, ortogonales |
| AC-8 Retorno unificado | §3.4 | `{"json","video","index"}`; `index=None` en seg |
| AC-9 Fachada delgada | §3.4 | enruta a `run_pipeline`/`track_video`, no reimplementa |
| AC-10 Reuso de modelo | §3.4, §3.5 | `bundle` propagado a ambos modos (amplía `run_pipeline`) |
| AC-11 Sin cambios colaterales | §3.6 | esquema y módulos vecinos intactos; solo `pipeline.py` aditivo |
| AC-12 Verificación | §5.1, §5.2 | validación local + smoke ambos modos (pod) |

---

## 7. Riesgos y consideraciones

- **`sampling` como tercera fuente de muestreo:** se introduce a **nivel de fachada**
  y se traduce a los controles internos ya existentes (`all_frames` en seg,
  `max_frames` en tracking); no se crea un mecanismo de muestreo nuevo en las
  implementaciones. La tabla §3.3 es el único punto de verdad de esa traducción.
- **Ampliación de `run_pipeline`:** aunque la tarea se describe como "fachada
  delgada", tocar `run_pipeline` es necesario para el reuso de modelo en ambos modos
  (decisión explícita, punto 10-A). El cambio es aditivo y con defaults `None`, así
  que no altera el comportamiento de las llamadas/tests actuales.
- **`max_frames` ignorado en segmentación:** podría sorprender a quien lo pase
  esperando un cap; se documenta en el docstring y se cubre implícitamente (seg usa
  cuota de config). No se levanta error para no fragmentar la API (es un control que
  simplemente no aplica, igual que `all_frames` no aplicaba a tracking).
- **Retorno `Path | None`:** la clave `"video"` puede ser `None` (render OFF) y
  `"index"` puede ser `None` (segmentación); los consumidores (tests, batch) deben
  tolerar ambos. Se documenta en el docstring.
- **Alcance:** esta tarea entrega la fachada de **un video**; la orquestación de
  lotes, el skip-done y el aislamiento de errores son de `batch_inference` (tarea 4).
