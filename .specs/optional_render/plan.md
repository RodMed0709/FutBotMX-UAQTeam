# Plan técnico — Render de mp4 opcional vía flag (`optional_render`)

- **Tarea atómica:** `optional_render`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso de referencia:** roadmap del pipeline de inferencia unificado + batch (tarea 2)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo introducir un flag **`render_video`** en `run_pipeline`
(`pipeline.py`, seg-only) y `track_video` (`tracking.py`, tracking) que decida, por
llamada, si se genera el mp4 anotado. El JSON del esquema común
(`inference_schema`) sigue siendo el entregable **siempre**. Cuando el render está
apagado, se **salta todo el trabajo de visualización** (overlay + escritor de
video + acumulación de frames), no solo la escritura final. No se toca la lógica de
detección/tracking, el esquema, ni los módulos `overlay`/`video_writer`/
`frame_extraction`.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Control de flujo del escritor en tracking:** `contextlib.nullcontext` (stdlib)
  para que el bucle de streaming corra con o sin escritor de video bajo el **mismo**
  `with`, sin anidar ni duplicar el bucle.
- **Sin dependencias nuevas** (`requirements.txt` no cambia). `imageio`/`cv2` siguen
  con import perezoso donde ya estaban; con render OFF ni siquiera se invoca el
  escritor.
- **Sin cambios** en `inference_schema.py`, `overlay.py`, `video_writer.py`,
  `frame_extraction.py`, `segmentation`, `load_sam3`, ByteTrack ni el muestreo.

---

## 3. Diseño

### 3.1 Estructura de archivos

```
src/core/pipeline.py             # MOD: run_pipeline gana render_video; salta write_video
src/core/tracking.py             # MOD: track_video gana render_video; bucle sin writer si OFF
testing/test_optional_render.py  # NUEVO: valida ON/OFF en ambos modos (GPU/pod)
```

Cambios **aditivos y locales** a las dos firmas; ningún módulo nuevo.

### 3.2 Contrato del flag (común a ambos modos)

- **Nombre/tipo:** `render_video: bool`.
- **Default:** `True` (uso de un solo video). No depende del modo. La futura capa
  `batch_inference` lo pasará `False` explícitamente.
- **Ortogonal** a `mode`, `all_frames`/`max_frames` e `include_masks`: cualquier
  combinación es válida.
- **Retorno:** se mantiene la forma del `dict` actual. La clave `"video"` **siempre
  está presente**; vale `Path` del mp4 si se renderizó y **`None`** si no. En
  tracking se conserva además `"index"`.

### 3.3 Integración en `run_pipeline` (seg-only)

Hoy (ver `pipeline.py`) el bucle compone `overlay_detections` por frame, acumula en
`composed` y al final llama `write_video(np.stack(composed), ...)`. Cambios:

- **Firma:**
  ```python
  def run_pipeline(video_path, output_path=None, all_frames=False,
                   mode="per_frame", include_masks=False,
                   render_video=True) -> dict[str, Path | None]:
  ```
- **Bucle:** la composición del overlay y el `append` a `composed` quedan **dentro
  de `if render_video:`**. El `frame_record(...)` (entregable) se construye
  **siempre**, fuera de ese `if`.
  ```python
  for i, frame in enumerate(frames):
      dets = detect_classes_in_frame(frame, classes=classes, bundle=bundle)
      if render_video:
          composed.append(overlay_detections(frame, dets, classes=classes))
      records.append(frame_record(int(source_indices[i]), dets, include_masks))
  ```
- **Escritura del mp4:** condicional.
  ```python
  mp4_out = write_video(np.stack(composed), mp4_path, fps=fps) if render_video else None
  ```
- **Cabecera y JSON:** sin cambios (se escriben siempre). `fps`, `resolution`,
  `num_frames` se siguen calculando igual (la cabecera los necesita
  independientemente del render).
- **Retorno:** `{"json": json_path, "video": mp4_out}` (`mp4_out` es `None` si OFF).

Nota: `mp4_path` se sigue **derivando** (de `output_path` o de `inference_paths`)
porque de él sale el nombre del JSON; con render OFF simplemente no se escribe.

### 3.4 Integración en `track_video` (tracking)

Hoy (ver `tracking.py`) el bucle de streaming vive dentro de
`with open_video_writer(mp4_path, fps=fps) as append:` y llama `append(composed)`
por frame. Para correr el **mismo** bucle con o sin escritor:

- **Firma:**
  ```python
  def track_video(video_path, output_path=None, classes=None, max_frames=None,
                  bundle=None, include_masks=False, render_video=True) -> dict:
  ```
- **Apertura condicional del escritor** con `nullcontext`:
  ```python
  from contextlib import nullcontext
  ...
  writer_cm = open_video_writer(mp4_path, fps=fps) if render_video else nullcontext(None)
  with writer_cm as append:
      for frame_index, frame in iter_frames(video_path, max_frames):
          ...
          # asociación ByteTrack y construcción de per_frame: SIN cambios
          if render_video:
              composed = overlay_detections(frame, per_frame, classes=classes)
              append(composed)
          frames_records.append(frame_record(frame_index, per_frame, include_masks))
  ```
  - Con render OFF, `nullcontext(None)` **no abre `imageio`** ni crea el archivo mp4;
    `append` queda `None` y nunca se invoca (protegido por el `if render_video`).
  - El resto del bucle (máscara→caja, `trackers[name].update`, `obj_id` estables,
    actualización de `tracks`/`TrackObservation`, `frame_record`) es **idéntico**: el
    tracking y el entregable no dependen del render.
- **Cabecera y JSON unificado** (`frames` + `tracks`): sin cambios; se escriben
  siempre.
- **Retorno:** `{"json": json_path, "video": (mp4_path if render_video else None),
  "index": tracks}`.

`mp4_path` se sigue derivando para nombrar el JSON; con render OFF no se escribe.

### 3.5 Manejo de `output_path` cuando no se renderiza

- `output_path` solo afecta a **dónde** irían los archivos. Su rol de **derivar la
  ruta del JSON** se mantiene intacto (igual que hoy: `json_path =
  mp4_path.with_name(f"{mp4_path.stem}.json")`).
- Con render OFF **no se crea ningún archivo de video**, se haya pasado o no
  `output_path`. No es error pasar `output_path` con `render_video=False`: la ruta de
  video simplemente se ignora para escritura.

### 3.6 Lo que NO cambia (anti-alcance técnico)

- El esquema y sus builders (`inference_schema.py`): el render no es parte del
  contrato del dato → **no** se añade campo al JSON ni se toca `SCHEMA_VERSION`.
- `overlay.py`, `video_writer.py`, `frame_extraction.py`, `segmentation`,
  `load_sam3`, la asociación ByteTrack y el muestreo de frames.
- Los defaults de la sección `tracking` de la config y `visualization.output_fps`.

---

## 4. Cambios de configuración y dependencias

- **`requirements.txt`:** sin cambios.
- **Config (`configs/00_testing_config.json`):** sin cambios. `render_video` es
  **parámetro de función**, no clave de config (mismo criterio que `include_masks`).

---

## 5. Validación (`testing/test_optional_render.py`)

> Ambas funciones invocan SAM3, así que la validación de comportamiento corre en
> **GPU/pod** (filosofía de tests del repo: smoke funcional con caso real). La parte
> local se limita a introspección de firma (sin modelo).

### 5.1 Parte A — local, **sin GPU**

- **Firma:** `inspect.signature(run_pipeline)` y `signature(track_video)` incluyen
  `render_video` con default `True`. (No invoca el modelo.)

### 5.2 Parte B — **GPU/pod**, clip corto

- **seg-only ON:** `run_pipeline(video, all_frames=False, render_video=True)` →
  retorno con `"video"` = `Path` existente; el mp4 existe en
  `outputs/inference/<stem>/<stem>.mp4`; el JSON existe.
- **seg-only OFF:** `run_pipeline(video, all_frames=False, render_video=False)` →
  retorno con `"video"` is `None`; el mp4 **no** existe; el JSON **sí** existe y es
  válido (misma forma que con render ON).
- **tracking ON:** `track_video(video, max_frames=<pequeño>, render_video=True)` →
  `"video"` = `Path` existente; mp4 + JSON (con `frames` y `tracks`) presentes.
- **tracking OFF:** `track_video(video, max_frames=<pequeño>, render_video=False)` →
  `"video"` is `None`; mp4 **no** existe; JSON (con `frames` y `tracks`) presente e
  idéntico en forma al de render ON.
- **Ortogonalidad:** un caso `render_video=False, include_masks=True` → JSON con
  `rle` y **sin** mp4 (combinación que usará la exportación de predicciones).

### 5.3 Calidad

- `ruff check .` y `black .` sin hallazgos.
- Importabilidad intacta: `from src.core.pipeline import run_pipeline`,
  `from src.core.tracking import track_video`.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Flag en ambos modos | §3.2, §3.3, §3.4 | `render_video` en las dos firmas |
| AC-2 Default single = ON | §3.2 | default `True`, no rompe llamadas actuales |
| AC-3 Independiente del modo | §3.2 | flag ortogonal a `mode` |
| AC-4 JSON siempre | §3.3, §3.4 | cabecera+JSON fuera del `if render_video` |
| AC-5 mp4 condicional | §3.3, §3.4 | `write_video`/`open_video_writer` bajo el flag |
| AC-6 Ahorro real | §3.3, §3.4 | overlay/escritor/acumulación dentro del `if` |
| AC-7 Retorno estable | §3.2 | `"video"` siempre presente (`Path`/`None`) |
| AC-8 Ortogonal a `include_masks` | §3.2, §5.2 | caso OFF+masks validado |
| AC-9 Sin cambios colaterales | §3.6 | esquema y módulos vecinos intactos |
| AC-10 Verificación | §5.1, §5.2 | firma (local) + ON/OFF ambos modos (pod) |

---

## 7. Riesgos y consideraciones

- **`nullcontext(None)` en tracking:** el riesgo es invocar `append` cuando es
  `None`; se evita porque toda llamada a `append` queda bajo `if render_video`. Es la
  forma más limpia de no duplicar el bucle de streaming.
- **Derivación del JSON vía `mp4_path`:** se conserva el cálculo de `mp4_path` aun con
  render OFF (de él sale el nombre del JSON); solo se omite la **escritura** del
  video. No se introduce un archivo de video vacío.
- **`np.stack(composed)` en seg-only:** con render OFF, `composed` queda vacío y
  nunca se llama `write_video`/`np.stack`, evitando el `ValueError` de "N=0" del
  validador de `write_video`.
- **Compatibilidad de retorno:** el tipo de `"video"` pasa de `Path` a `Path | None`;
  los consumidores actuales (tests, notebooks) que asumían `Path` deben tolerar
  `None` cuando se apaga el render. Documentar en el docstring.
- **Alcance:** esta tarea **no** unifica los caminos ni crea la batch; solo habilita
  el opt-in. El default OFF para lotes lo fija `batch_inference` al pasar el flag.
