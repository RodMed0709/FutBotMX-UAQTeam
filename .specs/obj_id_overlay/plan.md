# Plan técnico — Overlay por `obj_id` (`obj_id_overlay`)

- **Tarea atómica:** `obj_id_overlay`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso de referencia:** roadmap de siguientes pasos (tarea 1)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo construir un **post-pase desacoplado** que, dado un **JSON de tracking** +
el **video fuente**, escriba un **mp4 nuevo** donde la identidad de cada objeto sea
visible: **caja + etiqueta `nombre #id` + trayectoria** (color **por clase**), con
relleno de máscara **opcional** (si el JSON trae `rle`). No re-infiere, no carga SAM3 y
no toca `track_video`, `overlay_detections`, el esquema ni `pipeline.py`.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Sin dependencias nuevas** (`requirements.txt` no cambia). `cv2` (OpenCV) ya está en
  el stack; se importa de forma **perezosa**.
- **Módulo nuevo `src/core/track_overlay.py`** (no se mezcla con `overlay.py`, que es
  numpy-only para el overlay en vivo).
- **Reuso de bloques existentes:**
  - `frame_extraction.iter_frames(video_path, max_frames=None)` → recorrido en
    streaming `(frame_index, frame)`.
  - `video_writer.open_video_writer(path, fps)` → escritura incremental del mp4 (sin
    OOM en videos largos).
  - `inference_schema.decode_rle(rle)` → máscara booleana (solo si se pide relleno).
- **Dibujo con `cv2`:** `rectangle`, `putText`, `getTextSize`, `polylines`/`circle`.
- **Sin cambios** en `overlay.py`, `track_video`, `inference_schema`
  (incl. `SCHEMA_VERSION`), `pipeline.py`, `segmentation`, ByteTrack.

---

## 3. Diseño

### 3.1 Estructura de archivos

```
src/core/track_overlay.py          # NUEVO: render_obj_id_overlay (driver + dibujo)
testing/test_obj_id_overlay.py     # NUEVO: Parte A local (sintético, sin SAM3) + JSON real
```

### 3.2 Firma del driver

```python
def render_obj_id_overlay(
    json_path: Path | str,
    video_path: Path | str | None = None,
    output_path: Path | None = None,
    draw_masks: bool = False,
    trajectory_window: int | None = None,
    excluded_classes: list[str] | None = None,
) -> Path:
```

- **`json_path`**: JSON de tracking de entrada.
- **`video_path`**: si es `None`, se toma de la cabecera del JSON (`payload["video"]`);
  permite override si el video se movió.
- **`output_path`**: si es `None`, se deriva como `<json_stem>_obj_id.mp4` **junto al
  JSON** (no pisa el mp4 de inferencia `<stem>.mp4`).
- **`draw_masks`**: toggle del relleno de máscara (default **OFF**).
- **`trajectory_window`**: N (frames) de la ventana de estela; `None` ⇒ de config
  (`visualization.trajectory_window`).
- **`excluded_classes`**: clases a no dibujar; `None` ⇒ de config
  (`visualization.overlay_excluded_classes`, default `["green_floor"]`).
- **Devuelve** la ruta (`Path`) del mp4 escrito.

### 3.3 Lectura y validación del JSON

```python
payload = json.loads(Path(json_path).read_text("utf-8"))
if payload.get("mode") != "tracking":
    raise ValueError("render_obj_id_overlay requiere un JSON de mode='tracking'.")
```

- **fps**: `payload["fps"]`. **video**: `video_path or payload["video"]`.
- **Colores por clase**: del **snapshot embebido** `payload["config"]["classes"]`
  (mapa `name -> tuple(color)`), para reproducir los colores exactos de la corrida.
- **`trajectory_window` / `excluded_classes`**: del parámetro si llega, si no de la
  **config activa** (claves nuevas; son decisiones de *overlay*, no de la inferencia).

### 3.4 Lookups (frames + trayectorias)

- **Por frame:** `frame_by_index = {f["frame_index"]: f for f in payload["frames"]}`
  (registro frame-indexed: `detections[clase] -> [{obj_id, bbox, centroid, ...}]`).
- **Trayectorias precomputadas:** desde `payload["tracks"]`, por `obj_id` la lista
  **ordenada** de `(frame_index, centroid, class)`. En el frame `f` se dibuja el tramo
  con `frame_index` en `(f − N, f]` (**rango de `frame_index`**, no "últimas N
  observaciones"). Los centroides se redondean a `int` para `cv2`.

### 3.5 Dibujo por frame (orden y primitivas)

Recorrido **único** en streaming con `iter_frames`; por cada `(frame_index, frame)`:

1. **Relleno de máscara** (si `draw_masks` **y** el detection trae `rle`): `decode_rle`
   → mezcla alpha con `visualization.overlay_alpha` (mismo criterio que
   `overlay_detections`), color de clase. Si se pidió `draw_masks` pero el JSON no trae
   `rle` (`include_masks=False`), se **avisa una vez** y se omite.
2. **Trayectorias:** `cv2.polylines` de los centroides de cada `obj_id` no excluido en
   la ventana, color de su clase.
3. **Cajas:** `cv2.rectangle` con el `bbox = [x, y, w, h]` de la vista `frames`
   (`(x, y)`–`(x+w, y+h)`), color de clase.
4. **Etiquetas:** `nombre #id` (o solo `nombre` si `obj_id == -1`, warm-up) **encima**
   de la esquina superior izquierda, sobre un **rectángulo de fondo relleno** del color
   de clase (vía `getTextSize`); color del texto **negro o blanco** según la luminancia
   del color de clase (`0.299R+0.587G+0.114B`).

Grosor de línea y escala de fuente **derivados de la resolución** (factor sobre
`max(H, W)`, con default en config). Las clases en `excluded_classes` se saltan por
completo (sin caja/etiqueta/estela/máscara). Frames sin registro o sin detecciones se
escriben **tal cual**.

### 3.6 Color y formato (RGB de punta a punta)

- `iter_frames` entrega frames **RGB** y `open_video_writer` escribe **RGB**; los
  colores de `classes[].color` (RGB) se pasan **tal cual** a `cv2` (agnóstico al orden
  de canal). **No** se hace conversión BGR↔RGB → se evita el bug de colores invertidos.

### 3.7 Salida

```python
with open_video_writer(output_path, fps=fps) as append:
    for frame_index, frame in iter_frames(video_path):
        composed = _draw_frame(frame, frame_index, ...)  # copia, no muta
        append(composed)
return output_path
```

- mp4 nuevo junto al JSON; `fps`/resolución de la fuente; el JSON y el mp4 de
  inferencia **no** se tocan.

### 3.8 Claves de configuración nuevas

Bajo `visualization` (todas con default en código y sobreescribibles por parámetro):

- `trajectory_window` (int, p. ej. `60`) — N frames de la estela.
- `overlay_excluded_classes` (list[str], default `["green_floor"]`).
- (opcional) `overlay_line_scale` / `overlay_font_scale` para el factor de
  grosor/fuente; si no están, se usan defaults en código.

Se añadirán al `configs/00_testing_config.json` con valores por defecto.

### 3.9 Lo que NO cambia (anti-alcance técnico)

- `overlay.py` (`overlay_detections`/`show_overlay`), `track_video`, `pipeline.py`,
  `inference_schema.py` (incl. `SCHEMA_VERSION`), `segmentation`, ByteTrack.
- No se añade color por `obj_id`, ni renderizador unificado seg+tracking, ni post-pase
  de seg con RLE.

---

## 4. Cambios de configuración y dependencias

- **`requirements.txt`:** sin cambios.
- **Config (`configs/00_testing_config.json`):** se añaden las claves de §3.8 bajo
  `visualization` (con defaults). El resto de parámetros del overlay son **argumentos
  de función**.
- **`CLAUDE.md`:** al implementar, documentar el post-pase
  `track_overlay.py::render_obj_id_overlay` en la sección de arquitectura.

---

## 5. Validación (`testing/test_obj_id_overlay.py`)

> El post-pase **no necesita SAM3**, así que casi todo es testeable **localmente sin
> GPU** con datos sintéticos. Se complementa con una corrida sobre un JSON real.

### 5.1 Parte A — local, **sin GPU** (datos sintéticos)

- **Construir un mini-JSON de tracking** (a mano): cabecera con `mode="tracking"`,
  `fps`, `resolution`, `config` con `classes` (con `color`) incluyendo `green_floor`;
  `frames` con 2-3 objetos (uno `obj_id=-1` warm-up) y `tracks` con sus centroides; y
  un **video sintético** pequeño (p. ej. frames generados con numpy/`write_video`)
  cuyas dimensiones coincidan con `resolution`.
- **Casos:**
  - `render_obj_id_overlay(json, video)` → escribe el mp4 (existe, no vacío).
  - JSON con `mode="segmentation"` → `ValueError` (no escribe).
  - `excluded_classes=["green_floor"]` → la clase excluida no se dibuja (verificable
    p. ej. comprobando que no se invoca el dibujo para esa clase, o por inspección del
    flujo; a nivel funcional basta con que el mp4 se genera sin esa clase).
  - `draw_masks=True` sobre un JSON **sin** `rle` → aviso + mp4 solo con
    cajas/estela (no falla).
  - warm-up: un detection con `obj_id=-1` se dibuja con etiqueta solo de nombre.

### 5.2 Parte B — sobre un **JSON de tracking real**

- Tomar un JSON de tracking ya generado (p. ej. de las corridas de `batch_inference`
  en el pod) y correr `render_obj_id_overlay` → mp4 con cajas/etiquetas/estelas; si el
  JSON se generó con `include_masks=True`, probar `draw_masks=True`. (Puede correrse
  **local**, no requiere GPU, solo el JSON + el video.)

### 5.3 Calidad

- `ruff check .` y `black .` sin hallazgos.
- Importabilidad: `from src.core.track_overlay import render_obj_id_overlay`.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Post-pase desacoplado | §3.2, §3.7 | driver JSON+video → mp4, sin re-inferir |
| AC-2 Validación de modo | §3.3 | `mode!="tracking"` → `ValueError` |
| AC-3 Caja + etiqueta por objeto | §3.5 (3,4) | `rectangle` + `nombre #id` |
| AC-4 Warm-up | §3.5 (4) | `obj_id=-1` → etiqueta solo nombre |
| AC-5 Trayectorias (ventana N) | §3.4, §3.5 (2) | rango `(f−N, f]` desde `tracks` |
| AC-6 Filtro de clases | §3.3, §3.5 | `excluded_classes` (default `green_floor`) |
| AC-7 Color por clase | §3.3, §3.6 | colores de `payload["config"]["classes"]` |
| AC-8 Máscara opcional | §3.5 (1) | `draw_masks` + `rle`; aviso si falta |
| AC-9 Salida no destructiva | §3.2, §3.7 | `<stem>_obj_id.mp4`, no pisa inferencia |
| AC-10 Sin SAM3 | §2, §3 | solo JSON + frames + cv2 |
| AC-11 Sin cambios colaterales | §3.9, §4 | módulo nuevo + config + CLAUDE.md |
| AC-12 Verificación | §5.1, §5.2 | sintético local + JSON real |

---

## 7. Riesgos y consideraciones

- **Alineación `frame_index` ↔ frames del video:** en tracking los frames son un
  prefijo contiguo `0..num_frames-1` y `iter_frames` los entrega en ese orden; el
  lookup por `frame_index` tolera huecos (frame sin registro → se escribe tal cual).
- **Coherencia de `bbox`:** la caja usa el `bbox` de la vista `frames` (derivado de la
  máscara), no el `xyxy` de ByteTrack; ambos describen el objeto, y `frames` es el que
  trae `obj_id` por detección. La trayectoria usa los `centroid` de `tracks`.
- **`draw_masks` sin `rle`:** no es error; se avisa y se degrada a solo cajas/estela
  (el JSON ligero es el caso común).
- **RGB/BGR:** se mantiene RGB en todo el flujo y se pasan los colores tal cual a
  `cv2`; documentar para que nadie meta una conversión que invierta colores.
- **Colores desde el snapshot del JSON:** se leen de `payload["config"]["classes"]`
  (auto-descriptivo); si una clase del JSON no tuviera color, se cae a un color por
  defecto o se reporta (decisión menor de implementación).
- **Alcance:** solo tracking; el renderizador unificado y el post-pase de seg con RLE
  quedan como evolución futura (idea 8 del banco).
