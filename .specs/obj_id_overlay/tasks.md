# Tasks — Overlay por `obj_id` (`obj_id_overlay`)

- **Tarea atómica:** `obj_id_overlay`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** el post-pase **no requiere SAM3**, así que toda la
> verificación funcional (Fase D/E) corre **localmente sin GPU**: la mayoría con
> JSON+video **sintéticos**, y una corrida adicional sobre un **JSON de tracking real**
> (también local). El lint estático corre en cualquier entorno.

---

## Fase A — Esqueleto del driver (mp4 de punta a punta)

- [x] **T1 — Crear `src/core/track_overlay.py` con el driver y la lectura del JSON**
  - Módulo nuevo con `render_obj_id_overlay(json_path, video_path=None,
    output_path=None, draw_masks=False, trajectory_window=None,
    excluded_classes=None) -> Path`. Lee el JSON, resuelve `video` (cabecera u
    override), `fps`, colores de `payload["config"]["classes"]`, y los defaults de
    config (`trajectory_window`, `overlay_excluded_classes`). Imports perezosos.
  - **Verificación:** `from src.core.track_overlay import render_obj_id_overlay`.
  - **Plan:** §3.2, §3.3. **Spec:** AC-1, AC-7.

- [x] **T2 — Recorrido en streaming y escritura del mp4 (esqueleto)**
  - `with open_video_writer(output_path, fps)` + `iter_frames(video)`: recorre y
    escribe cada frame (aún sin trazos o con dibujo mínimo). `output_path` default
    `<json_stem>_obj_id.mp4` junto al JSON. Frames sin registro → tal cual.
  - **Verificación:** sobre JSON+video sintéticos se **escribe un mp4 no vacío**.
  - **Plan:** §3.4, §3.7. **Spec:** AC-1, AC-9.

---

## Fase B — Validación de entrada

- [x] **T3 — Validar `mode="tracking"`**
  - Si `payload["mode"] != "tracking"` → `ValueError` explícito **antes** de abrir el
    writer o el video (no escribe salida).
  - **Verificación:** un JSON `mode="segmentation"` levanta `ValueError` y no crea mp4.
  - **Plan:** §3.3. **Spec:** AC-2.

---

## Fase C — Primitivas de dibujo (cada una verificable)

- [x] **T4 — Cajas por objeto (color por clase)**
  - Por frame, `cv2.rectangle` con el `bbox` de la vista `frames` (`(x,y)`–`(x+w,y+h)`),
    color de la clase. RGB de punta a punta (sin conversión BGR).
  - **Verificación:** el mp4 sintético muestra cajas en las posiciones esperadas.
  - **Plan:** §3.5 (3), §3.6. **Spec:** AC-3, AC-7.

- [x] **T5 — Etiquetas `nombre #id` + warm-up**
  - Texto `nombre #id` encima de la esquina sup-izq, sobre rectángulo de fondo del
    color de clase; texto negro/blanco por luminancia. `obj_id=-1` → etiqueta solo de
    nombre. Grosor/escala derivados de la resolución.
  - **Verificación:** etiquetas legibles; un detection `obj_id=-1` sale sin `#id`.
  - **Plan:** §3.5 (4). **Spec:** AC-3, AC-4.

- [x] **T6 — Trayectorias por `obj_id` (ventana N)**
  - Precomputar por `obj_id` los `(frame_index, centroid)` de `payload["tracks"]`; en el
    frame `f` dibujar `cv2.polylines` del tramo en `(f−N, f]`, color de clase, centroides
    a `int`.
  - **Verificación:** la estela aparece y se acota a la ventana N.
  - **Plan:** §3.4, §3.5 (2). **Spec:** AC-5.

- [x] **T7 — Relleno de máscara opcional (`draw_masks`)**
  - Con `draw_masks` y `rle` presente: `decode_rle` + mezcla alpha
    (`visualization.overlay_alpha`), color de clase, **debajo** de trazos/cajas. Sin
    `rle` pidiéndolo: **avisar una vez** y degradar a cajas/estela.
  - **Verificación:** con `rle` se pinta la máscara; sin `rle` avisa y no falla.
  - **Plan:** §3.5 (1). **Spec:** AC-8.

- [x] **T8 — Filtro de clases excluidas**
  - Las clases en `excluded_classes` (default `green_floor`) se saltan por completo
    (sin caja/etiqueta/estela/máscara).
  - **Verificación:** la clase excluida no aparece dibujada en el mp4 sintético.
  - **Plan:** §3.3, §3.5. **Spec:** AC-6.

---

## Fase D — Configuración

- [x] **T9 — Claves de config nuevas (con defaults) + override por parámetro**
  - Añadir a `configs/00_testing_config.json` bajo `visualization`:
    `trajectory_window` y `overlay_excluded_classes` (y opcional escala de
    fuente/grosor). El parámetro de función **sobreescribe** la config.
  - **Verificación:** sin parámetro se usan los valores de config; con parámetro,
    el valor pasado.
  - **Plan:** §3.8, §4. **Spec:** AC-5, AC-6.

---

## Fase E — Test, calidad, anti-alcance, doc y commit

- [x] **T10 — Crear `testing/test_obj_id_overlay.py`**
  - **Parte A (local, sin GPU, sintético):** mini-JSON de tracking (cabecera
    `mode="tracking"`, `config.classes` con color incl. `green_floor`, `frames` con un
    `obj_id=-1`, `tracks` con centroides) + video sintético del mismo `resolution`.
    Casos: escribe mp4; `mode="segmentation"` → `ValueError`; `excluded_classes` omite
    la clase; `draw_masks=True` sin `rle` → aviso + cajas/estela; warm-up sin `#id`.
  - **Parte B (JSON real, local):** correr sobre un JSON de tracking real (p. ej. de
    `batch_inference`); si trae `rle`, probar `draw_masks=True`.
  - **Verificación:** el script existe; la Parte A pasa en local.
  - **Plan:** §5.1, §5.2. **Spec:** AC-12.

- [x] **T11 — Ejecutar el test en local**
  - Correr `test_obj_id_overlay.py` (Parte A; Parte B si hay un JSON real a mano).
  - **Verificación:** todas las aserciones pasan sin GPU.
  - **Plan:** §5.1, §5.2. **Spec:** AC-2, AC-3, AC-4, AC-5, AC-6, AC-8, AC-9, AC-12.

- [x] **T12 — Anti-alcance (no-regresión)**
  - Sin tocar `overlay.py`, `track_video`, `inference_schema` (incl. `SCHEMA_VERSION`),
    `pipeline.py`, `segmentation`, ByteTrack. `requirements.txt` sin cambios.
  - **Verificación:** `git diff` limitado a `track_overlay.py` (nuevo), el test, el
    config y `CLAUDE.md`; `SCHEMA_VERSION` intacto.
  - **Plan:** §3.9, §4. **Spec:** AC-10, AC-11.

- [x] **T13 — Calidad e importabilidad**
  - `ruff check .` y `black .` sin hallazgos; `from src.core.track_overlay import
    render_obj_id_overlay` OK.
  - **Verificación:** lint limpio; import correcto.
  - **Plan:** §5.3. **Spec:** AC-11.

- [x] **T14 — Documentación de cierre (`CLAUDE.md` + docstring)**
  - Documentar el post-pase `track_overlay.py::render_obj_id_overlay` en la
    arquitectura de `CLAUDE.md`.
  - **Verificación:** `CLAUDE.md` refleja el post-pase de overlay por `obj_id`.
  - **Plan:** §4. **Spec:** AC-1.

- [ ] **T15 — Commit (requiere confirmación)**
  - Commitear `src/core/track_overlay.py`, el test, el config y `CLAUDE.md`. **El
    agente NO commitea por iniciativa propia:** pregunta y espera confirmación
    (constitución §11). Conventional Commits en inglés, scope `obj_id_overlay`.
  - **Verificación:** tras tu confirmación, el commit existe.
  - **Plan:** —. **Spec:** —

---

## Trazabilidad resumida

| Tarea                                | Plan        | Spec (AC)                              |
| ------------------------------------ | ----------- | -------------------------------------- |
| T1 driver + lectura JSON             | §3.2, §3.3  | AC-1, AC-7                             |
| T2 streaming + escritura mp4         | §3.4, §3.7  | AC-1, AC-9                             |
| T3 validar `mode="tracking"`         | §3.3        | AC-2                                   |
| T4 cajas por clase                   | §3.5, §3.6  | AC-3, AC-7                             |
| T5 etiquetas `nombre #id` + warm-up  | §3.5        | AC-3, AC-4                             |
| T6 trayectorias (ventana N)          | §3.4, §3.5  | AC-5                                   |
| T7 máscara opcional                  | §3.5        | AC-8                                   |
| T8 filtro de clases                  | §3.3, §3.5  | AC-6                                   |
| T9 config nueva + override           | §3.8, §4    | AC-5, AC-6                             |
| T10 crear test (A + B)               | §5.1, §5.2  | AC-12                                  |
| T11 ejecutar test (local)            | §5.1, §5.2  | AC-2,3,4,5,6,8,9,12                    |
| T12 anti-alcance                     | §3.9, §4    | AC-10, AC-11                           |
| T13 calidad/import                   | §5.3        | AC-11                                  |
| T14 documentación de cierre          | §4          | AC-1                                   |
| T15 commit (confirmación)            | —           | —                                      |

---

> **Fuera de esta tarea (futuro):** renderizador unificado seg+tracking y post-pase de
> segmentación con RLE (idea 8 del banco); hacer configurable qué clases se **trackean**
> (lado `track_video`).
