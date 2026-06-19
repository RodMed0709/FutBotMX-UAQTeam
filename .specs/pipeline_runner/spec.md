# Spec — Primer pipeline ejecutable (`pipeline_runner`)

- **Tarea atómica:** `pipeline_runner`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** una función que orqueste de punta a punta `video → frames →
> detección por clase → overlay → video anotado`, generando además un JSON de
> detecciones,
> **para** tener el **primer pipeline ejecutable** del MVP (SAM3-only, por-frame),
> que se corre como "el pipeline" y no como un spike de notebook.

---

## 2. Motivación (por qué)

- Ya existen todas las piezas (`extract_frames`, `load_sam3`,
  `detect_classes_in_frame`, `overlay_detections`, `write_video`) y la
  configuración de clases; falta **ensamblarlas** en un único punto de entrada.
- El MVP necesita un artefacto que produzca el entregable concreto: el **mp4
  anotado** ("video original → proceso → video anotado") y un **JSON** con las
  detecciones para análisis posterior.
- Centralizar la orquestación permite **cargar el modelo una sola vez**, fijar las
  convenciones de salida (`outputs/`) y dejar preparada la extensión a tracking.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Definir en un nuevo módulo `src/core/pipeline.py` una función
  **`run_pipeline(...)`** que:
  - extrae frames del video (modo cuota o completo),
  - por cada frame detecta todas las clases y compone el overlay,
  - escribe el **mp4 anotado** y un **JSON de detecciones**,
  - carga el modelo SAM3 **una sola vez** y lo reutiliza.
- Auto-nombrar las salidas bajo `working_dirs.outputs_dir`.
- Dejar un parámetro `mode` preparado para el futuro tracking (solo `per_frame`
  implementado) y un parámetro `all_frames` (cuota vs completo).
- Exportar `run_pipeline` desde `src/core/__init__.py`.
- Un script para dispararlo (`testing/test_pipeline.py`).

### 3.2 Fuera de alcance

- **Implementación del tracking** (tarea 5); aquí solo queda el parámetro `mode`.
- **fps real de la fuente** para el modo completo "100% real": **no** se cablea
  aquí (requiere tocar `extract_frames`); es la **tarea siguiente**.
- **Export a COCO** y máscaras dentro del JSON.
- **CLI elaborado** (la constitución pide simplicidad): basta la función + un
  script.
- El **cómo técnico** (firma y tipos exactos, estructura concreta del JSON,
  composición de rutas, `np.stack`): corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- **Entrada:** un **video** por path (relativo a `PROJECT_ROOT` o absoluto; se
  delega en `extract_frames`), y parámetros opcionales (`output_path`,
  `all_frames`, `mode`).
- **Orquestación:**
  1. Carga el `Sam3Bundle` con `load_sam3()` (una vez).
  2. `extract_frames(video, all_frames=...)`.
  3. Por cada frame: `detect_classes_in_frame(frame, bundle=...)` →
     `overlay_detections(frame, dets)`; acumula los frames compuestos.
  4. `write_video(frames_compuestos, ruta_mp4, fps=...)`.
  5. Escribe el **JSON de detecciones**.
- **Salida 1 — mp4 anotado:** video con el overlay multi-clase por frame.
- **Salida 2 — JSON de detecciones (sin máscaras):** metadatos del run (video,
  `mode`, `all_frames`, fps, nº de frames, clases) y, por frame y por clase, una
  lista de `{obj_id, score}` (la cuenta se deriva del largo).
- **Modos:**
  - `all_frames=False` (**cuota**) → testeo / generación de frames para
    fine-tuning; fps de config.
  - `all_frames=True` (**completo**) → uso real; **por ahora** también fps de
    config (el fps real de la fuente llega en la tarea siguiente).
  - `mode="per_frame"` es el único implementado.
- **Ubicación de salida:** bajo `working_dirs.outputs_dir`, auto-nombrado
  `outputs/<stem>_annotated.mp4` y `outputs/<stem>_detections.json`; un
  `output_path` explícito sobreescribe el del mp4.
- **Retorno:** las rutas generadas (mp4 y JSON).

---

## 5. Criterios de aceptación

1. **AC-1 — Módulo y función:** existe `src/core/pipeline.py` con `run_pipeline`,
   exportada desde `src/core/__init__.py`.
2. **AC-2 — Orquestación end-to-end:** dado un video, produce un **mp4 anotado**
   con el overlay multi-clase.
3. **AC-3 — JSON de detecciones:** produce un JSON con metadatos del run y, por
   frame y clase, `{obj_id, score}` (sin máscaras).
4. **AC-4 — Modelo una sola vez:** el `Sam3Bundle` se carga una vez y se reutiliza
   en todos los frames.
5. **AC-5 — Modos cuota/completo:** `all_frames` selecciona cuota (default) o
   completo; ambos producen salida.
6. **AC-6 — `mode` preparado:** existe el parámetro `mode` con `per_frame` como
   único valor implementado.
7. **AC-7 — Salidas en `outputs/`:** mp4 y JSON se escriben bajo
   `working_dirs.outputs_dir`, auto-nombrados; `output_path` override funciona.
8. **AC-8 — Entrada flexible:** acepta path de video relativo a `PROJECT_ROOT` o
   absoluto.
9. **AC-9 — Validación:** un script (`testing/test_pipeline.py`) dispara el
   pipeline sobre un video real; **se ejecuta en RunPod (GPU)** y confirma que se
   generan mp4 y JSON.

---

## 6. Supuestos y notas

- **Dependencias:** integra **todas** las piezas previas (`frame_extraction`,
  `sam3_loader`, `classes_config`, `text_segmentation`, `segmentation_overlay`,
  `video_writer`). Es la tarea integradora del MVP por-frame.
- **Tarea de seguimiento:** el modo completo "100% real" se cierra en la
  **siguiente tarea**, que cableará el **fps real de la fuente** (hoy el modo
  completo usa el fps de config como placeholder).
- **Rendimiento:** corre el modelo (varias inferencias por frame); en **CPU es
  inviable**. La ejecución real va en **RunPod/GPU**; el agente crea el código y
  el script y solo verifica lo ligero (lint, importabilidad), sin ejecutar el
  pipeline.
- **JSON sin máscaras:** se guardan `obj_id` y `score` por detección (las máscaras
  son pesadas y van implícitas en el mp4); bastan para análisis/depuración.
- Esta especificación **no** define el *cómo* técnico (firma y tipos exactos,
  esquema concreto del JSON, composición de rutas, manejo de `np.stack`, ni el
  formato del script); todo ello corresponde al `plan.md` de esta misma carpeta.
