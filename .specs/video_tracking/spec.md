# Spec — Tracking por detección per-frame + ByteTrack (`video_tracking`)

- **Tarea atómica:** `video_tracking`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** MVP SAM3-only (ver
  `.specs/drafts/mvp_sam3_only_roadmap.md`, tarea 5). **Depende de:** el pipeline
  per-frame existente (`detect_classes_in_frame`) y `sam3_loader`.

---

## 1. Requisito (historia de usuario)

> **Como** persona que construye el pipeline de análisis de fútbol robótico,
> **quiero** un módulo de tracking que asocie las detecciones per-frame de SAM3 en
> **trayectorias con `obj_id` consistentes** a lo largo de un video, usando
> **ByteTrack** como capa de asociación,
> **para** obtener identidades estables (robot #1 sigue siendo #1) reutilizando el
> pipeline per-frame que ya existe y dejando el tracker **reutilizable** por el
> futuro pipeline `yolo→sam3→bytetrack`.

---

## 2. Motivación (por qué)

- El pipeline per-frame actual segmenta cada frame de forma independiente: sus
  `obj_id` **no** persisten entre frames. El tracking aporta la **identidad
  temporal** (trayectorias, conteo consistente, asociación frame a frame).
- **Por qué ByteTrack y no la sesión SAM3-video:** la constitución (§1) define el
  tracking de ambos pipelines oficiales como **ByteTrack**. La asociación por
  detección (a) **reutiliza** el per-frame existente en vez de construir un camino de
  inferencia nuevo; (b) **no acumula** features de todo el clip → soporta **video
  completo** sin el tope de memoria de la sesión SAM3-video; (c) maneja **objetos
  que aparecen después** del primer frame; y (d) es **código reutilizable**: el
  futuro pipeline YOLO solo cambia el detector, no la capa de tracking.
- Diseñar la salida **agnóstica al tracker** (índice de tracks estándar) permite que
  `yolo→sam3→bytetrack` emita la misma forma y reutilice overlay/trayectorias.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Nuevo módulo **`src/core/tracking.py`** con una función pública de tracking que:
  - recorre el video **en streaming** (frame a frame),
  - por frame, obtiene detecciones reusando **`detect_classes_in_frame`**,
  - convierte cada máscara a **caja** y la asocia con **ByteTrack** (un tracker por
    clase), asignando `obj_id` **estables y globalmente únicos**,
  - produce, al vuelo, el frame con overlay y lo escribe a un **mp4** (reusando
    `video_writer`),
  - retiene un **índice de tracks agnóstico** (`obj_id→clase`; por frame:
    centroide/caja/score, **sin máscara**) y lo persiste como **JSON**.
- **Utilidades derivadas:** trayectorias (centroides por `obj_id`).
- **Config:** parámetros de ByteTrack y `max_frames` opcional (§4.4).
- **Dependencia:** ByteTrack vía `trackers.ByteTrackTracker` + `supervision`
  (`sv.Detections`); **ambas ya en `requirements.txt`** (`supervision.ByteTrack`
  quedó deprecado en 0.28 → se usa el paquete `trackers`).
- **Script de prueba manual** `testing/test_tracking.py` (estilo standalone).

### 3.2 Fuera de alcance

- **No** se modifica `detect_classes_in_frame` ni el modo per-frame.
- **No** se conecta el `mode="tracking"` de `pipeline.py` (tarea aparte); este módulo
  expone la función lista para enchufar.
- **No** hay evaluación cuantitativa del tracking (no existe GT de tracking; sería su
  propia tarea con clips densos + IDs). Solo **verificación cualitativa**.
- **No** se retienen en memoria las máscaras de todos los frames (inviable en video
  completo); se usan al vuelo y se descartan.
- El **cómo técnico** (firmas/tipos exactos, helper de streaming, estructura del
  índice, manejo de errores, detalle del test) corresponde al `plan.md`.

---

## 4. Comportamiento esperado

### 4.1 Entrada y recorrido

- **Entrada:** un video (path **relativo a `PROJECT_ROOT` o absoluto**), como
  `extract_frames`.
- **Recorrido:** frames **consecutivos a fps nativo**, en **streaming** (uno a la
  vez), **todo el video** por defecto; `max_frames` opcional para pruebas.

### 4.2 Detección, asociación e identidad

- Por frame se obtienen detecciones con **`detect_classes_in_frame`** (todas las
  clases del config por defecto; conjunto **configurable** vía `classes`).
- Cada máscara se convierte a **caja** (`(x, y, w, h)`) + `score`.
- La asociación usa **ByteTrack, un tracker por clase**; el `obj_id` resultante es
  **estable entre frames** y **globalmente único** (namespacing por clase). La
  **clase** de cada track se conoce **por construcción** (el tracker que lo produjo).
- **Objetos nuevos** (aparecen después del inicio) se trackean: el per-frame los
  detecta y ByteTrack abre un track.
- **Detección ausente** en un frame: ByteTrack mantiene el track "perdido" hasta
  `track_buffer` frames; al reaparecer, re-asocia o abre track nuevo según ByteTrack.

### 4.3 Salidas

- **mp4** con overlay (escrito **incrementalmente**), a **fps real** de la fuente
  (`get_video_fps`), bajo `outputs/` (git-ignored).
- **Índice de tracks agnóstico** (retenido / devuelto): `obj_id→clase` y, por frame,
  **centroide/caja/score** (sin máscara).
- **JSON** persistido con ese índice (análogo al JSON del pipeline per-frame, que
  también omite máscaras).
- **Trayectorias:** utilidad derivada que calcula centroides por `obj_id` en el
  tiempo a partir del índice.

### 4.4 Configuración

- **Parámetros de ByteTrack** (`track_thresh`, `track_buffer`, `match_thresh`) con
  valores por defecto, **configurables** desde el config global.
- **`max_frames`** opcional (clip de prueba). Nada hardcodeado.

### 4.5 Acople con el pipeline actual y futuros

- La vista por-frame `dict[clase, list[Detection]]` (con `obj_id` ahora estable)
  alimenta **`overlay_detections`** y **`video_writer`** sin cambios.
- El **índice de tracks agnóstico** es el contrato reutilizable: un futuro
  `yolo→sam3→bytetrack` produce la misma forma cambiando solo el detector.

---

## 5. Criterios de aceptación

1. **AC-1 — Módulo presente:** existe `src/core/tracking.py` con la función pública
   de tracking, importable como paquete editable.
2. **AC-2 — Reúso del per-frame:** usa `detect_classes_in_frame` por frame; **no**
   modifica el modo per-frame ni `segment_with_text`.
3. **AC-3 — Identidad estable:** un mismo objeto conserva su `obj_id` a lo largo de
   los frames donde ByteTrack lo asocia; los `obj_id` son **globalmente únicos**.
4. **AC-4 — Clase por construcción:** cada track tiene su clase correcta (tracker por
   clase); no hay cruce de clases.
5. **AC-5 — Genérico sobre clases:** por defecto trackea todas las clases del config;
   acepta un subconjunto vía parámetro, sin tocar código.
6. **AC-6 — Video completo / streaming:** procesa el video frame a frame sin retener
   todas las máscaras ni todos los frames en memoria; `max_frames` acota si se indica.
7. **AC-7 — Objetos nuevos:** un objeto que aparece después del primer frame recibe
   un track.
8. **AC-8 — Salidas:** genera mp4 con overlay (a fps real), el índice de tracks
   agnóstico y su JSON (sin máscaras).
9. **AC-9 — Trayectorias:** la utilidad derivada produce centroides por `obj_id` en
   el tiempo.
10. **AC-10 — Config:** parámetros de ByteTrack y `max_frames` se leen del config;
    sin valores hardcodeados.
11. **AC-11 — Acople:** la vista por-frame funciona con `overlay_detections` y
    `video_writer` sin modificarlos.
12. **AC-12 — Dependencia declarada:** la librería de ByteTrack queda en
    `requirements.txt`.
13. **AC-13 — Verificación:** `testing/test_tracking.py` corre el tracking sobre un
    clip corto y comprueba identidad estable, clases, objetos nuevos y salidas
    (cualitativo; sin GT). Ejecutado en GPU/pod.

---

## 6. Supuestos y notas

- **ByteTrack como tracker (no SAM3-video):** decisión explícita por reuso del
  per-frame, soporte de video completo, manejo de objetos nuevos y alineación con la
  constitución (§1). El tracking SAM3-video del notebook 03 queda **descartado** para
  este MVP (era un atajo de spike).
- **Asociación por caja, no por máscara:** ByteTrack asocia cajas (IoU + Kalman). Para
  objetos rápidos y pequeños (el balón) podría perder/intercambiar IDs más que una
  memoria de apariencia; **riesgo asumido** porque este MVP **no evalúa** tracking.
- **`obj_id` cambia de semántica vs per-frame:** en per-frame el `obj_id` de
  `Detection` no es estable; en tracking **sí** lo es. Debe documentarse en el código.
- **Streaming obligatorio:** mantener todas las máscaras/frames de un video completo
  en RAM es inviable; por eso se procesa y descarta al vuelo, reteniendo solo el
  índice ligero.
- **Sin evaluación, por diseño:** no hay GT de tracking y no se crea en este MVP; la
  evaluación cuantitativa (HOTA/IDF1 + GT denso con IDs) es trabajo futuro.
- **Costo de cómputo:** SAM3 per-frame es lo caro (varias sesiones por frame);
  ByteTrack es CPU liviano. Video completo es lento pero acotado en memoria.
- Esta especificación **no** define el *cómo* técnico (firmas, helper de streaming,
  estructura exacta del índice, librería final de ByteTrack, manejo de errores ni el
  detalle del test); todo ello corresponde al `plan.md`.
