# Spec — Tracker BoT-SORT intercambiable (`botsort_tracker`)

- **Tarea atómica:** `botsort_tracker`
- **Paso de la metodología:** 2 (Especificación)
- **Proceso:** cuarta tarea de la secuencia que integra el pipeline YOLO + SAM3 a
  `src/`. Añade un **segundo tracker** junto al ByteTrack actual, montado sobre el
  mismo punto del tracking.
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** poder **elegir el tracker** (el ByteTrack actual o **BoT-SORT**, que
> añade compensación de movimiento de cámara — GMC),
> **para** obtener trayectorias e `obj_id` más robustos ante cámara en movimiento,
> sin reimplementar el bucle de tracking ni perder ByteTrack como opción.

---

## 2. Motivación (por qué)

- El tracking de `src` usa hoy **ByteTrack** (paquete `trackers` de Roboflow), fijo:
  un tracker por clase alimentado con las detecciones de cada frame. Funciona, pero
  no compensa el **movimiento de cámara**, frecuente en estos videos (cámara en mano
  / Meta Glasses), lo que fragmenta tracks.
- **BoT-SORT** es más robusto: incorpora **GMC** (Global Motion Compensation) y mejor
  asociación. Ya viene **integrado en `ultralytics`** (que el proyecto usa para YOLO),
  así que **no añade dependencia nueva**; su interfaz, eso sí, difiere de la del
  `ByteTrackTracker` de Roboflow y necesita un **adaptador**.
- El tracker es una pieza **intercambiable** por naturaleza: si se abstrae detrás de
  una interfaz común, ByteTrack y BoT-SORT conviven y se eligen por config/parámetro,
  reutilizando el resto del tracking (schema, overlay, `obj_id`).

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Definir una interfaz común de tracker**: un objeto con
  `.update(detections, frame) -> detections` que devuelve `tracker_id` y **preserva**
  el mapeo a la detección original (la `src` que `track_video` usa para recuperar la
  máscara). Es la interfaz que ya provee el `ByteTrackTracker` actual.
- **Estrenar el subpaquete `src/core/trackers/`** con:
  - `bytetrack`: la factory para `"bytetrack"` devuelve el `ByteTrackTracker` actual
    con los mismos parámetros de hoy (**ByteTrack byte-idéntico**, no-regresión).
  - `botsort`: **adaptador** de `ultralytics.trackers.BOTSORT` a la interfaz común
    (conversión de entrada/salida y recuperación del `src`).
  - un **factory** `get_tracker(name, frame_rate, config)` con validación de nombre.
- **Refactor de `track_video`** para construir los trackers vía la factory en vez de
  instanciar `ByteTrackTracker` directo. El resto del bucle
  (`mask→bbox→update→obj_id estable`, schema, overlay) **no cambia**.
- **Selección por parámetro + config**: `track_video` y `run_inference` ganan
  `tracker` (paralelo a `detector`); `None` → config `tracking.tracker` → default
  `"bytetrack"`. Validación temprana (un nombre inválido falla **antes** de cargar
  modelos).
- **Crecer el config de la fase**: selector `tracking.tracker` y una sección
  `botsort` (espejo de los parámetros de BoT-SORT de ultralytics, con **GMC activo**
  por defecto).
- **Un tracker por clase** (como hoy ByteTrack), aceptando que **GMC se calcula por
  clase**.
- **Test smoke** (script manual, pod) sobre el **mismo video** que el smoke anterior
  (`data/raw/17Abril/Cámaras/IMG_9871.MOV`), comparando estabilidad de `obj_id`
  BoT-SORT vs ByteTrack.

### 3.2 Fuera de alcance

- **ReID** (modelo de apariencia de BoT-SORT): desactivado; se difiere.
- **Tracker global multi-clase** (un solo tracker para todas las clases, más natural
  para GMC): se difiere; por ahora **un tracker por clase**.
- **Tuneo fino** de los parámetros de BoT-SORT: se dejan defaults sensatos.
- **Paridad de batch** (propagar `detector`/`tracker` en `run_batch`): es la tarea
  siguiente.
- **El detector**: ortogonal. Esta tarea no toca detectores; cualquier detector
  (`sam3_text` | `yolo_sam3`) combina con cualquier tracker.
- Cambios al esquema JSON, a `overlay`/`track_overlay` o a la lógica de
  `mask→bbox→obj_id`: el cambio es **solo en la capa de asociación**.
- La definición del **cómo técnico** (interfaz concreta, conversión BOTSORT,
  construcción de args, factory): corresponde al `plan.md`.

---

## 4. Comportamiento esperado (criterios de aceptación)

1. **Tracker intercambiable**: `track_video` construye sus trackers vía la factory y
   acepta `tracker` (nombre); el resto del bucle no cambia.
2. **No-regresión**: con `tracker="bytetrack"` (default), el resultado es **idéntico**
   al actual (mismos `obj_id`, JSON, overlay).
3. **BoT-SORT operativo**: con `tracker="botsort"`, `track_video`/`run_inference`
   devuelven la **misma forma** de resultado (`{"json","video","index"}`), con
   `obj_id` estable; BoT-SORT aplica **GMC**.
4. **Interfaz común respetada**: el adaptador BoT-SORT devuelve `tracker_id` y
   **preserva el `src`**, de modo que cada track se asocia de vuelta a su máscara
   (igual que ByteTrack).
5. **Selección por config y validación**: `tracking.tracker` elige el tracker cuando
   no se pasa parámetro (default `"bytetrack"`); un nombre desconocido lanza
   `ValueError` **antes** de cargar modelos.
6. **Config crecido**: existe `tracking.tracker` y la sección `botsort` (con GMC
   activo); ByteTrack sigue leyendo sus parámetros actuales sin cambios.
7. **Ortogonalidad detector×tracker**: cualquier combinación
   (`sam3_text`/`yolo_sam3` × `bytetrack`/`botsort`) corre por la misma vía.
8. **Verificación A/B (pod)**: el smoke corre `track_video(detector="yolo_sam3",
   tracker="botsort")` sobre `IMG_9871.MOV` y produce JSON con `tracks` + mp4; los
   `obj_id` son al menos tan estables como con ByteTrack (menos fragmentación
   esperada por GMC).

---

## 5. Dependencias y relación con otras tareas

- **Depende de:** `ultralytics` (ya instalado, trae BoT-SORT + GMC); el tracking
  existente (`track_video`, schema, overlay); `detector_strategy` (de donde viene el
  patrón de selección por nombre + config + validación temprana).
- **Habilita:** la **paridad de batch** (tarea siguiente), que propagará tanto
  `detector` como `tracker` en `run_batch`.
- **No** depende del detector concreto: detector y tracker son ortogonales.
