# Spec — Fachada única de inferencia (`unified_inference`)

- **Tarea atómica:** `unified_inference`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Pipeline de inferencia unificado + batch (roadmap
  del pipeline unificado, tarea 3). **Depende de:** `inference_schema` (tarea 1,
  completa) y `optional_render` (tarea 2, completa), cuyos JSON común y flag
  `render_video` esta fachada hereda sin tocar.
- **Habilita:** `batch_inference` (tarea 4), que orquestará N videos sobre esta
  única puerta de entrada y consumirá su valor de retorno unificado.

---

## 1. Requisito (historia de usuario)

> **Como** persona que corre el pipeline de análisis de fútbol robótico,
> **quiero** una **única puerta de entrada por video** que reciba el modo
> (`segmentation` o `tracking`) y resuelva por mí el muestreo de frames, el render y
> el esquema de salida,
> **para** no tener que conocer dos funciones distintas con firmas y formatos
> divergentes, y para que la futura capa de lotes itere sobre una sola interfaz
> estable.

---

## 2. Motivación (por qué)

- **Hoy hay dos caminos separados, no una fachada.** `run_pipeline(mode="per_frame")`
  vive en `pipeline.py` y `track_video` vive **fuera** del pipeline, en
  `tracking.py`. El `mode="tracking"` de `run_pipeline` es un **stub**
  (`NotImplementedError`). Quien quiere trackear debe saber que existe otra función,
  con otra firma.
- **La asimetría de muestreo está sin resolver de cara al usuario.** Segmentación usa
  **cuota equiespaciada** (cobertura para pruebas); tracking exige **prefijo
  contiguo** (continuidad temporal de ByteTrack). Hoy esa diferencia se expresa con
  parámetros distintos en funciones distintas (`all_frames` vs. `max_frames`); falta
  un lugar único que la resuelva por defecto según el modo y la valide.
- **El retorno es asimétrico.** `run_pipeline` devuelve `{"json", "video"}`;
  `track_video` devuelve `{"json", "video", "index"}`. Una capa de lotes que itere N
  videos no debería lidiar con dos formas.
- **Las tareas 1 y 2 ya dejaron lo común.** Ambas funciones emiten el **mismo JSON**
  (esquema común) y aceptan `include_masks` y `render_video` con idénticos defaults.
  Lo único que falta para cerrar la unificación es **la fachada delgada** que las
  envuelva tras una sola firma y cablee el modo tracking, hoy stub.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Una sola función fachada** en `src/core/` (la única puerta de entrada por video)
  con un parámetro `mode` de dos valores: **`"segmentation"`** y **`"tracking"`**.
- **Resolución del muestreo por modo, con ambos controles expuestos:**
  - `segmentation` → por defecto **cuota equiespaciada**; puede forzarse el video
    completo (control de "todos los frames" disponible).
  - `tracking` → por defecto **prefijo contiguo completo**; puede acotarse a un tope
    de N frames contiguos.
  - **Cuota equiespaciada + tracking es inválido** (ByteTrack necesita continuidad):
    se rechaza con error explícito.
- **Cableado del modo tracking.** La fachada hace que pedir tracking ejecute el
  camino de `track_video` (resuelve el stub histórico `mode="tracking"`).
- **Herencia de esquema (tarea 1) y render (tarea 2) sin cambios:** mismo JSON
  común, `include_masks` y `render_video` con sus mismos defaults (un solo video →
  render activado).
- **Valor de retorno unificado:** la fachada devuelve siempre la **misma forma**, con
  las claves del JSON, del mp4 (o ausencia de video) y del índice de tracks; en
  `segmentation` la entrada del índice indica "sin tracks", en `tracking` lleva el
  índice estable. Una sola forma para que `batch_inference` la consuma.
- **Reuso de los bloques compartidos:** la fachada **no** reimplementa el bucle de
  inferencia; delega en las implementaciones existentes (`run_pipeline` /
  `track_video`), que ya reutilizan `load_sam3`, `detect_classes_in_frame`,
  `overlay_detections`, `iter_frames`/`extract_frames`, `open_video_writer`.
- **Reuso del modelo:** la fachada acepta un `bundle` SAM3 ya cargado y lo pasa hacia
  abajo, para que la capa de lotes lo cargue una sola vez.
- **Verificación:** script manual en `testing/` que ejerza la fachada en ambos modos
  (smoke), incluida la validación del caso inválido (cuota + tracking). Las pruebas
  que invocan SAM3 corren en el pod.

### 3.2 Fuera de alcance

- **No** se construye la capa de lotes (`batch_inference`, tarea 4): esta tarea solo
  entrega la fachada de **un video**.
- **No** se borran ni se reescriben `run_pipeline` ni `track_video`: conservan su
  **firma pública actual** (no se rompen `testing/test_pipeline.py` /
  `test_tracking.py`); la fachada es **aditiva** y las envuelve.
- **No** se modifica el esquema del entregable (`inference_schema`), ni `overlay.py`,
  `video_writer.py`, `frame_extraction.py`, la detección, la asociación ByteTrack ni
  la lógica de muestreo de cada camino.
- **No** se unifica la lectura de configuración entre ambos caminos (hoy cada uno lee
  su sección): un helper compartido de config es **trabajo futuro**, no requisito de
  esta tarea.
- **No** se añade muestreo disperso a tracking ni continuidad a segmentación: la
  asimetría se **resuelve exponiéndola y validándola**, no eliminándola.
- **No** se introduce control de la fachada vía config/`.env`: el modo y los flags
  son **argumentos de función**, decididos por llamada.
- El **cómo técnico** (nombre y firma exactos de la fachada, mapeo de parámetros a
  cada implementación, mensaje y tipo de error del caso inválido, normalización del
  retorno, detalle del test) corresponde al `plan.md`.

---

## 4. Comportamiento esperado

### 4.1 Modo `segmentation` (por defecto)

- Es el **modo por defecto** (camino más barato y ya estable).
- Por defecto muestrea **cuota equiespaciada**; puede pedirse el **video completo**.
- Ejecuta el camino per-frame (detección por frame, `obj_id` **inestable**), escribe
  el JSON común y, si el render está activado, el mp4 anotado.
- En el retorno, la entrada del índice de tracks indica explícitamente **"sin
  tracks"** (no aplica en este modo).

### 4.2 Modo `tracking`

- Por defecto recorre el **prefijo contiguo completo** del video; puede **acotarse**
  a N frames contiguos.
- Ejecuta el camino de tracking (detección per-frame + ByteTrack por clase,
  `obj_id` **estable y globalmente único**), escribe el JSON común con la vista
  frame-indexed **y** el índice de tracks, y si el render está activado, el mp4.
- En el retorno, la entrada del índice lleva el **índice estable** `obj_id→Track`.

### 4.3 Resolución y validación del muestreo

- El **modo decide el muestreo por defecto** (cuota para segmentación, completo para
  tracking), pero **ambos controles se exponen** en la firma para forzar lo
  contrario cuando es válido (segmentación sobre video completo; tracking acotado).
- **Combinación inválida:** pedir **cuota equiespaciada con tracking** se rechaza con
  un **error explícito** (ByteTrack requiere continuidad temporal). El control que no
  aplica a un modo se ignora de forma documentada (p. ej. el tope contiguo no afecta
  a segmentación en cuota).

### 4.4 Herencia de render y máscaras

- `render_video` e `include_masks` son **ortogonales** al modo y conservan sus
  defaults de la tarea 2 / tarea 1 (un solo video → render activado; máscaras OFF).
- La fachada **no** reinterpreta estos flags: los pasa tal cual a la implementación
  correspondiente.

### 4.5 Retorno unificado

- La fachada devuelve **siempre la misma forma** con las tres dimensiones del
  resultado: ruta del JSON (siempre), del mp4 (o marca de "sin video"), e índice de
  tracks (índice real en tracking, marca de "sin tracks" en segmentación).
- Esa forma única es la que `batch_inference` agregará por video sin ramificar por
  modo.

---

## 5. Criterios de aceptación

1. **AC-1 — Puerta única:** existe una sola función fachada en `src/core/` que recibe
   `mode ∈ {"segmentation", "tracking"}` como única puerta de entrada por video.
2. **AC-2 — Modo por defecto:** sin especificar modo, la fachada corre
   `segmentation`.
3. **AC-3 — Tracking cableado:** pedir `mode="tracking"` ejecuta el camino de
   tracking (deja de ser un `NotImplementedError`).
4. **AC-4 — Muestreo por modo:** por defecto, `segmentation` muestrea cuota
   equiespaciada y `tracking` recorre el prefijo contiguo completo.
5. **AC-5 — Controles expuestos:** se puede forzar `segmentation` sobre el video
   completo y acotar `tracking` a N frames contiguos desde la firma de la fachada.
6. **AC-6 — Caso inválido:** pedir cuota equiespaciada con `tracking` produce un
   error explícito (no se ejecuta inferencia).
7. **AC-7 — Herencia de flags:** `render_video` e `include_masks` se comportan igual
   que en las tareas 2 y 1 y son ortogonales al modo.
8. **AC-8 — Retorno unificado:** la fachada devuelve la **misma forma** en ambos
   modos (JSON, video-o-ausencia, índice-o-ausencia); en `segmentation` la entrada
   del índice marca "sin tracks", en `tracking` lleva el índice estable.
9. **AC-9 — Fachada delgada:** no se reimplementa el bucle de inferencia; la fachada
   delega en `run_pipeline` / `track_video`, que conservan su firma pública.
10. **AC-10 — Reuso de modelo:** la fachada acepta un `bundle` SAM3 cargado y lo pasa
    hacia abajo (habilita cargar el modelo una sola vez en lotes).
11. **AC-11 — Sin cambios colaterales:** no se altera el esquema
    (`inference_schema`), ni `overlay.py`/`video_writer.py`/`frame_extraction.py`, ni
    la lógica de detección/tracking/muestreo de cada camino.
12. **AC-12 — Verificación:** un script en `testing/` ejerce la fachada en ambos
    modos y el caso inválido (smoke; las invocaciones a SAM3 corren en el pod).

---

## 6. Supuestos y notas

- **Fachada aditiva y delgada.** `run_pipeline` y `track_video` quedan como
  implementaciones internas detrás de la fachada; no se borran ni cambian su firma.
  El stub `mode="tracking"` de `run_pipeline` se resuelve **redirigiendo** al camino
  de tracking, no reimplementándolo.
- **Default por uso heredado.** Al ser un solo video, el render queda **activado** por
  defecto (criterio de la tarea 2); la capa de lotes (tarea 4) lo apagará.
- **La asimetría de muestreo se resuelve exponiéndola, no eliminándola.** El modo fija
  el muestreo por defecto y ambos controles quedan disponibles; la única combinación
  prohibida (cuota + tracking) se valida con error explícito.
- **Lectura de config sin unificar (trabajo futuro).** Cada camino sigue leyendo su
  sección de config; refactorizar eso a un helper compartido se anota como mejora
  posterior y **no** forma parte de esta tarea.
- **Importaciones perezosas** (torch/cv2/supervision/trackers) se mantienen dentro de
  funciones, como en el código actual.
- **Documentación de cierre:** al implementar se actualizará `CLAUDE.md` (arquitectura
  y el estado del stub `mode="tracking"`); el docstring de la fachada va en español,
  como el resto del código.
- Esta especificación **no** define el *cómo* técnico (nombre/firma exactos, mapeo de
  parámetros, tipo/mensaje del error, normalización del retorno, detalle del test);
  todo ello corresponde al `plan.md`.
