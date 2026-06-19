# Spec — Overlay por `obj_id` para hacer visible el tracking (`obj_id_overlay`)

- **Tarea atómica:** `obj_id_overlay`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Roadmap de siguientes pasos (tarea 1). Origen:
  follow-up (b) de `video_tracking`. **Depende de:** `inference_schema` (la vista
  `frames` + `tracks` del JSON de tracking) y `video_tracking` (que produce ese JSON).
- **Habilita:** `end_to_end_demo` (tarea 2 del roadmap), que mostrará el tracking ya
  visible.

---

## 1. Requisito (historia de usuario)

> **Como** persona que revisa los resultados de tracking,
> **quiero** un mp4 donde **se vea la identidad de cada objeto** (su caja, su nombre y
> su recorrido) en vez de un relleno de máscaras por clase,
> **para** confirmar de un vistazo que el tracking sigue correctamente a cada robot y
> al balón a lo largo del video, sin tener que leer el JSON.

---

## 2. Motivación (por qué)

- **El mp4 de tracking hoy parece segmentación.** `track_video` rinde el video
  rellenando máscaras **por clase**; dos robots quedan como manchas del mismo color, y
  la identidad estable (`obj_id`) — que es justo lo que aporta el tracking — **no se
  ve**. La calidad del tracking solo es visible leyendo el JSON.
- **El dato ya tiene todo lo necesario.** El JSON de tracking trae, por frame, las
  detecciones con `obj_id`/`bbox`/`centroid`/clase (vista `frames`) y, por `obj_id`,
  toda su trayectoria de centroides (vista `tracks`). Falta **dibujarlo**.
- **Desacoplar de la inferencia.** Re-correr SAM3 solo para cambiar la visualización
  es caro. Un **post-pase** que lee el JSON + el video y reescribe un mp4 permite
  iterar la visualización **sin volver a inferir**.
- **La identidad se ve por forma, no por color.** Se conserva el **color por clase**
  (consistente con el resto del proyecto); lo que hace visible la identidad es **una
  caja por objeto + su trayectoria**, no un color por `obj_id`.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Un post-pase** que toma un **JSON de tracking** + el **video fuente** y escribe un
  **mp4 nuevo** con el overlay por objeto. No re-infiere ni modifica `track_video`.
- **Validación de entrada:** el JSON debe ser de `mode="tracking"`; un JSON de
  segmentación (sin `tracks`, `obj_id` inestable) se rechaza con error explícito.
- **Dibujo por frame** (color **por clase**, reusando `classes[].color`):
  - una **caja** por objeto detectado en ese frame;
  - una **etiqueta `nombre #id`** (p. ej. `robot #3`) para distinguir varios objetos
    de la misma clase;
  - en **warm-up** (`obj_id = -1`, sin identidad estable aún) se dibuja la caja con
    etiqueta solo de nombre (sin `#id`), no se omite.
- **Trayectorias:** por cada `obj_id`, una **polilínea de sus centroides** (de la
  vista `tracks`), del color de su clase, limitada a una **ventana deslizante de los
  últimos N frames** (N configurable) para no saturar videos largos.
- **Filtro de clases configurable** (default excluye `green_floor`): las clases
  excluidas **no** se dibujan (ni caja, ni etiqueta, ni trayectoria, ni máscara).
- **Relleno de máscara opcional** (toggle, default **OFF**): con el toggle activo y
  si el JSON trae `rle`, además del trazo se pinta la máscara (color de clase); sin
  `rle` se ignora con aviso y se dibujan solo cajas/estela.
- **Salida:** mp4 ubicado junto al JSON con un nombre **distinto** al mp4 de
  inferencia (no lo pisa); `fps` y resolución tomados de la cabecera/video fuente
  (sin reescalar).
- **Verificación:** script en `testing/` con parte local (JSON+video sintéticos, **sin
  SAM3**) y parte sobre un JSON de tracking real.

### 3.2 Fuera de alcance

- **No** se modifica `overlay_detections` (el overlay **en vivo** por clase, usado por
  `run_pipeline` y `track_video`), que queda intacto.
- **No** se toca `track_video`, `inference_schema`, `pipeline.py` ni el esquema del
  JSON (`SCHEMA_VERSION`).
- **No** se construye el **renderizador unificado seg+tracking** ni el **post-pase de
  segmentación con RLE** (idea 8 del banco): son evolución futura.
- **No** se hace configurable qué clases se **trackean** (eso es del lado de
  `track_video`, tarea aparte); aquí el filtro es **solo de dibujo**.
- **No** se añade color por `obj_id`: el color es **por clase**.
- El **cómo técnico** (nombre/firma del post-pase y su driver, claves de config
  exactas, formato del lookup por `frame_index`, grosores/tipografía, detalle del
  test) corresponde al `plan.md`.

---

## 4. Comportamiento esperado

### 4.1 Entrada y validación

- Entra la **ruta de un JSON de tracking** (y, por defecto, el video fuente que indica
  su cabecera; con opción de override de la ruta del video).
- Si el JSON no es de `mode="tracking"` → **error explícito** (sin escribir nada).

### 4.2 Render por frame

- Se recorre el video alineando cada frame con su registro por `frame_index` del JSON.
- Para cada objeto **no excluido** del frame: caja (color de clase) + etiqueta
  `nombre #id` (o solo `nombre` en warm-up).
- Para cada `obj_id` no excluido: su **trayectoria** (centroides de los últimos N
  frames), color de clase.
- Si el toggle de máscara está activo y hay `rle`: relleno de la máscara (color de
  clase) bajo los trazos.

### 4.3 Salida

- Un **mp4 nuevo** junto al JSON, con nombre distinto al de inferencia, mismo `fps` y
  resolución que la fuente. El JSON y el mp4 de inferencia originales **no** se
  modifican.

### 4.4 Independencia

- El post-pase **no** carga SAM3 ni re-infiere: solo lee JSON + frames del video y
  dibuja. Funciona con cualquier JSON de tracking, tenga o no `rle`.

---

## 5. Criterios de aceptación

1. **AC-1 — Post-pase desacoplado:** existe una función que, dado un JSON de tracking
   + el video, escribe un mp4 con el overlay, **sin** re-inferir ni tocar
   `track_video`.
2. **AC-2 — Validación de modo:** un JSON que no sea de `mode="tracking"` produce un
   error explícito y no escribe salida.
3. **AC-3 — Caja + etiqueta por objeto:** cada objeto no excluido se dibuja con caja
   (color de clase) y etiqueta `nombre #id`.
4. **AC-4 — Warm-up:** las detecciones con `obj_id = -1` se dibujan con etiqueta solo
   de nombre (sin `#id`).
5. **AC-5 — Trayectorias:** cada `obj_id` no excluido lleva su polilínea de centroides
   (color de clase) limitada a los últimos N frames (N configurable).
6. **AC-6 — Filtro de clases:** las clases excluidas (default `green_floor`) no se
   dibujan en absoluto.
7. **AC-7 — Color por clase:** todo el dibujo usa `classes[].color`; no hay color por
   `obj_id`.
8. **AC-8 — Máscara opcional:** con el toggle activo y `rle` presente, se rellena la
   máscara (color de clase); sin `rle`, se avisa y se omite (solo cajas/estela). Por
   defecto el toggle está OFF.
9. **AC-9 — Salida no destructiva:** el mp4 se escribe junto al JSON con nombre
   distinto al de inferencia; el JSON y el mp4 originales no se alteran. `fps`/
   resolución de la fuente.
10. **AC-10 — Sin SAM3:** el post-pase no carga el modelo ni re-infiere; funciona con
    JSON con o sin `rle`.
11. **AC-11 — Sin cambios colaterales:** no se modifica `overlay_detections`,
    `track_video`, `inference_schema`/`SCHEMA_VERSION` ni `pipeline.py`.
12. **AC-12 — Verificación:** un script en `testing/` valida (local, sin SAM3) la
    escritura del mp4 sobre JSON+video sintéticos, el rechazo de `mode≠"tracking"` y
    el filtro de clases; y corre sobre un JSON de tracking real.

---

## 6. Supuestos y notas

- **Color por clase, identidad por forma:** decisión explícita; reusa `classes[].color`
  y muestra identidad con caja + etiqueta `#id` + trayectoria.
- **Post-pase, no en vivo:** lee el JSON + el video; el overlay en vivo (`track_video`)
  sigue existiendo aparte y no se toca.
- **Máscara como extra opcional:** las máscaras aportan forma, no identidad; por eso el
  relleno es opt-in y solo cuando el JSON las trae (`include_masks=True` en su día).
- **Filtro solo de dibujo:** excluir `green_floor` aquí no cambia qué se trackea;
  hacer configurable el tracking es otra tarea.
- **Testeable sin GPU:** el post-pase no necesita SAM3, así que su comportamiento se
  valida localmente con datos sintéticos; solo la generación de un JSON real requirió
  el pod.
- Esta especificación **no** define el *cómo* técnico (nombre/firma, claves de config,
  lookup por `frame_index`, parámetros de dibujo, detalle del test); todo ello
  corresponde al `plan.md`.
