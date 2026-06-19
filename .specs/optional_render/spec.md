# Spec — Render de mp4 opcional vía flag (`optional_render`)

- **Tarea atómica:** `optional_render`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Pipeline de inferencia unificado + batch (roadmap
  del pipeline unificado, tarea 2). **Depende de:** `inference_schema` (tarea 1,
  completa), que ya hace del JSON el entregable y reubica el mp4 a la carpeta por
  video.
- **Habilita:** `unified_inference` (tarea 3) y `batch_inference` (tarea 4), que
  heredan este flag (la batch lo querrá apagado por defecto).

---

## 1. Requisito (historia de usuario)

> **Como** persona que corre el pipeline de análisis de fútbol robótico sobre uno o
> muchos videos,
> **quiero** poder **decidir por llamada si se genera el mp4 anotado**, manteniendo
> el JSON estructurado como salida siempre presente,
> **para** no desperdiciar CPU ni disco escribiendo video cuando solo me interesa el
> dato (lotes, evaluación), y seguir obteniéndolo sin esfuerzo cuando inspecciono un
> solo video.

---

## 2. Motivación (por qué)

- **El mp4 se escribe siempre hoy**, en ambos caminos de inferencia
  (`run_pipeline` y `track_video`): no hay forma de desactivarlo. Para lotes de 20+
  videos y para la exportación de predicciones de evaluación, renderizar video es
  **I/O y cómputo desperdiciados** (componer overlay y codificar mp4 por cada frame).
- **El entregable es el dato, el video es accesorio.** La tarea `inference_schema`
  ya estableció el JSON como producto real; falta el interruptor que haga del mp4 un
  **opt-in**. Esta tarea cierra ese desacople.
- **La capa batch lo necesita.** `batch_inference` querrá el render apagado por
  defecto; dejar el flag listo ahora (con el default correcto para uso de un solo
  video) evita rehacer las firmas después.
- **Ahorro real, no cosmético.** Apagar el render debe **saltarse el trabajo de
  visualización completo** (overlay + writer), no solo omitir la escritura final.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Añadir un flag de render** a `run_pipeline` (`pipeline.py`) y a `track_video`
  (`tracking.py`) que controla si se genera el mp4 anotado.
- **Default por uso, no por modo:** en la invocación de **un solo video** (las
  funciones actuales) el render está **activado por defecto**; ambos modos pueden
  renderizar o no. El render **nunca** depende del modo (`segmentation`/`tracking`).
- **Saltar todo el trabajo de visualización** cuando el render está apagado: no
  componer overlay, no abrir el escritor de video, no acumular frames compuestos en
  memoria. Solo detección/tracking + escritura del JSON.
- **JSON siempre presente**, en su ubicación canónica por video
  (`outputs/inference/<video_stem>/<video_stem>.json`), con o sin render.
- **Valor de retorno estable:** las funciones siguen devolviendo el mismo `dict`;
  cuando no se renderiza, la clave del mp4 se conserva con valor que indique
  "sin video" (no se omite la clave). `track_video` mantiene además su índice de
  tracks.
- **Verificación:** script manual en `testing/` que corra cada función con render
  ON y OFF y compruebe que (a) el JSON existe siempre, (b) el mp4 existe solo con
  render ON, (c) el retorno refleja correctamente la ausencia de video. Las pruebas
  que invocan SAM3 corren en el pod.

### 3.2 Fuera de alcance

- **No** se construye la fachada unificada (`unified_inference`) ni la capa de lotes
  (`batch_inference`); esta tarea solo introduce el flag en las dos funciones que ya
  existen.
- **No** se cambia el default a "apagado" en ningún flujo de lotes (aún no existe);
  el default de un solo video es **render activado**.
- **No** se modifica el esquema del entregable (`inference_schema`), ni
  `overlay.py`, `video_writer.py`, `frame_extraction.py`, la detección, la
  asociación ByteTrack ni el muestreo de frames.
- **No** se añade un campo nuevo al JSON para registrar si hubo render: el render no
  forma parte del contrato del dato y no altera `schema_version`.
- **No** se introduce control del flag vía config/`.env`: es **argumento de
  función**, decidido por llamada (igual criterio que `include_masks`).
- El **cómo técnico** (nombre y firma exactos del flag, su valor de retorno cuando
  no hay video, reestructura del bucle de streaming para no anidar el escritor,
  manejo de `output_path` cuando no se renderiza, detalle del test) corresponde al
  `plan.md`.

---

## 4. Comportamiento esperado

### 4.1 Render activado (default de un solo video)

- Comportamiento idéntico al actual: se compone el overlay por frame y se escribe el
  mp4 anotado **junto** al JSON, en la carpeta por video.
- El `dict` de retorno trae la ruta del mp4 generado (más el JSON; en tracking,
  también el índice de tracks).

### 4.2 Render desactivado

- **No** se realiza ningún trabajo de visualización: ni overlay, ni escritor de
  video, ni acumulación de frames compuestos.
- Se ejecuta solo la inferencia (detección per-frame o tracking) y se escribe el
  **JSON**, en su ubicación canónica.
- El `dict` de retorno conserva la misma forma; la entrada correspondiente al mp4
  indica explícitamente que **no se generó video**.

### 4.3 Independencia de modo y de `include_masks`

- El flag de render es **ortogonal** al modo (`segmentation`/`tracking`) y a
  `include_masks`: cualquier combinación es válida (p. ej. render OFF + máscaras ON
  para exportar predicciones de evaluación sin video).
- El cálculo del fps real de la fuente **no** depende del render (el JSON lo
  necesita en su cabecera siempre).

### 4.4 `output_path` cuando no se renderiza

- El JSON sigue derivándose y ubicándose como hoy. La ruta de salida del mp4 se
  **ignora** para efectos de escritura de video cuando el render está apagado (no se
  crea ningún archivo de video). El detalle exacto lo fija el `plan.md`.

---

## 5. Criterios de aceptación

1. **AC-1 — Flag presente en ambos modos:** `run_pipeline` y `track_video` aceptan
   un flag que controla la generación del mp4.
2. **AC-2 — Default de un solo video = render ON:** sin especificar el flag, las
   funciones generan el mp4 como hoy (no rompe llamadas existentes).
3. **AC-3 — Independiente del modo:** el render se decide por el flag, nunca por el
   modo; ambos modos pueden correr con render ON u OFF.
4. **AC-4 — JSON siempre:** con render ON u OFF, el JSON del esquema común se escribe
   en `outputs/inference/<video_stem>/<video_stem>.json`.
5. **AC-5 — mp4 condicional:** el mp4 se escribe **solo** con render ON; con render
   OFF no existe archivo de video.
6. **AC-6 — Ahorro real:** con render OFF no se invoca el overlay ni el escritor de
   video ni se acumulan frames compuestos (se salta el trabajo de visualización,
   no solo la escritura final).
7. **AC-7 — Retorno estable:** el `dict` de retorno mantiene su forma en ambos
   casos; con render OFF la entrada del mp4 indica explícitamente la ausencia de
   video. En tracking se conserva el índice de tracks.
8. **AC-8 — Ortogonalidad con `include_masks`:** cualquier combinación de
   render × `include_masks` produce el resultado esperado (render OFF + máscaras ON
   genera JSON con RLE y sin mp4).
9. **AC-9 — Sin cambios colaterales:** no se altera el esquema (`inference_schema`),
   ni `overlay.py`/`video_writer.py`/`frame_extraction.py`, ni la lógica de
   detección/tracking/muestreo.
10. **AC-10 — Verificación:** un script en `testing/` valida render ON/OFF en ambos
    modos (JSON siempre, mp4 condicional, forma del retorno).

---

## 6. Supuestos y notas

- **Default por uso, no global ni por modo:** la invocación de un solo video
  (estas funciones) usa render ON; la futura capa batch lo dejará OFF. Aquí solo se
  deja el flag listo con el default de un solo video.
- **El flag es parámetro de función**, no config ni `.env`: se decide por llamada,
  como `include_masks`. Quien orqueste lotes apagará el render explícitamente.
- **El render no es parte del contrato del dato:** por eso no se añade campo al JSON
  ni se toca `schema_version`. El JSON es idéntico se renderice o no.
- **Sin retrocompatibilidad especial:** los outputs son git-ignored y desechables;
  el flag con default "ON" preserva el comportamiento de las llamadas actuales.
- **El streaming de tracking** abre hoy el escritor de video como context manager
  alrededor del bucle; apagar el render implica que el bucle no dependa de ese
  escritor. La forma concreta de reestructurarlo la define el `plan.md`.
- Esta especificación **no** define el *cómo* técnico (nombre/firma del flag, valor
  de retorno sin video, reestructura del bucle, manejo de `output_path`, detalle del
  test); todo ello corresponde al `plan.md`.
