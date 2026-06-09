# Spec — Orquestación de inferencia por lotes (`batch_inference`)

- **Tarea atómica:** `batch_inference`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Pipeline de inferencia unificado + batch (roadmap del
  pipeline unificado, tarea 4). **Depende de:** `unified_inference` (tarea 3,
  completa), cuya fachada `run_inference` es la única puerta por video que esta capa
  itera; hereda también `inference_schema` (tarea 1) y `optional_render` (tarea 2).
- **Habilita:** correr la inferencia sobre el dataset (o un subconjunto) de forma
  reproducible y reanudable; es el paso previo al ejemplo end-to-end de cierre del
  roadmap y a flujos de evaluación masiva.

---

## 1. Requisito (historia de usuario)

> **Como** persona que analiza el dataset de fútbol robótico,
> **quiero** correr la inferencia sobre **muchos videos de una sola vez** (un split o
> una lista), cargando el modelo una sola vez, saltando lo ya procesado y sin que un
> video defectuoso detenga todo,
> **para** generar el dato estructurado de todo el conjunto de forma reproducible y
> reanudable, sin orquestar manualmente video por video.

---

## 2. Motivación (por qué)

- **Hoy solo hay inferencia por video.** `run_inference` (tarea 3) procesa **un**
  video; para correr 20+ habría que orquestar a mano un bucle, recargando el modelo
  en cada llamada y sin política de reanudación ni de errores.
- **El modelo debe cargarse una sola vez.** SAM3 tarda decenas de segundos en cargar;
  para un lote eso es prohibitivo si se repite por video. La tarea 3 ya dejó
  `run_inference` capaz de recibir un `bundle` precargado en **ambos** modos — falta
  la capa que lo aproveche.
- **Los lotes necesitan robustez.** Un video corrupto, un fallo de decodificación o un
  OOM puntual **no** deben tumbar el lote entero; deben registrarse y dejar que el
  resto continúe.
- **Reanudación.** En corridas largas (o tras un fallo) se quiere **saltar lo ya
  hecho** y no recomputar JSONs existentes, salvo que se pida explícitamente
  reprocesar.
- **El video es accesorio en lote.** Para 20+ videos, renderizar mp4 es I/O y cómputo
  desperdiciado; el lote debe tener el **render apagado por defecto** (el dato es el
  producto), sobreescribible.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Un orquestador de lotes** que itera N videos invocando `run_inference` **una vez
  por video**, con un único `mode` para todo el lote.
- **Fuente de videos = manifiesto `db_metadata.csv`:**
  - **Filtro por `split`** (0=reserva, 1=fine-tuning, 2=testing) para seleccionar el
    subconjunto; default razonable = testing.
  - **Lista explícita** de videos (rutas/ids) que **tiene prioridad** sobre el filtro
    de split, para acotar el lote a voluntad.
  - La ruta que se pasa a `run_inference` es la columna `ruta` (project-relative).
- **Carga única del modelo:** un `bundle` SAM3 cargado una sola vez y reutilizado en
  cada `run_inference` del lote (ambos modos).
- **Skip-done (reanudación):** si el JSON de salida del video ya existe, se **salta**;
  un flag de sobrescritura fuerza el reproceso.
- **Aislamiento de errores:** una excepción procesando un video se **registra y el
  lote continúa**; un video malo no detiene la corrida.
- **Render apagado por defecto** en lote (sobreescribible por llamada). Hereda
  `include_masks`, `sampling`/`max_frames` de `run_inference` aplicados a todo el lote.
- **Resumen estructurado** del lote: por video, su estado (procesado / saltado /
  fallido), las rutas de salida y el mensaje de error si lo hubo; más un conteo
  agregado. Logging por video a stdout.
- **Verificación:** script manual en `testing/` con parte local (selección de videos,
  skip-done, firma — sin SAM3) y parte pod (lote corto real).

### 3.2 Fuera de alcance

- **No** se añade **paralelismo**: la iteración es **secuencial** (SAM3/GPU es el
  cuello de botella). El paralelismo queda como trabajo futuro explícito.
- **No** se modifica `run_inference`, `run_pipeline`, `track_video`, el esquema
  (`inference_schema`) ni los módulos de overlay/escritura/extracción.
- **No** se construye el **ejemplo end-to-end sobre un video real completo** (cierre
  del roadmap); esta tarea entrega la maquinaria de lotes, no la demo final.
- **No** se aborda `prediction_export` (pertenece al roadmap de evaluación, corre
  sobre el set congelado `testing_frames`, no sobre videos muestreados) ni los
  follow-ups de tracking (overlay por `obj_id`, tuning de ByteTrack).
- **No** se cambia el manifiesto ni la lógica de splits (`src/data/`): el lote solo
  **lee** `db_metadata.csv`.
- El **cómo técnico** (nombre y firma exactos del orquestador, estructura del resumen,
  forma del skip-done, manejo de errores, detalle del test) corresponde al `plan.md`.

---

## 4. Comportamiento esperado

### 4.1 Selección de videos

- Sin argumentos de selección → lote = videos del split por defecto (testing).
- Con `split` → lote = videos de ese split del manifiesto.
- Con lista explícita de videos → ese conjunto exacto (ignora el filtro de split).
- El orden es **determinista** (p. ej. por `id`/`ruta`), para corridas reproducibles.

### 4.2 Carga única y modo

- El modelo se carga **una sola vez** antes del bucle y se pasa a cada
  `run_inference`. Todo el lote corre con el **mismo `mode`** y los mismos parámetros
  de muestreo/máscaras/render.

### 4.3 Skip-done y sobrescritura

- Antes de procesar un video, si su JSON de salida ya existe se marca **saltado** y no
  se recomputa.
- Con la sobrescritura activada, se reprocesa aunque exista el JSON.

### 4.4 Aislamiento de errores

- Si `run_inference` lanza una excepción para un video, el orquestador la **captura**,
  registra el video como **fallido** con su mensaje, y **continúa** con el siguiente.
- El lote termina siempre con un resumen; un fallo individual no propaga.

### 4.5 Render y herencia de flags

- `render_video` por defecto **apagado** en lote; sobreescribible. `include_masks`,
  `sampling` y `max_frames` se heredan de `run_inference` y aplican a todo el lote.

### 4.6 Resumen y logging

- Por video se emite a stdout su progreso (`i/N`, ruta, estado).
- Al final, un **resumen** con conteos (procesados / saltados / fallidos) y, por
  video, su estado y rutas/errores, devuelto también como valor de retorno
  estructurado.

---

## 5. Criterios de aceptación

1. **AC-1 — Orquestador único:** existe una función de lote que itera N videos
   llamando a `run_inference` una vez por video, con un solo `mode`.
2. **AC-2 — Fuente desde el manifiesto:** el lote selecciona videos de
   `db_metadata.csv` por `split`, usando la columna `ruta`.
3. **AC-3 — Lista explícita prioritaria:** una lista explícita de videos acota el lote
   y tiene prioridad sobre el filtro de split.
4. **AC-4 — Carga única del modelo:** el `bundle` SAM3 se carga una sola vez y se
   reutiliza en todas las llamadas del lote.
5. **AC-5 — Render OFF por defecto:** en lote, `render_video` es `False` por defecto y
   es sobreescribible.
6. **AC-6 — Skip-done:** un video cuyo JSON ya existe se salta; un flag de
   sobrescritura fuerza el reproceso.
7. **AC-7 — Aislamiento de errores:** un video que falla se registra y el lote
   continúa; ningún fallo individual detiene la corrida.
8. **AC-8 — Resumen estructurado:** el lote devuelve, por video, estado
   (procesado/saltado/fallido), rutas de salida y error si lo hubo, con conteos
   agregados; y loguea el progreso a stdout.
9. **AC-9 — Herencia de flags:** `mode`, `sampling`/`max_frames`, `include_masks` se
   aplican uniformemente a todo el lote vía `run_inference`.
10. **AC-10 — Secuencial:** la iteración es secuencial (sin paralelismo); el orden es
    determinista.
11. **AC-11 — Sin cambios colaterales:** no se altera `run_inference`/`run_pipeline`/
    `track_video`, el esquema, ni la lógica de `src/data/` (el lote solo lee el CSV).
12. **AC-12 — Verificación:** un script en `testing/` valida selección, skip-done y
    firma en local (sin SAM3) y un lote corto real en el pod (skip-done, aislamiento
    de errores, resumen).

---

## 6. Supuestos y notas

- **El lote es una capa delgada sobre `run_inference`:** no reimplementa inferencia,
  solo selecciona videos, orquesta el bucle, gestiona modelo/errores/skip y agrega el
  resumen.
- **Default de selección = testing (`split=2`):** es el subconjunto de uso más
  inmediato; cualquier split o lista explícita es válida.
- **Skip-done por existencia del JSON de salida:** el JSON es el entregable (tarea 1),
  así que su presencia marca "ya procesado". La forma exacta de derivar esa ruta la
  fija el `plan.md` (coherente con la ubicación por video de la tarea 1).
- **Secuencial por diseño:** SAM3 satura la GPU con un solo proceso; el paralelismo no
  aporta hoy y añade complejidad. Queda como trabajo futuro.
- **El lote solo lee `db_metadata.csv`:** no recalcula metadatos ni splits; reusa el
  manifiesto como fuente de verdad.
- Esta especificación **no** define el *cómo* técnico (nombre/firma del orquestador,
  estructura del resumen, skip-done, manejo de errores, detalle del test); todo ello
  corresponde al `plan.md`.
