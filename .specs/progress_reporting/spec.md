# Spec — Barra de progreso en inferencia (`progress_reporting`)

- **Tarea atómica:** `progress_reporting`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Mejora de la experiencia de uso del pipeline (UX de
  ejecución), transversal a segmentación y tracking.
- **Depende de:** nada nuevo; `tqdm>=4.66` ya está en `requirements.txt`.
- **Habilita:** ver el avance real de las corridas largas (sobre todo tracking) en los
  notebooks de benchmark y en el pod.

---

## 1. Requisito (historia de usuario)

> **Como** persona que corre inferencia (segmentación o tracking) sobre videos,
> **quiero** una **barra de progreso** con ETA y velocidad en vez de un `print` por
> frame (y que tracking, que hoy no muestra nada, también la tenga),
> **para** ver el estado de avance de una corrida larga sin saturar la salida.

---

## 2. Motivación (por qué)

- **Segmentación es ruidosa.** `run_pipeline` imprime `frame {i+1}/{total}` **una
  línea por frame** (`pipeline.py`), lo que llena la salida y no da ETA ni velocidad.
- **Tracking es ciego.** `track_video` recorre el video en streaming
  (`iter_frames`) **sin imprimir nada** de progreso; en un video largo no hay forma de
  saber cuánto falta.
- **La herramienta ya está.** `tqdm` está en las dependencias; falta cablearlo. Una
  barra única por video da avance, ETA y frames/s, y se puede silenciar.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **`progress: bool = True`** en `run_pipeline` y `track_video`.
- **Segmentación:** reemplazar el `print` por-frame por una barra `tqdm` (total
  conocido = `len(frames)`).
- **Tracking:** envolver `iter_frames(...)` en una barra `tqdm`; para el total se
  añade un helper **`get_frame_count(video_path)`** en `frame_extraction.py` (decord
  `len`, metadata barata). `total = count` o `min(max_frames, count)`.
- **`tqdm.auto`** (notebook + terminal), import perezoso, `desc` con el stem del
  video, `leave=False`, `disable=not progress`.
- **Hilvanar `progress`** por `run_inference` (ambas ramas) y `run_batch`
  (default `True`).
- **Smoke test** (sin GPU): firmas con `progress=True` por defecto y `get_frame_count`
  sobre un video real.
- **Actualizar `CLAUDE.md`**.

### 3.2 Fuera de alcance

- **Barras a nivel de clase/sub-paso** dentro de un frame, y **logging estructurado**.
- **El streaming de `pipeline.py`** (la cura de RAM de `all_frames`): tarea condicional
  aparte, puede no llegar a hacerse.
- **El esquema JSON, las rutas, el muestreo y la lógica de inferencia:** no se tocan.

---

## 4. Comportamiento esperado

### 4.1 `progress=True` (default)

Una **barra por video** con avance, ETA y velocidad (frames/s), tanto en segmentación
(`desc="seg <stem>"`) como en tracking (`desc="track <stem>"`). En tracking la barra
tiene total (vía `get_frame_count`), así que muestra ETA real.

### 4.2 `progress=False`

Silencio total: sin barra ni prints de progreso. Útil en lotes/CI o cuando la salida
debe quedar limpia.

### 4.3 En lotes (`run_batch`)

El print `[i/n] ruta -> status` por video **se conserva**; la barra del video en curso
aparece debajo mientras corre y desaparece al terminar (`leave=False`).

### 4.4 Retrocompatibilidad

Las firmas ganan un parámetro con default `True` → las llamadas existentes siguen
igual. El único cambio de salida es que el verbose por-frame de segmentación pasa de
`print` a barra.

---

## 5. Criterios de aceptación

1. `run_pipeline` y `track_video` aceptan `progress: bool = True`.
2. Segmentación muestra una barra `tqdm` (total = `len(frames)`) y **ya no** imprime
   `frame {i+1}/{total}`.
3. Tracking muestra una barra `tqdm` con total derivado de `get_frame_count`
   (`min(max_frames, count)` cuando hay tope).
4. `get_frame_count(video_path)` existe en `frame_extraction.py` y devuelve el nº de
   frames (decord `len`), aceptando ruta relativa o absoluta como el resto del módulo.
5. `progress=False` desactiva la barra por completo.
6. `progress` se propaga por `run_inference` y `run_batch` (default `True`).
7. La barra usa `tqdm.auto`, import perezoso, `desc` con el stem, `leave=False`.
8. No cambia el esquema JSON, rutas, muestreo ni lógica de inferencia.
9. Smoke test (sin GPU) cubre firmas y `get_frame_count`.
10. `CLAUDE.md` menciona las barras, el flag `progress` y `get_frame_count`.

---

## 6. Supuestos y notas

- Lista completa de supuestos acordada con el usuario (técnicos, funcionales y de
  proceso); **ninguno rechazado**. Decisiones fijadas: parámetro **`progress`**,
  **`tqdm.auto`**, helper **`get_frame_count`**, `leave=False`, hilvanado del flag.
- La **barra visual** solo se confirma corriendo inferencia (pod/GPU); lo testeable sin
  GPU son las firmas y `get_frame_count` sobre un video real.
- Independiente de `config_aware_output_paths` y `detector_in_segmentation`; toca los
  mismos archivos pero en regiones distintas (loops de inferencia, no rutas).
