# Spec — Paridad de batch para detector/tracker (`batch_detector_tracker`)

- **Tarea atómica:** `batch_detector_tracker`
- **Paso de la metodología:** 2 (Especificación)
- **Proceso:** quinta y **última** tarea de la secuencia que integra el pipeline
  YOLO + SAM3 a `src/`. Cierra la integración llevando las perillas `detector` y
  `tracker` —ya presentes en la fachada `run_inference`— a la capa de lotes
  `run_batch`.
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** poder **elegir el detector y el tracker para todo un lote** desde
> `run_batch` (igual que ya puedo hacerlo por video en `run_inference`),
> **para** generar de una sola corrida los outputs de una configuración concreta
> (p. ej. `yolo_sam3` × `botsort`) sobre el conjunto de testing, sin llamar a
> `run_inference` video por video ni perder la carga única de SAM3, el skip-done
> y el aislamiento de errores.

---

## 2. Motivación (por qué)

- Las tareas previas de la integración (`box_prompt`, `yolo_detector`,
  `detector_strategy`, `botsort_tracker`) dotaron a `run_inference` de dos perillas
  ortogonales: `detector` (`sam3_text` | `yolo_sam3`) y `tracker`
  (`bytetrack` | `botsort`). Hoy se pueden elegir **por video**.
- `run_batch` —la capa secuencial sobre la fachada— **se quedó atrás**: su firma
  no expone `detector` ni `tracker`, así que un lote **siempre** usa los defaults
  del config. No hay forma de correr el conjunto de testing con BoT-SORT o con el
  detector YOLO sin renunciar a las ventajas del batch (carga única de SAM3,
  skip-done, aislamiento de errores, resumen estructurado).
- La invariante de diseño del proyecto es que **`run_batch` es una capa delgada
  sobre `run_inference`**: ambas deben exponer la **misma superficie de
  configuración**. Cerrar este hueco restablece esa paridad y deja el batch listo
  como motor para generar, por configuración, los outputs que después se comparan
  (eficiencia / comportamiento; la instrumentación de tiempo y memoria es tarea
  aparte).

---

## 3. Alcance

### 3.1 Dentro de alcance

- **`run_batch` gana dos parámetros**: `detector: str | None = None` y
  `tracker: str | None = None`, con la **misma semántica** que en `run_inference`
  (`None` ⇒ usar el default del config). Se ubican al **final** de la firma para no
  romper llamadas posicionales existentes.
- **Aplican a todo el lote por igual**: una corrida de `run_batch` usa el mismo
  `detector` y el mismo `tracker` para todos los videos seleccionados (no hay
  override por video).
- **Propagación directa**: ambos se pasan tal cual a la llamada `run_inference(...)`
  existente dentro del bucle.
- **Validación temprana**: si `detector`/`tracker` traen un nombre inválido,
  `run_batch` falla con `ValueError` **antes** de cargar SAM3 (la carga única), no
  por video ni después de la carga. Reutiliza los mecanismos existentes
  (`KNOWN_TRACKERS` para el tracker; `get_detector(...)` para el detector).
- **Documentación**: el docstring de `run_batch` describe ambos parámetros y su
  semántica de "solo tracking" (ver fuera de alcance).

### 3.2 Fuera de alcance

- **Semántica de detector/tracker en cada modo**: ambos solo tienen efecto en
  `mode="tracking"`; en `mode="segmentation"` se **ignoran** (es el comportamiento
  ya definido por `run_inference`). Esta tarea **no** cambia esa semántica, solo la
  propaga y la documenta.
- **Override por video** (un detector/tracker distinto por cada video del lote): se
  descarta; el lote es homogéneo en configuración.
- **Cambiar la forma del resumen** (`list[dict]` por video): las llaves se mantienen
  (`id`, `ruta`, `status`, `json`, `video`, `error`). **No** se añade tiempo ni
  memoria.
- **Instrumentación de eficiencia** (medir tiempo de inferencia y VRAM por video):
  es una tarea atómica **separada y posterior**.
- **Versionar el skip-done por detector/tracker**: el skip-done sigue basándose solo
  en la existencia del JSON de salida en su ruta canónica; cambiar de configuración
  sobre un output existente requiere `overwrite=True`. (No se altera el naming.)
- **Cambios de config**: no se añaden ni modifican llaves en los JSON de `configs/`;
  los defaults de detector/tracker ya viven ahí de tareas previas.
- **Tocar `run_inference`, `track_video`, el subpaquete `trackers`/`detectors`, el
  schema o `src/data`**: el único archivo de código que cambia es `src/core/batch.py`.
- **Correr el lote real** con SAM3: esta tarea deja el código listo y verificado en
  su parte sin-GPU; la corrida real es en el pod.
- El **cómo técnico** (líneas exactas, orden de la validación): corresponde al
  `plan.md`.

---

## 4. Comportamiento esperado (criterios de aceptación)

1. **Firma ampliada**: `run_batch` acepta `detector` y `tracker` (ambos
   `str | None`, default `None`), al final de la firma; las llamadas posicionales
   previas siguen funcionando.
2. **Propagación a todo el lote**: ambos se pasan a `run_inference` en cada video, de
   modo que el lote completo corre con esa configuración.
3. **Default conservado**: con `detector=None` y `tracker=None` (default), el
   comportamiento es **idéntico** al actual (los defaults del config).
4. **Validación temprana**: un `tracker` o `detector` con nombre inválido levanta
   `ValueError` **antes** de cargar SAM3 (no se llega a procesar ningún video).
5. **Solo-tracking documentado**: el docstring deja claro que detector/tracker solo
   tienen efecto en `mode="tracking"` y se ignoran en `mode="segmentation"`.
6. **Resumen intacto**: la forma de cada entrada del resultado (`id`, `ruta`,
   `status`, `json`, `video`, `error`) no cambia.
7. **Aislamiento preservado**: el skip-done, la carga única de SAM3 y el aislamiento
   de errores por video siguen funcionando igual.
8. **Verificación sin GPU**: un script de prueba ejercita la nueva validación
   temprana (nombre inválido ⇒ `ValueError` sin cargar SAM3) y la presencia de los
   parámetros en la firma; la corrida real de un lote con SAM3 queda para el pod.

---

## 5. Dependencias y relación con otras tareas

- **Depende de:** `detector_strategy` (registro `get_detector` + selección por
  nombre con `ValueError`), `botsort_tracker` (`KNOWN_TRACKERS` + selección de
  tracker), y la fachada `run_inference` (que ya recibe y propaga `detector`/
  `tracker`). `batch_inference` (la versión actual de `run_batch`) es la base que se
  amplía.
- **Habilita:** generar, por configuración (detector × tracker), los outputs del
  conjunto de testing en una sola corrida — insumo de la futura tarea de
  **instrumentación/benchmark de eficiencia** (tiempo + memoria) y de la comparación
  de comportamiento entre modelos.
- **No** depende de GT: esta tarea es puramente de orquestación; la evaluación de
  exactitud contra ground-truth sigue su proceso aparte.
- **Cierra** la secuencia de integración YOLO + SAM3 (quinta de cinco tareas).
