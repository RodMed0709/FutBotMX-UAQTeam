# Spec — Segmentación por caja con SAM3 (`sam3_box_prompt`)

- **Tarea atómica:** `sam3_box_prompt`
- **Paso de la metodología:** 2 (Especificación)
- **Proceso:** primera tarea de la secuencia que integra el pipeline YOLO + SAM3
  (SAM3-céntrico) al módulo `src/`.
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline YOLO + SAM3 (SAM3-céntrico),
> **quiero** un building block que, dado un frame y un conjunto de **cajas**,
> obtenga las **máscaras finas** de esos objetos usando SAM3 en modo **box-prompt**
> (`Sam3TrackerModel`),
> **para** que un detector rápido (YOLO, tarea posterior) solo tenga que localizar
> con cajas y SAM3 —que sigue siendo el centro— produzca todas las máscaras, sin
> recorrer toda la imagen con text-prompt.

---

## 2. Motivación (por qué)

- SAM3 por **text-prompt** busca el concepto en **toda la imagen** → lento. El
  notebook `fase_2_YOLO_SAM3` validó la alternativa: una **caja** le dice a SAM3
  *dónde* está el objeto y el **box-prompt** segmenta solo ahí → rápido y confiable.
  La caja no necesita ser perfecta; **SAM3 hace la máscara buena**. Así SAM3
  produce todas las máscaras (sigue siendo el centro) y el detector solo lo acelera.
- Es la **segunda cara** del mismo checkpoint `assets/sam3`: además de
  `Sam3VideoModel` + `add_text_prompt` (texto, ya usado en `segmentation.py`),
  existe `Sam3TrackerModel` + `input_boxes` (geometría, estilo SAM2). Hoy esa cara
  **no está expuesta** en `src/`; vive suelta en `notebooks/fase_2_YOLO_SAM3/pipeline_yolo_sam3.py::boxes_to_masks`,
  con rutas hardcoded y carga ad-hoc del modelo.
- Es el **building block base** del pipeline YOLO + SAM3: las tareas siguientes
  (detector YOLO; detector inyectable en el tracking) lo consumen. Conviene tenerlo
  como pieza única, bien definida, alineada con las convenciones del repo
  (`Detection` como moneda común, carga vía `sam3_loader`, imports perezosos).

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Exponer la carga de la 2ª cara de SAM3** (`Sam3TrackerModel` /
  `Sam3TrackerProcessor`) **extendiendo `sam3_loader`**, desde el **mismo**
  checkpoint que ya resuelve la configuración (`working_dirs.sam3_dir`), con las
  **mismas convenciones** de carga (device del bundle, `bfloat16`, modo evaluación).
  La carga de la cara tracker es **opt-in/perezosa**: no debe encarecer ni alterar
  el `load_sam3()` actual ni a sus llamadores.
- **Definir un building block de box-prompt**: dado un **frame** y una lista de
  **cajas** (xyxy en píxeles absolutos), devolver una **máscara fina por caja** a
  resolución del frame, empaquetada en la moneda común `Detection`.
- **Devolver una lista de `Detection` alineada 1:1 y en el mismo orden** que las
  cajas de entrada. Cada `Detection` lleva la máscara booleana `(H, W)`, el `score`
  que se le pase (el del detector; sin score → valor por defecto) y un `obj_id`
  **posicional** (per-frame, **inestable** — la identidad estable la asigna el
  tracker en una tarea posterior).
- **Caso vacío**: si no se reciben cajas, devolver **lista vacía sin invocar al
  modelo**.
- **Documentar el warning benigno** `sam3_video → sam3_tracker` que emite la carga
  del tracker (los pesos del tracker sí están; verificado que produce máscaras
  precisas): se **acepta y se documenta**, no se silencia con código.
- **Estrenar el subpaquete** `src/core/detectors/` donde vivirá este y los demás
  detectores/segmentadores del pipeline (estructura acordada en el roadmap).
- **Test smoke** (script manual, no pytest) que ejercite el box-prompt sobre un
  frame real con cajas conocidas; por depender de SAM3 + GPU, **corre en el pod**
  (coherente con la filosofía de tests del repo).

### 3.2 Fuera de alcance

- **El detector que produce las cajas (YOLO).** Aquí las cajas son una **entrada
  dada**; de dónde salen (YOLO, manual, otra fuente) es de la tarea `yolo_detector`.
- **El cableado al tracking / pipeline / fachada.** Este building block **no** se
  conecta a `track_video`, `run_pipeline`, `run_inference` ni al batch; eso es la
  tarea `detector_strategy`. No agrega `mode` nuevo ni toca el batch.
- **`green_floor` por text-prompt.** Ya existe vía
  `segmentation.detect_classes_in_frame`; no se reimplementa aquí.
- **Propagación SAM3-video** (`propagation.py`): no se porta en esta tarea.
- **Identidad estable de objetos** (`obj_id` consistente entre frames): la pone el
  tracker, no este building block.
- **Cambios de configuración** propios de YOLO/BoT-SORT (ruta de pesos, selección
  de tracker, GMC): pertenecen a sus tareas y se agregan **sobre la marcha** en el
  config de la fase (`configs/01_yolo_sam3_config.json`). Esta tarea **no** requiere
  claves de config nuevas (usa el `sam3_dir` existente).
- La definición del **cómo técnico** (nombres exactos de función/clase y firma,
  campos concretos que se añaden al `Sam3Bundle`, mecanismo de carga perezosa,
  API exacta de `Sam3TrackerModel`/`post_process_masks`, casteo de dtype): es del
  `plan.md`.

---

## 4. Comportamiento esperado (criterios de aceptación)

1. **Carga de la cara tracker disponible y reutilizable**: existe una forma única,
   vía `sam3_loader`, de obtener `Sam3TrackerModel` + `Sam3TrackerProcessor` listos
   para inferir, resolviendo sola la ruta (`working_dirs.sam3_dir`) y usando el
   **mismo device** que el resto del bundle. No se introducen rutas hardcoded ni
   symlinks.
2. **No regresión del loader actual**: `load_sam3()` y sus llamadores existentes
   siguen funcionando igual; cargar la cara tracker es opt-in y no encarece
   `import src.core` ni la carga por defecto.
3. **Máscara por caja**: dado un frame `(H, W, 3)` y N cajas xyxy válidas, se
   obtienen **N máscaras** booleanas a resolución `(H, W)`, **una por caja** y en el
   **mismo orden**.
4. **Empaque en `Detection`**: la salida es `list[Detection]` con N elementos;
   cada uno con `mask` booleana, `score` (el provisto, o el valor por defecto si no
   se provee) y `obj_id` posicional del frame.
5. **Caso vacío**: con la lista de cajas vacía, la salida es `[]` y **no** se invoca
   al modelo.
6. **Caja con máscara vacía/degenerada**: la `Detection` correspondiente **se
   devuelve igual** (se preserva el 1:1 con las cajas); descartar máscaras vacías es
   responsabilidad del consumidor (p. ej. `inference_schema`), no de este building
   block.
7. **Reuso de building blocks**: la pieza se apoya en `sam3_loader` (carga) y en la
   moneda común `Detection`; mantiene el estilo de **imports perezosos** del repo
   (torch/transformers dentro de las funciones).
8. **Verificación**: el script smoke produce, sobre un frame real con cajas
   conocidas, máscaras no vacías y coherentes con las cajas (inspección visual /
   conteo de píxeles), corriendo en el pod.

---

## 5. Dependencias y relación con otras tareas

- **Depende de:** `sam3_loader` (se extiende), `segmentation` (de donde proviene la
  moneda `Detection`). Ambas ya implementadas.
- **Habilita:** `yolo_detector` (tarea 2, produce las cajas) y `detector_strategy`
  (tarea 3, compone detector → box-prompt → tracker dentro de `track_video`).
- **No** depende de YOLO ni de ultralytics.
