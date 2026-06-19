# Spec — Detector de cajas YOLO (`yolo_detector`)

- **Tarea atómica:** `yolo_detector`
- **Paso de la metodología:** 2 (Especificación)
- **Proceso:** segunda tarea de la secuencia que integra el pipeline YOLO + SAM3
  (SAM3-céntrico) al módulo `src/`. Las cajas que aquí se producen alimentan al
  box-prompt (`sam3_box_prompt`, ya implementado).
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline YOLO + SAM3 (SAM3-céntrico),
> **quiero** un building block que, dado un frame, use el detector **YOLO11**
> ya entrenado (`best.pt`) para localizar rápido los objetos y devolver sus
> **cajas** agrupadas por clase del repo,
> **para** que SAM3 —que sigue siendo el centro— solo tenga que convertir esas
> cajas en máscaras (box-prompt), sin recorrer toda la imagen con text-prompt.

---

## 2. Motivación (por qué)

- SAM3 por text-prompt es **preciso pero lento** (busca el concepto en toda la
  imagen). El detector YOLO11 destilado en fase_1 es **rápido y deployable**
  (~130 FPS) y localiza con cajas; alimentar esas cajas a SAM3 box-prompt
  (tarea `sam3_box_prompt`) da el pipeline rápido y SAM3-céntrico validado en los
  notebooks `fase_2_YOLO_SAM3`.
- Hoy la inferencia YOLO vive **suelta y hardcoded** en
  `notebooks/fase_2_YOLO_SAM3/pipeline_yolo_sam3.py` (ruta del `best.pt` junto al
  notebook, `CLASS_NAMES`/`conf`/`imgsz` incrustados, RGB/BGR ambiguo). Falta una
  pieza **config-driven**, reutilizable y alineada con las convenciones del repo.
- Es el **detector** que la tarea siguiente (`detector_strategy`) inyectará en el
  tracking. Conviene tenerlo como building block único y bien definido, que emita
  cajas listas para box-prompt.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Cargar el modelo YOLO** (`best.pt`) de forma **config-driven**: la ruta de los
  pesos se lee del config (`working_dirs.yolo_weights`) y se resuelve con
  `get_abs_path`. Carga **cacheada** (singleton) e **independiente** del
  `Sam3Bundle` (es otro modelo y otra librería), con selección de device automática
  (GPU si está, si no CPU) y forzable.
- **Inferir cajas por frame**: dado un frame `(H, W, 3)` RGB, ejecutar el detector
  y obtener cajas, clases y scores.
- **Mapear la clase YOLO → clase del repo** mediante el `yolo_id` de cada clase en
  el config, y **agrupar la salida por nombre de clase del repo**.
- **Devolver las cajas como building block reutilizable**, en una estructura ligera
  caja+score (no máscara), que es justo lo que consume el box-prompt.
- **Crecer el config de la fase** (`configs/01_yolo_sam3_config.json`) con lo que
  esta tarea necesita y que es persistente:
  - nueva clave `working_dirs.yolo_weights` (ruta del `best.pt`);
  - la clase **`yellow_zone`** (con su `color` y `coco_id`);
  - un campo **`yolo_id`** en las clases detectables por YOLO (`robot`,
    `orange_ball`, `yellow_zone`); `green_floor` **no** lo recibe;
  - una sección **`yolo`** con parámetros de inferencia (`conf`, `imgsz`).
- **Parámetros de inferencia config-driven** (`conf`, `imgsz`) con defaults
  sensatos (`0.4`, `960`), sobreescribibles por argumento.
- **Test smoke** (script manual, no pytest) que ejercite el detector sobre un frame
  real; corre donde está `best.pt` (el pod; admite CPU).

### 3.2 Fuera de alcance

- **Producir máscaras.** YOLO da **cajas**; la máscara la hace SAM3 box-prompt
  (`sam3_box_prompt`, ya implementada). Esta tarea no segmenta.
- **El cableado** del detector a `boxes_to_masks`, al tracking, a `run_inference` o
  al batch: es la tarea `detector_strategy`. No agrega `mode`, no toca el batch, no
  emite mp4 ni JSON.
- **Entrenar / re-exportar** YOLO (es fase_1) y **descargar/aprovisionar** el
  `best.pt` (setup de entorno; futuro `bootstrap_data`). Se asume el peso ya en
  disco.
- **`green_floor`**: no es clase YOLO; se segmenta por text-prompt (ya existe). No
  aparece en la salida de este detector.
- El detector **SAM3-text** existente: no se toca aquí (su extracción/refactor es de
  la tarea `detector_strategy`).
- La definición del **cómo técnico** (nombres de funciones/clases y firmas exactas,
  dataclass concreta de salida, mecanismo de caché, API exacta de ultralytics):
  corresponde al `plan.md`.

---

## 4. Comportamiento esperado (criterios de aceptación)

1. **Carga config-driven y reutilizable**: existe una forma única de obtener el
   modelo YOLO listo para inferir, resolviendo sola la ruta (`working_dirs.yolo_weights`
   vía `get_abs_path`) y el device; la carga se **cachea** (singleton) y es
   independiente de la carga de SAM3. Sin rutas hardcoded ni symlinks.
2. **Import perezoso**: `import src.core` no arrastra `ultralytics`; se importa solo
   al invocar la carga/inferencia.
3. **Cajas por frame**: dado un frame `(H, W, 3)` RGB con objetos, el detector
   devuelve sus **cajas xyxy** (píxeles absolutos) con su **score**, agrupadas por
   **nombre de clase del repo**.
4. **Mapeo de clase correcto**: cada caja YOLO se asocia a la clase del repo según
   su `yolo_id` en el config; las clases del config **sin** `yolo_id` (`green_floor`)
   **no** aparecen en la salida.
5. **Clase no mapeada**: una clase YOLO que no exista en el config se **descarta**
   sin romper.
6. **Sin detecciones**: un frame sin objetos devuelve la estructura con listas
   vacías (o vacía), sin error.
7. **Parámetros config-driven**: `conf` e `imgsz` se leen de la sección `yolo` del
   config (defaults `0.4`/`960`) y pueden sobreescribirse por argumento.
8. **Config crecido**: el config de la fase incluye `working_dirs.yolo_weights`, la
   clase `yellow_zone`, los `yolo_id` (`robot=0`, `orange_ball=1`, `yellow_zone=2`)
   y la sección `yolo`; `green_floor` permanece sin `yolo_id`.
9. **Verificación**: el script smoke produce, sobre un frame real, cajas coherentes
   (conteo por clase y scores razonables), corriendo donde está `best.pt`.

---

## 5. Dependencias y relación con otras tareas

- **Depende de:** el `best.pt` entrenado (artefacto de fase_1) presente en disco; el
  config de la fase (se extiende); `get_abs_path` (rutas).
- **Habilita:** `detector_strategy` (tarea 3), que compondrá
  `yolo_detector` (cajas) → `boxes_to_masks` (máscaras, ya implementado) → tracker,
  emitiendo `obj_id` estable, JSON unificado y overlay reutilizando lo existente.
- **No** produce máscaras ni identidad estable: eso es de tareas posteriores.
