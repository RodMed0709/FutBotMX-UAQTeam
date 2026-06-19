# Spec — Clases del modelo en la configuración (`classes_config`)

- **Tarea atómica:** `classes_config`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún (esta tarea solo edita un archivo de configuración `.json`).

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** que las clases a detectar (su nombre, prompt(s) de texto para SAM3,
> color de visualización y categoría COCO) estén definidas de forma centralizada
> en el archivo de configuración,
> **para** tener una única fuente de verdad —en vez del bloque `CLASSES`
> duplicado y divergente entre notebooks— que pueda **crecer** (añadir objetos
> nuevos, p. ej. porterías) y permita **experimentar con varios prompts** por
> clase sin tocar el código.

---

## 2. Motivación (por qué)

- Hoy las clases están definidas como una variable `CLASSES` **copy-pasteada** en
  los notebooks de `fase_0/`, y además **divergente**: el notebook 02 usa un
  diccionario `{prompt: color}`, el 04 una lista de dicts con `coco_id`/`name`/
  `sam3_prompt`/`color`. No hay una definición única ni estable.
- La constitución exige que la configuración viva **fuera del código**, en
  archivos `.json` versionados. Las clases del modelo son configuración global del
  proyecto y deben estar ahí.
- El proyecto necesita **iterar sobre los prompts** (SAM3 reconoce conceptos
  visuales, no contexto deportivo: `"green floor"` y `"green playing surface with
  lines"` funcionan, `"soccer field"`/`"grass"` no). Conviene poder guardar y
  probar **varios candidatos de prompt** por clase.
- El conjunto de objetos a detectar **puede crecer** (hoy 3; mañana quizá
  porterías u otros). El esquema debe admitir añadir clases sin rediseñarse.
- Centralizar las clases desbloquea de forma limpia a las tareas consumidoras:
  segmentación por texto (3), overlay (4), tracking (5) y, en fase 1,
  `coco_autoannotate`.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Definir, dentro del archivo de configuración** `configs/00_testing_config.json`,
  un bloque que centralice las clases del proyecto como una **lista** (orden
  significativo).
- Cada clase incluye los campos:
  - **`name`** — identificador de la clase (estilo COCO, p. ej. `green_floor`).
  - **`sam3_prompts`** — **lista** de prompts de texto candidatos para SAM3; por
    **convención, el primero es el prompt activo**.
  - **`color`** — color RGB (0–255) para visualización.
  - **`coco_id`** — id de categoría COCO (para fase 1 / `coco_autoannotate`).
- Poblar el bloque con las **3 clases actuales del MVP** (robot, balón naranja,
  piso verde) con los valores ya ejercitados en los notebooks.
- El esquema debe soportar un **número arbitrario de clases**: añadir un objeto
  nuevo (p. ej. portería) es solo agregar otra entrada con sus campos.

### 3.2 Fuera de alcance

- **Cualquier código nuevo.** Esta tarea **solo edita la configuración**. El
  *cómo* se leen las clases desde el código (lectura directa del JSON o un helper
  en `utils`, tipos, validación) corresponde al `plan.md` y/o a la tarea
  consumidora (`text_segmentation`).
- **Lógica de detección, segmentación, overlay o tracking** que use estas clases.
- **Distinguir robots aliados vs enemigos** (decisión abierta del roadmap, fuera
  del MVP base).
- **Añadir clases nuevas** más allá de las 3 del MVP (porterías u otras): el
  esquema lo habilita, pero poblarlas es trabajo futuro.

---

## 4. Comportamiento esperado

- **Definición declarativa:** las clases quedan descritas en el `.json`; ningún
  consumidor necesita redefinirlas en código.
- **Lista ordenada:** las clases se guardan como lista; el **orden se preserva** y
  es significativo (consumidores futuros, como la heurística `obj_id → clase` del
  tracking, pueden depender de él). Orden del MVP: `robot` → `orange_ball` →
  `green_floor`.
- **Prompts por clase:** cada clase expone una **lista** `sam3_prompts`; el
  **primero** es el que se usa por defecto. Para **experimentar con otros
  prompts** se editan/añaden entradas en `sam3_prompts` (esto se documenta para
  quien consuma la config).
- **Extensible:** agregar una clase nueva (p. ej. portería) consiste en añadir una
  entrada más con los mismos campos, sin cambiar la estructura.
- **Valores del MVP** (de los notebooks 02/04):
  - `robot` — prompts `["robot"]`, color (60,130,255), coco_id 1.
  - `orange_ball` — prompts `["orange ball"]`, color (255,100,0), coco_id 2.
  - `green_floor` — prompts `["green playing surface with lines", "green floor"]`,
    color (50,220,70), coco_id 3. (El primero es el activo; el segundo se conserva
    como candidato alternativo validado.)

---

## 5. Criterios de aceptación

1. **AC-1 — Bloque de clases en la config:** `configs/00_testing_config.json`
   contiene un bloque que centraliza las clases como **lista**.
2. **AC-2 — Campos por clase:** **cada** clase de la lista incluye `name`,
   `sam3_prompts` (lista no vacía), `color` (RGB) y `coco_id`.
3. **AC-3 — Clases del MVP:** el bloque incluye las 3 clases del MVP (robot,
   orange_ball, green_floor) con los valores indicados en §4.
4. **AC-4 — Prompt activo por convención:** el primer elemento de `sam3_prompts`
   de cada clase es el prompt activo; para `green_floor` es
   `"green playing surface with lines"`.
5. **AC-5 — Orden preservado:** las clases aparecen en el orden robot →
   orange_ball → green_floor.
6. **AC-6 — Extensible a N clases:** el esquema admite añadir más clases con solo
   agregar entradas; la validación de aceptación comprueba que **cada** clase
   tiene los campos requeridos, **no** un número fijo de clases.
7. **AC-7 — JSON válido:** tras la edición, el archivo sigue siendo un JSON válido
   y el resto de su contenido (`working_dirs`, `preprocess`) permanece intacto.
8. **AC-8 — Sin código nuevo:** la tarea no añade ni modifica código fuente; solo
   el archivo de configuración.

---

## 6. Supuestos y notas

- Es **cimiento** (junto con `sam3_loader`): no depende de ninguna otra tarea y
  **desbloquea** segmentación (3), overlay (4), tracking (5) y `coco_autoannotate`.
- **`coco_id` se incluye ya** aunque el MVP por-frame no exporte COCO: es barato,
  ya existe en el notebook 04 y alimenta la fase 1 (`coco_autoannotate`), evitando
  re-tocar la config más adelante.
- **Discrepancia con el roadmap resuelta aquí:** el roadmap cita `"green floor"`
  como prompt validado; los notebooks 02/04 evolucionaron a `"green playing
  surface with lines"`. Se adopta este último como **activo** y se conserva
  `"green floor"` como candidato en la lista.
- Esta especificación **no** define el *cómo* técnico (estructura exacta de claves
  JSON, representación del color —lista vs objeto—, ni el mecanismo de lectura
  desde el código); todo ello corresponde al `plan.md` de esta misma carpeta.
