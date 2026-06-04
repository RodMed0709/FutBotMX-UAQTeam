# Spec — Visualización multi-clase de detecciones (`segmentation_overlay`)

- **Tarea atómica:** `segmentation_overlay`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** una función que pinte las detecciones de todas las clases sobre un
> frame (color por clase) y **devuelva el frame compuesto**, más otra que lo
> **muestre** con leyenda,
> **para** inspeccionar visualmente los resultados de la segmentación y, a la vez,
> tener el frame anotado listo para alimentar el escritor de video del MVP.

---

## 2. Motivación (por qué)

- La segmentación (`text_segmentation`) ya produce `{name: [Detection]}` con
  máscaras full-res, pero no hay forma estándar de **verlas pintadas** sobre el
  frame. Los notebooks lo hacen con `multi_class_overlay` suelto.
- El MVP por-frame necesita el **frame compuesto como array** para escribirlo a
  un mp4 (la tarea siguiente). Por eso el overlay no puede ser solo "display":
  debe **devolver el array**.
- Conviene separar la **composición** (devuelve array, reutilizable para mp4 y
  para mostrar) de la **visualización** (matplotlib + leyenda, para inspección),
  apoyándose en los colores ya centralizados en la configuración
  (`classes_config`).

---

## 3. Alcance

### 3.1 Dentro de alcance

- Definir en un nuevo módulo `src/core/overlay.py`:
  - **`overlay_detections(...)`** — pinta las máscaras de cada clase sobre el
    frame y **devuelve** el frame compuesto como array `uint8 (H, W, 3)` RGB.
  - **`show_overlay(...)`** — muestra el frame compuesto **con leyenda**
    (color ↔ nombre de clase) vía matplotlib (display-only).
- Tomar el **color por clase** de la configuración (`classes[].color`) y el
  **alpha** por defecto de la configuración, con override por parámetro.
- Añadir a la configuración el valor por defecto del alpha del overlay.
- Exportar las piezas públicas desde `src/core/__init__.py`.
- Dos artefactos de validación: un **script headless** (sin gráficos) y un
  **notebook** de inspección visual.

### 3.2 Fuera de alcance

- **Escritura de video (mp4)** o de imágenes a disco (es la tarea siguiente /
  `pipeline_runner`).
- **Segmentación / tracking / export COCO.**
- El **cómo técnico** (firmas y tipos exactos, librería de dibujo de máscaras,
  fórmula de mezcla, lectura de config, backend de matplotlib, ubicación del
  notebook): corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- **Entrada:** un frame `np.ndarray uint8 (H, W, 3)` RGB y un diccionario
  `{name: [Detection]}` (la salida de `detect_classes_in_frame`).
- **`overlay_detections`:**
  - Pinta, para cada clase, las máscaras de sus `Detection` con el **color de esa
    clase** (de la config), mezclando con transparencia (**alpha**).
  - La mezcla se calcula internamente con precisión (float) pero el resultado se
    **devuelve como `uint8 (H, W, 3)`** RGB, listo para mp4 e `imshow`.
  - **No escribe a disco.** Si el dict está vacío o una clase no tiene
    detecciones, devuelve el frame sin esas anotaciones (sin error).
- **`show_overlay`:** compone (vía `overlay_detections`) y **muestra** el
  resultado con **leyenda** color↔clase. **Display-only**: no escribe a disco ni
  devuelve el array como salida funcional.
- **Colores y alpha:** color por clase desde la config (`classes[].color`, por
  `name`); alpha por defecto desde la config (`visualization.overlay_alpha`),
  sobreescribible por parámetro.
- **Máscaras:** se asumen a tamaño del frame (full-res, garantizado por
  `text_segmentation`); se admite un chequeo defensivo barato.

---

## 5. Criterios de aceptación

1. **AC-1 — Módulo y piezas:** existe `src/core/overlay.py` con
   `overlay_detections` y `show_overlay`, exportadas desde `src/core/__init__.py`.
2. **AC-2 — Devuelve array compuesto:** `overlay_detections` devuelve un
   `np.ndarray uint8 (H, W, 3)` RGB con las máscaras pintadas por color de clase.
3. **AC-3 — Color por clase desde config:** el color de cada clase se toma de
   `classes[].color` por `name`.
4. **AC-4 — Alpha desde config con override:** el alpha por defecto se lee de
   `visualization.overlay_alpha` y puede sobreescribirse por parámetro.
5. **AC-5 — Display con leyenda:** `show_overlay` muestra el frame compuesto con
   una leyenda color↔nombre de clase; no escribe a disco.
6. **AC-6 — Sin escritura a disco:** `overlay_detections` no persiste nada;
   devuelve el array.
7. **AC-7 — Casos vacíos:** dict vacío o clase sin detecciones → no es error; se
   devuelve/ muestra el frame sin esas anotaciones.
8. **AC-8 — Validación doble:**
   - un **script headless** (sin gráficos) valida `overlay_detections` con
     detecciones **sintéticas**: forma `(H, W, 3)`, `dtype uint8`, que los píxeles
     bajo máscara cambian hacia el color de la clase;
   - un **notebook** muestra `show_overlay` con matplotlib para inspección visual.

---

## 6. Supuestos y notas

- **Dependencias:** depende de `text_segmentation` (3, `Detection` y el dict de
  detecciones) y `classes_config` (2, colores); **desbloquea** el escritor mp4 y
  `pipeline_runner` (6).
- **Desviación del roadmap (documentada):** el roadmap ubicaba el overlay en
  `src/utils.py` (gemela de `show_frames`); se decide ponerlo **en `src/core/`**
  (módulo `overlay.py`) con array + display juntos, para evitar un import circular
  `utils → core` (core ya depende de utils) y mantener el overlay cohesionado.
- **Validación sin GPU:** el overlay es pura composición y **no usa el modelo**,
  por lo que el script headless se valida en local con datos sintéticos; la
  inspección visual con detecciones reales se hace en el notebook (idealmente con
  salida de `detect_classes_in_frame` corrida en RunPod).
- **El array es la pieza clave:** `overlay_detections` devolviendo `uint8` es lo
  que conecta con la tarea del escritor mp4 (la siguiente del MVP por-frame).
- Esta especificación **no** define el *cómo* técnico (firmas y tipos exactos,
  dibujo de máscaras, fórmula de mezcla, backend de matplotlib, ubicación del
  notebook ni la estructura del bloque `visualization` en la config); todo ello
  corresponde al `plan.md` de esta misma carpeta.
