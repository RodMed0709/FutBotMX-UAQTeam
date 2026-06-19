# Spec — Función de visualización de un conjunto de frames

- **Tarea atómica:** `frame_visualization`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** una función general de utilidades que muestre un conjunto de frames
> en una cuadrícula (hasta 6 frames; o todos si hay menos de 6),
> **para** inspeccionar visualmente y de forma rápida los frames obtenidos —por
> ejemplo, los que produce la extracción— y verificar que las etapas previas del
> trabajo funcionan como espero.

---

## 2. Motivación (por qué)

- La extracción de frames (tarea `frame_extraction`) entrega los frames **en
  memoria**, sin escribirlos a disco. Hace falta una forma cómoda de
  **inspeccionarlos visualmente** para validar que la extracción es correcta.
- Una cuadrícula con unos pocos frames representativos permite **revisar de un
  vistazo** el resultado sin saturar la pantalla ni procesar todo el conjunto.
- Al ser una utilidad **general** y de **solo visualización**, sirve como
  herramienta de depuración reutilizable a lo largo de todo el proyecto, separada
  de la lógica del pipeline (detección → segmentación → tracking).

---

## 3. Alcance

### 3.1 Dentro de alcance

- Definir una **función general en utilidades** que **muestre** un conjunto de
  frames recibidos como entrada.
- **Mostrar hasta 6 frames** dispuestos en una **cuadrícula**.
- Si el conjunto tiene **menos de 6 frames**, mostrar **todos** los disponibles,
  adaptando la cuadrícula a esa cantidad.
- Si el conjunto tiene **más de 6 frames**, seleccionar **6 repartidos
  uniformemente** a lo largo del conjunto (no los primeros 6), para que la muestra
  sea representativa.
- Respetar el **orden** en que llegan los frames (orden temporal, tal como los
  entrega la extracción).

### 3.2 Fuera de alcance

- **Guardar los frames a disco** (escritura de imágenes, formato de salida,
  nomenclatura): la función **solo muestra**, no persiste nada.
- **Leer o extraer** los frames desde un vídeo o desde disco: la función recibe
  los frames ya en memoria (la extracción es otra tarea, `frame_extraction`).
- Cualquier lógica de **detección, segmentación, tracking** o anotación sobre los
  frames (cajas, máscaras, etiquetas).
- La definición del **cómo técnico** (librería de ploteo, tipos exactos de los
  frames y de la entrada, módulo destino dentro de `src/`, disposición concreta de
  la cuadrícula a nivel de implementación): corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- **Entrada:** un **conjunto de frames** ya en memoria (p. ej. el resultado de la
  función de extracción de frames). La función **no** lee vídeos ni archivos.
- **Salida:** su efecto es **visual** (mostrar la cuadrícula). La función **no**
  guarda los frames en disco y **no** entrega los frames como resultado funcional.
- **Cantidad mostrada:**
  - Si hay **6 o más** frames: se muestran **6**, seleccionados de forma
    **uniforme** a lo largo del conjunto.
  - Si hay **menos de 6** frames: se muestran **todos** los disponibles.
- **Disposición:** los frames se presentan en una **cuadrícula**, que se adapta a
  la cantidad efectivamente mostrada.
- **Orden:** se conserva el orden de llegada de los frames.
- **Entrada vacía (0 frames):** la función **no muestra nada** y **avisa** de que
  no hay frames que visualizar, sin interrumpir abruptamente el flujo con un error
  inesperado.

---

## 5. Criterios de aceptación

1. **AC-1 — Función presente:** existe una función general de utilidades para
   visualizar un conjunto de frames.
2. **AC-2 — Máximo 6 en cuadrícula:** cuando el conjunto tiene 6 o más frames, la
   función muestra exactamente **6** frames dispuestos en una **cuadrícula**.
3. **AC-3 — Menos de 6:** cuando el conjunto tiene menos de 6 frames, la función
   muestra **todos** los disponibles, adaptando la cuadrícula.
4. **AC-4 — Selección uniforme:** cuando hay más de 6 frames, los 6 mostrados
   están **repartidos uniformemente** a lo largo del conjunto (no son los primeros
   6).
5. **AC-5 — Orden preservado:** los frames se muestran en el **orden** en que
   llegan en la entrada.
6. **AC-6 — Solo visualiza:** la función **únicamente muestra** los frames; **no**
   los escribe a disco ni los devuelve como salida funcional.
7. **AC-7 — Entrada vacía:** ante un conjunto sin frames, la función **avisa** y
   no muestra nada, sin error inesperado.
8. **AC-8 — Validación manual:** se demuestra de forma **exploratoria** (script
   suelto o notebook) que, sobre un conjunto real de frames, la cuadrícula se
   muestra correctamente en los casos de **más de 6**, **exactamente 6** y **menos
   de 6** frames.

---

## 6. Supuestos y notas

- La función es una **utilidad general** (en `utils`), reutilizable y desacoplada
  del pipeline; su fin es la **inspección visual / depuración rápida**, no formar
  parte de la salida de producción del pipeline.
- Su uso previsto es **interactivo**, en **notebooks o scripts exploratorios**.
- Entronca de forma natural con la tarea `frame_extraction`: la entrada típica
  serán los frames que esa función entrega en memoria.
- Recordatorio de la constitución y del proyecto: los vídeos solo resuelven
  **dentro del contenedor**; por tanto, cualquier **validación manual** que parta
  de frames extraídos de vídeos reales debe ejecutarse en el contenedor.
- Esta especificación **no** define el *cómo* técnico (librería de ploteo, tipos
  de los parámetros, formato/representación de cada frame, módulo destino dentro de
  `src/`, ni la disposición exacta de la cuadrícula); todo ello corresponde al
  `plan.md` de esta misma carpeta.

---

## 7. Siguientes pasos (metodología)

1. Elaborar `plan.md` con el detalle técnico de implementación.
2. Derivar `tasks.md` con las tareas ejecutables.
3. Implementar (paso 5) únicamente después de los anteriores.
