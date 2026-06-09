# Spec — Anotación manual del ground-truth de segmentación (`gt_annotation`)

- **Tarea atómica:** `gt_annotation`
- **Paso de la metodología:** 2 (Especificación)
- **Naturaleza:** tarea de **proceso, no de código**. Este `spec.md` funciona como
  **protocolo de anotación**; el `plan.md`/`tasks.md` serán un **checklist operativo
  para el humano**. No gatea implementación de código (constitución §6.2: spec "por
  tarea o proceso").
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código.
- **Proceso al que pertenece:** evaluación del pipeline SAM3-only (roadmap de
  evaluación, tarea 2). **Depende de** la tarea 1 (`eval_frame_export`, completa).

---

## 1. Requisito (historia de usuario)

> **Como** persona que evalúa el pipeline de análisis de fútbol robótico,
> **quiero** anotar manualmente las máscaras de segmentación de los 600 frames del
> set de evaluación y exportarlas en formato COCO,
> **para** disponer de un **ground-truth independiente del modelo** contra el cual
> medir, de forma comparable y reproducible, el rendimiento de segmentación del
> pipeline.

---

## 2. Motivación (por qué)

- Las métricas de segmentación (mIoU, Boundary IoU, Dice) solo tienen sentido contra
  un **ground-truth fiable**. Sin él no hay forma de comparar el pipeline consigo
  mismo en el tiempo ni contra los otros pipelines del proyecto.
- El GT debe ser **independiente de SAM3**: si se generara con el mismo modelo que se
  evalúa, la medición sería circular y los números, inválidos.
- Anotar a **granularidad fina** (robots por equipo) una sola vez permite reusar el
  mismo GT para el MVP actual (colapsando a `robot`) y para un pipeline equipo-aware
  en fase 1, sin re-anotar.
- Preservar la **trazabilidad** entre cada imagen y su origen (`assets/testing_frames.csv`)
  es lo que permitirá, después, emparejar GT y predicciones frame a frame.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Anotar manualmente** (segmentación) los **600 frames** de `data/testing_frames/`
  listados en `assets/testing_frames.csv`, en **Roboflow**.
- Anotar las **clases finas** definidas en §4.1, incluida la clase para equipo
  ambiguo.
- Producir una **guía de anotación** (criterios de §4.2) y aplicarla de forma
  consistente.
- **Control de calidad** humano (§4.4).
- **Exportar** el resultado en **COCO instance segmentation** y **almacenarlo** como
  dato pesado git-ignored en el repo (§4.5).

### 3.2 Fuera de alcance

- **No** se escribe ni modifica código (tarea de proceso).
- **No** se carga el COCO a código ni se calculan métricas (tareas `gt_loader` y
  `segmentation_metrics`).
- **No** se **colapsan** las subclases de robot a `robot`: eso ocurre después, en
  `gt_loader` (aquí se exporta la granularidad fina).
- **No** se anota tracking (clips densos con identidad temporal), ni otros videos, ni
  el split de fine-tuning.
- **No** se automatiza la descarga/colocación del COCO (se documenta el cómo; la
  automatización, si se quisiera, sería otra tarea).

---

## 4. Protocolo de anotación (comportamiento esperado)

### 4.1 Clases a anotar (granularidad fina)

| Clase | Descripción |
|---|---|
| `robot_aliado` | Robot del equipo propio (distinguible por marcador/color). |
| `robot_rival` | Robot del equipo contrario (distinguible por marcador/color). |
| `robot_desconocido` | Robot cuyo equipo **no** se distingue de forma fiable (ver §4.3). |
| `orange_ball` | Balón naranja. |
| `green_floor` | Superficie de juego verde. |

- Se anota **una instancia por objeto** (segmentación de instancias).
- El **colapso** `robot_aliado`/`robot_rival`/`robot_desconocido` → `robot` para el
  MVP actual se hace en `gt_loader`, **no** aquí.

### 4.2 Guía de anotación (criterios de consistencia)

La consistencia entre anotadores define la validez del GT. Reglas mínimas:

- **Robots parciales en el borde:** se anota la parte visible (criterio único a
  fijar: anotar si ≥ X % visible — definir en el `plan.md`/guía).
- **Oclusiones:** se anota solo la región visible del objeto ocluido (sin "rellenar"
  la parte tapada).
- **Balón borroso por movimiento (motion blur):** se anota incluyendo el halo de
  desenfoque hasta donde sea razonablemente identificable como balón.
- **Límites de `green_floor`:** decisión explícita a fijar en la guía — si incluye
  líneas pintadas y si excluye gradas/exterior de la cancha.
- **Anti-circularidad:** **prohibido** auto-anotar con **SAM3**. Si se usa asistencia
  (p. ej. Smart Polygon), **toda** máscara debe ser **revisada y corregida por un
  humano**.

### 4.3 Regla para equipo ambiguo

- Cuando el equipo de un robot **no** se distingue de forma fiable (caso típico en
  los **2 videos cenitales / cámara superior**), se etiqueta como
  `robot_desconocido` en lugar de adivinar `aliado`/`rival`.
- Esto no afecta el MVP actual (todo colapsa a `robot`) y preserva la información
  para fase 1.

### 4.4 Control de calidad

- **Piloto previo:** anotar primero un **subconjunto pequeño** para validar la guía y
  el flujo antes de anotar los 600 completos.
- **Revisión:** al menos una **muestra de verificación** revisada por una segunda
  persona; doble revisión completa deseable si los recursos lo permiten.

### 4.5 Export y almacenamiento

- **Formato:** **COCO instance segmentation** (polígonos), mismo formato del
  notebook 04.
- **Resolución:** anotación a **tamaño nativo** (las imágenes se exportaron sin
  resize).
- **Trazabilidad:** los nombres de imagen en Roboflow se mantienen como
  `<video_id>_<frame_index>.png`, de modo que el COCO sea enlazable con
  `assets/testing_frames.csv` (clave `video_id`/`frame_index`).
- **Almacenamiento:** el COCO se **descarga de Roboflow** y se coloca en una
  ubicación **git-ignored** del repo (propuesta `data/gt/`, a precisar), análoga a
  `data/raw` (dato pesado, fuera del remoto).

### 4.6 Anti-leakage

- Los 600 frames pertenecen al split **testing**; **no** se usan como labels de
  fine-tuning. El split ya los aísla a nivel de video.

---

## 5. Criterios de aceptación

1. **AC-1 — Cobertura:** los **600 frames** de `assets/testing_frames.csv` quedan
   anotados (todas las instancias visibles de las clases de §4.1).
2. **AC-2 — Clases finas:** la anotación usa las clases de §4.1, incluida
   `robot_desconocido`; **no** se colapsa a `robot` en esta etapa.
3. **AC-3 — Anti-circularidad:** ninguna máscara proviene de SAM3 sin verificación
   humana; toda asistencia fue revisada/corregida por una persona.
4. **AC-4 — Regla de ambiguos:** los robots de equipo no distinguible (cenitales)
   están como `robot_desconocido`, no adivinados.
5. **AC-5 — Export COCO:** existe un export en **COCO instance segmentation** con
   polígonos y las categorías de §4.1.
6. **AC-6 — Trazabilidad:** los nombres de imagen permiten enlazar cada anotación con
   su fila en `assets/testing_frames.csv` (`<video_id>_<frame_index>`).
7. **AC-7 — Almacenamiento:** el COCO está en la ubicación git-ignored acordada y
   **no** se sube al remoto.
8. **AC-8 — QC:** se realizó el piloto y al menos una muestra de verificación
   revisada.
9. **AC-9 — Guía:** existe la guía de anotación con las reglas de §4.2–§4.3
   resueltas (umbrales/decisiones concretas).

---

## 6. Supuestos y notas

- **Tarea de proceso:** el valor es el **asset COCO + la guía**, no código. El
  `plan.md` detallará el flujo en Roboflow y el `tasks.md` será el checklist humano.
- **Granularidad fina y colapsable** (principio del roadmap §6): se anota por equipo;
  el colapso a `robot` vive en `gt_loader`.
- **Trazabilidad como eje:** preservar `<video_id>_<frame_index>` es lo que hará
  posible alinear GT↔predicción en `segmentation_metrics`; es el detalle más fácil de
  romper al subir/bajar imágenes de Roboflow.
- **Cuello de botella:** es la tarea más costosa en tiempo humano del proceso de
  evaluación (~10–40 h para 600 imágenes según asistencia e instancias/frame); de ahí
  el piloto y la posible repartición entre el equipo.
- **Dependencia dura:** requiere la tarea 1 (`eval_frame_export`) completa, que ya lo
  está; las imágenes ya viven en el volumen compartido.
- Esta especificación **no** define el detalle operativo (pasos exactos en Roboflow,
  umbrales finales de la guía, ubicación definitiva del COCO); eso es del `plan.md`.
