# Plan operativo — Anotación manual del ground-truth de segmentación (`gt_annotation`)

- **Tarea atómica:** `gt_annotation`
- **Paso de la metodología:** 3 (Planificación)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Proceso de referencia:** roadmap de evaluación del pipeline SAM3-only (tarea 2)
- **Naturaleza:** tarea de **proceso, no de código**. Este `plan.md` describe el
  **flujo operativo en Roboflow** y **cierra las decisiones de la guía** que el spec
  dejó abiertas. No define implementación de código.
- **Estado:** Diseño del proceso. **No** implica crear ni modificar código.

---

## 1. Objetivo del plan

Definir, a nivel operativo, **cómo** producir el ground-truth: configuración del
proyecto en Roboflow, carga de imágenes, anotación con la guía cerrada, control de
calidad, export COCO y colocación del asset en el repo. El resultado es el COCO de
GT + la guía aplicada, no código.

---

## 2. Herramienta y estructura del proyecto

- **Roboflow**, proyecto tipo **Instance Segmentation** (polígonos por instancia).
- **Proyecto dedicado** al GT de evaluación, separado de cualquier proyecto de
  fine-tuning. Nombre propuesto: `futbot-eval-gt`.
- **Clases** (creadas exactamente, granularidad fina):
  `robot_aliado`, `robot_rival`, `robot_desconocido`, `orange_ball`, `green_floor`.
- **Sin** uso de la partición train/valid/test interna: al ser GT, todas las
  imágenes van a un único conjunto.

---

## 3. Guía de anotación (decisiones cerradas)

> Estas son las reglas operativas que resuelven las decisiones abiertas del spec
> §4.2–§4.3. Son la fuente de consistencia entre anotadores.

### 3.1 Robots

- **Parciales en el borde:** se anota el robot si **≥ 25 %** de su cuerpo es visible;
  por debajo de eso se omite. Se anota solo la **región visible** (sin extrapolar
  fuera del frame).
- **Oclusiones:** se anota **solo la parte visible** del robot ocluido; no se
  "rellena" la zona tapada ni se unen fragmentos separados por la oclusión en una
  sola máscara (cada región visible del mismo robot se incluye en su instancia).
- **Equipo:** `robot_aliado` / `robot_rival` según marcador/color. Si el equipo
  **no** es distinguible de forma fiable → `robot_desconocido` (no adivinar). Caso
  típico: los **2 videos cenitales / cámara superior**.

### 3.2 Balón (`orange_ball`)

- Se anota incluyendo el **halo de motion blur** hasta donde sea razonablemente
  identificable como balón; si el desenfoque hace ambiguo el contorno, se prioriza
  **no inflar** la máscara más allá del núcleo claramente reconocible.

### 3.3 Superficie (`green_floor`)

- **Incluye** las **líneas pintadas** sobre la cancha (son parte de la superficie de
  juego; no se recortan).
- **Excluye** todo lo que no es la superficie verde de juego: gradas, paredes,
  exterior de la cancha, objetos sobre el piso (robots y balón se anotan como sus
  propias clases, encima).

### 3.4 Anti-circularidad

- **Prohibido** auto-anotar con **SAM3**. Se permite la asistencia de Roboflow
  (Smart Polygon) **solo** si **cada** máscara se **revisa y corrige** a mano; la
  decisión final es siempre humana.

---

## 4. Flujo operativo en Roboflow

### 4.1 Carga de imágenes

- Subir los **600 PNG** de `data/testing_frames/` (desde una copia local del volumen
  compartido), **preservando** el nombre `<video_id>_<frame_index>.png`.
- Verificar que Roboflow **no** renombra ni reordena las imágenes (el `file_name`
  debe conservarse para la trazabilidad).

### 4.2 Anotación

- Anotar con la **herramienta de polígono** siguiendo la guía §3.
- Asignar a cada instancia su clase de §2.

### 4.3 Generación de la versión y export

- Generar **una versión** del dataset (Roboflow lo exige para exportar) con:
  - **Preprocessing:** desactivar **resize** y **auto-orient** (mantener tamaño
    nativo; evitar drift de coordenadas).
  - **Augmentation:** **ninguna** (el GT no se aumenta).
- **Exportar** en **COCO Segmentation (JSON)** con polígonos.
- Cada re-export incrementa la versión en Roboflow (queda registro).

### 4.4 Trazabilidad

- Confirmar que las entradas `images` del COCO conservan
  `file_name = <video_id>_<frame_index>.png`, lo que enlaza cada anotación con su
  fila en `assets/testing_frames.csv` (`video_id`/`frame_index`).
- Los `category_id` asignados por Roboflow se **documentan** (tabla id→clase); la
  correspondencia con las clases/`coco_id` del proyecto se resuelve en `gt_loader`,
  no aquí.

---

## 5. Asset resultante

- **Ubicación:** `data/gt/eval_coco/` (git-ignored), análogo a `data/raw` (dato
  pesado, fuera del remoto).
- **Contenido mínimo:** el **JSON COCO** de anotaciones. (Las imágenes ya viven en
  `data/testing_frames/`; no es necesario duplicarlas, pero si el zip de Roboflow las
  trae, se ignoran o se mantienen junto al JSON sin versionar.)
- **Obtención:** **descarga manual** del zip COCO desde Roboflow y extracción en
  `data/gt/eval_coco/` sobre el **volumen compartido** (pod). No se automatiza.

---

## 6. Control de calidad

### 6.1 Piloto

- Anotar y exportar primero un **subconjunto de 30–50 imágenes** (mezclando frames
  `aleatorio` y `cenital`) para validar el flujo §4 y la guía §3 end-to-end antes de
  los 600. Si el COCO piloto carga bien en una prueba manual y la trazabilidad se
  sostiene, se procede al total.

### 6.2 Revisión

- **Muestra de verificación:** una segunda persona revisa una muestra de los frames
  anotados (mínimo el piloto + un % del total); doble revisión completa deseable si
  los recursos lo permiten.
- Criterios de revisión: clases correctas, máscaras ajustadas al objeto, reglas de
  §3 aplicadas de forma consistente, equipo ambiguo marcado `robot_desconocido`.

---

## 7. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Cobertura (600) | §4.1, §4.2 | todas las imágenes subidas y anotadas |
| AC-2 Clases finas | §2, §4.2 | 5 clases, sin colapsar |
| AC-3 Anti-circularidad | §3.4 | sin SAM3; asistencia con corrección humana |
| AC-4 Regla de ambiguos | §3.1 | cenitales → `robot_desconocido` |
| AC-5 Export COCO | §4.3 | COCO Segmentation con polígonos |
| AC-6 Trazabilidad | §4.1, §4.4 | `file_name = <video_id>_<frame_index>` |
| AC-7 Almacenamiento | §5 | `data/gt/eval_coco/` git-ignored |
| AC-8 QC | §6 | piloto + muestra de verificación |
| AC-9 Guía | §3 | umbrales y límites resueltos |

---

## 8. Riesgos y consideraciones

- **Pérdida de trazabilidad:** es el riesgo más fácil de cometer — si Roboflow
  renombra/recomprime las imágenes o se aplica auto-orient, el enlace con
  `testing_frames.csv` se rompe. Mitigación: verificar `file_name` y desactivar
  preprocessing (§4.1, §4.3).
- **Inconsistencia entre anotadores:** mitigada por la guía cerrada (§3) y el QC
  (§6); el piloto sirve para detectar divergencias temprano.
- **Costo humano:** ~10–40 h para 600 imágenes; el piloto y el reparto entre el
  equipo acotan el riesgo de invertir mal el esfuerzo.
- **Re-export:** si se corrige la anotación, se genera una versión nueva en Roboflow
  y se re-descarga el COCO; el asset en `data/gt/` se sobrescribe.
- **Colapso diferido:** mantener las 5 clases finas es intencional; `gt_loader`
  colapsará a `robot` para el MVP. No colapsar aquí preserva el trabajo para fase 1.
