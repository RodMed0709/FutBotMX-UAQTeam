# Tasks — Anotación manual del ground-truth de segmentación (`gt_annotation`)

- **Tarea atómica:** `gt_annotation`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Naturaleza:** tarea de **proceso, no de código**. Este `tasks.md` es un
  **checklist operativo para el equipo humano**; no hay implementación de código.
- **Estado:** Lista de tareas. El proceso de anotación comienza tras aprobar este
  documento.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** trabajo **humano** en Roboflow; el asset resultante (COCO)
> vive en el **volumen compartido** (`data/gt/eval_coco/`, git-ignored). Esta tarea
> **no versiona código**; los artefactos versionados son `spec.md` y `plan.md`.

---

## Fase A — Configuración del proyecto

- [ ] **T1 — Crear el proyecto Roboflow de GT**
  - Proyecto tipo **Instance Segmentation**, dedicado al GT de evaluación
    (separado de fine-tuning). Nombre propuesto: `futbot-eval-gt`.
  - Crear las clases: `robot_aliado`, `robot_rival`, `robot_desconocido`,
    `orange_ball`, `green_floor`.
  - **Verificación:** el proyecto existe con las 5 clases y tipo segmentación.
  - **Plan:** §2. **Spec:** AC-2.

- [ ] **T2 — Subir las imágenes preservando el nombre**
  - Subir los **600 PNG** de `data/testing_frames/` (copia local del volumen);
    confirmar que Roboflow **no** renombra ni reordena (`<video_id>_<frame_index>.png`).
  - **Verificación:** 600 imágenes cargadas con el `file_name` intacto.
  - **Plan:** §4.1. **Spec:** AC-1, AC-6.

---

## Fase B — Piloto (compuerta)

- [ ] **T3 — Anotar el piloto (30–50 imágenes)**
  - Anotar un subconjunto mezclando frames `aleatorio` y `cenital`, aplicando la
    guía del plan §3 (parciales ≥ 25 %, oclusiones solo visible, balón con blur,
    `green_floor` con líneas y sin gradas, ambiguos → `robot_desconocido`).
  - **Verificación:** el piloto queda anotado siguiendo la guía.
  - **Plan:** §3, §6.1. **Spec:** AC-4, AC-9.

- [ ] **T4 — Exportar y validar el piloto (gate)**
  - Generar versión **sin preprocessing/augmentation**, exportar **COCO
    Segmentation**, descargar y comprobar que carga y que el `file_name` enlaza con
    `assets/testing_frames.csv`.
  - **Verificación:** COCO piloto válido y trazabilidad sostenida. **Solo si pasa**
    se continúa a la Fase C.
  - **Plan:** §4.3, §4.4, §6.1. **Spec:** AC-5, AC-6.

---

## Fase C — Anotación completa

- [ ] **T5 — Anotar los frames restantes**
  - Completar la anotación del resto hasta los **600**, con la guía única. El trabajo
    **puede repartirse** entre el equipo manteniendo los mismos criterios.
  - **Verificación:** los 600 frames anotados (todas las instancias visibles).
  - **Plan:** §3, §4.2. **Spec:** AC-1, AC-2.

- [ ] **T6 — Verificar anti-circularidad**
  - Confirmar que **ninguna** máscara provino de SAM3 sin corrección humana (la
    asistencia tipo Smart Polygon, si se usó, fue revisada a mano).
  - **Verificación:** todas las máscaras tienen decisión final humana.
  - **Plan:** §3.4. **Spec:** AC-3.

---

## Fase D — Control de calidad

- [ ] **T7 — Muestra de verificación (segunda persona)**
  - Una segunda persona revisa una muestra del total (mínimo el piloto + un % del
    resto): clases correctas, máscaras ajustadas, reglas de la guía aplicadas,
    ambiguos como `robot_desconocido`.
  - **Verificación:** muestra revisada y correcciones aplicadas.
  - **Plan:** §6.2. **Spec:** AC-8.

---

## Fase E — Export final y asset

- [ ] **T8 — Export COCO final**
  - Generar la versión final (sin preprocessing/augmentation) y exportar **COCO
    Segmentation** con polígonos de las 5 clases.
  - **Verificación:** existe el export COCO final con las categorías de las 5 clases.
  - **Plan:** §4.3. **Spec:** AC-5.

- [ ] **T9 — Colocar el asset en el volumen compartido**
  - Descargar el zip y extraer el JSON en `data/gt/eval_coco/` (git-ignored) sobre el
    **volumen compartido** del pod.
  - **Verificación:** el COCO está en `data/gt/eval_coco/` y **no** aparece para
    versionar (git lo ignora).
  - **Plan:** §5. **Spec:** AC-7.

- [ ] **T10 — Documentar el mapeo de categorías**
  - Registrar la tabla `category_id` → clase que asignó Roboflow (insumo para
    `gt_loader`).
  - **Verificación:** existe la tabla id→clase accesible para la siguiente tarea.
  - **Plan:** §4.4. **Spec:** AC-2, AC-6.

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 proyecto + clases | §2 | AC-2 |
| T2 subir imágenes | §4.1 | AC-1, AC-6 |
| T3 anotar piloto | §3, §6.1 | AC-4, AC-9 |
| T4 export + validar piloto (gate) | §4.3, §4.4, §6.1 | AC-5, AC-6 |
| T5 anotar resto | §3, §4.2 | AC-1, AC-2 |
| T6 anti-circularidad | §3.4 | AC-3 |
| T7 QC muestra | §6.2 | AC-8 |
| T8 export final | §4.3 | AC-5 |
| T9 colocar asset | §5 | AC-7 |
| T10 mapeo categorías | §4.4 | AC-2, AC-6 |
