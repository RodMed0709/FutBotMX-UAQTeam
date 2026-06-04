# Tasks — Clases del modelo en la configuración (`classes_config`)

- **Tarea atómica:** `classes_config`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se modifica el archivo de configuración.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar. Esta tarea **no** añade código fuente: solo edita la config.

---

## Fase A — Edición de la configuración

- [x] **T1 — Añadir la clave `"classes"` a `configs/00_testing_config.json`**
  - Agregar la clave de nivel superior `"classes"` como **array ordenado**,
    hermana de `working_dirs` y `preprocess`, con las 3 clases del MVP y el
    esquema `name` / `sam3_prompts` (lista, primero = activo) / `color` `[r,g,b]`
    / `coco_id`, según §3.3 del plan:
    ```json
    "classes": [
      { "name": "robot",       "sam3_prompts": ["robot"],       "color": [60, 130, 255], "coco_id": 1 },
      { "name": "orange_ball", "sam3_prompts": ["orange ball"], "color": [255, 100, 0],  "coco_id": 2 },
      { "name": "green_floor", "sam3_prompts": ["green playing surface with lines", "green floor"], "color": [50, 220, 70], "coco_id": 3 }
    ]
    ```
  - Mantener `working_dirs` y `preprocess` **intactos** (edición aditiva).
  - **Verificación:** el archivo contiene `classes` con las 3 clases; cada una con
    `name`, `sam3_prompts` (no vacío), `color` (RGB) y `coco_id`; orden robot →
    orange_ball → green_floor; el primer prompt de `green_floor` es
    `"green playing surface with lines"`.
  - **Plan:** §3.1, §3.2, §3.3. **Spec:** AC-1, AC-2, AC-3, AC-4, AC-5, AC-8.

---

## Fase B — Validación

- [x] **T2 — Validar el JSON**
  - Ejecutar `python -m json.tool configs/00_testing_config.json` (o equivalente)
    y confirmar que no hay errores de sintaxis.
  - Confirmar que `working_dirs` y `preprocess` siguen presentes e intactos.
  - **Verificación:** el comando no reporta errores; el archivo sigue siendo JSON
    válido y conserva el resto de su contenido.
  - **Plan:** §5. **Spec:** AC-6, AC-7.

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 añadir `classes` | §3.1–§3.3 | AC-1…AC-5, AC-8 |
| T2 validar JSON | §5 | AC-6, AC-7 |
