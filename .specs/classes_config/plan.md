# Plan técnico — Clases del modelo en la configuración (`classes_config`)

- **Tarea atómica:** `classes_config`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** no hay draft previo para este plan.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente
  (esta tarea solo edita un archivo de configuración `.json`).

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo centralizar las clases del proyecto en
`configs/00_testing_config.json`: la estructura JSON exacta (clave, esquema por
clase, representación del color), el bloque concreto con las 3 clases del MVP, el
patrón de acceso que usarán los consumidores (sin añadir código en esta tarea) y
la validación ligera del resultado.

---

## 2. Stack técnico

- **Archivo de configuración:** `configs/00_testing_config.json` (JSON, ya
  existente). Edición **aditiva**.
- **Validación:** `python -m json.tool` (módulo estándar; no se crea código nuevo).
- **Patrón de acceso (consumidores, fuera de esta tarea):** lectura de
  `CONFIG_FILENAME` desde `.env` con `strip()`, `src/utils.py::get_abs_path` para
  resolver `configs/<CONFIG_FILENAME>` y `json` para parsear — la misma convención
  que ya usan `frame_extraction.py` y `sam3_loader.py`.

> Esta tarea **no** introduce dependencias ni código; solo datos de configuración.

---

## 3. Diseño del bloque de configuración

### 3.1 Ubicación y tipo

- Nueva clave de **nivel superior** `"classes"` en
  `configs/00_testing_config.json`, hermana de `working_dirs` y `preprocess`.
- `"classes"` es un **array JSON** (lista **ordenada**; el orden es significativo).

### 3.2 Esquema por clase

Cada elemento del array es un objeto con:

| Campo | Tipo | Notas |
|---|---|---|
| `name` | string | Identificador estilo COCO (p. ej. `"green_floor"`). |
| `sam3_prompts` | array de strings (≥1) | Candidatos de prompt; **el primero es el activo**. |
| `color` | array `[r, g, b]` (int 0–255) | JSON no tiene tuplas; el consumidor convierte a `tuple(...)` si lo requiere. |
| `coco_id` | int | Categoría COCO (para fase 1 / `coco_autoannotate`). |

### 3.3 Bloque concreto (3 clases del MVP)

```json
"classes": [
  { "name": "robot",       "sam3_prompts": ["robot"],       "color": [60, 130, 255], "coco_id": 1 },
  { "name": "orange_ball", "sam3_prompts": ["orange ball"], "color": [255, 100, 0],  "coco_id": 2 },
  { "name": "green_floor", "sam3_prompts": ["green playing surface with lines", "green floor"], "color": [50, 220, 70], "coco_id": 3 }
]
```

Resultado del archivo completo tras la edición (aditiva; `working_dirs` y
`preprocess` intactos):

```json
{
  "working_dirs": {
    "dataset_dir": "data/raw",
    "sam3_dir": "assets/sam3"
  },
  "preprocess": {
    "fps": "1",
    "frame_quota": 30
  },
  "classes": [
    { "name": "robot",       "sam3_prompts": ["robot"],       "color": [60, 130, 255], "coco_id": 1 },
    { "name": "orange_ball", "sam3_prompts": ["orange ball"], "color": [255, 100, 0],  "coco_id": 2 },
    { "name": "green_floor", "sam3_prompts": ["green playing surface with lines", "green floor"], "color": [50, 220, 70], "coco_id": 3 }
  ]
}
```

### 3.4 Patrón de acceso (documentación, sin código en esta tarea)

Los consumidores (tareas 3, 4, 5, `coco_autoannotate`) leerán las clases así:

```python
# convención ya usada en el repo
classes = config["classes"]                  # lista ordenada
for cls in classes:
    active_prompt = cls["sam3_prompts"][0]    # primero = activo
    color = tuple(cls["color"])               # [r,g,b] -> (r,g,b)
    name, coco_id = cls["name"], cls["coco_id"]
```

- Para **experimentar con otros prompts** se editan/reordenan las entradas de
  `sam3_prompts` (el activo es siempre el primero).
- Si una tarea consumidora quiere encapsular esta lectura en un helper
  (`get_classes()` en `utils`), eso pertenece **a esa tarea**, no a `classes_config`
  (que es "sin código nuevo").

### 3.5 Extensibilidad

- Añadir un objeto nuevo a detectar es **agregar otra entrada** al array `classes`
  con los mismos campos. El esquema no cambia y los consumidores que iteran
  `config["classes"]` lo recogen automáticamente.

---

## 4. Cambios de configuración

- **`configs/00_testing_config.json`**: agregar la clave `"classes"` (§3.3). Es el
  único cambio de la tarea.

---

## 5. Validación

- Ejecutar `python -m json.tool configs/00_testing_config.json` (o equivalente)
  para confirmar que el archivo sigue siendo **JSON válido**.
- Inspeccionar que `classes` contiene las **3 clases** y que **cada una** tiene
  `name`, `sam3_prompts` (no vacío), `color` (RGB) y `coco_id`.
- Confirmar que `working_dirs` y `preprocess` permanecen intactos.
- Validación **manual y ligera**; no se crea ningún script.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Bloque de clases (lista) | §3.1 | clave `"classes"` como array |
| AC-2 Campos por clase | §3.2 | `name`, `sam3_prompts`, `color`, `coco_id` |
| AC-3 Clases del MVP | §3.3 | robot, orange_ball, green_floor |
| AC-4 Prompt activo por convención | §3.2, §3.4 | `sam3_prompts[0]`; green = "green playing surface with lines" |
| AC-5 Orden preservado | §3.1, §3.3 | array ordenado robot→orange_ball→green_floor |
| AC-6 Extensible a N clases | §3.5 | agregar entradas; validación por campos, no por conteo |
| AC-7 JSON válido | §5 | `python -m json.tool` |
| AC-8 Sin código nuevo | §2, §3.4 | solo edición del `.json` |

---

## 7. Riesgos y consideraciones

- **Color como lista, no tupla:** JSON no soporta tuplas; los consumidores que
  esperan `tuple` (p. ej. para matplotlib/cv2) deben convertir con `tuple(...)`.
  Se documenta en §3.4 para evitar errores de tipo aguas abajo.
- **Discrepancia de prompt con el roadmap:** se adopta `"green playing surface
  with lines"` como activo (notebooks 02/04) y se conserva `"green floor"` como
  candidato; documentado en el spec §6.
- **`coco_id` sin uso en el MVP por-frame:** se incluye ahora para no re-tocar la
  config al llegar a la fase 1 (`coco_autoannotate`); no afecta al MVP.
- **Acoplamiento por orden:** consumidores futuros (heurística `obj_id → clase` del
  tracking) pueden depender del orden de `classes`; mantenerlo estable.
