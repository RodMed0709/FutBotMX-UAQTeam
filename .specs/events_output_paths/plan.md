# Plan — Rutas de salida dedicadas para eventos (`events_output_paths`)

- **Tarea atómica:** `events_output_paths`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Estado:** Define el *cómo*. **No** implica escribir código todavía (eso es el paso 5,
  habilitado por `tasks.md`).
- **Spec de referencia:** `.specs/events_output_paths/spec.md`.

---

## 1. Enfoque general

Replicar el patrón ya probado de `inference_schema.inference_paths`: una **función pura**
que construye una ruta a partir del `stem` del video, sin tocar el disco. Vive en un
**módulo nuevo** `src/core/events_schema.py` para no mezclar dominios con el esquema de
inferencia. Luego se **migran** los módulos fase_5 para que su ruta por defecto
(`output_path=None`) salga de este helper en vez de escribir plano en `outputs/`.

La migración es de **bajo riesgo**: los módulos ya aceptan un `output_path` explícito; solo
cambia **cómo se calcula el default** cuando no se pasa ruta. La firma pública de cada
función no cambia (sigue siendo `output_path: ... | None = None`).

---

## 2. Diseño del helper

Archivo: `src/core/events_schema.py`

```python
from pathlib import Path
from src.utils import PROJECT_ROOT

def events_paths(
    stem: str,
    kind: str,
    ext: str,
    *,
    outputs_dir: str = "outputs",
    namespace: str | None = None,
) -> Path:
    """Ruta de un producto de eventos: outputs/eventos/[<namespace>/]<stem>/<stem>_<kind>.<ext>."""
    base = PROJECT_ROOT / outputs_dir / "eventos"
    if namespace:
        base = base / namespace
    base = base / stem
    return base / f"{stem}_{kind}.{ext}"
```

Decisiones:
- **Función pura**, sin `mkdir` (igual que `inference_paths`); el escritor de cada módulo
  ya hace `path.parent.mkdir(parents=True, exist_ok=True)`.
- **`outputs_dir` por parámetro** con default `"outputs"` (la convención del repo); resuelto
  contra `PROJECT_ROOT`. No hardcodea rutas absolutas.
- **`ext` sin punto** (`"json"`, `"mp4"`, `"png"`); el helper agrega el punto.
- **`namespace`** opcional, insertado antes del `<stem>` (simetría con `inference_paths`).
- Sin dependencias nuevas (solo `pathlib` y `PROJECT_ROOT`).

---

## 3. Migración de los módulos fase_5

Para cada módulo, el cambio es el mismo patrón: cuando `output_path is None`, derivar la
ruta con `events_paths(stem, kind, ext)` en vez de `Path("outputs") / f"{kind}_{stem}.ext"`.
El `stem` se obtiene del `tracks_json`/`source` que ya recibe cada función.

| Módulo | Función que escribe | `kind` propuesto | ext |
|---|---|---|---|
| `metric_positions.py` | (json + png de viz) | `metric_positions` | `json`, `png` |
| `event_goal_geometric.py` | `write_geometric_goals_json` (+ png) | `goal_geometric` | `json`, `png` |
| `metric_kinematics.py` | (json + png) | `metric_speed_distance` | `json`, `png` |
| `metric_heatmap.py` | (png balón / robot) | `heatmap_ball`, `heatmap_robot` | `png` |
| `metric_field_zones.py` | (json + png por esquema) | `field_zones_<esquema>` | `json`, `png` |
| `demo_overlay.py` | `compose_demo` | `demo` | `mp4` |

Notas de migración:
- **No cambiar la firma pública**: cada función conserva su `output_path=None`; solo cambia
  el cálculo del default interno.
- Donde hoy el **test** pasa la ruta a mano (`outputs/...`), pasar a **no pasar ruta** y
  dejar que el módulo derive el default (así se ejercita el helper); o pasar la ruta ya
  construida con `events_paths(...)` cuando el test necesite controlarla.
- El `stem` para el `kind` con sufijo variable (heatmap balón/robot, zonas por esquema) se
  arma en el módulo (p. ej. `f"heatmap_{cual}"`, `f"field_zones_{esquema}"`) y se pasa como
  `kind` a `events_paths`.

---

## 4. Test manual

Archivo: `testing/test_events_output_paths.py` (estilo del repo: script directo, sin pytest,
sin GPU). Verifica:
1. Estructura: la ruta termina en `outputs/eventos/<stem>/<stem>_<kind>.<ext>`.
2. `kind` y `ext` aparecen correctos en el nombre del archivo.
3. `namespace` se inserta antes del `<stem>` cuando se pasa.
4. La ruta es **absoluta** y está bajo `PROJECT_ROOT`.
5. El helper **no** crea carpetas (la ruta padre no existe tras llamarlo, salvo que ya
   existiera).

Sin tocar SAM3/GPU ni leer videos: es puramente construcción de rutas.

---

## 5. Riesgos y mitigación

- **Romper un test fase_5 que asume la ruta plana.** Mitigación: migrar módulo y test en el
  mismo cambio; revisar cada `testing/test_*` afectado.
- **Salidas viejas en la raíz de `outputs/`.** Quedan como están (git-ignored); no se borran
  en esta tarea. Las corridas nuevas escriben ya en `outputs/eventos/<stem>/`.
- **Doble fuente de verdad de `outputs_dir`.** Se mantiene el mismo default `"outputs"` que
  usa el resto; si una capa superior ya tiene el `outputs_dir` del config, se lo pasa.

---

## 6. Archivos afectados

- **Nuevo:** `src/core/events_schema.py`, `testing/test_events_output_paths.py`.
- **Modificados (default de ruta):** `src/core/metric_positions.py`,
  `event_goal_geometric.py`, `metric_kinematics.py`, `metric_heatmap.py`,
  `metric_field_zones.py`, `demo_overlay.py` y sus `testing/test_*` correspondientes.
- **Sin tocar:** `inference_schema.py` y todo `outputs/inference/...`.
