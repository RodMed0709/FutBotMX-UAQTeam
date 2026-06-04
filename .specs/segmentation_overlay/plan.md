# Plan técnico — Visualización multi-clase de detecciones (`segmentation_overlay`)

- **Tarea atómica:** `segmentation_overlay`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** no hay draft previo para este plan.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar el overlay multi-clase: una función
`overlay_detections` que pinta las máscaras por color de clase y **devuelve** el
frame compuesto como `uint8 (H,W,3)`, y `show_overlay` que lo muestra con leyenda
(matplotlib). Colores y alpha provienen de la configuración. Además, definir el
cambio de config (alpha por defecto) y los dos artefactos de validación (script
headless + notebook).

---

## 2. Stack técnico

- **Python:** 3.11.
- **Composición:** `numpy` (mezcla alpha en float). `overlay_detections` es
  **numpy-only** (sin torch/cv2/PIL).
- **Display:** `matplotlib` (`imshow` + `matplotlib.patches.Patch` para la
  leyenda), importado **perezosamente** dentro de `show_overlay`.
- **Configuración:** `json` (estándar) + `src.utils.get_abs_path` para leer
  `classes[].color` y `visualization.overlay_alpha`, con la convención del repo.
- **Tipo de detección:** `Detection` de `src.core.segmentation` (solo se usa
  `det.mask`).

> `import src.core` sigue ligero: `numpy`/`warnings` a nivel de módulo; matplotlib
> solo dentro de `show_overlay`.

---

## 3. Diseño

### 3.1 Ubicación y módulo

- Archivo nuevo: `src/core/overlay.py`.
- Exportación en `src/core/__init__.py`:
  `from src.core.overlay import overlay_detections, show_overlay` y sumarlos a
  `__all__`.

### 3.2 Firmas

```python
def overlay_detections(
    frame: np.ndarray,
    detections_by_class: dict[str, list[Detection]],
    classes: list[dict] | None = None,
    alpha: float | None = None,
) -> np.ndarray: ...  # uint8 (H, W, 3)

def show_overlay(
    frame: np.ndarray,
    detections_by_class: dict[str, list[Detection]],
    classes: list[dict] | None = None,
    alpha: float | None = None,
) -> None: ...
```

### 3.3 Lectura de configuración (colores + alpha)

`_load_overlay_config()` lee la config (leer `CONFIG_FILENAME` del `.env` con
`strip()` → `get_abs_path(f"configs/{...}")` → `json.load`) y devuelve:

- `classes`: lista de clases (para el mapa `name → color`).
- `default_alpha`: `config["visualization"]["overlay_alpha"]`.

Resolución de parámetros en las funciones:

- `classes = classes if classes is not None else <de config>`.
- `alpha = alpha if alpha is not None else default_alpha`.
- Mapa de color: `{cls["name"]: tuple(cls["color"]) for cls in classes}`
  (color `[r,g,b]` 0–255 → se normaliza a 0–1 al mezclar).

### 3.4 `overlay_detections` — composición

1. **Validar** `frame` (§3.6) → `H, W`.
2. Resolver `classes`, `alpha` y el mapa de colores (§3.3).
3. **Copiar** a float: `out = frame.astype(np.float32) / 255.0` (no muta la
   entrada).
4. Por cada `name, dets` en `detections_by_class` y cada `det`:
   - `color01 = np.array(color_map[name], np.float32) / 255.0`.
   - `mask = det.mask`; si `mask.shape != (H, W)` → `warnings.warn` y **omitir**.
   - `out[mask] = (1 - alpha) * out[mask] + alpha * color01`.
5. **Devolver** `(out * 255.0).round().clip(0, 255).astype(np.uint8)`.

Si `detections_by_class` está vacío, el bucle no pinta nada y se devuelve la
copia del frame (en uint8).

### 3.5 `show_overlay` — display con leyenda

1. `composed = overlay_detections(frame, detections_by_class, classes, alpha)`.
2. Importar matplotlib **dentro** de la función.
3. `plt.imshow(composed)`, ejes ocultos.
4. **Leyenda:** una `mpatches.Patch(color=<color01>, label=<name>)` por clase
   presente; `plt.legend(handles=...)`.
5. `plt.show()`. No devuelve nada ni escribe a disco.

### 3.6 Manejo de errores

| Situación | Comportamiento |
|---|---|
| `frame` no `ndarray` o no `(H, W, 3)` | `ValueError` |
| `detections_by_class` vacío | devuelve copia del frame (no error) |
| `mask.shape != (H, W)` | `warnings.warn` + se omite esa detección |
| Clase sin color en config | `KeyError` (mensaje claro) |
| `CONFIG_FILENAME`/config/clave ausente | `ValueError`/`FileNotFoundError`/`KeyError` (propagados) |

---

## 4. Cambios de configuración

- **`configs/00_testing_config.json`**: agregar el bloque `visualization` con el
  alpha por defecto (edición aditiva; el resto intacto).

```json
"visualization": { "overlay_alpha": 0.55 }
```

---

## 5. Validación

### 5.1 Script headless — `testing/test_overlay.py`

- **Sin gráficos ni modelo.** Construye un frame sintético (p. ej. `np.zeros`/ruido
  `(H, W, 3) uint8`) y un `detections_by_class` con `Detection` de máscaras
  **sintéticas** (rectángulos booleanos), pasando `classes` explícitas (colores de
  prueba) para no depender de la config.
- Llama `overlay_detections` y verifica:
  - forma `(H, W, 3)` y `dtype uint8`;
  - los píxeles **bajo la máscara** cambian respecto al frame original y se acercan
    al color de la clase; los de fuera no cambian;
  - el frame de entrada **no se muta**.
- **No** llama a `show_overlay` (evita matplotlib). El **agente lo ejecuta** en
  local.

### 5.2 Notebook — `notebooks/fase_0/06_segmentation_overlay_check.ipynb`

- Inspección **visual**: extrae un frame real, obtiene detecciones (de
  `detect_classes_in_frame`, en RunPod/GPU) y llama `show_overlay` para ver el
  resultado con leyenda. Lo **crea el agente**; lo ejecuta el usuario.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Módulo y piezas | §3.1, §3.2 | `overlay.py` + export |
| AC-2 Devuelve array compuesto | §3.4 | `uint8 (H,W,3)` |
| AC-3 Color por clase desde config | §3.3 | `classes[].color` por `name` |
| AC-4 Alpha desde config con override | §3.3, §4 | `visualization.overlay_alpha` |
| AC-5 Display con leyenda | §3.5 | `show_overlay` + `Patch` |
| AC-6 Sin escritura a disco | §3.4, §3.5 | devuelve array / solo muestra |
| AC-7 Casos vacíos | §3.4, §3.6 | dict vacío → copia del frame |
| AC-8 Validación doble | §5.1, §5.2 | script headless + notebook |

---

## 7. Riesgos y consideraciones

- **Indexado booleano `out[mask]`:** asume `mask` booleana `(H, W)`; el chequeo
  defensivo (§3.4) descarta máscaras de forma incorrecta para no romper el
  indexado ni desalinear.
- **uint8 tras mezcla:** el `round().clip(0,255)` evita desbordes/truncamientos;
  la mezcla se hace en float para no perder precisión (bandas de color).
- **Colores solo de clases presentes:** la leyenda y el pintado usan las clases que
  aparecen en `detections_by_class`; clases sin detecciones no estorban.
- **Desviación del roadmap:** overlay en `src/core/overlay.py` (no en `utils`),
  documentado en el spec §6, para evitar el ciclo `utils → core`.
- **Notebook requiere GPU para datos reales:** la parte visual con detecciones
  reales depende de correr la segmentación en RunPod; el script headless no.
