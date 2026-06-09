# Plan técnico — Función de visualización de un conjunto de frames

- **Tarea atómica:** `frame_visualization`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar una función general de utilidades que
**muestre** un conjunto de frames en una **cuadrícula** —hasta 6 frames; todos si
hay menos de 6— recibiendo como entrada una **matriz NumPy 4D** `(N, H, W, 3)` y
usando `matplotlib`. La función **solo visualiza** (no guarda ni retorna frames).
Además, definir la **notebook** de validación manual.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Visualización:** `matplotlib` (`matplotlib.pyplot`) — agregar a
  `requirements.txt` si no está presente.
- **Cálculo numérico / entrada:** `numpy` (ya en `requirements.txt`); la entrada es
  un `np.ndarray` 4D.
- **Sin dependencia de torch** ni de lectura de vídeo (la función opera sobre
  frames ya en memoria).
- **Sin lectura de configuración** (`.json`/`.env`): esta función no resuelve
  rutas ni cuotas.

---

## 3. Diseño de la función

### 3.1 Ubicación y módulo

- Es una **utilidad general**, por lo que vive en `src/utils.py` (junto a
  `get_abs_path`), no en `src/core/`.

### 3.2 Firma

```python
def show_frames(frames: np.ndarray) -> None:
    ...
```

- `frames: np.ndarray` — matriz **4D** con forma `(N, H, W, 3)` (frames RGB),
  típicamente el resultado de `src.core.frame_extraction.extract_frames`.
- **Retorno:** `None`. Su efecto es **visual** (render de la cuadrícula); no
  guarda en disco ni devuelve los frames.

### 3.3 Selección de los frames a mostrar

- Sea `N = frames.shape[0]`.
- **`N == 0`** → no hay nada que mostrar (ver §3.6).
- **`0 < N <= 6`** → mostrar **todos** los `N` frames.
- **`N > 6`** → seleccionar **6** índices **equiespaciados** con
  `np.linspace(0, N - 1, 6)` redondeados a entero (mismo criterio uniforme que
  `frame_extraction`), preservando el orden.

### 3.4 Disposición de la cuadrícula

- Se usa `matplotlib.pyplot.subplots(nrows, ncols)` con un máximo de **3
  columnas**:
  - `ncols = min(n_mostrados, 3)`
  - `nrows = ceil(n_mostrados / 3)`
- Caso de **6 frames** → cuadrícula **2×3**. Casos con menos frames adaptan filas
  y columnas (p. ej. `N=4` → 2×3 con un eje sobrante oculto; `N=2` → 1×2).
- Cada subgráfico muestra un frame con `ax.imshow(frame)` y oculta los ejes con
  `ax.axis("off")`; opcionalmente un título con el índice original del frame.
- Los **ejes sobrantes** (cuando `nrows*ncols > n_mostrados`) se ocultan.
- Render final con `plt.tight_layout()` y `plt.show()`.

### 3.5 Pseudocódigo

```python
def show_frames(frames: np.ndarray) -> None:
    # 1. Validar la entrada (ver §3.6)
    # 2. N = frames.shape[0]; si N == 0 -> avisar y return
    # 3. Seleccionar índices: todos si N <= 6, si no np.linspace(0, N-1, 6)
    # 4. n = len(indices); ncols = min(n, 3); nrows = ceil(n / 3)
    # 5. fig, axes = plt.subplots(nrows, ncols)
    # 6. Para cada eje: imshow(frame) + axis("off"); ocultar ejes sobrantes
    # 7. plt.tight_layout(); plt.show()
```

### 3.6 Validación de entrada y manejo de errores

| Situación | Comportamiento |
|---|---|
| Entrada no es `np.ndarray`, o no es 4D `(N,H,W,3)` | `ValueError` con mensaje claro |
| `N == 0` (array vacío) | **Aviso** (`print`/`warnings.warn`) y `return` sin mostrar nada; **no** lanza excepción |
| `0 < N < 6` | Muestra todos los frames |
| `N >= 6` | Muestra 6 frames equiespaciados |

- La validación de forma/tipo se realiza al inicio; el caso vacío se trata como
  flujo normal (aviso), no como error, conforme al spec (AC-7).

---

## 4. Cambios de configuración

- **Ninguno** en `.json`/`.env`: la función no lee rutas ni parámetros de
  configuración.
- **`requirements.txt`:** asegurar la presencia de `matplotlib`.

---

## 5. Validación manual (notebook)

- **Entregable:** una **notebook Jupyter** en `notebooks/` (p. ej.
  `notebooks/frame_visualization_demo.ipynb`) que contenga **todos los casos de
  prueba** para comprobar que la función es correcta. Se **crea** en el paso 5
  (implementación); **no se ejecuta** como parte de esta planificación.
- **Casos de prueba que debe contener la notebook:**
  1. **N > 6:** construir/obtener un conjunto con más de 6 frames y verificar que
     se muestran **6** repartidos uniformemente.
  2. **N == 6:** verificar la cuadrícula **2×3** completa.
  3. **0 < N < 6:** verificar que se muestran **todos** y la cuadrícula se adapta
     (p. ej. `N=4`, `N=2`, `N=1`).
  4. **N == 0:** verificar el **aviso** y que no se renderiza nada ni se lanza
     excepción.
  5. **Entrada inválida:** (no 4D / no `ndarray`) verificar que lanza
     `ValueError`.
- **Origen de los frames:** preferentemente reutilizar
  `extract_frames` sobre un `.MOV` real (recordar: solo resuelve **dentro del
  contenedor**); para los casos sin vídeo pueden usarse arrays sintéticos
  (`np.random`/`np.zeros`) con forma `(N, H, W, 3)`.
- **Entorno:** la notebook está pensada para entornos con **GUI / capacidad de
  render** o **Jupyter**, conforme al borrador.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Función presente | §3.1, §3.2 | `src/utils.py::show_frames` |
| AC-2 Máximo 6 en cuadrícula | §3.3, §3.4 | 6 frames, layout 2×3 |
| AC-3 Menos de 6 | §3.3, §3.4 | Muestra todos, cuadrícula adaptada |
| AC-4 Selección uniforme | §3.3 | `np.linspace(0, N-1, 6)` |
| AC-5 Orden preservado | §3.3 | Índices crecientes, sin reordenar |
| AC-6 Solo visualiza | §3.2 | Retorno `None`, sin escritura a disco |
| AC-7 Entrada vacía | §3.6 | Aviso + `return`, sin excepción |
| AC-8 Validación manual | §5 | Notebook con todos los casos |

---

## 7. Riesgos y consideraciones

- **Backend de matplotlib:** la visualización requiere un backend con capacidad de
  render (GUI o el backend *inline* de Jupyter). En un contenedor sin display, la
  ventana interactiva no aparecerá; por eso la validación se entrega como
  **notebook** (entorno con render), conforme al borrador.
- **Forma homogénea:** se asume que todos los frames del array comparten `(H, W)`
  (provienen de un mismo vídeo), condición ya garantizada por la salida de
  `extract_frames`.
- **Consumo de memoria:** la función no copia los frames; solo selecciona índices
  para mostrar, por lo que el coste adicional es mínimo.
- **Acoplamiento mínimo:** la función no depende de configuración ni de rutas, lo
  que la mantiene como utilidad general reutilizable.

---

## 8. Siguiente paso (metodología)

Elaborar `tasks.md` con la descomposición en tareas ejecutables. La
implementación (paso 5) ocurre únicamente después de definir las tareas.
