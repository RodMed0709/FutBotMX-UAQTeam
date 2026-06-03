# Tasks — Función de visualización de un conjunto de frames

- **Tarea atómica:** `frame_visualization`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Dependencias

- [x] **T1 — Asegurar `matplotlib` en `requirements.txt`**
  - Verificar que `matplotlib` esté listado en `requirements.txt`; agregarlo si
    falta.
  - **Verificación:** `requirements.txt` incluye `matplotlib` y `import
    matplotlib.pyplot` funciona en el entorno.
  - **Plan:** §2, §4. **Spec:** AC-1.

---

## Fase B — Función `show_frames`

- [x] **T2 — Definir la firma en `src/utils.py`**
  - Añadir `def show_frames(frames: np.ndarray) -> None:` en `src/utils.py`.
  - Importar `numpy` y `matplotlib.pyplot` (y `math.ceil` o equivalente).
  - **Verificación:** la función existe con la firma indicada y el módulo importa
    sin errores.
  - **Plan:** §3.1, §3.2. **Spec:** AC-1.

- [x] **T3 — Validación de entrada y caso vacío**
  - Comprobar que `frames` es un `np.ndarray` 4D `(N, H, W, 3)`; en caso contrario
    lanzar `ValueError` con mensaje claro.
  - Si `N == 0`, emitir un **aviso** y `return` sin mostrar nada (sin excepción).
  - **Verificación:** entrada no 4D / no `ndarray` lanza `ValueError`; array vacío
    avisa y retorna sin renderizar ni abortar.
  - **Plan:** §3.6. **Spec:** AC-7.

- [x] **T4 — Selección de los frames a mostrar**
  - `N = frames.shape[0]`. Si `N <= 6` usar todos; si `N > 6`, calcular 6 índices
    equiespaciados con `np.linspace(0, N - 1, 6)` redondeados a `int`, preservando
    el orden.
  - **Verificación:** con `N > 6` se eligen 6 índices repartidos uniformemente; con
    `0 < N <= 6` se usan todos, en orden de llegada.
  - **Plan:** §3.3. **Spec:** AC-2, AC-3, AC-4, AC-5.

- [x] **T5 — Construcción y render de la cuadrícula**
  - `ncols = min(n_mostrados, 3)`, `nrows = ceil(n_mostrados / 3)`; crear
    `plt.subplots(nrows, ncols)`.
  - Para cada eje: `ax.imshow(frame)` + `ax.axis("off")`; ocultar los ejes
    sobrantes. Cerrar con `plt.tight_layout()` y `plt.show()`.
  - **Verificación:** se muestra una cuadrícula (6 frames → 2×3) que se adapta a
    cantidades menores; la función retorna `None` y no escribe a disco.
  - **Plan:** §3.4, §3.5. **Spec:** AC-2, AC-3, AC-6.

---

## Fase C — Notebook de validación

- [x] **T6 — Crear `notebooks/02_frame_visualization_demo.ipynb`**
  - Crear la notebook con **todos** los casos de prueba (sin ejecutarla):
    1. **N > 6** → se muestran 6 frames uniformes.
    2. **N == 6** → cuadrícula 2×3 completa.
    3. **0 < N < 6** → todos los frames, cuadrícula adaptada (p. ej. `N=4`, `N=2`,
       `N=1`).
    4. **N == 0** → aviso, sin render ni excepción.
    5. **Entrada inválida** (no 4D / no `ndarray`) → `ValueError`.
  - Origen de frames: `extract_frames` sobre un `.MOV` real (solo en contenedor) y
    arrays sintéticos (`np.zeros`/`np.random`) con forma `(N, H, W, 3)` para los
    demás casos.
  - **Verificación:** la notebook existe en `notebooks/` y contiene los 5 casos
    descritos. **No se ejecuta** en esta tarea.
  - **Plan:** §5. **Spec:** AC-8.

---

## Fase D — Validación manual (a cargo del usuario)

- [ ] **T7 — Ejecutar y validar la notebook**
  - Ejecutar `notebooks/02_frame_visualization_demo.ipynb` en un entorno con render
    (GUI / Jupyter inline) y confirmar visualmente cada caso.
  - **Verificación:** los 5 casos se comportan como se espera; criterios AC-1 a
    AC-8 del spec satisfechos.
  - **Plan:** §5, §7. **Spec:** AC-8.
  - **Responsable:** usuario.

---

## Resumen de trazabilidad

| Tarea | Plan | Criterios de aceptación (spec) |
|---|---|---|
| T1 | §2, §4 | AC-1 |
| T2 | §3.1, §3.2 | AC-1 |
| T3 | §3.6 | AC-7 |
| T4 | §3.3 | AC-2, AC-3, AC-4, AC-5 |
| T5 | §3.4, §3.5 | AC-2, AC-3, AC-6 |
| T6 | §5 | AC-8 |
| T7 | §5, §7 | AC-8 |

---

## Nota de metodología

Este documento cierra el paso 4. La **implementación (paso 5)** de estas tareas
ocurrirá únicamente cuando se indique explícitamente; hasta entonces no se crea
ni modifica código fuente.
