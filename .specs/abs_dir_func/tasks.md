# Tasks — Utilidad de resolución de rutas absolutas (`src/utils.py`)

- **Tarea atómica:** `abs_dir_func`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Módulo y función

- [x] **T1 — Crear el esqueleto de `src/utils.py`**
  - Crear el módulo `src/utils.py` como contenedor de funciones generales.
  - Importar `pathlib.Path` y definir `PROJECT_ROOT` a partir de la ubicación del
    módulo: `Path(__file__).resolve().parents[1]`.
  - **Verificación:** `src/utils.py` existe y `PROJECT_ROOT` apunta a la raíz del
    proyecto.
  - **Plan:** §3.1, §3.2. **Spec:** AC-1, AC-4.

- [x] **T2 — Definir la firma de `get_abs_path`**
  - Declarar `get_abs_path(relative_path: str) -> Path`.
  - **Verificación:** la función existe con la firma `str → Path`.
  - **Plan:** §3.1. **Spec:** AC-2.

- [x] **T3 — Validación de entrada (`ValueError`)**
  - Lanzar `ValueError` si `relative_path` no es `str`, está vacío/solo espacios,
    o es una ruta **absoluta**.
  - **Verificación:** entradas inválidas lanzan `ValueError` y detienen el
    proceso.
  - **Plan:** §3.3 (paso 1), §3.4.

- [x] **T4 — Resolución y verificación de existencia (`FileNotFoundError`)**
  - Resolver `abs_path = (PROJECT_ROOT / relative_path).resolve()`.
  - Si `abs_path` no existe, lanzar `FileNotFoundError` (detiene el proceso).
  - Devolver `abs_path` (un `Path` absoluto).
  - **Verificación:** una ruta relativa válida y existente devuelve su `Path`
    absoluto; una inexistente lanza `FileNotFoundError`.
  - **Plan:** §3.3 (pasos 2-4), §3.4. **Spec:** AC-3, AC-5.

---

## Fase B — Script de prueba

- [x] **T5 — Crear `testing/test_abs_dir_func.py`**
  - Leer del `.env` la variable `CONFIG_FILENAME` aplicando `strip()` a clave y
    valor (el `.env` la tiene como `CONFIG_FILENAME =...`).
  - Construir la ruta relativa `configs/<CONFIG_FILENAME>` y resolverla a absoluta
    con `get_abs_path(...)`; leer el JSON de configuración.
  - Tomar de `working_dirs` las rutas `dataset_dir` y `sam3_dir` y resolverlas a
    absolutas con la función.
  - Intentar abrir un archivo `.MOV` dentro de `dataset_dir` con `cv2` y reportar
    si se pudo leer.
  - Imprimir en consola todas las rutas absolutas resueltas y reportar su
    existencia.
  - Capturar `FileNotFoundError` por ruta y reportar **sin abortar** el resto de
    la demostración (caso local: `data/raw`/`assets/sam3` son symlinks del
    contenedor).
  - **Verificación:** el script existe y, al ejecutarse, imprime las rutas
    absolutas, el estado de existencia y el resultado de la lectura del `.MOV`,
    sin abortar ante rutas faltantes.
  - **Plan:** §4. **Spec:** AC-6.

---

## Fase C — Validación manual (a cargo del usuario)

- [ ] **T6 — Ejecutar y validar manualmente**
  - Ejecutar `python testing/test_abs_dir_func.py` e inspeccionar la salida.
  - Confirmar que las rutas absolutas de configuración y de `working_dirs` son
    las esperadas.
  - **Verificación:** salida coherente; criterios AC-1 a AC-6 del spec
    satisfechos.
  - **Plan:** §4. **Spec:** AC-6.
  - **Responsable:** usuario.

---

## Resumen de trazabilidad

| Tarea | Plan | Criterios de aceptación (spec) |
|---|---|---|
| T1 | §3.1, §3.2 | AC-1, AC-4 |
| T2 | §3.1 | AC-2 |
| T3 | §3.3, §3.4 | — |
| T4 | §3.3, §3.4 | AC-3, AC-5 |
| T5 | §4 | AC-6 |
| T6 | §4 | AC-6 |

---

## Nota de metodología

Este documento cierra el paso 4. La **implementación (paso 5)** de estas tareas
ocurrirá únicamente cuando se indique explícitamente; hasta entonces no se crea
ni modifica código fuente.
