# Tasks — Aceptar rutas absolutas de vídeo en `extract_frames`

- **Tarea atómica:** `abs_video_path`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Corrección en `_resolve_video_path`

- [x] **T1 — Reescribir `_resolve_video_path` con lógica por ramas**
  - Conservar el chequeo inicial de tipo: si `video_path` no es `Path`, lanzar
    `ValueError`.
  - **Rama relativa** (`not video_path.is_absolute()`): delegar en
    `get_abs_path(str(video_path))` (comportamiento idéntico al actual).
  - **Rama absoluta** (`video_path.is_absolute()`): **eliminar** el intento de
    `relative_to(PROJECT_ROOT)` y el `ValueError` por "fuera del proyecto".
  - **Verificación:** una ruta absoluta a un vídeo existente fuera de
    `PROJECT_ROOT` ya **no** lanza `ValueError`; una ruta relativa sigue
    resolviéndose igual que antes.
  - **Plan:** §3.1, §3.3. **Spec:** AC-1, AC-2, AC-3.

- [x] **T2 — Validar la rama absoluta con `is_file()`**
  - En la rama absoluta: si `not video_path.is_file()` lanzar
    `FileNotFoundError` (cubre inexistente y directorio); devolver
    `video_path.resolve()` en caso válido.
  - **Verificación:** ruta absoluta inexistente → `FileNotFoundError`; ruta
    absoluta a un directorio → `FileNotFoundError`; ruta absoluta a archivo válido
    (incl. symlink con destino existente) → devuelve su ruta resuelta.
  - **Plan:** §3.3, §3.4, §3.5. **Spec:** AC-4, AC-5.

- [x] **T3 — Actualizar docstrings**
  - Reescribir el docstring de `_resolve_video_path` para el nuevo contrato
    (acepta relativas vía `get_abs_path` y absolutas a archivos válidos en
    cualquier ubicación).
  - Ajustar la sección Args/Raises de `extract_frames`: `video_path` admite rutas
    absolutas externas; las clases de excepción documentadas no cambian.
  - **Verificación:** los docstrings describen el comportamiento nuevo y no
    mencionan ya la restricción "dentro del proyecto" para rutas absolutas.
  - **Plan:** §3.6. **Spec:** AC-7.

---

## Fase B — Script de prueba

- [x] **T4 — Crear `testing/test_abs_video_path.py`**
  - Localizar un `.MOV` real (resolver `dataset_dir` con `get_abs_path` y `rglob`)
    y obtener su ruta **absoluta** con `.resolve()`.
  - Escenario 1 — **absoluta externa:** `extract_frames(abs_path, all_frames=False)`
    → reportar la forma del arreglo; no debe lanzar `ValueError`.
  - Escenario 2 — **relativa (regresión):** `extract_frames(Path("data/raw/.../x.MOV"))`
    → sigue funcionando.
  - Escenario 3 — **inexistente:** ruta absoluta falsa → esperar `FileNotFoundError`.
  - Escenario 4 — **directorio:** ruta absoluta de un directorio → esperar error.
  - Imprimir formas/`dtype`/excepciones capturadas; reportar **sin abortar** ante
    datos ausentes.
  - **Verificación:** el script existe e importa; al ejecutarse donde resuelvan los
    datos, imprime los 4 escenarios con el resultado esperado en cada uno.
  - **Plan:** §5. **Spec:** AC-9.

---

## Fase C — Validación manual (a cargo del usuario)

- [x] **T5 — Ejecutar y validar manualmente**
  - Ejecutar donde resuelvan los datos (contenedor o host con vídeos presentes):
    ```bash
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_abs_video_path.py
    ```
  - Confirmar: absoluta externa → frames; relativa → frames; inexistente →
    `FileNotFoundError`; directorio → error.
  - **Verificación:** salida coherente; criterios AC-1 a AC-9 del spec
    satisfechos.
  - **Plan:** §5, §7. **Spec:** AC-9.
  - **Responsable:** usuario.

---

## Resumen de trazabilidad

| Tarea | Plan | Criterios de aceptación (spec) |
|---|---|---|
| T1 | §3.1, §3.3 | AC-1, AC-2, AC-3 |
| T2 | §3.3, §3.4, §3.5 | AC-4, AC-5 |
| T3 | §3.6 | AC-7 |
| T4 | §5 | AC-9 |
| T5 | §5, §7 | AC-9 |

> **Nota:** AC-6 (tipo inválido → `ValueError`) y AC-8 (`get_abs_path` intacta) se
> cumplen por construcción: T1 conserva el chequeo de tipo y ninguna tarea
> modifica `get_abs_path`.

---

## Nota de metodología

Este documento cierra el paso 4. La **implementación (paso 5)** de estas tareas
ocurrirá únicamente cuando se indique explícitamente; hasta entonces no se crea
ni modifica código fuente.
