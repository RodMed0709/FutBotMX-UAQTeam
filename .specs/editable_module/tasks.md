# Tasks — Paquete `src` instalable en modo editable

- **Tarea atómica:** `editable_module`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Empaquetado

- [x] **T1 — Crear `src/__init__.py`**
  - Añadir `src/__init__.py` (vacío o con un docstring breve) para que `src` sea
    un paquete regular reconocible por `find_packages`.
  - **Verificación:** `src/__init__.py` existe; `import src` deja de depender de
    parches de `sys.path` una vez instalado el paquete.
  - **Plan:** §3.2. **Spec:** AC-1.

- [x] **T2 — Crear `setup.py` en la raíz**
  - Definir `setup(...)` con `name="futbotmx"`, `version="0.0.1"`,
    `python_requires=">=3.11"`, `packages=find_packages(include=["src", "src.*"])`
    y **sin** `install_requires` (las dependencias siguen en `requirements.txt`).
  - **Verificación:** `setup.py` existe; `find_packages(...)` incluye `src` y
    `src.core`; el nombre importable sigue siendo `src`.
  - **Plan:** §3.1. **Spec:** AC-2.

---

## Fase B — Instalación en Docker

- [x] **T3 — Añadir la instalación editable al `Dockerfile`**
  - Insertar `RUN pip install --no-cache-dir -e .` **después** de `COPY . .`.
  - **Verificación:** al reconstruir la imagen, el build instala el paquete en
    modo editable sin errores; el enlace queda en `/opt/venv`.
  - **Plan:** §4.1, §4.2. **Spec:** AC-5.

- [x] **T4 — Documentar/ejecutar la instalación editable en el venv local**
  - Documentar `pip install -e .` como paso único de preparación del venv local
    (ejecutarlo cuando se trabaje fuera del contenedor).
  - **Verificación:** tras `pip install -e .`, `import src.core` funciona desde un
    cwd distinto de la raíz en el venv local.
  - **Plan:** §5. **Spec:** AC-4.

---

## Fase C — Validación manual (a cargo del usuario)

- [x] **T5 — Validar en el contenedor**
  - Reconstruir y levantar, luego importar algo de `src`:
    ```bash
    docker compose --env-file .env -f docker/docker-compose.yml up --build -d
    docker compose --env-file .env -f docker/docker-compose.yml exec futbotmx26 \
      python -c "import src.core; from src.utils import get_abs_path; print('import OK')"
    ```
  - Opcional: ejecutar el import desde un cwd ajeno (p. ej. `/tmp`) para evidenciar
    que ya no depende de `sys.path`.
  - **Verificación:** la importación se completa sin error (imprime `import OK`);
    los cambios en `src/` se reflejan sin reinstalar (modo editable).
  - **Plan:** §6. **Spec:** AC-1, AC-3, AC-6.
  - **Responsable:** usuario.

---

## Resumen de trazabilidad

| Tarea | Plan | Criterios de aceptación (spec) |
|---|---|---|
| T1 | §3.2 | AC-1 |
| T2 | §3.1 | AC-2 |
| T3 | §4.1, §4.2 | AC-5 |
| T4 | §5 | AC-4 |
| T5 | §6 | AC-1, AC-3, AC-6 |

---

## Trabajo futuro (pendiente)

> Habilitado por esta tarea, pero **fuera de su alcance** (ver `spec.md` §3.2). Se
> aborda más adelante, idealmente como su propia tarea atómica con su flujo SDD.

- [ ] **TODO — Limpiar los scripts de `testing/` para no usar `sys.path`**
  - Ahora que `src` se instala como paquete editable, eliminar el parche
    `sys.path.insert(0, str(PROJECT_ROOT))` de los scripts existentes
    (`testing/test_abs_dir_func.py`, `testing/test_frame_extraction.py`) y dejar
    los `import src...` directos.
  - **Verificación:** los scripts se ejecutan correctamente en el contenedor sin
    el parche de `sys.path`.
  - **Pendiente:** definir como tarea atómica propia cuando se retome.

---

## Nota de metodología

Este documento cierra el paso 4. La **implementación (paso 5)** de estas tareas
ocurrirá únicamente cuando se indique explícitamente; hasta entonces no se crea
ni modifica código fuente.
