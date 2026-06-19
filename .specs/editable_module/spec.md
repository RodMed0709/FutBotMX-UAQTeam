# Spec — Paquete `src` instalable en modo editable

- **Tarea atómica:** `editable_module`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el proyecto,
> **quiero** que el módulo `src` y sus submódulos se instalen en el entorno como
> un **módulo editable**,
> **para** poder importarlos de forma sencilla y estable desde cualquier parte del
> proyecto (notebooks, scripts de prueba, futuros módulos) sin manipular
> `sys.path` ni depender del directorio de trabajo.

---

## 2. Motivación (por qué)

- Hoy, para importar `src` desde fuera de la raíz (p. ej. en `testing/`), los
  scripts insertan manualmente `PROJECT_ROOT` en `sys.path`. Es frágil y se
  repite en cada archivo.
- Un paquete **editable** habilita `from src.core import extract_frames` (y
  `from src.utils import get_abs_path`) desde **cualquier** ubicación, incluidos
  los **notebooks**, que son un caso de uso central del proyecto.
- "Editable" garantiza que los cambios en `src/` se reflejan **sin reinstalar**,
  lo que encaja con el desarrollo iterativo y mantiene la reproducibilidad
  (la instalación es parte del entorno, no un parche por archivo).

---

## 3. Alcance

### 3.1 Dentro de alcance

- Hacer que el paquete `src` (con sus submódulos, p. ej. `src.core`, `src.utils`)
  sea **importable como módulo editable** en el entorno.
- Que la instalación editable funcione **tanto en el contenedor como en el venv
  local**, de forma reproducible.
- Que la instalación editable forme parte de la **preparación del entorno del
  contenedor** (no un paso manual oculto).

### 3.2 Fuera de alcance

- **Reorganizar o renombrar** el código existente de `src/` (`utils.py`,
  `core/`); el nombre del paquete importable se conserva como `src`.
- Añadir nuevas funciones o lógica del pipeline.
- **Migrar obligatoriamente** los scripts de `testing/` para eliminar su parche de
  `sys.path` (queda habilitado como mejora, no es requisito de esta tarea).
- La definición del *cómo* técnico (archivo de empaquetado, comando de
  instalación, cambios concretos en Docker): corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- Tras la tarea, desde **cualquier** directorio del proyecto (incluido un
  notebook en `notebooks/` o un script en `testing/`), una importación del tipo
  `import src` / `from src.core import extract_frames` **funciona** sin
  insertar rutas en `sys.path`.
- El paquete importable **mantiene el nombre `src`**, de modo que el código ya
  escrito (`src/utils.py`, `src/core/frame_extraction.py`) sigue funcionando sin
  cambios en sus rutas de import.
- La instalación es **editable**: editar un archivo de `src/` cambia el
  comportamiento importado sin necesidad de reinstalar.
- En el **contenedor**, el paquete queda instalado como parte del arranque o de
  la construcción del entorno (sin pasos manuales adicionales por parte del
  usuario).

---

## 5. Criterios de aceptación

1. **AC-1 — Importable sin `sys.path`:** desde un directorio distinto de la raíz,
   `import src` y `from src.core import extract_frames` funcionan sin manipular
   `sys.path`.
2. **AC-2 — Nombre conservado:** el paquete importable sigue siendo `src`; el
   código existente no requiere cambiar sus sentencias de import.
3. **AC-3 — Editable real:** un cambio en un archivo de `src/` se refleja al
   reimportar sin reinstalar el paquete.
4. **AC-4 — Reproducible en ambos entornos:** la instalación editable funciona en
   el venv local y en el contenedor.
5. **AC-5 — Integrado en el entorno del contenedor:** la instalación editable
   ocurre como parte de la preparación del contenedor, sin pasos manuales ocultos.
6. **AC-6 — Validación manual:** se demuestra **levantando el contenedor** (o
   entrando con `exec` a uno ya activo) e **importando algo de `src`**
   (p. ej. `python -c "import src.core"`); la importación se completa sin error.

---

## 6. Supuestos y notas

- El caso de uso principal que motiva la tarea son los **notebooks**, donde el
  parche de `sys.path` es especialmente incómodo.
- Esta tarea es de **entorno/empaquetado**: no cambia la lógica del proyecto, solo
  cómo se accede a ella.
- Esta especificación **no** define el *cómo* técnico (archivo de empaquetado
  —el borrador sugiere `setup.py`—, comando `pip install -e`, ni los cambios
  concretos en `docker/`); todo ello corresponde al `plan.md` de esta carpeta.

---

## 7. Siguientes pasos (metodología)

1. Elaborar `plan.md` con el detalle técnico de implementación
   (empaquetado vía `setup.py`, instalación editable y cambios en Docker).
2. Derivar `tasks.md` con las tareas ejecutables.
3. Implementar (paso 5) únicamente después de los anteriores.
