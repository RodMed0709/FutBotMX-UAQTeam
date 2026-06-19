# Plan técnico — Paquete `src` instalable en modo editable

- **Tarea atómica:** `editable_module`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo empaquetar el proyecto para que `src` (y sus
submódulos `src.core`, `src.utils`) quede instalado en **modo editable** y sea
importable desde cualquier ubicación —incluidos los notebooks— sin manipular
`sys.path`, tanto en el **venv local** como en el **contenedor**.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Empaquetado:** `setup.py` (setuptools) en la raíz del proyecto, conforme al
  borrador.
- **Instalación:** `pip install -e .` (editable).
- **Entornos:** venv local (`pip install -e .` manual) y contenedor (instalación
  en la fase de build del `Dockerfile`).

---

## 3. Diseño del empaquetado

### 3.1 `setup.py` (raíz del proyecto)

- Contenido mínimo, sin declarar dependencias (siguen en `requirements.txt`):

```python
from setuptools import find_packages, setup

setup(
    name="futbotmx",
    version="0.0.1",
    python_requires=">=3.11",
    packages=find_packages(include=["src", "src.*"]),
)
```

- `find_packages(include=["src", "src.*"])` mantiene el **nombre importable `src`**
  (y `src.core`), de modo que el código existente no cambia sus imports (AC-2).

### 3.2 `src/__init__.py`

- Hoy `src/` **no** tiene `__init__.py`. Se añade uno (puede ir vacío o con un
  docstring breve) para que `find_packages` reconozca `src` como paquete regular
  y `import src` funcione de forma explícita (AC-1).
- `src/core/__init__.py` ya existe; no se toca.

### 3.3 Conservación de `PROJECT_ROOT`

- `src/utils.py` seguirá derivando
  `PROJECT_ROOT = Path(__file__).resolve().parents[1]`. La instalación editable
  **no** copia el código a `site-packages` (solo lo enlaza), por lo que
  `__file__` sigue apuntando a `src/utils.py` dentro del proyecto y la resolución
  de rutas no cambia (AC-3 y comportamiento de `get_abs_path` intactos).

---

## 4. Instalación en el contenedor (Docker)

### 4.1 Cambio en el `Dockerfile`

- Añadir, **después** de `COPY . .` (cuando `setup.py` y `src/` ya están en la
  imagen), la instalación editable:

```dockerfile
COPY . .
RUN pip install --no-cache-dir -e .
```

### 4.2 Por qué funciona con el bind-mount

- El bind-mount de `docker-compose.yml` monta el proyecto del host en
  `/${CONTAINER_WORKSPACE_DIR}` (`/futbot`), ocultando lo copiado en build bajo
  esa ruta.
- La instalación editable **no** escribe dentro de `/futbot`: registra el enlace
  (`.egg-link` / entrada `.pth`) en el venv **`/opt/venv`**, que está **fuera** del
  bind-mount y por tanto sobrevive.
- Ese enlace apunta a `/futbot`. En runtime, `/futbot` es el código del host
  montado, así que `import src` resuelve contra el **código real y editable**.
- Conclusión: instalar en build es suficiente y correcto; no se requiere instalar
  en el arranque ni modificar el `command` del compose.

### 4.3 Sin cambios en `docker-compose.yml`

- El servicio `futbotmx26`, los volúmenes y el comando de arranque (symlinks
  `data/raw`/`assets/sam3` + `tail -f`) **no cambian**.

---

## 5. Instalación en el venv local

- Paso único de preparación, documentado para quien reproduzca el proyecto:

```bash
pip install -e .
```

- Habilita `import src...` desde cualquier cwd dentro del venv local (AC-4).

---

## 6. Validación manual

- **Contenedor:** levantar (o entrar a uno activo) e importar algo de `src`:

```bash
docker compose --env-file .env -f docker/docker-compose.yml up --build -d
docker compose --env-file .env -f docker/docker-compose.yml exec futbotmx26 \
  python -c "import src.core; from src.utils import get_abs_path; print('import OK')"
```

- **Opcional (cwd ajeno):** ejecutar el import desde un directorio distinto de la
  raíz (p. ej. `/tmp`) para evidenciar que ya no depende de `sys.path`.

---

## 7. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Importable sin `sys.path` | §3.1, §3.2 | `setup.py` + `src/__init__.py` |
| AC-2 Nombre `src` conservado | §3.1 | `find_packages(include=["src","src.*"])` |
| AC-3 Editable real | §3.3, §4.2 | Enlace, no copia |
| AC-4 Reproducible en ambos entornos | §4, §5 | Docker (build) + venv local |
| AC-5 Integrado en el contenedor | §4.1 | `pip install -e .` en el `Dockerfile` |
| AC-6 Validación manual | §6 | `import src.core` en el contenedor |

---

## 8. Riesgos y consideraciones

- **Reconstruir la imagen:** el nuevo `RUN pip install -e .` solo surte efecto al
  reconstruir (`up --build`); una imagen previa no tendrá el enlace.
- **Falta de `src/__init__.py`:** sin él, `find_packages` no detecta `src` y el
  import falla; por eso §3.2 lo añade como parte de la tarea.
- **Nombre genérico `src`:** `src` como nombre importable es poco específico y
  podría colisionar si en el futuro se instalara otro paquete llamado `src`. Se
  asume aceptable para no romper los imports actuales (cambiar el nombre quedaría
  como decisión futura, fuera de alcance).
- **Orden en el `Dockerfile`:** la instalación editable debe ir tras `COPY . .`
  para que `setup.py` y `src/` existan en la imagen al momento del `pip install -e`.

---

## 9. Siguiente paso (metodología)

Elaborar `tasks.md` con la descomposición en tareas ejecutables. La
implementación (paso 5) ocurre únicamente después de definir las tareas.
