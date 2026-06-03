# Plan técnico — Aceptar rutas absolutas de vídeo en `extract_frames`

- **Tarea atómica:** `abs_video_path`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo lograr que `extract_frames` acepte una **ruta
absoluta** a un vídeo siempre que apunte a un **archivo existente y válido**,
aunque esté **fuera de `PROJECT_ROOT`**, sin romper el soporte actual de rutas
**relativas** ni modificar la firma pública, `get_abs_path`, la configuración o
los modos de extracción. El cambio se concentra en la función auxiliar
`_resolve_video_path`.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Rutas:** `pathlib.Path` (estándar). Métodos clave: `Path.is_absolute()`,
  `Path.is_file()`, `Path.resolve()`.
- **Resolución/validación de rutas relativas:** `src/utils.py::get_abs_path`
  (reutilizada **sin cambios**, solo para la rama relativa).
- **Sin nuevas dependencias** ni cambios en `requirements.txt`.
- **Sin cambios de configuración** (`configs/*.json` y `.env` intactos).

---

## 3. Diseño del cambio

### 3.1 Punto de cambio único

- Archivo: `src/core/frame_extraction.py`.
- Función afectada: **`_resolve_video_path`** (líneas ~76-106 actuales).
- **No se modifican:** la firma ni el cuerpo de `extract_frames`,
  `_load_frame_quota`, `_load_env`, ni `src/utils.py::get_abs_path`.

### 3.2 Contrato nuevo de `_resolve_video_path`

```python
def _resolve_video_path(video_path: Path) -> Path:
    ...
```

- **Entrada:** `Path` relativo a `PROJECT_ROOT` **o** `Path` absoluto a cualquier
  ubicación del sistema.
- **Salida:** `Path` absoluto verificado (existe y es archivo), listo para pasar a
  `decord.VideoReader(str(...))`.

### 3.3 Lógica por ramas

1. **Validación de tipo (se conserva):**
   ```python
   if not isinstance(video_path, Path):
       raise ValueError("Se esperaba una ruta de tipo Path, se recibio: ...")
   ```

2. **Rama relativa** (`not video_path.is_absolute()`):
   - Se delega en `get_abs_path(str(video_path))`, que resuelve contra
     `PROJECT_ROOT` y verifica existencia (lanza `FileNotFoundError`/`ValueError`).
   - **Comportamiento idéntico al actual.**

3. **Rama absoluta** (`video_path.is_absolute()`):
   - **Ya no** se intenta `relative_to(PROJECT_ROOT)` ni se rechaza por estar
     fuera del proyecto.
   - Se valida que sea un **archivo existente**:
     ```python
     if not video_path.is_file():
         raise FileNotFoundError(
             f"La ruta del video no existe o no es un archivo: {video_path}"
         )
     ```
   - Se devuelve la ruta resuelta: `return video_path.resolve()`.

### 3.4 Por qué `is_file()`

- Resuelve en una sola comprobación los criterios de la spec:
  - **existe** y **es archivo** → válido.
  - **no existe** o **es directorio** → `FileNotFoundError` (AC-4, AC-5).
- **Symlinks:** `is_file()` sigue el enlace; devuelve `True` si el destino existe
  y es archivo, lo que cubre los montajes/ubicaciones externas (AC-1). No se usa
  `resolve(strict=True)` para no romper escenarios de enlace.
- La validez del **contenido** como vídeo no se comprueba aquí; la determina
  `decord` al abrir el archivo (fuera de alcance de esta validación).

### 3.5 Manejo de errores

| Situación | Excepción | Rama |
|---|---|---|
| `video_path` no es `Path` | `ValueError` | común |
| Ruta **relativa** inexistente/inválida | `FileNotFoundError` / `ValueError` (vía `get_abs_path`) | relativa |
| Ruta **absoluta** inexistente o que es directorio | `FileNotFoundError` | absoluta |

Las excepciones se propagan (detienen el proceso), coherente con el estilo actual
de la función y de `get_abs_path`.

### 3.6 Documentación en el código

- Actualizar el **docstring** de `_resolve_video_path` para reflejar el nuevo
  contrato: acepta rutas relativas (resueltas vía `get_abs_path`) **y** rutas
  absolutas a archivos válidos en cualquier ubicación.
- Revisar el docstring de `extract_frames` (sección Args/Raises): `video_path`
  pasa a admitir rutas absolutas externas; las excepciones documentadas no
  cambian de clase.

---

## 4. Cambios fuera de `frame_extraction.py`

- **Ninguno** en código de producción: no se tocan `src/utils.py`, `configs/`,
  `.env`, `requirements.txt` ni `setup.py`.

---

## 5. Validación manual

- **Opción elegida:** nuevo script suelto `testing/test_abs_video_path.py`
  (ejecutable manual, no pytest), para no contaminar el script existente de
  `frame_extraction`.
- **Flujo:**
  1. **Ruta absoluta externa:** localizar un `.MOV` real (p. ej. resolviendo
     `dataset_dir` con `get_abs_path` y `rglob`, luego usar su ruta **absoluta**
     `.resolve()`), llamar `extract_frames(abs_path, all_frames=False)` y reportar
     la forma del arreglo. Debe producir frames sin `ValueError`.
  2. **Ruta relativa:** llamar `extract_frames(Path("data/raw/.../x.MOV"))` y
     verificar que sigue funcionando (regresión).
  3. **Inexistente:** llamar con una ruta absoluta falsa y verificar
     `FileNotFoundError`.
  4. **Directorio:** llamar con la ruta absoluta de un directorio y verificar el
     error.
  5. **Imprimir** formas/`dtype`/excepciones capturadas; reportar sin abortar.
- **Ejecución:** donde resuelvan los datos (contenedor con montajes, o host con
  los vídeos presentes):
  ```bash
  docker compose --env-file .env -f docker/docker-compose.yml \
    exec futbotmx26 python testing/test_abs_video_path.py
  ```

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Ruta absoluta externa aceptada | §3.3(3), §3.4 | `is_file()`, sin `relative_to` |
| AC-2 Ruta absoluta interna aceptada | §3.3(3) | misma rama absoluta |
| AC-3 Ruta relativa intacta | §3.3(2) | delega en `get_abs_path` |
| AC-4 Inexistencia → `FileNotFoundError` | §3.3, §3.5 | ambas ramas |
| AC-5 Directorio rechazado | §3.4, §3.5 | `is_file()` es `False` |
| AC-6 Tipo inválido → `ValueError` | §3.3(1) | chequeo conservado |
| AC-7 Firma y modos sin cambios | §3.1 | solo cambia `_resolve_video_path` |
| AC-8 `get_abs_path` intacta | §2, §4 | no se modifica |
| AC-9 Validación manual | §5 | `testing/test_abs_video_path.py` |

---

## 7. Riesgos y consideraciones

- **Pérdida de la "barrera" PROJECT_ROOT:** aceptar absolutas externas es
  justamente el objetivo; el contrato pasa a ser "archivo válido", no "dentro del
  proyecto". Documentarlo en el docstring evita malentendidos.
- **Rutas relativas sin cambios:** siguen exigiendo que el dato resuelva bajo
  `PROJECT_ROOT` (vía `get_abs_path`); en el host sin datos, lanzarán
  `FileNotFoundError` como hoy.
- **Symlinks rotos:** un symlink absoluto cuyo destino no exista hará `is_file()`
  → `False` → `FileNotFoundError`, comportamiento correcto.
- **Memoria y `decord`:** sin cambios respecto a la tarea original; este plan no
  altera la lógica de extracción.

---

## 8. Siguiente paso (metodología)

Elaborar `tasks.md` con la descomposición en tareas ejecutables. La
implementación (paso 5) ocurre únicamente después de definir las tareas.
