# Plan técnico — Utilidad de resolución de rutas absolutas (`src/utils.py`)

- **Tarea atómica:** `abs_dir_func`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## Nota de consistencia con el `spec.md`

✅ Resuelto. El `spec.md` fue actualizado: AC-5 pasó de "no requiere existencia"
a **"verifica existencia y lanza `FileNotFoundError` si la ruta no existe"**, y se
documentó el reparto de errores (`ValueError` para entrada inválida,
`FileNotFoundError` para ruta inexistente). Plan y spec quedan alineados.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar en `src/utils.py` una función general
que convierta una ruta relativa (`str`) en su ruta absoluta (`pathlib.Path`),
resolviéndola respecto a la raíz del proyecto, validando la entrada y verificando
la existencia de la ruta resultante. Además, definir un script de prueba manual
que valide la función usando las rutas de configuración del proyecto.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Función:** solo biblioteca estándar (`pathlib`).
- **Script de prueba:** `json` (estándar) para leer la config; `python-dotenv`
  (o parseo simple) para leer el `.env`; `cv2` (ya en `requirements.txt`) para
  intentar abrir un `.MOV`.

---

## 3. Diseño de la función

### 3.1 Ubicación y firma

- Archivo: `src/utils.py`.
- Firma: `get_abs_path(relative_path: str) -> Path`.

### 3.2 Resolución de la raíz del proyecto

- La raíz se deriva de la ubicación del propio módulo:
  `PROJECT_ROOT = Path(__file__).resolve().parents[1]` (dado que `src/` cuelga de
  la raíz del proyecto).
- Esto garantiza un resultado **estable** independientemente del `cwd` (AC-4).

### 3.3 Lógica

1. **Validar entrada** → si no es `str`, está vacía/solo espacios, o es una ruta
   **absoluta**, lanzar `ValueError` (la función espera una ruta **relativa**).
2. **Resolver** → `abs_path = (PROJECT_ROOT / relative_path).resolve()`.
3. **Verificar existencia** → si `abs_path` no existe, lanzar `FileNotFoundError`.
4. **Devolver** `abs_path` (objeto `Path` absoluto).

### 3.4 Manejo de errores (detiene el proceso)

| Situación | Excepción |
|---|---|
| Entrada no `str`, vacía, o ruta absoluta | `ValueError` |
| Ruta resuelta inexistente | `FileNotFoundError` |

Ambas excepciones se propagan sin capturar dentro de la función (detienen el
proceso), conforme al borrador.

---

## 4. Script de prueba manual

- Ubicación: `testing/test_abs_dir_func.py` (ejecutable manual, no pytest).
- **Flujo:**
  1. Leer del `.env` la variable `CONFIG_FILENAME` (aplicar `strip()` a clave y
     valor, pues el `.env` la tiene como `CONFIG_FILENAME =...` con espacio).
  2. Construir la ruta relativa `configs/<CONFIG_FILENAME>`.
  3. Resolver a absoluta con `get_abs_path(...)` y leer el JSON de configuración.
  4. Tomar de `working_dirs` las rutas relativas `dataset_dir` (`data/raw`) y
     `sam3_dir` (`assets/sam3`) y resolverlas a absolutas con la función.
  5. Aprovechando que `dataset_dir` apunta a la base de datos de vídeo, intentar
     **abrir un archivo `.MOV`** de esa ruta con `cv2` y **reportar** si se pudo
     leer.
  6. **Imprimir en consola** todas las rutas absolutas resueltas y reportar su
     existencia.
- **Robustez del script:** como la función lanza `FileNotFoundError` cuando una
  ruta no existe (caso local frecuente: `data/raw`/`assets/sam3` son symlinks que
  solo existen dentro del contenedor), el script **captura** esas excepciones por
  cada ruta y las **reporta sin abortar** el resto de la demostración. Así se
  cumplen las asunciones 10 y 11 sin contradecir el manejo de errores de la
  función.

---

## 5. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Módulo presente (`src/utils.py`) | §3.1 | |
| AC-2 Firma `str → Path` | §3.1 | |
| AC-3 `Path` absoluto | §3.2, §3.3 | |
| AC-4 Estable respecto al `cwd` | §3.2 | |
| AC-5 Verifica existencia (`FileNotFoundError`) | §3.3, §3.4 | Alineado con el spec actualizado |
| AC-6 Validación manual | §4 | |

---

## 6. Riesgos y consideraciones

- **Rutas inexistentes en local:** `data/raw` y `assets/sam3` son symlinks
  creados dentro del contenedor; en ejecución local pueden no existir y la
  función lanzará `FileNotFoundError`. El script lo maneja reportando por ruta.
- **Parseo del `.env`:** el espacio en `CONFIG_FILENAME =...` obliga a hacer
  `strip()`; si se usa `python-dotenv` esto se maneja de forma transparente.
- **Lectura de `.MOV`:** depende de que existan vídeos en `dataset_dir`; si no
  hay, se reporta sin considerarlo fallo.

---

## 7. Siguiente paso (metodología)

Elaborar `tasks.md` con la descomposición en tareas ejecutables. La
implementación (paso 5) ocurre únicamente después de definir las tareas.
