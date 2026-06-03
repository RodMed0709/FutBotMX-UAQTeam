# Plan técnico — Función de extracción de frames de un vídeo

- **Tarea atómica:** `frame_extraction`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** `.specs/drafts/frame_extraction/00_plan.md`
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar una función que extraiga frames de un
**único vídeo** en dos modos —**cuota** (por defecto) y **todos los frames**—,
leyendo la cuota desde el archivo de configuración `.json`, validando la ruta del
vídeo mediante la utilidad existente `get_abs_path`, y devolviendo los frames como
un arreglo de NumPy. Además, definir un script de validación manual.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Lectura de vídeo:** `decord` (`decord.VideoReader`), ya en `requirements.txt`.
- **Cálculo numérico y salida:** `numpy` (ya en `requirements.txt`).
- **Configuración:** `json` (estándar) para parsear el `.json`; lectura de
  `CONFIG_FILENAME` desde `.env` (parseo simple con `strip()` o `python-dotenv`).
- **Resolución/validación de rutas:** `src/utils.py::get_abs_path` (reutilizada).
- **Sin dependencia de torch** para esta función.

---

## 3. Diseño de la función

### 3.1 Ubicación y módulo

- Nuevo submódulo paquete `src/core/` con su `__init__.py`.
- Archivo: `src/core/frame_extraction.py`.

### 3.2 Firma

```python
def extract_frames(video_path: Path, all_frames: bool = False) -> np.ndarray:
    ...
```

- `video_path: Path` — ruta del vídeo (conforme al borrador).
- `all_frames: bool = False` — selector de modo. `False` → modo **cuota**;
  `True` → **todos** los frames.
- **Retorno:** `numpy.ndarray` apilado con forma `(N, H, W, 3)` (frames RGB).

### 3.3 Validación de la ruta del vídeo (reutilizando `get_abs_path`)

- `get_abs_path` espera un `str` **relativo** a `PROJECT_ROOT` y verifica
  existencia (lanza `FileNotFoundError`/`ValueError`).
- Como la función recibe un `Path` (posiblemente absoluto), el plan asume:
  1. Convertir `video_path` a una ruta **relativa a `PROJECT_ROOT`**
     (`Path.relative_to(PROJECT_ROOT)` o equivalente).
  2. Pasar esa ruta relativa como `str` a `get_abs_path` para obtener la ruta
     absoluta verificada.
- Así se centraliza la verificación de existencia en la utilidad del proyecto y se
  respeta la regla de "acceder a rutas vía la utilidad", sin duplicar lógica.

### 3.4 Lectura de la configuración y de la cuota

1. Leer `CONFIG_FILENAME` del `.env` aplicando `strip()` a clave y valor (el
   `.env` la escribe como `CONFIG_FILENAME =...` con espacio).
2. Construir la ruta relativa `configs/<CONFIG_FILENAME>` y resolverla con
   `get_abs_path(...)`.
3. Parsear el JSON con `json`.
4. Leer la **cuota** desde una nueva clave dentro de `preprocess`, p. ej.
   `preprocess.frame_quota`.

> **Cambio de configuración requerido:** añadir `"frame_quota": 30` (ejemplo) en
> `configs/00_testing_config.json`, dentro del bloque `preprocess`. El valor es
> ajustable por *trial* sin tocar el código.

### 3.5 Lógica de extracción

1. **Validar ruta** → obtener la ruta absoluta verificada (§3.3).
2. **Abrir el vídeo** → `vr = decord.VideoReader(str(abs_path))`;
   `total = len(vr)` (número total de frames).
3. **Modo completo** (`all_frames=True`) → índices `range(total)`.
4. **Modo cuota** (`all_frames=False`):
   - Leer y **validar la cuota** (entero positivo; ver §3.6).
   - Si `total <= quota` → devolver **todos** los frames (cuota = máximo, sin
     duplicar ni rellenar).
   - Si `total > quota` → calcular índices **equiespaciados** con
     `np.linspace(0, total - 1, quota)` redondeados a `int` y deduplicados si
     fuese necesario.
5. **Recuperar frames** → `frames = vr.get_batch(indices).asnumpy()`
   (configurar el bridge nativo de decord para salida NumPy:
   `decord.bridge.set_bridge('native')`).
6. **Devolver** `frames` como `np.ndarray` `(N, H, W, 3)`.

### 3.6 Manejo de errores

| Situación | Excepción |
|---|---|
| Ruta del vídeo inexistente / inválida | `FileNotFoundError` / `ValueError` (vía `get_abs_path`) |
| `CONFIG_FILENAME` ausente en `.env` | `KeyError` / `ValueError` (mensaje claro) |
| Clave de cuota ausente en la config (modo cuota) | `KeyError` |
| Cuota no entera o ≤ 0 | `ValueError` |

Las excepciones se propagan (detienen el proceso), coherente con el estilo de
`get_abs_path`.

---

## 4. Cambios de configuración

- `configs/00_testing_config.json`: agregar la clave de cuota.

```json
{
  "working_dirs": { "dataset_dir": "data/raw", "sam3_dir": "assets/sam3" },
  "preprocess": { "fps": "1", "frame_quota": 30 }
}
```

---

## 5. Script de validación manual

- Ubicación: `testing/test_frame_extraction.py` (ejecutable manual, no pytest).
- **Flujo:**
  1. Resolver el `dataset_dir` de la config con `get_abs_path` y localizar un
     `.MOV` de forma **recursiva** (`rglob`, pues los vídeos viven en subcarpetas
     fechadas).
  2. Llamar `extract_frames(video, all_frames=False)` y reportar la forma del
     arreglo (debe coincidir con la cuota o el total si el vídeo tiene menos).
  3. Llamar `extract_frames(video, all_frames=True)` y reportar el total.
  4. **Imprimir** formas, `dtype` y conteos; reportar sin abortar si una ruta no
     existe (caso local: symlinks de datos solo válidos en contenedor).
- **Ejecución:** dentro del contenedor (ahí se montan los vídeos):
  ```bash
  docker compose --env-file .env -f docker/docker-compose.yml \
    exec futbotmx26 python testing/test_frame_extraction.py
  ```

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Función presente | §3.1, §3.2 | `src/core/frame_extraction.py` |
| AC-2 Dos modos (cuota/completo) | §3.2, §3.5 | Selector `all_frames` |
| AC-3 Cuota desde configuración | §3.4, §4 | Clave `preprocess.frame_quota` |
| AC-4 Distribución uniforme | §3.5(4) | `np.linspace` equiespaciado |
| AC-5 Modo completo | §3.5(3) | Todos los frames |
| AC-6 Ruta verificada vía `get_abs_path` | §3.3, §3.6 | Reutiliza la utilidad |
| AC-7 Salida en memoria (sin disco) | §3.2, §3.5(6) | `np.ndarray` retornado |
| AC-8 Validación manual | §5 | Script en `testing/` |

---

## 7. Riesgos y consideraciones

- **Rutas solo válidas en contenedor:** `data/raw` es un symlink que solo resuelve
  dentro del contenedor; en el host `get_abs_path` lanzará `FileNotFoundError`. La
  validación con vídeos reales debe correr en el contenedor.
- **`video_path` fuera de `PROJECT_ROOT`:** si la ruta recibida no es relativa a
  la raíz del proyecto, `relative_to` fallará; el plan asume que los vídeos viven
  bajo `dataset_dir` (dentro del proyecto). Conviene documentar este supuesto.
- **`decord` y resolución variable:** los vídeos tienen resolución no fija; todos
  los frames de un mismo vídeo comparten `(H, W)`, por lo que el apilado en un
  único `ndarray` es válido **por vivo** (no se mezclan vídeos en esta tarea).
- **Memoria:** el modo completo sobre vídeos largos puede consumir mucha RAM al
  apilar todos los frames; es un riesgo aceptado para esta tarea (la cuota es el
  modo por defecto, precisamente para acotar el coste).
- **Parseo del `.env`:** el espacio en `CONFIG_FILENAME =...` obliga a `strip()`.

---

## 8. Siguiente paso (metodología)

Elaborar `tasks.md` con la descomposición en tareas ejecutables. La
implementación (paso 5) ocurre únicamente después de definir las tareas.
