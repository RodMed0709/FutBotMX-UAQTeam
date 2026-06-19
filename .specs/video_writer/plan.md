# Plan técnico — Escritor de video (`video_writer`)

- **Tarea atómica:** `video_writer`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** no hay draft previo para este plan.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar `write_video(frames, output_path,
fps=None)`: escribir un `np.ndarray (N,H,W,3) uint8` a un mp4 con `imageio`+ffmpeg,
con fps por defecto desde la config y override por parámetro, creando el
directorio de salida si falta. Definir también los cambios de config
(`outputs_dir`, `output_fps`) y el script de validación local.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Escritura de video:** `imageio` con backend **ffmpeg** (`imageio-ffmpeg`),
  ambos en `requirements.txt`. RGB-nativo (los frames del proyecto son RGB).
- **Arrays / rutas:** `numpy`, `pathlib`.
- **Configuración:** `json` (estándar) + `src.utils.get_abs_path` para leer
  `visualization.output_fps`, con la convención del repo.

> `imageio` se importa **perezosamente** dentro de la función para no encarecer
> `import src.core`.

---

## 3. Diseño

### 3.1 Ubicación y módulo

- Archivo nuevo: `src/core/video_writer.py`.
- Exportación en `src/core/__init__.py`: `from src.core.video_writer import
  write_video` y sumarlo a `__all__`.

### 3.2 Firma

```python
def write_video(
    frames: np.ndarray,
    output_path: Path | str,
    fps: float | None = None,
) -> Path: ...
```

- `frames`: `np.ndarray (N, H, W, 3) uint8` RGB.
- `output_path`: ruta **completa** del mp4 a escribir.
- `fps`: si `None`, se usa el default de config; si se pasa, prevalece.
- **Retorno:** `Path` del archivo escrito.

### 3.3 Resolución de fps

`_load_output_fps()` lee la config (leer `CONFIG_FILENAME` del `.env` con
`strip()` → `get_abs_path(f"configs/{...}")` → `json.load`) y devuelve
`config["visualization"]["output_fps"]`. En `write_video`:

```python
fps = fps if fps is not None else _load_output_fps()
```

### 3.4 Validación de entrada

```python
if not isinstance(frames, np.ndarray):
    raise ValueError(...)
if frames.ndim != 4 or frames.shape[-1] != 3:
    raise ValueError(...)          # se espera (N, H, W, 3)
if frames.shape[0] == 0:
    raise ValueError(...)          # nada que escribir
if frames.dtype != np.uint8:
    raise ValueError(...)          # mp4 espera uint8 0-255
```

### 3.5 Creación del directorio y escritura

```python
output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

import imageio
writer = imageio.get_writer(
    str(output_path), format="FFMPEG", mode="I",
    fps=fps, codec="libx264", pixelformat="yuv420p",
)
try:
    for frame in frames:
        writer.append_data(frame)
finally:
    writer.close()
return output_path
```

- `libx264` + `yuv420p` → amplia compatibilidad de reproducción.
- Se mantiene el `macro_block_size` por defecto (16) de imageio, que **ajusta
  dimensiones impares/no múltiplos** automáticamente (emite aviso). Ver §7.
- `get_abs_path` **no** se usa para `output_path` (la ruta puede no existir aún);
  por eso se crea el directorio con `mkdir`.

### 3.6 Ruta de outputs (responsabilidad del pipeline)

- `write_video` recibe la **ruta completa**; no lee `working_dirs.outputs_dir`.
- La clave `working_dirs.outputs_dir` se **añade a la config** (constitución §5.5)
  y la **consume el `pipeline_runner`** para componer la ruta del mp4 (p. ej.
  `PROJECT_ROOT / outputs_dir / <nombre>.mp4`), manteniendo el escritor "tonto".

### 3.7 Manejo de errores

| Situación | Excepción |
|---|---|
| `frames` no `ndarray` / no `(N,H,W,3)` / vacío / no `uint8` | `ValueError` |
| `CONFIG_FILENAME`/`visualization.output_fps` ausente (al leer default) | `ValueError`/`KeyError`/`FileNotFoundError` (propagados) |
| Fallo de ffmpeg al escribir | excepción de `imageio` propagada |

---

## 4. Cambios de configuración

`configs/00_testing_config.json` (ediciones aditivas; resto intacto):

- En `working_dirs`: `"outputs_dir": "outputs"`.
- En `visualization`: `"output_fps": 4`.

```json
{
  "working_dirs": { "dataset_dir": "data/raw", "sam3_dir": "assets/sam3", "outputs_dir": "outputs" },
  "preprocess": { "fps": "1", "frame_quota": 30 },
  "classes": [ ... ],
  "visualization": { "overlay_alpha": 0.55, "output_fps": 4 }
}
```

---

## 5. Validación

### 5.1 Script — `testing/test_video_writer.py`

- **Local, sin GPU ni modelo.** Construye frames sintéticos `(N, H, W, 3) uint8`
  (p. ej. un degradado/animación simple).
- Escribe el mp4 en **`outputs/test_video_maker/`** (carpeta dedicada,
  git-ignored): la crea si no existe y **deja** el archivo ahí en cada corrida
  (nombre fijo, se sobrescribe) para inspección manual.
- Verifica:
  - el archivo existe y su tamaño es > 0;
  - se puede **releer** con `imageio` y el nº de frames/ dimensiones son coherentes;
  - la **creación del directorio** funciona (parte de un dir inexistente);
  - entrada inválida (p. ej. `dtype` o forma incorrectos) lanza `ValueError`.
- **El agente lo ejecuta** en local.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Módulo y función | §3.1, §3.2 | `write_video` + export |
| AC-2 Escribe mp4 | §3.5 | imageio + ffmpeg (libx264) |
| AC-3 Crea el directorio | §3.5 | `mkdir(parents, exist_ok)` |
| AC-4 fps configurable | §3.3, §4 | `visualization.output_fps` + override |
| AC-5 Outputs en config | §3.6, §4 | `working_dirs.outputs_dir` |
| AC-6 Retorno | §3.2, §3.5 | devuelve `Path` |
| AC-7 Entrada inválida | §3.4, §3.7 | `ValueError` |
| AC-8 Validación | §5.1 | script headless local |

---

## 7. Riesgos y consideraciones

- **Dimensiones impares / no múltiplos de 16:** `yuv420p` exige dimensiones pares;
  imageio (`macro_block_size=16` por defecto) **reescala/pad** para cumplir y emite
  un aviso. Los frames del overlay conservan el tamaño del frame fuente; si una
  resolución diera problemas, se ajustaría `macro_block_size` puntualmente. No
  altera el diseño.
- **`outputs/` git-ignored:** los videos no se versionan (constitución §7); el
  subdir `outputs/test_video_maker/` queda cubierto por esa exclusión.
- **fps en modo completo:** lo pasa el `pipeline_runner` como override (fps real de
  la fuente); el escritor no decide el modo.
- **Tamaño del mp4:** con `libx264` el archivo es compacto; sin compresión sería
  mucho mayor. No se exponen parámetros de calidad en el MVP (defaults de imageio).
