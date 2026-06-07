# Plan técnico — Exportar y congelar el set de frames de evaluación (`eval_frame_export`)

- **Tarea atómica:** `eval_frame_export`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** [`../drafts/evaluation_sam3_only_roadmap.md`](../drafts/evaluation_sam3_only_roadmap.md) (tarea 1)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar el módulo `src/data/eval_frames.py` que:
lee los videos del split *testing* desde `assets/db_metadata.csv` (`split==2`),
extrae sus frames de cuota reusando `extract_frames`, los persiste como imágenes PNG
bajo `data/testing_frames/` (git-ignored) y genera un **CSV de control versionado**
en `assets/` con la procedencia y el grupo (`aleatorio`/`cenital`) de cada frame.
No toca `extract_frames`, el pipeline ni `db_metadata.csv`.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Extracción de frames:** `src.core.frame_extraction.extract_frames(path, all_frames=False)`
  (modo cuota, determinista). **No** se reimplementa el muestreo.
- **Índices de muestreo:** nuevo helper `get_frame_indices(path, all_frames=False)`
  en `frame_extraction.py` que devuelve los índices que el muestreo selecciona;
  `extract_frames` se refactoriza para reusarlo (misma salida y firma). Habilita
  registrar `frame_original` sin duplicar la lógica.
- **Escritura de imágenes:** `cv2.imwrite` (OpenCV, ya dependencia vía
  `src/core/segmentation.py` y `video_writer.py`). PNG sin pérdida.
- **Tabla / CSV:** `pandas` (ya en `requirements.txt`).
- **Rutas / config:** `src.utils.PROJECT_ROOT` y `get_abs_path`; lectura de `.env`
  + JSON con el mismo patrón de `metadata.py` / `pipeline.py`.
- Sin nuevas dependencias.

---

## 3. Diseño

### 3.1 Estructura de archivos

```
src/core/frame_extraction.py       # se AÑADE get_frame_indices (refactor aditivo)
src/data/eval_frames.py            # lógica de la tarea (nuevo)
src/data/__init__.py               # se amplía la API pública
testing/test_eval_frame_export.py  # script manual standalone (corre en el pod)
```

`src/data/__init__.py` exporta la API pública añadida:

```python
from src.data.eval_frames import export_testing_frames, validate_testing_frames_schema
```

### 3.2 Esquema (fuente única de verdad)

Constantes a nivel de módulo (esquema mutable desde un solo lugar):

```python
COLUMNS = [
    "id", "video_id", "video_ruta", "frame_index", "frame_original", "imagen", "grupo",
]
TESTING_SPLIT = 2          # split de testing en db_metadata.csv
GROUP_RANDOM = "aleatorio"
GROUP_CENITAL = "cenital"
IMAGE_EXT = ".png"
```

### 3.3 Carga de configuración

Helper local `_load_eval_frames_config()` (mismo patrón que `metadata.py`): lee
`CONFIG_FILENAME` del `.env`, abre el JSON y devuelve lo necesario.

```python
def _load_eval_frames_config() -> tuple[str, str, list[str]]:
    """Devuelve (testing_frames_dir, testing_frames_csv, forced_testing) y la ruta
    del metadata_csv desde la config global."""
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")
    config = json.loads(get_abs_path(f"configs/{config_filename}").read_text("utf-8"))

    working_dirs = config.get("working_dirs", {})
    for key in ("metadata_csv", "testing_frames_dir", "testing_frames_csv"):
        if key not in working_dirs:
            raise KeyError(f"Falta 'working_dirs.{key}' en la configuracion.")

    forced_testing = config.get("splits", {}).get("forced_testing", [])
    return (
        working_dirs["metadata_csv"],
        working_dirs["testing_frames_dir"],
        working_dirs["testing_frames_csv"],
        list(forced_testing),
    )
```

- `frame_quota` **no** se lee aquí: lo consume internamente `extract_frames`.
- `forced_testing` (rutas POSIX relativas) determina el grupo `cenital`.

### 3.4 Selección de los videos de testing

```python
def _load_testing_videos(metadata_csv: str) -> pandas.DataFrame:
    """Lee db_metadata.csv y devuelve las filas con split == TESTING_SPLIT."""
    csv_path = get_abs_path(metadata_csv)          # debe existir (no se construye)
    df = pandas.read_csv(csv_path)
    return df[df["split"] == TESTING_SPLIT][["id", "ruta"]].reset_index(drop=True)
```

- **Requiere** que `db_metadata.csv` exista; si falta, `get_abs_path` lanza
  `FileNotFoundError` (mensaje claro). Esta tarea **no** lo genera.
- Se conservan `id` (→ `video_id`) y `ruta` (→ `video_ruta`).

### 3.5 Grupo de cada video

```python
def _group_for(ruta: str, forced_testing: set[str]) -> str:
    return GROUP_CENITAL if ruta in forced_testing else GROUP_RANDOM
```

- Comparación directa de la ruta POSIX relativa contra el conjunto
  `splits.forced_testing` (los 2 videos de cámara superior).

### 3.5b Helper aditivo en `frame_extraction.py`

Para registrar `frame_original` sin duplicar la lógica de muestreo, se extrae la
selección de índices a un helper reusable y se expone públicamente. `extract_frames`
pasa a llamarlo (salida y firma **idénticas**).

```python
def _select_frame_indices(total: int, all_frames: bool) -> np.ndarray:
    """Índices de frame a extraer (extraído de extract_frames, sin cambios)."""
    if all_frames:
        return np.arange(total)
    quota = _load_frame_quota()
    if total <= quota:
        return np.arange(total)
    return np.unique(np.linspace(0, total - 1, quota).round().astype(int))


def get_frame_indices(video_path: Path, all_frames: bool = False) -> np.ndarray:
    """Devuelve los índices (en el video fuente) que el muestreo seleccionaría."""
    abs_path = _resolve_video_path(video_path)
    total = len(decord.VideoReader(str(abs_path)))
    return _select_frame_indices(total, all_frames)
```

- `extract_frames` se reescribe para usar `_select_frame_indices(total, all_frames)`
  en lugar del bloque `if/else` actual: **mismo resultado**, ahora compartido.
- `get_frame_indices` y `extract_frames` devuelven arrays **alineados por posición**:
  el frame posicional `i` de `extract_frames` corresponde a `get_frame_indices(...)[i]`.

### 3.6 Escritura de una imagen

```python
def _write_frame_image(frame_rgb: np.ndarray, dest: Path) -> None:
    import cv2
    cv2.imwrite(str(dest), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
```

- `extract_frames` devuelve RGB; OpenCV escribe BGR → conversión obligatoria.
- PNG (sin pérdida) para no introducir artefactos de compresión en el GT.
- **Nomenclatura:** `f"{video_id:04d}_{frame_index:04d}{IMAGE_EXT}"`, planas bajo
  `testing_frames_dir` (sin subcarpetas por video).

### 3.7 Función pública orquestadora

```python
def export_testing_frames(force: bool = False) -> pandas.DataFrame:
    """Extrae y persiste los frames de cuota de los 20 videos de testing + CSV.

    Idempotente: si el CSV existe, pasa el handler de validación y force=False,
    se devuelve sin re-extraer. En cualquier otro caso se regenera por completo.
    """
    metadata_csv, frames_dir, frames_csv, forced = _load_eval_frames_config()
    csv_path = PROJECT_ROOT / frames_csv           # versionado (puede no existir aún)

    if not force and csv_path.exists() and validate_testing_frames_schema(csv_path):
        return pandas.read_csv(csv_path)

    forced_set = set(forced)
    videos = _load_testing_videos(metadata_csv)
    out_dir = PROJECT_ROOT / frames_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, fid = [], 0
    for video_id, ruta in videos.itertuples(index=False):
        frames = extract_frames(Path(ruta), all_frames=False)        # (N,H,W,3) cuota
        originals = get_frame_indices(Path(ruta), all_frames=False)  # alineado por posición
        grupo = _group_for(ruta, forced_set)
        for frame_index, frame in enumerate(frames):
            img_name = f"{int(video_id):04d}_{frame_index:04d}{IMAGE_EXT}"
            _write_frame_image(frame, out_dir / img_name)
            rows.append({
                "id": fid,
                "video_id": int(video_id),
                "video_ruta": ruta,
                "frame_index": frame_index,
                "frame_original": int(originals[frame_index]),
                "imagen": (Path(frames_dir) / img_name).as_posix(),
                "grupo": grupo,
            })
            fid += 1

    df = pandas.DataFrame(rows, columns=COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
```

- **`frame_index`** = índice posicional del array de `extract_frames`;
  **`frame_original`** = `get_frame_indices(...)[frame_index]`, el índice real en el
  video fuente. Ambos arrays están alineados por posición (§3.5b).
- **Caso borde** (video con menos frames que la cuota): `extract_frames` ya
  devuelve los que haya; el bucle `enumerate(frames)` lo refleja sin lógica extra.
- `imagen` y `video_ruta` quedan **relativas a `PROJECT_ROOT`** en POSIX.
- `csv_path = PROJECT_ROOT / frames_csv` (no `get_abs_path`: el CSV puede no existir
  aún, lo estamos creando).

### 3.8 Handler de validación (independiente, esquema mutable)

```python
def validate_testing_frames_schema(csv_path: Path) -> bool:
    """True si el CSV existe y sus columnas coinciden con COLUMNS (orden incluido)."""
    if not csv_path.exists():
        return False
    try:
        header = pandas.read_csv(csv_path, nrows=0)
    except (pandas.errors.ParserError, pandas.errors.EmptyDataError, OSError):
        return False
    return list(header.columns) == COLUMNS
```

- Valida solo el esquema (no fila a fila), espejo de `validate_metadata_schema`.

### 3.9 Manejo de errores

| Situación | Excepción |
|---|---|
| `CONFIG_FILENAME` ausente en `.env` | `ValueError` |
| Faltan `metadata_csv` / `testing_frames_dir` / `testing_frames_csv` | `KeyError` |
| `db_metadata.csv` inexistente | `FileNotFoundError` (vía `get_abs_path`) |
| Un video de testing no existe | `FileNotFoundError` (vía `extract_frames`) |
| Video corrupto al abrir con decord | excepción de `decord` propagada |
| CSV ilegible en el handler | `validate_testing_frames_schema` → `False` (regenera) |

---

## 4. Cambios de configuración

En `configs/00_testing_config.json`:

```jsonc
"working_dirs": {
  "dataset_dir": "data/raw",
  "sam3_dir": "assets/sam3",
  "outputs_dir": "outputs",
  "metadata_csv": "assets/db_metadata.csv",
  "testing_frames_dir": "data/testing_frames",        // <-- nuevo (imágenes, git-ignored)
  "testing_frames_csv": "assets/testing_frames.csv"   // <-- nuevo (manifiesto, versionado)
}
```

- `data/testing_frames/` se añade a `.gitignore` (dato pesado); `assets/` ya existe
  y el CSV se versiona.

---

## 5. Validación

### 5.1 `testing/test_eval_frame_export.py` (se ejecuta **en el pod**)

A diferencia de otras tareas, este script **se corre en el pod (RunPod)**, no en
local: así las imágenes quedan en el **volumen compartido** disponible para todo el
equipo (no se regeneran por cada quien). No usa modelo ni GPU.

1. Ejecuta `export_testing_frames(force=True)` y confirma que crea el CSV en
   `assets/testing_frames.csv`.
2. Lee el CSV y comprueba: columnas == `COLUMNS` (orden), `id` secuencial `0..M-1`,
   `video_id` ∈ ids de `db_metadata.csv` con `split==2`, exactamente 20 videos
   distintos, `grupo` ∈ {`aleatorio`, `cenital`}.
3. Verifica que cada `imagen` del CSV **existe** en disco y abre como PNG válido.
3b. **Frame original:** para un video de muestra, comprueba que la columna
   `frame_original` coincide con `get_frame_indices(ruta)` alineado por posición con
   `frame_index`, y que `extract_frames` mantiene su salida (misma forma `(N,H,W,3)`).
4. Comprueba que los 2 videos de `splits.forced_testing` quedan marcados `cenital`
   y el resto `aleatorio`.
5. **Idempotencia:** tras generar, `export_testing_frames(force=False)` no reescribe
   (mtime del CSV sin cambios) y `validate_testing_frames_schema` es `True`; tras
   corromper el header, es `False` y se regenera.

### 5.2 Operativo (consecuencia de correr en el pod)

- Como las **imágenes** son git-ignored pero el **CSV es versionado**, tras correr
  el script en el pod hay que **commitear `assets/testing_frames.csv`** (desde el
  pod o sincronizándolo) para que la procedencia llegue al equipo. Las imágenes se
  quedan en el volumen compartido.

### 5.3 Calidad

- `ruff check .` y `black .` sin hallazgos.
- Importabilidad: `from src.data import export_testing_frames, validate_testing_frames_schema`.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Utilidad presente | §3.1 | `src/data/eval_frames.py` + `__init__` |
| AC-2 Origen correcto | §3.4 | filtro `split==2` de `db_metadata.csv` |
| AC-3 Frames en `data/testing_frames/` | §3.6, §3.7, §4 | `testing_frames_dir` |
| AC-4 Cuota desde config | §2, §3.7 | vía `extract_frames` (`preprocess.frame_quota`) |
| AC-5 CSV de control versionado | §3.7, §4 | `assets/testing_frames.csv` |
| AC-6 Identidad estable | §3.7 | `(video_id, frame_index)` |
| AC-6b Frame original | §3.5b, §3.7 | `get_frame_indices` (sin duplicar lógica) |
| AC-7 Grupos correctos | §3.5 | `forced_testing` → `cenital` |
| AC-8 Rutas relativas | §3.7 | `as_posix()` relativo a `PROJECT_ROOT` |
| AC-9 Reproducibilidad | §3.7 | `extract_frames` determinista |
| AC-10 Idempotencia | §3.7, §3.8 | handler + `force` |
| AC-11 Caso borde | §3.7 | `enumerate(frames)` sobre lo disponible |
| AC-12 Sin efectos colaterales | §3.5b | extract_frames con misma salida/firma; solo helper aditivo |
| AC-13 Validación | §5.1 | `testing/test_eval_frame_export.py` (en el pod) |

---

## 7. Riesgos y consideraciones

- **Dos índices, una sola lógica:** se registran `frame_index` (posicional, alinea
  GT y predicción) y `frame_original` (índice en el video fuente, trazabilidad). El
  original se obtiene de `get_frame_indices` (helper aditivo en `frame_extraction`),
  **no** por lógica duplicada: así no hay riesgo de divergencia con el muestreo de
  `extract_frames`. La alineación posicional GT↔predicción sigue dependiendo de usar
  la misma cuota y el mismo `extract_frames`.
- **Doble apertura del video:** `extract_frames` y `get_frame_indices` abren el
  contenedor por separado (decord). Es una lectura de solo-metadatos extra por video
  (20 videos) — coste despreciable; se prioriza reusar la API sobre micro-optimizar.
- **Acoplamiento a la cuota:** si `preprocess.frame_quota` cambia, el set de
  evaluación cambia y el GT anotado dejaría de alinear. Mitigación operativa: la
  cuota se fija antes de anotar y no se toca después (el CSV versionado deja
  constancia del set congelado).
- **Idempotencia por esquema, no por contenido:** si alguien borra imágenes pero
  deja el CSV, `force=False` no las regenera. Aceptable (mismo criterio que
  `build_metadata_csv`); usar `force=True` para reconstruir.
- **Ejecución en el pod:** elegida para que las imágenes vivan en el volumen
  compartido. Implica el paso operativo de commitear el CSV versionado (§5.2).
- **Resolución por imagen:** los frames se guardan a la resolución nativa del video
  (sin resize), para que el GT se anote a tamaño real.
- **Sin GPU ni modelo:** solo lectura de video (decord) + escritura PNG (cv2).
