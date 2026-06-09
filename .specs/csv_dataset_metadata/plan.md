# Plan técnico — Gestión y organización de metadatos del dataset (`csv_dataset_metadata`)

- **Tarea atómica:** `csv_dataset_metadata`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar el submódulo `src/data/metadata.py` que
descubre los videos `.MOV` bajo `dataset_dir`, extrae sus metadatos con `decord`,
asigna *splits* reproducibles y persiste/valida `assets/db_metadata.csv`, sin tocar
`extract_frames` ni el pipeline.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Lectura de video / metadatos:** `decord` (`decord.VideoReader`), ya usado en
  `src/core/frame_extraction.py`.
- **Tabla / CSV:** `pandas` (ya en `requirements.txt`).
- **Muestreo reproducible:** `numpy.random.default_rng(seed)`.
- **Rutas / config:** `src.utils.PROJECT_ROOT` y `get_abs_path`; lectura de
  `.env` + JSON con el mismo patrón de `frame_extraction._load_env` /
  `pipeline._load_pipeline_config`.
- Sin nuevas dependencias.

---

## 3. Diseño

### 3.1 Estructura de archivos

```
src/data/__init__.py        # nuevo submódulo (paquete)
src/data/metadata.py        # lógica de la tarea
testing/test_metadata.py    # script manual standalone
```

`src/data/__init__.py` exporta la API pública:

```python
from src.data.metadata import build_metadata_csv, validate_metadata_schema
__all__ = ["build_metadata_csv", "validate_metadata_schema"]
```

### 3.2 Esquema (fuente única de verdad)

Constantes a nivel de módulo, para que el handler y el escritor compartan una sola
definición y el esquema sea **mutable** desde un solo lugar:

```python
COLUMNS = ["id", "ruta", "nombre", "duracion", "ancho", "alto", "fps_average", "split"]
VIDEO_EXTENSIONS = {".mov"}      # comparación en minúsculas (case-insensitive)
SPLIT_FINETUNING = 1             # 23 videos
SPLIT_TESTING = 2                # 20 videos
SPLIT_RESERVE = 0                # resto
SPLIT_SIZES = {SPLIT_FINETUNING: 23, SPLIT_TESTING: 20}
```

### 3.3 Carga de configuración

Helper local `_load_metadata_config()` (mismo patrón que `pipeline.py`): lee
`CONFIG_FILENAME` del `.env`, abre el JSON y devuelve lo necesario:

```python
def _load_metadata_config() -> tuple[str, str, int]:
    """Devuelve (dataset_dir, metadata_csv, split_seed) desde la config global."""
    # ... parseo .env -> CONFIG_FILENAME -> get_abs_path(f"configs/{...}")
    working_dirs = config.get("working_dirs", {})
    if "dataset_dir" not in working_dirs:
        raise KeyError("Falta 'working_dirs.dataset_dir' en la configuracion.")
    if "metadata_csv" not in working_dirs:
        raise KeyError("Falta 'working_dirs.metadata_csv' en la configuracion.")
    seeds = config.get("seeds", {})
    if "split" not in seeds:
        raise KeyError("Falta 'seeds.split' en la configuracion.")
    return working_dirs["dataset_dir"], working_dirs["metadata_csv"], int(seeds["split"])
```

### 3.4 Descubrimiento determinista de videos

```python
def _discover_videos(dataset_dir: str) -> list[Path]:
    """rglob recursivo de .MOV bajo PROJECT_ROOT/dataset_dir, orden alfabético."""
    base = get_abs_path(dataset_dir)
    videos = [
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda p: p.relative_to(PROJECT_ROOT).as_posix())
```

- `get_abs_path(dataset_dir)` resuelve y valida la existencia del directorio base.
- El **orden alfabético por ruta POSIX relativa** garantiza un `id` estable e
  independiente del sistema de archivos / cwd.

### 3.5 Extracción de metadatos por video

```python
def _extract_video_metadata(abs_path: Path) -> dict:
    reader = decord.VideoReader(str(abs_path))
    n_frames = len(reader)
    fps = float(reader.get_avg_fps())
    frame0 = reader[0]                      # (H, W, 3); bridge nativo -> numpy
    alto, ancho = int(frame0.shape[0]), int(frame0.shape[1])
    duracion = float(n_frames / fps) if fps > 0 else 0.0
    return {"duracion": duracion, "ancho": ancho, "alto": alto, "fps_average": fps}
```

- `decord.bridge.set_bridge("native")` ya se hace en `frame_extraction`; en este
  módulo se establece igual al importar para no depender del orden de imports.
- Solo se lee el primer frame para las dimensiones (decord no expone alto/ancho sin
  leer un frame); el resto son metadatos del contenedor (sin decodificar todo).

### 3.6 Asignación de splits (reproducible)

```python
def _assign_splits(n: int, seed: int) -> list[int]:
    """Devuelve una lista de splits (len n) alineada al orden determinista."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)               # índices barajados de forma reproducible
    splits = [SPLIT_RESERVE] * n
    cursor = 0
    for split_id, size in ((SPLIT_FINETUNING, SPLIT_SIZES[SPLIT_FINETUNING]),
                            (SPLIT_TESTING, SPLIT_SIZES[SPLIT_TESTING])):
        for idx in perm[cursor:cursor + size]:
            splits[idx] = split_id
        cursor += size
    return splits
```

- Disjunto y sin reemplazo por construcción (una permutación, cortes contiguos).
- Si `n < 23 + 20`, se lanza `ValueError` con mensaje claro (no hay videos
  suficientes para los splits fijos).

### 3.7 Función pública orquestadora

```python
def build_metadata_csv(force: bool = False) -> pandas.DataFrame:
    """Descubre -> extrae -> asigna splits -> valida -> escribe assets/db_metadata.csv.

    Idempotente: si el CSV ya existe y pasa el handler de validación y force=False,
    se devuelve sin reescribir. En cualquier otro caso (ausente, esquema inválido o
    force=True) se regenera y sobrescribe por completo.
    """
    dataset_dir, metadata_csv, seed = _load_metadata_config()
    csv_path = PROJECT_ROOT / metadata_csv

    if not force and csv_path.exists() and validate_metadata_schema(csv_path):
        return pandas.read_csv(csv_path)

    videos = _discover_videos(dataset_dir)
    if len(videos) < sum(SPLIT_SIZES.values()):
        raise ValueError(f"Videos insuficientes para los splits: {len(videos)}")

    splits = _assign_splits(len(videos), seed)
    rows = []
    for idx, abs_path in enumerate(videos):
        meta = _extract_video_metadata(abs_path)
        rows.append({
            "id": idx,
            "ruta": abs_path.relative_to(PROJECT_ROOT).as_posix(),
            "nombre": abs_path.name,
            **meta,
            "split": splits[idx],
        })
    df = pandas.DataFrame(rows, columns=COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
```

- `index=False`: la columna `id` es el índice lógico; no se añade índice de pandas.
- `csv_path = PROJECT_ROOT / metadata_csv` (no `get_abs_path`, que exigiría que el
  archivo ya exista; aquí lo estamos creando).

### 3.8 Handler de validación (independiente, esquema mutable)

```python
def validate_metadata_schema(csv_path: Path) -> bool:
    """True si el CSV existe y su estructura coincide con COLUMNS (orden incluido).

    No valida el contenido fila a fila; solo el esquema (existencia + columnas).
    Diseñado para evolucionar si COLUMNS cambia en el futuro.
    """
    if not csv_path.exists():
        return False
    try:
        header = pandas.read_csv(csv_path, nrows=0)
    except (pandas.errors.ParserError, OSError):
        return False
    return list(header.columns) == COLUMNS
```

- Función **separada** de la generación: `build_metadata_csv` la consume para
  decidir si regenera. Cumple el requisito del draft de encapsular la validación.

### 3.9 Manejo de errores

| Situación | Excepción |
|---|---|
| `CONFIG_FILENAME` ausente en `.env` | `ValueError` |
| Faltan claves `dataset_dir`/`metadata_csv`/`seeds.split` | `KeyError` |
| `dataset_dir` inexistente | `FileNotFoundError` (vía `get_abs_path`) |
| Videos insuficientes para splits | `ValueError` |
| Video corrupto al abrir con decord | excepción de `decord` propagada |
| CSV ilegible en el handler | `validate_metadata_schema` → `False` (regenera) |

---

## 4. Cambios de configuración

En `configs/00_testing_config.json`:

```jsonc
"working_dirs": {
  "dataset_dir": "data/raw",
  "sam3_dir": "assets/sam3",
  "outputs_dir": "outputs",
  "metadata_csv": "assets/db_metadata.csv"   // <-- nuevo
},
"seeds": {                                    // <-- nueva sección
  "split": 42
},
```

- La carpeta `assets/` ya existe en el repo (scaffolding); el CSV se versiona.

---

## 5. Validación

### 5.1 `testing/test_metadata.py` (agente, local)

Script standalone (estilo de los demás `test_*.py`) que el **agente ejecuta en
local** (no usa modelo ni GPU; hay videos reales en `data/raw`):

1. Ejecuta `build_metadata_csv(force=True)` y confirma que crea
   `assets/db_metadata.csv`.
2. Lee el CSV y comprueba: columnas == `COLUMNS` (orden), `id` secuencial
   `0..N-1`, una fila por video, `ancho/alto` enteros > 0, `fps_average`/`duracion`
   floats > 0, `ruta` resoluble con `get_abs_path`.
3. Comprueba los conteos de splits (23 / 20 / resto) y que son disjuntos y cubren
   todos los videos.
4. **Reproducibilidad:** dos corridas con `force=True` producen la misma columna
   `split`.
5. **Idempotencia:** tras generar, `build_metadata_csv(force=False)` no reescribe
   (verificable por mtime sin cambios) y `validate_metadata_schema(csv_path)` es
   `True`; tras corromper el header, es `False` y se regenera.

### 5.2 Calidad

- `ruff check .` y `black .` sin hallazgos.
- Importabilidad: `from src.data import build_metadata_csv, validate_metadata_schema`.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Módulo presente | §3.1 | `src/data/metadata.py` + `__init__` |
| AC-2 CSV generado | §3.7 | `build_metadata_csv` → `to_csv` |
| AC-3 Una fila por video | §3.4, §3.7 | `rglob` + `enumerate` |
| AC-4 Metadatos correctos | §3.5 | decord (fps, n_frames, frame0) |
| AC-5 Ruta relativa | §3.7 | `relative_to(PROJECT_ROOT).as_posix()` |
| AC-6 Splits correctos | §3.6 | permutación + cortes contiguos |
| AC-7 Reproducibilidad | §3.6 | `default_rng(seed)` |
| AC-8 Config | §3.3, §4 | `working_dirs.metadata_csv`, `seeds.split` |
| AC-9 Handler | §3.7, §3.8 | `validate_metadata_schema` + regeneración |
| AC-10 Pipeline intacto | (no se toca) | sin cambios en core |
| AC-11 Validación local | §5.1 | `testing/test_metadata.py` |

---

## 7. Riesgos y consideraciones

- **Dimensiones vía primer frame:** decord no expone alto/ancho sin leer un frame;
  leer `reader[0]` decodifica un único frame (coste despreciable). Se asume
  resolución constante por video (cierto para `.MOV` de cámara).
- **`get_avg_fps` promedio:** para videos de fps variable es una aproximación; es la
  mejor referencia disponible y suficiente para el manifiesto.
- **`duracion = n_frames / fps`:** coherente con los metadatos de decord; puede
  diferir levemente de la duración del contenedor, pero es consistente y reproducible.
- **Estabilidad de `id`:** depende del orden alfabético de rutas relativas; añadir
  videos nuevos puede desplazar `id` al regenerar. Aceptable: el CSV se regenera de
  forma explícita y el `id` es un índice del manifiesto, no una clave persistente
  externa.
- **Seed única para los dos splits:** una sola permutación reproducible cubre
  fine-tuning y testing de forma disjunta; basta una entrada `seeds.split`.
- **Sin paralelizar:** 123 lecturas de solo-metadatos son rápidas; mantener el
  código secuencial prioriza la simplicidad (acordado en el spec).
