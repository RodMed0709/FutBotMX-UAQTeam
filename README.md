# Proyecto FutBot MX 2026 — UAQ Team

Detección, segmentación, seguimiento y análisis de videos de fútbol robótico
(Copa FutBotMX). Pipeline base: **YOLO** (detección) → **SAM3** (segmentación) →
**ByteTrack** (tracking). En esta etapa el MVP corre **SAM3** por frame.

> Documento para el equipo. El detalle de cada tarea vive en `.specs/<tarea>/`.
> Antes de tocar código, lee `.specs/constitution.md` (flujo SDD obligatorio).

## Requisitos

- **Python 3.11** en un entorno aislado (venv local o conda `futbot26`).
- **Torch aparte** (según el pod):
  - CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  - GPU (RTX 5090/Blackwell): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- **SAM3** (no está en PyPI): `pip install git+https://github.com/facebookresearch/sam3.git`

## Instalación

```bash
pip install -r requirements.txt          # deps (torch y sam3 van aparte, ver arriba)
pip install -e .                         # src/ como paquete editable -> import src
```

Tras `pip install -e .`, `import src` funciona desde cualquier directorio (notebooks
incluidos), sin parches de `sys.path`.

## Configuración

- Los configs viven en `configs/{NN}_{EXP}.json` y centralizan **rutas relativas** y
  parámetros. El activo se elige con `CONFIG_FILENAME` en `.env`
  (actual: `00_testing_config.json`).
- Las rutas se resuelven con `src.utils.get_abs_path(...)` contra la raíz del
  proyecto — **nunca** hardcodear rutas absolutas.
- Datos (reales, git-ignored): videos en `data/raw/`, modelo en `assets/sam3/`.

## Datos

- 123 videos `.MOV` bajo `data/raw/` (subcarpetas por fecha; búsqueda recursiva).
- Manifiesto del dataset: `assets/db_metadata.csv` (id, ruta, nombre, duración,
  resolución, fps, split). Splits: `0` reserva, `1` fine-tuning, `2` testing.

## Uso

### Pipeline (video → mp4 anotado + JSON)

```python
from src.core import run_pipeline

# Modo cuota (por defecto): N frames muestreados (preprocess.frame_quota).
out = run_pipeline("data/raw/17Abril/Cámaras/IMG_9779.MOV")
print(out)  # {"video": <...>_annotated.mp4, "detections": <...>_detections.json}

# Modo completo: todos los frames, al fps real del video.
run_pipeline("data/raw/17Abril/Cámaras/IMG_9779.MOV", all_frames=True)

# Salida con nombre/ruta explícita.
run_pipeline("data/raw/.../x.MOV", output_path="outputs/demo.mp4")
```

La salida se auto-nombra bajo `working_dirs.outputs_dir` (`<stem>_annotated.mp4` y
`<stem>_detections.json`) si no se pasa `output_path`. El modelo SAM3 carga una sola
vez por llamada. Requiere GPU para la inferencia.

### Manifiesto de metadatos del dataset

```python
from src.data import build_metadata_csv

df = build_metadata_csv()              # crea assets/db_metadata.csv si falta
df = build_metadata_csv(force=True)    # regenera (sobrescribe)
```

Splits reproducibles con `seeds.split`. Videos que deben quedar siempre en testing:
listarlos en `splits.forced_testing` del config.

### Extracción de frames (NumPy en memoria)

```python
from src.core import extract_frames, get_video_fps
from src.utils import show_frames

frames = extract_frames("data/raw/.../x.MOV")                   # (N, H, W, 3), cuota
frames = extract_frames("data/raw/.../x.MOV", all_frames=True)  # todos
fps = get_video_fps("data/raw/.../x.MOV")
show_frames(frames)                                             # vista en grilla
```

## Pruebas (scripts manuales, no pytest)

```bash
python testing/test_env.py             # imports + versiones + torch.cuda
python testing/test_abs_dir_func.py    # get_abs_path contra los configs
python testing/test_frame_extraction.py
python testing/test_metadata.py        # genera y valida db_metadata.csv
```

## Docker

```bash
docker compose --env-file .env -f docker/docker-compose.yml up --build -d
docker compose --env-file .env -f docker/docker-compose.yml exec futbotmx26 \
  python testing/test_metadata.py
```

## Calidad de código

```bash
ruff check .      # lint
black .           # formato
```

## Estructura

```
src/core/    lógica del pipeline (frame_extraction, sam3_loader, segmentation,
             overlay, video_writer, pipeline)
src/data/    preparación del dataset (metadata)
configs/     {NN}_{EXP}.json
data/raw/    videos .MOV (git-ignored)
assets/      sam3/ (modelo, git-ignored) + db_metadata.csv
testing/     scripts manuales
notebooks/   exploración (fase_0/)
.specs/      Spec-Driven Development (una carpeta por tarea)
```
