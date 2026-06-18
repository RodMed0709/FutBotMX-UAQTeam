# plan.md — `bootstrap_data`

## Enfoque

Script de paquete (`src/bootstrap_data.py`) **declarativo**: toda la verdad de "qué
bajar / a dónde / desde qué ID de Drive" vive en `assets/bootstrap_manifest.json`; el
script solo lee el manifiesto, filtra por paquete, verifica presencia y descarga lo
faltante. Imports pesados perezosos (`gdown`, `questionary`) dentro de las funciones.

## Artefactos

### 1. `assets/bootstrap_manifest.json` (versionado, fuente de verdad)

```json
{
  "schema_version": 1,
  "items": [
    { "nombre": "IMG_9933_5m30", "paquetes": ["demo"], "vista": "superior",
      "recursos": [
        { "tipo": "clip", "drive_id": "<ID>", "destino": "data/raw/demos/IMG_9933_5m30.mp4" },
        { "tipo": "tracking_json", "drive_id": "<ID>",
          "destino": "outputs/inference/yolo_sam3+bytetrack/IMG_9933_5m30/IMG_9933_5m30.json" }
      ] },
    { "nombre": "sam3_weights", "paquetes": ["all", "demo"],
      "recursos": [ { "tipo": "dir", "drive_id": "<FOLDER_ID>", "destino": "assets/sam3" } ] },
    { "nombre": "yolo_best", "paquetes": ["all", "demo"],
      "recursos": [ { "tipo": "file", "drive_id": "<ID>", "destino": "assets/yolo/best.pt" } ] },
    { "nombre": "dataset_completo", "paquetes": ["all"],
      "recursos": [ { "tipo": "dir", "drive_id": "<FOLDER_ID>", "destino": "data/raw" } ] }
  ]
}
```

- `tipo`: `clip` | `tracking_json` | `file` (archivo único) | `dir` (carpeta de Drive).
- `destino`: ruta **relativa** a `PROJECT_ROOT`.
- `drive_id`: acepta el **ID pelón** o la **URL completa** de compartir; el descargador
  normaliza (extrae el ID de `/d/<ID>/` o `/folders/<ID>`).
- Agregar un demo = agregar un ítem; cero código.

> **Límite de gdown en carpetas → dataset manual:** `gdown.download_folder` baja ~50
> archivos máx. por carpeta sin autenticar, y `17Abril` tiene 88 videos. Por eso los
> `dir` del dataset llevan `manual: true`: el bootstrap solo verifica presencia y, si
> falta, imprime el enlace + instrucciones para bajarlo a mano. La reproducibilidad la
> cubre el **paquete demo** (gdown sobre archivos sueltos, sin límite).

### 2. `.env.example` (versionado, plantilla del `.env`)

```
CONFIG_FILENAME=01_yolo_sam3_config.json
CONTAINER_WORKSPACE_DIR=futbot
```

### 3. `src/bootstrap_data.py`

Funciones puras y testeables, separando lógica de IO:

- `load_manifest(path=...) -> dict` — lee y valida el manifiesto.
- `select_package(items, package) -> list` — filtra ítems por `"all"`/`"demo"`.
- `is_present(recurso) -> bool` — verifica presencia (archivo existe; `dir` con ≥1
  archivo; `clip`/`tracking_json` archivo existe).
- **Recursos `manual: true`** (dataset de la convocatoria): `download_resource` **no**
  intenta gdown; si está ausente, acumula una entrada "pendiente (manual)" con el enlace
  y la ruta destino para imprimir instrucciones. Si está presente, "ok".
- `ensure_env(project_root) -> str` — crea `.env` desde `.env.example` si falta;
  si existe, no toca y reporta llaves faltantes. Devuelve estado (`creado`/`presente`).
- `download_resource(recurso)` — `gdown` (import perezoso); `dir` → `gdown.download_folder`,
  `file`/`clip`/`tracking_json` → `gdown.download`. Crea el dir padre.
- `run_bootstrap(package, *, dry_run=False) -> Report` — orquesta: ensure_env → filtra
  → por recurso ausente descarga → acumula reporte.
- `prompt_package() -> str` — menú `questionary.select` (`demos` recomendado);
  retorna `"all"`/`"demo"` o `None` (salir).
- `main()` — `prompt_package()` → `run_bootstrap()` → imprime reporte (rich).

### 4. `requirements.txt`

- Añadir `gdown` (documentar que descarga de Drive).

### 5. Documentación (entregable de reproducibilidad)

- `README.md`: sección de provisión de datos + el flujo de reproducibilidad
  (demos → Capa B local sin GPU; `--overwrite` → rehace de cero).
- `docs/`: nota equivalente en la fase correspondiente.

## Riesgos y mitigaciones

- **IDs de Drive aún inexistentes:** el manifiesto se versiona con placeholders y se
  llena cuando el equipo suba los archivos; `run_bootstrap(dry_run=True)` permite probar
  el filtrado/verificación sin descargar.
- **`dir` de Drive grande (dataset/SAM3):** `gdown.download_folder` con reanudación;
  idempotencia evita re-bajar.
- **`.env` existente con llaves viejas:** no se sobreescribe; solo se reporta lo
  faltante (evita pisar configuración local del usuario).

## Validación

- Smoke sin red: manifiesto de prueba → `select_package`, `is_present`,
  `run_bootstrap(dry_run=True)` reportan correcto; `ensure_env` crea `.env` en un tmp y
  respeta uno existente.
- `ruff check .` y `black .`.
