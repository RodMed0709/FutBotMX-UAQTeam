# tasks.md — `bootstrap_data`

Script interactivo de provisión de datos desde Google Drive, declarativo sobre un
manifiesto versionado, con generación de `.env` y dos paquetes (todos / demos). Orden:

- [x] **T1 — Manifiesto versionado.**
  - Crear `assets/bootstrap_manifest.json` (`schema_version`, `items[]` con `nombre`,
    `paquetes`, `vista?`, `recursos[]` = `{tipo, drive_id, destino}`). IDs como
    placeholders hasta que el equipo suba los archivos (opción A: IDs en el manifiesto).
  - Poblar las entradas demo (clips superiores `IMG_9933`/`IMG_9938` + genéricos
    `*_singular_display`) con sus `tracking_json`, más `sam3_weights`, `yolo_best` y
    `dataset_completo`.

- [x] **T2 — Plantilla `.env.example`.**
  - Crear `.env.example` versionado con `CONFIG_FILENAME=01_yolo_sam3_config.json` y
    `CONTAINER_WORKSPACE_DIR=futbot`.

- [x] **T3 — Lógica pura del script.**
  - `src/bootstrap_data.py`: `load_manifest`, `select_package`, `is_present`,
    `ensure_env` (crea `.env` desde la plantilla si falta; no-destructivo).

- [x] **T4 — Descarga e idempotencia.**
  - `download_resource` (gdown, import perezoso; `dir` vs `file`/`clip`/`tracking_json`;
    `drive_id` acepta URL o ID y se normaliza) y `run_bootstrap(package, dry_run=False)`
    que salta lo presente y acumula reporte.
  - **Recursos `manual: true`** (dataset de la convocatoria): no gdown; si ausente,
    reporta "pendiente (manual)" con enlace + ruta destino (descarga a mano).

- [x] **T5 — Menú interactivo + reporte.**
  - `prompt_package` (`questionary.select`, demos recomendado) y `main()` con reporte
    rich (qué encontró / qué bajó / dónde). Registrar entry point `python -m
    src.bootstrap_data`.

- [x] **T6 — Dependencias.**
  - Añadir `gdown` a `requirements.txt` con nota de uso.

- [x] **T7 — Documentación de reproducibilidad (entregable).**
  - Sección en `README.md` + nota en `docs/`: provisión de datos y el flujo
    "demos → Capa B local sin GPU; `main … --overwrite` → rehace de cero para validar
    reproducibilidad".

- [x] **T8 — Smoke (sin red).**
  - `testing/test_bootstrap_data.py`: manifiesto de prueba → `select_package`,
    `is_present`, `run_bootstrap(dry_run=True)`, `ensure_env` (crea/respeta `.env` en
    tmp). `ruff check .` y `black .`.

## Notas / dependencias

- **Comparte el manifiesto** con la tarea `main_demo_flag` (el `--demo` lo lee para
  listar demos presentes). Definir el esquema del manifiesto aquí primero.
- Los IDs reales de Drive son insumo humano; hasta entonces el script se valida con
  `dry_run` y un manifiesto de prueba.
