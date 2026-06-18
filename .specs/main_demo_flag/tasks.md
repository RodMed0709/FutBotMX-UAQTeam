# tasks.md — `main_demo_flag`

Flag `--demo` en `main.py`: demo guiada cuyo primer paso es elegir el clip demo del
manifiesto. Depende del esquema del manifiesto definido en `bootstrap_data`. Orden:

- [x] **T1 — Flag y posicional opcional.**
  - `parse_args`: añadir `--demo` (`store_true`); `video` pasa a `nargs="?"`
    (default `None`). `--help` documenta que `--demo` sobreescribe la selección de
    entrada (`--default`/`--vista`/`video`) y es combinable con `--overwrite`.

- [x] **T2 — Lector de demos desde el manifiesto.**
  - `load_demo_choices()`: lee `assets/bootstrap_manifest.json`, filtra `paquetes`
    incluye `"demo"`, resuelve el `clip` y devuelve solo los **presentes** en disco
    (`{nombre, vista, clip_path}`). Captura `FileNotFoundError` (manifiesto ausente).

- [x] **T3 — Selector interactivo.**
  - `choose_demo(console)`: sin demos presentes → `SystemExit(2)` sugiriendo
    `python -m src.bootstrap_data`; con demos → `questionary.select` (`nombre (vista)`)
    → devuelve `{clip_path, vista}`.

- [x] **T4 — Inyección en la orquestación.**
  - `run()`: si `args.demo`, antes de `validate_video`/`choose_pipeline`, elegir demo,
    validar su clip y forzar `choose_pipeline` interactivo con `vista_arg = demo.vista`
    (ignorando `--default`/`--vista`). `--overwrite` intacto. Si no `--demo` y `video`
    es `None` → error claro.

- [x] **T5 — Smoke (sin GPU).**
  - `testing/test_main_demo_flag.py`: manifiesto de prueba → `load_demo_choices` lista
    solo presentes; `choose_demo` (monkeypatch del select) devuelve ruta/vista;
    sin demos → `SystemExit(2)`; `--demo --overwrite` conserva `overwrite=True`.
  - `ruff check .` y `black .`.

## Notas / dependencias

- **Depende de** `bootstrap_data` para el esquema y la presencia del manifiesto
  `assets/bootstrap_manifest.json`. Implementar tras (o coordinado con) esa tarea.
- No auto-descarga: si faltan demos, dirige al bootstrap (separación de
  responsabilidades).
