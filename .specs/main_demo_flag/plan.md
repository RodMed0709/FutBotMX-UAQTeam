# plan.md — `main_demo_flag`

## Enfoque

Cambio acotado en `main.py`: una flag nueva, un selector que lee el manifiesto, y un
punto de inyección **antes** de `choose_pipeline`/`validate_video` que, si `--demo`
está activo, resuelve el video y la vista desde el demo elegido. `--overwrite` queda
ortogonal e intacto. Imports pesados (`questionary`) perezosos, como ya hace el hub.

## Cambios por archivo

### `main.py`

- **`parse_args`**: añadir `--demo` (`action="store_true"`). Documentar que sobreescribe
  la selección de entrada (`--default`/`--vista`/`video`) y es combinable con
  `--overwrite`. Volver el posicional `video` **opcional** (`nargs="?"`, default `None`).

- **Lector de demos** (nuevo helper, p. ej. `load_demo_choices()`):
  - Lee `assets/bootstrap_manifest.json` (vía `get_abs_path`).
  - Filtra ítems con `"demo"` en `paquetes`.
  - Por cada uno, resuelve la ruta del clip (`recursos[tipo=="clip"].destino`) y
    **comprueba que exista en disco**.
  - Devuelve lista de `{nombre, vista, clip_path}` solo de los **presentes**.

- **`choose_demo()`** (nuevo): si no hay demos presentes → `SystemExit(2)` con mensaje
  que sugiere el bootstrap. Si hay → `questionary.select` mostrando `nombre (vista)`;
  devuelve el `{clip_path, vista}` elegido.

- **`run()`** (orquestación): si `args.demo`:
  1. `demo = choose_demo(console)` **antes** de `validate_video`/`choose_pipeline`.
  2. `video = validate_video(str(demo.clip_path), console)` (reusa la validación).
  3. `choice = choose_pipeline(default=False, vista_arg=demo.vista, config, console)`
     — fuerza modo interactivo para detector/tracker/overlays pero **inyecta la vista**
     del demo (no se pregunta). `--default`/`--vista` se ignoran bajo `--demo`.
  4. El resto del pipeline (incluido el respeto a `args.overwrite`) sigue igual.
  - Si **no** `args.demo`: comportamiento actual sin cambios; si `video` es `None`
    (no se pasó y no hay `--demo`) → error claro pidiendo una ruta o `--demo`.

- **`choose_pipeline`**: aceptar que `vista_arg` venga ya resuelto (del demo) y, en ese
  caso, no preguntar la vista. La firma ya recibe `vista_arg`; basta pasar la del demo
  y saltarse el prompt cuando viene dado (comportamiento actual: si `vista_arg` no es
  `None` no pregunta — reutilizable).

## Riesgos y mitigaciones

- **Manifiesto ausente** (bootstrap aún no corrido): `load_demo_choices` captura
  `FileNotFoundError` y `choose_demo` reporta el mismo error accionable (correr el
  bootstrap).
- **Precedencia confusa:** documentar en `--help` y en el código que `--demo` solo
  pisa la **selección de entrada**, no `--overwrite`.
- **No-TTY:** `questionary` requiere TTY; replicar el guard actual del hub
  (mensaje pidiendo TTY).

## Validación

- Smoke con manifiesto de prueba apuntando a un clip local existente
  (`outputs/fase5_clips/IMG_9933_5m30.mp4` como demo de prueba): `load_demo_choices`
  lista solo los presentes; `choose_demo` (monkeypatch del select) devuelve la ruta y
  vista correctas.
- Smoke de error: sin demos presentes → `SystemExit(2)` con el mensaje del bootstrap.
- Precedencia: `--demo --overwrite` mantiene `args.overwrite=True` en la corrida.
- `ruff check .` y `black .`.
