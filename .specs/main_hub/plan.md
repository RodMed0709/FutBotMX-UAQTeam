# plan.md — `main_hub`

> Paso 3 del SDD. Redacción **técnica** de la solución: stack, arquitectura, contratos y
> decisiones de implementación que satisfacen el `spec.md`. No es la lista ejecutable
> (eso es `tasks.md`). Documento en español.

## 1. Stack técnico

- **Python 3.11** (entorno del proyecto), `import src` editable.
- **Consola**: `questionary` (menús/selección interactiva) + `rich` (paneles, color,
  tabla de resumen, barra de progreso de etapas). Se añaden a `requirements.txt`.
- **Argumentos**: `argparse` de la stdlib (sin dependencia extra) para
  `<ruta_video> [--default] [--overwrite]`.
- **CV/validación de video**: `cv2` vía las utilidades existentes
  (`frame_extraction.get_frame_count`, `get_video_fps`) — importadas **lazy**.
- **Orquestación**: fachadas existentes `run_inference`, `render_obj_id_overlay`,
  `render_broadcast_overlay`; helpers `inference_schema.inference_paths`,
  `events_schema.events_paths`, `utils.get_abs_path`/`PROJECT_ROOT`.

> `questionary` degrada mal en terminales no interactivas; si `stdin` no es TTY y no se
> pasó `--default`, el `main` aborta con un mensaje que sugiere `--default` (ver §7).

## 2. Arquitectura del `main.py`

Archivo único en la **raíz**, con funciones pequeñas y una capa `main()`/`run()`. Imports
pesados (`src.core.*`, cv2) **dentro** de las funciones que los usan (RNF-2). Estructura:

```
main.py
├── parse_args()                      -> argparse.Namespace (video, default, overwrite)
├── validate_video(path)              -> Path  (RF-5/6; lanza SystemExit≠0 si inválido)
├── choose_pipeline(default: bool)    -> PipelineChoice (detector, tracker, overlays)
│      · default=True  -> elección por defecto, sin preguntar (RF-3/12)
│      · default=False -> questionary: detector -> tracker -> ¿overlays? (RF-7)
├── derive_run_label(choice)          -> str  ("<detector>+<tracker>")  (RF-21/§5)
├── plan_outputs(video, run_label)    -> OutputPaths (rutas nativas esperadas)
├── stage_inference(...)              -> dict   (reusa o corre run_inference; RF-16)
├── stage_individual_overlays(...)    -> dict   (opcional; RF-12)
├── stage_broadcast(...)              -> dict   (reusa o corre render_broadcast_overlay)
├── report(results)                   -> None   (rich.Table con rutas + generado/reusado)
└── run(args) / main()                -> int    (orquesta etapas, try/except por etapa)
```

Tipos ligeros (dataclasses o dicts):
- `PipelineChoice(detector: str, tracker: str, want_overlays: bool, default: bool)`.
- `OutputPaths(tracking_json, tracking_video, seg_json, seg_video, obj_overlay,
  broadcast_mp4, broadcast_png)` — todas `Path` en sus rutas **nativas**.
- Cada `stage_*` devuelve `{"status": "generado"|"reusado"|"omitido", "paths": {...}}`.

## 3. Validación de la entrada (RF-5/6)

`validate_video(path)`:
1. Resolver: si es relativa, `get_abs_path(path)`; si es absoluta y existe, usarla.
2. Comprobar `is_file()` y extensión en un set configurable
   `{.mov, .mp4, .avi, .mkv, .m4v}` (case-insensitive).
3. Abrir con cv2 (`get_frame_count(path) >= 1`); capturar excepción → error claro.
4. Si `get_frame_count` supera el umbral `MAX_FRAMES_WARN` (derivado de
   `frame_quota`/fps; p. ej. ~1 min) → **rich warning** y continúa (RF-6).
5. Cualquier fallo → `rich` con el motivo + `raise SystemExit(2)` **antes** de cargar
   nada pesado (no se importa SAM3).

## 4. Selección de piezas (RF-7/8/9/12)

`choose_pipeline(default)`:
- **`--default`** ⇒ `PipelineChoice(detector=<config|sam3_text>, tracker="bytetrack",
  want_overlays=False, default=True)`. El detector por defecto se lee del config activo
  (clave `detector`); si no existe, fallback `sam3_text` (la convención del repo).
- **Interactivo** ⇒ `questionary.select`:
  1. **Detector/segmentador**: opciones de `list(src.core.detectors._DETECTORS)`
     (introspección del registro; hoy `sam3_text`, `yolo_sam3`).
  2. **Tracker**: opciones de `list(src.core.trackers.KNOWN_TRACKERS)`
     (hoy `bytetrack`, `botsort`).
  3. **¿Overlays individuales (segmentación + tracking)?**: `questionary.confirm`.
- **Fijos no preguntados (RF-9)**: homografía `"lines"`, `use_kalman=True` (ya es el
  default tras la decisión 19), `goal_source="strict"`, `layout=2`. El `main` los pasa
  explícitos al broadcast para dejar la intención registrada.

## 5. Namespacing y rutas nativas (RF-19)

- `run_label = f"{detector}+{tracker}"` (p. ej. `sam3_text+bytetrack`). Aísla salidas por
  configuración y mantiene el skip-done **por config** (RF-16/21).
- `plan_outputs` calcula las rutas **sin** crear carpetas:
  - Tracking: `inference_paths(stem, outputs_dir, namespace=run_label)`
    → `outputs/inference/<run_label>/<stem>/<stem>.{json,mp4}`.
  - Segmentación (overlay opcional): `inference_paths(stem, outputs_dir,
    namespace=f"{run_label}/seg")` para no colisionar con el JSON de tracking.
  - Overlay obj_id: `<tracking_json_stem>_obj_id.mp4` junto al JSON (lo nombra
    `render_obj_id_overlay`).
  - Broadcast: `events_paths(stem, "broadcast", "mp4")` y `(... , "png")`
    → `outputs/eventos/<stem>/<stem>_broadcast.{mp4,png}`.
- `outputs_dir` se lee del config activo (`working_dirs.outputs_dir`, default `outputs`).

## 6. Etapas (orquestación e idempotencia A — RF-10..18)

Todas las etapas comprueban **existencia en ruta nativa** antes de ejecutar; con
`--overwrite` se fuerza (se ignora la comprobación).

1. **`stage_inference`** (tracking, RF-10):
   - Si `tracking_json.exists()` y no `--overwrite` ⇒ `reusado` (no carga SAM3).
   - Si no ⇒ `run_inference(video, mode="tracking", detector=..., tracker=...,
     run_label=run_label, include_masks=False, render_video=True, progress=True)`.
   - `include_masks=False`: el broadcast/metric no necesitan RLE; ahorra peso. (Si más
     adelante se quisiera el overlay obj_id con relleno de máscara, se evalúa entonces.)
2. **`stage_individual_overlays`** (solo si `want_overlays`, RF-12):
   - **Tracking**: `render_obj_id_overlay(tracking_json, video_path=<video_crudo>)`
     → `<stem>_obj_id.mp4`. Se pasa el **video crudo** como fuente (no el segmentado).
   - **Segmentación**: `run_inference(video, mode="segmentation", detector=...,
     run_label=f"{run_label}/seg", render_video=True)`. Idempotente por su propio JSON.
   - Cada sub-overlay respeta su skip-done por ruta nativa.
3. **`stage_broadcast`** (entregable, RF-11/13/21):
   - Si `broadcast_mp4.exists()` y no `--overwrite` ⇒ `reusado`.
   - Si no ⇒ `render_broadcast_overlay(tracking_json, clip=<video_crudo>, layout=2,
     goal_source="strict", use_kalman=True, progress=True)`.
   - **`clip=`** es el parámetro nuevo (ver §8): evita el `<stem>.mp4` segmentado y se
     reenvía a la métrica interna.

El post-proceso CPU (homografía/métrica/eventos) ocurre **dentro** de
`render_broadcast_overlay` y **se recalcula** siempre (RF-17): no se persiste ni saltea
granularmente.

## 7. UX de consola y manejo de errores (RF-14/15, RNF-4)

- **Cabecera** `rich` con el video, el modo (interactivo/`--default`) y la config elegida.
- **Progreso**: las fachadas ya muestran su `tqdm` (`progress=True`); el `main` enmarca
  cada etapa con un encabezado `rich` ("▶ Inferencia (tracking)…", etc.).
- **Resumen final** `report()`: `rich.Table` con columnas *Artefacto | Estado
  (generado/reusado/omitido) | Ruta*. Es la materialización de HU-5/RF-20.
- **Errores por etapa**: cada `stage_*` envuelta en `try/except Exception`; al fallar,
  imprime el motivo con `rich`, marca la etapa como fallida y `run()` retorna **≠ 0**
  sin borrar artefactos válidos. La validación de entrada falla **antes** de cargar SAM3.
- **No-TTY sin `--default`**: si `not sys.stdin.isatty()`, abortar con mensaje que sugiere
  `--default` (los prompts no funcionarían).

## 8. Cambio mínimo en `src/` (RF-21, RNF-1)

Único cambio permitido, **retro-compatible**, en
`src/core/event_broadcast_overlay.py::render_broadcast_overlay`:

- Añadir parámetro `clip: str | Path | None = None` (default `None` ⇒ comportamiento
  actual intacto).
- Si `clip is not None`:
  - usarlo como el clip del lienzo/fps en lugar de
    `tracks_json.parent / f"{stem}.mp4"` (con validación `exists()` ya presente);
  - **reenviarlo** a `compute_metric_positions(tracks_json, video=clip, …)`
    (`compute_metric_positions` ya acepta `video=` y lo prioriza, vía `_resolve_clip`).
- Si `clip is None`: ruta actual sin cambios (notebooks y demos siguen igual).

Es un cambio de pocas líneas, sin tocar la lógica de render ni de eventos. Se documenta
en `docs/10_eventos.md` (firma) como nota menor.

## 9. Config y constantes

- Sin rutas absolutas: todo vía `get_abs_path`/`PROJECT_ROOT` y `working_dirs` del config
  (RNF-3).
- Constantes del `main` (en cabecera del archivo): `VIDEO_EXTS`, `MAX_FRAMES_WARN`,
  `DEFAULT_TRACKER="bytetrack"`. El detector por defecto se deriva del config.

## 10. Pruebas (alineadas a la filosofía del repo)

- **Smoke local sin GPU** `testing/test_main_hub.py` (script manual, estilo del repo):
  - `validate_video` acepta un `.MOV` real de `data/raw` y **rechaza** (código≠0) una ruta
    inexistente y un no-video (p. ej. un `.txt`).
  - `choose_pipeline(default=True)` devuelve la elección esperada sin prompts.
  - `plan_outputs` arma las rutas nativas esperadas (string match).
  - Con un **tracking JSON ya existente** (fixture o uno de `outputs/`), `stage_inference`
    reporta `reusado` **sin** importar SAM3 (verifica idempotencia A).
- **Validación visual / etapas pesadas**: en el **pod** (GPU), corrida real end-to-end
  sobre un clip corto; se valida que el broadcast usa el clip **crudo** (sin máscaras) y
  que relanzar reusa. (Se hará cuando se pida, no en este paso.)

## 11. Riesgos y mitigaciones

- **`questionary`/`rich` nuevas deps** → se fijan en `requirements.txt`; son ligeras y
  puras Python. Mitigación: import lazy y fallback no-TTY a `--default`.
- **Header `video` del JSON apunta al pod** (`/workspace/...`) → no se usa para resolver
  el clip; el `main` pasa `clip=` explícito (§8) y, para el overlay obj_id, `video_path=`
  explícito. Mitigado.
- **Detector por defecto del config** podría no existir → fallback `sam3_text`
  documentado (convención del repo).
- **Skip-done por `run_label`** → si el usuario cambia detector/tracker, el `run_label`
  cambia y se genera en otra subcarpeta (no colisiona, no "reusa" indebidamente). Es el
  comportamiento deseado.

## 12. Trazabilidad spec → plan

| Requisito (spec) | Cubierto en |
|---|---|
| RF-1..4 (args) | §1, §2 (`parse_args`) |
| RF-5/6 (validación) | §3 |
| RF-7/8/9/12 (selección/fijos) | §4 |
| RF-10/11/13 (etapas/orden) | §6 |
| RF-16/17/18 (idempotencia A) | §6 |
| RF-19/20 (no mover + reporte) | §5, §7 (`report`) |
| RF-21 (clip crudo) | §6.3, §8 |
| RNF-1 (cambio src acotado) | §8 |
| RNF-2 (lazy imports) | §2 |
| RNF-3 (config-driven) | §5, §9 |
| RNF-4 (errores por etapa) | §7 |
| RNF-5 (español) | todo el `main` |
