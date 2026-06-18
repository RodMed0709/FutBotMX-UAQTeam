# tasks.md — `main_hub`

> Paso 4 del SDD. Descompone el `plan.md` en tareas **ejecutables y verificables**. La
> implementación (paso 5) solo empieza una vez aprobado este archivo. Documento en español.

## Convenciones
- Cada tarea es atómica y tiene **criterio de hecho** (DoD).
- Orden recomendado; las dependencias se anotan con `⟵`.
- "Verificación" = comando o chequeo concreto (la mayoría smoke local sin GPU).

---

### T0 — Dependencias de consola
- [ ] Añadir `questionary` y `rich` a `requirements.txt` (sección de tooling, con
  comentario "consola del hub `main.py`").
- **DoD**: `pip install -r requirements.txt` resuelve; `python -c "import questionary,
  rich"` funciona en el entorno activo.

### T1 — Cambio mínimo en `src/` (param `clip=`)  ⟵ ninguno
- [ ] En `src/core/event_broadcast_overlay.py::render_broadcast_overlay` añadir
  `clip: str | Path | None = None` (default `None`).
- [ ] Si `clip is not None`: usarlo como clip del lienzo/fps (en vez de
  `tracks_json.parent/<stem>.mp4`), validando `exists()`; y **reenviarlo** a
  `compute_metric_positions(tracks_json, video=clip, …)`.
- [ ] Si `clip is None`: comportamiento actual intacto.
- [ ] Actualizar docstring de la función y la firma citada en `docs/10_eventos.md`.
- **DoD**: llamada sin `clip=` se comporta igual que hoy (retro-compatible); con
  `clip=<ruta>` el lienzo y la métrica usan ese video. Verificación: `ruff check
  src/core/event_broadcast_overlay.py` limpio + revisión de que `compute_metric_positions`
  recibe `video=clip`.

### T2 — Esqueleto de `main.py` y CLI  ⟵ T0
- [ ] Crear `main.py` en la raíz con cabecera de constantes (`VIDEO_EXTS`,
  `MAX_FRAMES_WARN`, `DEFAULT_TRACKER="bytetrack"`).
- [ ] `parse_args()` con `argparse`: `video` (posicional), `--default`, `--overwrite`.
- [ ] `main()`/`run(args)` que encadena las etapas y retorna **código de salida int**.
- [ ] Imports pesados (`src.core.*`, `cv2`) **dentro** de funciones (lazy).
- **DoD**: `python main.py --help` muestra los 3 parámetros; `python main.py` sin ruta
  falla con mensaje de uso (código≠0).

### T3 — Validación de la entrada (`validate_video`)  ⟵ T2
- [ ] Resolver ruta (relativa → `get_abs_path`; absoluta existente → usar).
- [ ] Chequear `is_file()` + extensión en `VIDEO_EXTS` (case-insensitive).
- [ ] Abrir con cv2 (`get_frame_count >= 1`); excepción → error claro.
- [ ] Si frames > `MAX_FRAMES_WARN` → warning `rich` y continúa.
- [ ] Fallos → `rich` con motivo + `SystemExit(2)` **antes** de cargar SAM3.
- **DoD**: rechaza ruta inexistente y un `.txt`; acepta un `.MOV` real; el warning de
  video largo aparece pero no detiene. Verificación: `testing/test_main_hub.py` (T9).

### T4 — Selección de piezas (`choose_pipeline`)  ⟵ T2
- [ ] `--default` ⇒ `PipelineChoice(detector=<config|sam3_text>, tracker="bytetrack",
  want_overlays=False, default=True)` sin prompts.
- [ ] Interactivo ⇒ `questionary.select` para detector (de `_DETECTORS`), tracker (de
  `KNOWN_TRACKERS`) y `questionary.confirm` para overlays.
- [ ] Detector por defecto leído del config activo (clave `detector`), fallback
  `sam3_text`.
- [ ] Guard no-TTY: si `not sys.stdin.isatty()` y no `--default` ⇒ abortar sugiriendo
  `--default`.
- **DoD**: `choose_pipeline(default=True)` devuelve la elección esperada sin prompts;
  las opciones interactivas salen de los registros (no hardcodeadas).

### T5 — Rutas nativas (`derive_run_label` + `plan_outputs`)  ⟵ T4
- [ ] `run_label = f"{detector}+{tracker}"`.
- [ ] `plan_outputs` calcula (sin crear carpetas): tracking
  `inference_paths(stem, outputs_dir, namespace=run_label)`; segmentación
  `namespace=f"{run_label}/seg"`; obj_id overlay `<stem>_obj_id.mp4` junto al JSON;
  broadcast `events_paths(stem, "broadcast", "mp4"|"png")`.
- [ ] `outputs_dir` leído de `working_dirs.outputs_dir` (default `outputs`).
- **DoD**: las rutas devueltas coinciden (string match) con el esquema del `plan.md` §5.

### T6 — Etapa inferencia (`stage_inference`, idempotente)  ⟵ T5
- [ ] Si `tracking_json.exists()` y no `--overwrite` ⇒ `reusado` (no importa SAM3).
- [ ] Si no ⇒ `run_inference(video, mode="tracking", detector, tracker,
  run_label, include_masks=False, render_video=True, progress=True)`.
- [ ] Devuelve `{"status", "paths"}`.
- **DoD**: con un JSON existente reporta `reusado` sin tocar GPU; sin él, invoca
  `run_inference` con los args correctos (revisión).

### T7 — Etapa overlays individuales (`stage_individual_overlays`)  ⟵ T6
- [ ] Solo si `want_overlays`.
- [ ] Tracking: `render_obj_id_overlay(tracking_json, video_path=<video_crudo>)`.
- [ ] Segmentación: `run_inference(video, mode="segmentation", detector,
  run_label=f"{run_label}/seg", render_video=True)`.
- [ ] Respeta skip-done por ruta nativa de cada sub-overlay.
- **DoD**: con `want_overlays=False` la etapa se marca `omitido`; con `True` genera
  ambos overlays usando el **video crudo** como fuente.

### T8 — Etapa broadcast (`stage_broadcast`, entregable)  ⟵ T1, T6
- [ ] Si `broadcast_mp4.exists()` y no `--overwrite` ⇒ `reusado`.
- [ ] Si no ⇒ `render_broadcast_overlay(tracking_json, clip=<video_crudo>, layout=2,
  goal_source="strict", use_kalman=True, progress=True)`.
- **DoD**: genera el broadcast usando el clip **crudo** (sin máscaras quemadas); relanzar
  lo reporta `reusado`.

### T9 — Reporte y manejo de errores (`report` + try/except)  ⟵ T6, T7, T8
- [ ] `report()` imprime `rich.Table` *Artefacto | Estado (generado/reusado/omitido) |
  Ruta* con todas las salidas (RF-20/HU-5).
- [ ] Cada `stage_*` envuelta en `try/except`; al fallar, motivo con `rich`, etapa
  marcada fallida y `run()` retorna ≠ 0; no borra artefactos válidos.
- [ ] Cabecera `rich` inicial (video, modo, config elegida).
- **DoD**: una corrida exitosa termina con la tabla de rutas y código 0; un fallo
  simulado en una etapa produce código ≠ 0 y mensaje claro.

### T10 — Smoke test local (`testing/test_main_hub.py`)  ⟵ T3..T9
- [ ] Script manual estilo repo (no pytest): import directo `from main import ...`
  (o `import main`).
- [ ] Casos: `validate_video` acepta `.MOV` real y rechaza inexistente/`.txt`;
  `choose_pipeline(default=True)` sin prompts; `plan_outputs` string-match; con un
  tracking JSON fixture, `stage_inference` ⇒ `reusado` **sin** importar SAM3.
- **DoD**: `python testing/test_main_hub.py` corre en local **sin GPU** y todos los
  asserts pasan.

### T11 — Lint, doc y cierre  ⟵ todas
- [ ] `ruff check .` y `black .` limpios sobre `main.py`, el cambio en `src/` y el test.
- [ ] Nota breve de uso del `main.py` (dónde se documentará: README final / docs). En
  este paso basta un docstring de módulo en `main.py` con ejemplos de invocación.
- [ ] Revisar trazabilidad: todos los RF/RNF del spec tienen tarea asociada.
- **DoD**: lint/format OK; `python main.py --help` y el smoke test verdes; checklist de
  trazabilidad completa.

---

## Resumen de archivos tocados (paso 5)
- **Nuevo**: `main.py` (raíz), `testing/test_main_hub.py`.
- **Modificado (mínimo)**: `src/core/event_broadcast_overlay.py` (param `clip=`),
  `docs/10_eventos.md` (firma), `requirements.txt` (`questionary`, `rich`).

## Orden de ejecución sugerido
T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11.

## Verificación final (criterios de aceptación del spec)
- CA-1/CA-2: corridas interactiva y `--default` producen el broadcast y reportan rutas.
- CA-3: relanzar no re-infiere ni re-renderiza; `--overwrite` sí.
- CA-4: entrada inválida → código ≠ 0 sin cargar SAM3.
- CA-5: rutas nativas mostradas en pantalla; nada se mueve.
- CA-6: broadcast con clip crudo (sin máscaras).
