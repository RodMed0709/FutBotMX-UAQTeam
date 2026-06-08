# Tasks — Tracking por detección per-frame + ByteTrack (`video_tracking`)

- **Tarea atómica:** `video_tracking`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** la **ejecución de las pruebas (Fase E) corre EXCLUSIVAMENTE
> en el pod (GPU)** — requiere el modelo SAM3 y GPU; **no se corre en local**. El
> lint estático (`ruff`/`black`) sí puede correr en cualquier entorno.

---

## Fase A — Dependencia y configuración

- [x] **T1 — Añadir `supervision` a `requirements.txt`**
  - Declarar la dependencia `supervision` (ByteTrack); instalarla en el entorno.
  - **Verificación:** `supervision` aparece en `requirements.txt` e importa en el
    entorno del pod.
  - **Plan:** §2, §4. **Spec:** AC-12.

- [x] **T2 — Añadir la sección `tracking` al config**
  - En `configs/00_testing_config.json`, añadir `"tracking": {track_thresh,
    track_buffer, match_thresh, max_frames}` (con defaults; `max_frames: null`).
  - **Verificación:** el JSON sigue válido; la sección y sus claves están presentes.
  - **Plan:** §4. **Spec:** AC-10.

---

## Fase B — Helpers aditivos (sin tocar las funciones existentes)

- [x] **T3 — `iter_frames` en `src/core/frame_extraction.py`**
  - Generador `iter_frames(video_path, max_frames=None)` que hace
    `yield (frame_index, frame_rgb)` vía decord, sin cargar todo el video; reusa
    `_resolve_video_path`.
  - **Verificación:** itera frames de un video real; `extract_frames` y
    `get_frame_indices` mantienen su salida/firma.
  - **Plan:** §3.4. **Spec:** AC-6.

- [x] **T4 — `open_video_writer` en `src/core/video_writer.py`**
  - Context manager incremental (abre el writer `imageio`, `yield append`, cierra),
    reusando codec/fps de `write_video`.
  - **Verificación:** escribe un mp4 frame a frame; `write_video` (batch) intacto.
  - **Plan:** §3.5. **Spec:** AC-6, AC-8, AC-11.

---

## Fase C — Núcleo de tracking (`src/core/tracking.py`)

- [x] **T5 — Modelo de datos y carga de config**
  - Dataclasses `TrackObservation` y `Track`; helper `_load_tracking_config()` que
    devuelve `{track_thresh, track_buffer, match_thresh, max_frames}` con defaults.
  - **Verificación:** las dataclasses se construyen; `_load_tracking_config()` lee
    la sección `tracking` real con defaults ante claves ausentes.
  - **Plan:** §3.2, §3.3. **Spec:** AC-10.

- [x] **T6 — Máscara→caja + ByteTrack por clase + `obj_id` global**
  - `_mask_to_xyxy` (boundingRect; `None` si vacía); un `sv.ByteTrack` por clase;
    construir `sv.Detections(xyxy, confidence, data={"src": idx})`, `update_with_
    detections`, recuperar el `Detection` origen y mapear `(clase, tracker_id) ->
    obj_id` global único.
  - **Verificación:** sobre detecciones de prueba, un mismo objeto conserva su
    `obj_id` entre frames; `obj_id` únicos entre clases; clase correcta por
    construcción.
  - **Plan:** §3.6, §3.7. **Spec:** AC-3, AC-4, AC-7.

- [x] **T7 — Orquestador `track_video` (streaming)**
  - Recorre con `iter_frames`; por frame: `detect_classes_in_frame` → T6 →
    `dict[clase, list[Detection]]` con `obj_id` estable → `overlay_detections` →
    `open_video_writer.append`. Auto-naming de salidas bajo `outputs/`. Reúsa
    `load_sam3` y `get_video_fps`. Sin retener máscaras de todos los frames.
  - **Verificación:** corre end-to-end sobre un clip; genera mp4; memoria acotada
    (no acumula máscaras/frames).
  - **Plan:** §3.8. **Spec:** AC-1, AC-2, AC-5, AC-6, AC-8, AC-11.

- [x] **T8 — JSON del índice + `get_trajectories` + API**
  - Serializar `tracks` a JSON (sin máscaras); `get_trajectories(tracks)`; exponer
    `track_video` y `get_trajectories` (imports). Manejo de errores (§3.10).
  - **Verificación:** el JSON se escribe y recarga; `get_trajectories` devuelve
    centroides; `from src.core.tracking import track_video, get_trajectories` OK.
  - **Plan:** §3.8, §3.9, §3.10. **Spec:** AC-8, AC-9.

---

## Fase D — Test

- [x] **T9 — Crear `testing/test_tracking.py` (dos pruebas)**
  - **Prueba A (clip corto):** `track_video(video, max_frames=pequeño)`; verifica
    mp4+JSON, ≥1 `obj_id` en varios frames, clases válidas, `obj_id` únicos, índice
    sin máscaras, `get_trajectories`.
  - **Prueba B (video real no-forzado):** selección **determinista** de un video que
    **NO** esté en `splits.forced_testing` (p. ej. menor `id` no forzado en
    `db_metadata.csv`); `track_video(video, max_frames=None)`; verifica que completa
    sin OOM, mp4+JSON, tracks a lo largo del tiempo, objetos nuevos con track.
  - **Verificación:** el script existe y sus comprobaciones son ejecutables.
  - **Plan:** §5.1, §5.2. **Spec:** AC-13.

---

## Fase E — Ejecución en el pod y calidad

- [ ] **T10 — Ejecutar ambas pruebas en el pod (GPU)**
  - Correr `testing/test_tracking.py` **en el pod** (modelo SAM3 + GPU). **No se
    corre en local.** La prueba B puede ser de corrida larga.
  - **Verificación:** ambas pruebas pasan en el pod; salidas presentes
    (mp4 + JSON); identidad estable y objetos nuevos confirmados.
  - **Plan:** §5.1, §5.2. **Spec:** AC-13.

- [x] **T11 — Calidad e importabilidad**
  - `ruff check .` y `black .` sin hallazgos; import de la API correcto.
  - **Verificación:** lint limpio; import OK (en el entorno con `supervision`).
  - **Plan:** §5.3. **Spec:** AC-1.

- [ ] **T12 — Commit (requiere confirmación)**
  - Commitear el código/config/requirements. **El agente NO commitea por iniciativa
    propia:** pregunta y espera confirmación (constitución §11). Conventional
    Commits en inglés, scope `video_tracking`.
  - **Verificación:** tras tu confirmación, el commit existe.
  - **Plan:** —. **Spec:** —

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 dependencia `supervision` | §2, §4 | AC-12 |
| T2 config `tracking` | §4 | AC-10 |
| T3 `iter_frames` | §3.4 | AC-6 |
| T4 `open_video_writer` | §3.5 | AC-6, AC-8, AC-11 |
| T5 modelo de datos + config | §3.2, §3.3 | AC-10 |
| T6 máscara→caja + ByteTrack | §3.6, §3.7 | AC-3, AC-4, AC-7 |
| T7 orquestador `track_video` | §3.8 | AC-1, AC-2, AC-5, AC-6, AC-8, AC-11 |
| T8 JSON + trayectorias + API | §3.8, §3.9, §3.10 | AC-8, AC-9 |
| T9 crear test (A + B) | §5.1, §5.2 | AC-13 |
| T10 ejecutar en pod | §5.1, §5.2 | AC-13 |
| T11 calidad/import | §5.3 | AC-1 |
| T12 commit (confirmación) | — | — |
