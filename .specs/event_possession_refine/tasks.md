# Tasks — Posesión vs control (`event_possession_refine`)

- **Tarea atómica:** `event_possession_refine`
- **Paso de la metodología:** 4 (Descomposición en tareas) → habilita el paso 5
  (implementación).
- **Spec/plan de referencia:** `.specs/event_possession_refine/{spec,plan}.md`.

> A partir de aquí (y **solo** aquí) se autoriza escribir/modificar código.

---

## T1 · Estructuras y base reutilizada

- [x] Crear `src/core/event_possession_refine.py` con `PossessionRefineResult`
      (`posesion_por_frame`, `control_por_frame`, `resumen`).
- [x] Importar de `events`/`events_core` lo reutilizable: `load_frame_objects`,
      `ball_centroid`, `_raw_possession`, `_nearest_robot`, `_bbox_diagonal`, `ROBOT_CLASS`.
- [x] Docstring de módulo en español (rol: separar posesión de control + histéresis
      adaptativa por cambio radical; Capa A en píxeles, sin GPU/homografía).

**Verificación:** `python -c "import src.core.event_possession_refine"` sin error.

---

## T2 · Posesión con histéresis adaptativa

- [x] Serie **cruda** de posesión vía `_raw_possession(by_frame, gate_k)`.
- [x] Serie del balón por frame (`ball_centroid`) y diagonal del robot poseedor
      (`_bbox_diagonal`) para los umbrales.
- [x] **Cambio radical**: al aparecer un poseedor distinto del actual, es radical si
      (a) salto del balón `dist(ball[f], ball[f_prev]) > jump_k·diag` **o** (b) cercanía clara
      (nuevo poseedor a `< clear_factor ·` la distancia del actual). Radical ⇒ confirma en 1
      frame; si no ⇒ racha `min_frames` (histéresis normal). Balón no visible ⇒ `None`.

**Verificación:** un salto grande del balón a otro robot cambia la posesión en ese frame;
un parpadeo de cercanía no la cambia hasta `min_frames`.

---

## T3 · Capa de control

- [x] Por frame con poseedor `oid`, evaluar la **ventana** `control_window`: el balón se
      mueve si su desplazamiento en la ventana ≥ `move_k·diag(oid)`.
- [x] `control[f] = oid` si hay poseedor **y** el balón se mueve; si no, `None` (posesión
      pasiva). Muestras insuficientes en la ventana ⇒ no se declara control.
- [x] Garantizar `control ⊆ posesión` por construcción.

**Verificación:** un tramo de balón quieto junto a un robot ⇒ posesión sin control.

---

## T4 · API pública y resumen

- [x] `compute_possession_refine(by_frame, *, gate_k, min_frames, control_window, move_k,
      jump_k, clear_factor, fps)` → `PossessionRefineResult` (firma del plan §6).
- [x] `resumen`: `posesion_por_obj`/`control_por_obj` (frames, segundos, pct), totales
      (`pct_posesion_total`, `pct_control_total`, `pct_libre`, `pct_no_visible`),
      `cambios_de_posesion`, `cambios_de_control`, `fps`, `params`.
- [x] `write_possession_refine_json(result, path)` (estilo `write_*` de fase_5).
- [x] Imports perezosos; sin GPU.

**Verificación:** `compute_possession_refine(<by_frame>)` devuelve resultado coherente
(`pct_control_total ≤ pct_posesion_total`); el JSON se escribe.

---

## T5 · Test manual + viz

- [x] Crear `testing/test_event_possession_refine.py` (script directo, sin pytest, sin GPU),
      default sobre `IMG_9933_5m30`.
- [x] Invariantes: `control ⊆ posesión` por frame; `pct_control ≤ pct_posesion` (total y por
      obj); cobertura de frames (control + posesión-sin-control + libre + no_visible = 100%).
- [x] Caso de coherencia: en el tramo estable de poseedor más largo, su control < su posesión.
- [x] Viz: línea de tiempo de 2 filas (posesión / control), color por `obj_id`, →
      `events_paths(stem, "possession_refine", "png")`; JSON → `…"json"`.

**Verificación:** `python testing/test_event_possession_refine.py` termina OK (local).

---

## T6 · Cierre

- [x] `ruff check` limpio en los archivos nuevos.
- [x] Confirmar que `events.py`/`events_core.py` quedaron intactos (solo importados).
- [x] Confirmar con el usuario antes de cualquier commit (constitución §7.1/§11).

---

## Orden sugerido

T1 → T2 (histéresis adaptativa) → T3 (control) → T4 (API/resumen) → T5 (test/viz) → T6.

---

## Fuera de alcance (recordatorio del spec)

- No introduce equipos/bandos ni usa homografía/cm (movimiento del balón en píxeles).
- No conecta el resultado al overlay (es `event_broadcast_overlay`).
- No cambia ni deprecia `events.compute_possession` (se consume).
