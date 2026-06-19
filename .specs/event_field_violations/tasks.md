# Tasks — Violaciones de campo (`event_field_violations`)

- **Tarea atómica:** `event_field_violations`
- **Paso de la metodología:** 4 (Descomposición en tareas) → habilita el paso 5
  (implementación).
- **Spec/plan de referencia:** `.specs/event_field_violations/{spec,plan}.md`.

> A partir de aquí (y **solo** aquí) se autoriza escribir/modificar código.

---

## T1 · Estructuras y geometría base

- [x] Crear `src/core/event_field_violations.py` con `FieldViolationEvent` y
      `FieldViolationsResult` (campos del plan §2).
- [x] Importar lo reutilizable: `compute_metric_positions`/`MetricResult`, `field_template`
      (rectángulo de líneas, boca, `_penalty_outline_cm`), `_events_from_series`,
      `load_frame_objects`/`ball_centroid`, `ROBOT_CLASS`/`BALL_CLASSES`.
- [x] Constantes de campo (cm): `FIELD_X0/X1/Y0/Y1`, boca `[61,121]`, polígonos de área chica
      (yellow/blue) cerrados.
- [x] Docstring de módulo en español (3 detectores; fuera=cm geométrico, lack/pushing=px prob.).

**Verificación:** `python -c "import src.core.event_field_violations"` sin error.

---

## T2 · Geometría: predicados `fuera`

- [x] `_out_of_field(xy)`: fuera del rectángulo de líneas **y** no en la boca de portería.
- [x] `_in_penalty(xy) -> "yellow"|"blue"|None`: punto-en-polígono del área chica (D) con
      `cv2.pointPolygonTest`, tolerancia `line_margin_cm`.
- [x] `es_fuera(xy)` + `causa`/`zona` (salida_campo vs area_chica).

**Verificación:** un robot en la boca **no** es fuera; un punto en el polígono del área chica
sí (`causa="area_chica"`).

---

## T3 · Detector `fuera` (Capa B, cm)

- [x] Posiciones de robots en cm (`compute_metric_positions`, `cls=="robot"`), por `obj_id` y
      frame; serie `near[f]=es_fuera` por robot (huecos = no visible).
- [x] `_events_from_series` por robot → episodios; `causa`/`zona`/`ref` del frame de apertura;
      `probabilidad=1.0`.
- [x] Si no hay cm fiable, omitir `fuera` y marcar `fuera_disponible=False`.

**Verificación:** sobre `IMG_9933_5m30` los `fuera` (si los hay) tienen `causa` y `ref` en cm
coherentes; sin cm, no se emiten.

---

## T4 · Detector `lack_of_progress` (Capa A, prob.)

- [x] Balón en px (`ball_centroid`); ventana `lop_window`; estancado si paso medio <
      `lop_move_thresh_k · diagonal del balón`.
- [x] Serie con relleno de huecos (`gap_frames`) → `_events_from_series` (min_frames ≈
      ventana) → episodios.
- [x] `probabilidad ∈ (0,1]` creciente con duración y quietud (fórmula documentada);
      `obj_ids=[]` (opcional: poseedor dominante del tramo).

**Verificación:** el tramo de balón parado (≈ f502-676) sale como `lack_of_progress` con
probabilidad alta.

---

## T5 · Detector `pushing` (Capa A, prob.) — solo en área chica

- [x] Pares de robots en contacto por frame (IoU > `push_iou` **o** centroides <
      `push_k·(radio_i+radio_j)`).
- [x] Restricción: contacto **dentro del área chica** (cm con `_in_penalty`, o proxy px con
      bbox de zona si no hay cm). **Sin** requisito de desplazamiento (los empujones son casi
      estáticos); el desplazamiento (`push_move_k·diag`) solo suma a la probabilidad.
- [x] Serie por par → `_events_from_series` → episodios; `obj_ids=[i,j]`, `zona`,
      `probabilidad ∈ (0, 0.95]` por fuerza de contacto + duración + bonus de desplazamiento.

**Verificación:** pushing solo aparece si el contacto ocurre dentro del área chica.

---

## T6 · API pública y resumen

- [x] `compute_field_violations(source, *, line_margin_cm, lop_window, lop_move_thresh_k,
      push_iou, push_k, push_move_k, min_frames, exit_frames, cooldown_frames, gap_frames,
      fps)` → `FieldViolationsResult` (firma del plan §7).
- [x] `resumen`: `fuera_disponible`, conteos por tipo/causa, total, fps, params, nota.
- [x] `write_field_violations_json(result, path)` (estilo `write_*` de fase_5).
- [x] Imports perezosos; sin GPU.

**Verificación:** `compute_field_violations(<json>)` devuelve resultado coherente; el JSON se
escribe.

---

## T7 · Test manual + viz

- [x] Crear `testing/test_event_field_violations.py` (script directo, sin pytest, sin GPU),
      default sobre `IMG_9933_5m30`.
- [x] Invariantes: `causa` sii `tipo=="fuera"`; `probabilidad==1.0` sii `fuera`; `pushing`
      siempre con `zona`; frames válidos.
- [x] Coherencia: balón parado ⇒ `lack_of_progress` (prob. alta). Casos borde de geometría
      (boca no es fuera; punto en área chica sí).
- [x] Viz: línea de tiempo por tipo + posiciones de `fuera` sobre la cancha →
      `events_paths(stem, "field_violations", "png")`; JSON → `…"json"`.

**Verificación:** `python testing/test_event_field_violations.py` termina OK (local).

---

## T8 · Cierre

- [x] `ruff check` limpio en los archivos nuevos.
- [x] Confirmar que `metric_positions.py`/`field_template.py`/`events*.py`/`event_goals.py`
      quedaron intactos (solo importados).
- [x] Confirmar con el usuario antes de cualquier commit (constitución §7.1/§11).

---

## Orden sugerido

T1 → T2 (geometría) → T3 (fuera) → T4 (lack-of-progress) → T5 (pushing) → T6 (API/resumen) →
T7 (test/viz) → T8.

---

## Fuera de alcance (recordatorio del spec)

- No introduce equipos/bandos ni la clase humano (interferencia humana fuera).
- No conecta el resultado al overlay (es `event_broadcast_overlay`).
- No cambia ni deprecia los módulos consumidos (se construye encima).
