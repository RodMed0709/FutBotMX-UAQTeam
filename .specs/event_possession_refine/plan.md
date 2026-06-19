# Plan — Posesión vs control (`event_possession_refine`)

- **Tarea atómica:** `event_possession_refine`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Estado:** Define el *cómo*. **No** implica escribir código todavía (eso es el paso 5,
  habilitado por `tasks.md`).
- **Spec de referencia:** `.specs/event_possession_refine/spec.md`.

---

## 1. Enfoque general

Módulo nuevo `src/core/event_possession_refine.py` que **consume** `events.compute_possession`
y añade dos cosas: (a) una **capa de control** (subconjunto estricto de la posesión) y
(b) una **histéresis adaptativa** que actualiza la posesión de inmediato ante un **cambio
radical**. No reescribe la posesión por cercanía: reutiliza la base (`load_frame_objects`,
`ball_centroid`, `_nearest_robot`/`_raw_possession`, `_bbox_diagonal`) de `events`/`events_core`.

Tres series por frame, sobre el timeline contiguo del balón:
- **posesión** (`obj_id | None`): cercanía + histéresis adaptativa (radical ⇒ 1 frame).
- **control** (`obj_id | None`): posesión **y** balón en movimiento en la ventana.
- (interno) **cruda** (`obj_id | None`): poseedor sin histéresis, para detectar el cambio
  radical.

---

## 2. Estructuras

```python
@dataclass
class PossessionRefineResult:
    posesion_por_frame: dict[int, int | None]   # obj_id poseedor (con histéresis adaptativa)
    control_por_frame: dict[int, int | None]    # obj_id que controla (None si balón quieto)
    resumen: dict                               # % posesión/control por obj, libre, no_visible
```

> No se redefine `FrameObject`/`PossessionResult`; se importan/consumen de `events`.

---

## 3. Posesión con histéresis adaptativa

Reusa `_raw_possession(by_frame, gate_k)` (de `events`) para la serie **cruda** y
`_nearest_robot`/`_bbox_diagonal` para medir distancias. La histéresis adaptativa sustituye a
`_apply_hysteresis` **solo en este módulo** (no se toca `events`):

Por frame, con la serie cruda `raw[f]` y la posición del balón `ball[f]`:
- **Cambio radical** si, al detectar un poseedor distinto del actual, se cumple **alguna**:
  - **salto del balón**: `dist(ball[f], ball[f_prev]) > jump_thresh` (default `jump_k`·diagonal
    media del robot poseedor, en 1–2 frames);
  - **cercanía clara**: el nuevo poseedor está a una distancia `< clear_factor ·` la del
    poseedor actual al balón (cambio inequívoco, no empate).
- **Si radical** ⇒ se adopta el nuevo poseedor **en 1 frame**.
- **Si no** ⇒ se aplica la histéresis normal: el nuevo valor debe sostenerse `min_frames`
  frames para confirmarse (igual que `_apply_hysteresis`).
- Balón no visible ⇒ `None` (se mantiene la lógica actual de huecos).

> Implementación: una sola pasada que combina la confirmación adaptativa (radical ⇒ inmediata,
> ambiguo ⇒ racha `min_frames`) sobre `raw`.

---

## 4. Capa de control

Sobre la serie de **posesión** ya confirmada y el balón por frame:
- Para cada frame con poseedor `oid`, se mira la **ventana** `[f, f+control_window)` (o
  centrada; se fija en el plan): el balón **se mueve** si su desplazamiento acumulado/medio en
  la ventana ≥ `move_thresh` (default `move_k`·diagonal del bbox de `oid`, por frame).
- **control[f] = oid** si hay poseedor **y** el balón se mueve en la ventana; si no,
  `control[f] = None` (posesión pasiva, balón quieto).
- Balón no visible en parte de la ventana: se evalúa con las muestras disponibles; si no hay
  suficientes, no se declara control.

> El control es por diseño un **subconjunto** de la posesión ⇒ `control ⊆ posesión`.

---

## 5. Resumen (no engañoso)

```python
resumen = {
    "n_frames": int,
    "posesion_por_obj": {oid: {"frames": n, "segundos": s, "pct": p}},
    "control_por_obj":  {oid: {"frames": n, "segundos": s, "pct": p}},
    "pct_posesion_total": float,   # algún robot posee
    "pct_control_total": float,    # algún robot controla (<= posesion_total)
    "pct_libre": float,            # balón visible sin poseedor
    "pct_no_visible": float,
    "cambios_de_posesion": int,
    "cambios_de_control": int,
    "fps": fps,
    "params": { gate_k, min_frames, control_window, move_k, jump_k, clear_factor },
}
```

Invariante reportable: `pct_control_total ≤ pct_posesion_total` y, por robot,
`control_por_obj[oid].pct ≤ posesion_por_obj[oid].pct`.

---

## 6. API pública

```python
def compute_possession_refine(
    by_frame: dict[int, list[FrameObject]],
    *,
    gate_k: float = 1.5,            # heredado de events
    min_frames: int = 3,           # histéresis para cambios ambiguos
    control_window: int = 5,       # ventana para medir movimiento del balón
    move_k: float = 0.15,          # umbral de movimiento (·diagonal robot / frame)
    jump_k: float = 1.0,           # salto radical del balón (·diagonal robot)
    clear_factor: float = 0.6,     # cambio radical por cercanía clara
    fps: float | None = None,
) -> PossessionRefineResult: ...

def write_possession_refine_json(result: PossessionRefineResult, path) -> Path: ...
```

- Entrada: `by_frame` de `load_frame_objects` (igual que `compute_possession`).
- Imports perezosos (`numpy` ya lo usa `events`; matplotlib solo en el viz del test).
  Sin GPU, sin homografía.

---

## 7. Test manual

`testing/test_event_possession_refine.py` (script directo, sin pytest, sin GPU), sobre
`IMG_9933_5m30`:
1. Imprime `% posesión` y `% control` por robot + totales.
2. Invariantes: `control ⊆ posesión` por frame; `pct_control ≤ pct_posesion`; cobertura de
   frames (posesión-sin-control + control + libre + no_visible = 100%).
3. Caso de coherencia: un tramo de balón quieto junto a un robot debe dar **posesión sin
   control** (se localiza el tramo más largo con poseedor estable y se verifica que su control
   es menor que su posesión).
4. Viz: línea de tiempo con dos filas (posesión / control), color por `obj_id`, a
   `events_paths(stem, "possession_refine", "png")`; JSON a `…"json"`.

---

## 8. Riesgos y mitigación

- **Umbrales sensibles** (`move_k`, `jump_k`): defaults conservadores + configurables; el test
  imprime las cifras para calibrar contra el video real (igual que en `event_shot_vs_goal`).
- **Balón parpadeante** (huecos de detección): la ventana de control usa las muestras
  disponibles; la posesión hereda el manejo de huecos de la base.
- **Cambio radical mal disparado** (H/tracking ruidoso): requiere salto **grande** o cercanía
  **clara** (factores con margen), no cualquier diferencia.
- **Compatibilidad:** no se toca `events.py`; este módulo lo consume y re-deriva la cruda con
  el mismo `gate_k`.

---

## 9. Archivos afectados

- **Nuevo:** `src/core/event_possession_refine.py`, `testing/test_event_possession_refine.py`.
- **Sin tocar:** `events.py`, `events_core.py` (solo se importan/consumen).
