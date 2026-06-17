# Plan — Violaciones de campo (`event_field_violations`)

- **Tarea atómica:** `event_field_violations`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Estado:** Define el *cómo*. **No** implica escribir código todavía (eso es el paso 5,
  habilitado por `tasks.md`).
- **Spec de referencia:** `.specs/event_field_violations/spec.md`.

---

## 1. Enfoque general

Módulo nuevo `src/core/event_field_violations.py` con tres detectores que comparten el motor
de estados `_events_from_series` (de `event_goals`):

- **`fuera`** (Capa B, cm) — robot fuera del rectángulo de líneas **o** dentro del área chica,
  con `causa`. Usa `compute_metric_positions` + geometría de `field_template`.
- **`lack_of_progress`** (Capa A, px, prob.) — balón casi inmóvil en ventana larga.
- **`pushing`** (Capa A, px, prob.) — contacto robot-robot **dentro del área chica**.

Mismo patrón que el resto de fase_5: por frame se construye un booleano por entidad, se
segmenta con `_events_from_series` (debounce/cierre/cooldown) y se hace un post-paso de
clasificación/etiquetado. Imports perezosos (`cv2` solo para punto-en-polígono y el viz).

---

## 2. Estructuras

```python
@dataclass
class FieldViolationEvent:
    tipo: str                 # "fuera" | "lack_of_progress" | "pushing"
    causa: str | None         # fuera: "salida_campo" | "area_chica"; otros: None
    obj_ids: list[int]        # robot(s) involucrado(s); [] si no aplica (balón)
    zona: str | None          # "yellow" | "blue" para area_chica/pushing; None si no aplica
    frame_inicio: int
    frame_fin: int
    dur_frames: int
    dur_s: float | None
    ref: tuple[float, float] | None   # cm (fuera) o px (lack/pushing)
    probabilidad: float       # 1.0 geométrico; (0,1) probabilístico

@dataclass
class FieldViolationsResult:
    eventos: list[FieldViolationEvent]
    resumen: dict
```

---

## 3. Geometría (cm) reutilizada de `field_template`

- **Rectángulo de líneas** (campo legal): `x∈[FIELD_X0, FIELD_X1]`, `y∈[FIELD_Y0, FIELD_Y1]`
  con `FIELD_X0=LINE_BORDER_CM=12`, `FIELD_X1=LENGTH_CM-12=231`, `FIELD_Y0=12`,
  `FIELD_Y1=WIDTH_CM-12=170`.
- **Boca de portería** (excepción de salida): `y∈[_GOAL_TOP_Y_CM, _GOAL_BOTTOM_Y_CM]=[61,121]`
  en `x<FIELD_X0` o `x>FIELD_X1`.
- **Área chica** (polígono D): `_penalty_outline_cm(goal_x, inner_x)` cerrado por la línea de
  gol, para `yellow` (`goal_x=12, inner_x=37`) y `blue` (`goal_x=231, inner_x=206`).
  Punto-en-polígono con `cv2.pointPolygonTest` (contorno en cm escalado a un sistema fijo, o
  test analítico). Tolerancia `line_margin_cm`.

Predicados por robot/frame (sobre su `xy_cm`):
- `_out_of_field(xy)`: fuera del rectángulo **y** no dentro de la boca ⇒ `causa="salida_campo"`.
- `_in_penalty(xy)` → `zona` o `None`: dentro del polígono de algún área chica ⇒
  `causa="area_chica"`.
- `fuera[f] = _out_of_field or (_in_penalty is not None)`; la `causa`/`zona` se toma del frame
  de apertura del episodio.

---

## 4. Detector `fuera` (Capa B)

1. `compute_metric_positions(source)` → posiciones en cm; filtrar `cls == "robot"`,
   agrupadas por `obj_id` y por frame (puede haber varias por frame).
2. Para cada robot (`obj_id`), serie `near[f] = es_fuera(xy_cm[f])` sobre su timeline; huecos
   = no visible.
3. `_events_from_series(near, str(obj_id), min_frames, exit_frames, cooldown, fps)` →
   episodios; para cada uno se determina `causa`/`zona`/`ref` del frame de apertura.
4. `probabilidad = 1.0` (geométrico).

> Si la homografía no es fiable (sin `metric_positions`), se omite `fuera` y se anota en el
> resumen (`fuera_disponible=False`).

---

## 5. Detector `lack_of_progress` (Capa A, prob.)

Sobre el balón en px (`ball_centroid` por frame):
- Ventana deslizante de `lop_window` frames (≈ `min_secs·fps`); el balón está **estancado** si
  el **desplazamiento neto** (o paso medio) en la ventana < `lop_move_thresh` (escala = diagonal
  del balón, como en `event_possession_refine`).
- Serie `near[f] = estancado(f)` (con relleno de huecos `gap_frames`) →
  `_events_from_series` (con `min_frames ≈ lop_window`) → episodios.
- **Probabilidad** = función acotada que crece con la **duración** (más allá del mínimo) y la
  **quietud** (cuán por debajo del umbral): p.ej.
  `prob = clamp(0.5 + 0.5·min(1, (dur - min)/min) ... )` (fórmula concreta en implementación,
  documentada y configurable). No hay actores robot obligatorios (`obj_ids=[]`); opcional:
  anotar el robot poseedor dominante del tramo si lo hay.

---

## 6. Detector `pushing` (Capa A, prob.) — solo en área chica

- Por frame, pares de robots en **contacto**: IoU de bboxes > `push_iou` **o** distancia de
  centroides < `push_k · (radio_i + radio_j)` (radio = ½ diagonal).
- **Restricción de zona**: el contacto debe ocurrir **dentro del área chica**. En Capa B se usa
  `_in_penalty(xy_cm)` del punto de contacto; si no hay cm, se aproxima con el bbox de
  `yellow_zone`/`blue_zone` en px (proxy) — documentado como indicativo.
- **Empuje sin requisito de desplazamiento**: los robots casi no se mueven en un empujón
  (validado: desplazamiento/diag ~0.02, muy por debajo de cualquier umbral razonable). El
  contacto sostenido en el área chica **es** la señal; el desplazamiento solo suma confianza.
- Serie por par `(i,j)` `near[f] = contacto ∧ en_area` → `_events_from_series` → episodios;
  `obj_ids=[i,j]`, `zona` del área.
- **Probabilidad** ∈ (0, 0.95]: crece con la **fuerza del contacto** (`strength` = solape IoU o
  cercanía de centroides), la **duración** y un **bonus** por fracción de frames con
  desplazamiento del empujado (`push_move_k`).

---

## 7. API pública

```python
def compute_field_violations(
    source: str | Path | MetricResult,
    *,
    line_margin_cm: float = 3.0,
    lop_window: int = 60,            # ventana de lack-of-progress (frames)
    lop_move_thresh_k: float = 0.10, # ·diagonal del balón (paso medio)
    push_iou: float = 0.05,
    push_k: float = 1.0,             # cercanía = push_k·(radio_i+radio_j)
    push_move_k: float = 0.15,       # desplazamiento del empujado ·diagonal robot
    min_frames: int = 3,
    exit_frames: int = 3,
    cooldown_frames: int = 15,
    gap_frames: int = 20,
    fps: float | None = None,
) -> FieldViolationsResult: ...

def write_field_violations_json(result: FieldViolationsResult, path) -> Path: ...
```

- `source`: ruta a tracks_json (deriva `metric_positions` para cm **y** `load_frame_objects`
  para px) o un `MetricResult` ya calculado (px se re-obtiene del tracks_json embebido si está;
  si no, se omiten lack/pushing — se decide en implementación, default = ruta a json).
- Imports perezosos; sin GPU.

---

## 8. Resumen

```python
resumen = {
    "fuera_disponible": bool,          # hubo cm fiable
    "conteo": {"fuera": {"salida_campo": n, "area_chica": n},
               "lack_of_progress": n, "pushing": n},
    "total_eventos": int,
    "fps": fps,
    "params": {...},
    "nota": "fuera = geométrico (cm); lack/pushing = probabilístico (px, indicativo)",
}
```

---

## 9. Test manual

`testing/test_event_field_violations.py` (script directo, sin pytest, sin GPU), sobre
`IMG_9933_5m30`:
1. Imprime eventos por tipo/causa con su probabilidad.
2. Invariantes: `causa` presente sii `tipo=="fuera"`; `probabilidad==1.0` sii `fuera`;
   `pushing` siempre con `zona` de área chica; frames válidos.
3. Coherencia: el tramo de balón parado (≈ f502-676) aparece como `lack_of_progress` con
   probabilidad alta.
4. Casos borde (geometría): un robot en la boca de portería **no** es fuera; un punto dentro
   del polígono del área chica sí.
5. Viz: línea de tiempo por tipo / mapa de la cancha con las posiciones de los eventos `fuera`
   → `events_paths(stem, "field_violations", "png")`; JSON → `…"json"`.

---

## 10. Riesgos y mitigación

- **Homografía ruidosa** ⇒ falsos `fuera` cerca de las líneas: `line_margin_cm` y el debounce
  de `_events_from_series` los amortiguan.
- **Pushing difícil** ⇒ se restringe al área chica + exige desplazamiento + es probabilístico
  (margen de error explícito).
- **Lack-of-progress sin audio** ⇒ heurístico por quietud del balón, probabilístico.
- **Compatibilidad:** no se tocan los módulos consumidos; este construye encima.

---

## 11. Archivos afectados

- **Nuevo:** `src/core/event_field_violations.py`, `testing/test_event_field_violations.py`.
- **Sin tocar:** `metric_positions.py`, `field_template.py`, `events.py`, `events_core.py`,
  `event_goals.py` (solo se importan).
