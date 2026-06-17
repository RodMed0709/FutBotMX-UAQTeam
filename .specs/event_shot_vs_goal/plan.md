# Plan — Tiro a gol vs gol (`event_shot_vs_goal`)

- **Tarea atómica:** `event_shot_vs_goal`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Estado:** Define el *cómo*. **No** implica escribir código todavía (eso es el paso 5,
  habilitado por `tasks.md`).
- **Spec de referencia:** `.specs/event_shot_vs_goal/spec.md`.

---

## 1. Enfoque general

Módulo nuevo `src/core/event_shot_goal.py` que **segmenta cada lance cerca de una portería**
y lo **clasifica** `tipo ∈ {"tiro","gol"}`. No reimplementa el motor de estados: reutiliza
`_events_from_series` (de `event_goals`) para segmentar y la geometría/posiciones ya
existentes.

Dos rutas, misma salida (`ShotGoalEvent`):
- **Ruta cm (cámara superior, autoridad):** posiciones del balón en cm de
  `metric_positions` + líneas de gol/boca de `field_template`. Es la buena.
- **Ruta px (universal, proxy):** centroide del balón vs bbox de la zona, refinada (bbox
  encogido + regla de 3/4). Para tomas parciales sin cm fiable.

Principio de clasificación (igual en ambas rutas): se segmenta **un evento por lance**
(balón en la región de portería, con dirección hacia ella); dentro del lance, si **alguna**
muestra **cruzó la línea de gol** ⇒ `gol`; si no ⇒ `tiro`. Así no hay doble conteo: un gol
no emite además un tiro.

---

## 2. Estructuras

```python
@dataclass
class ShotGoalEvent:
    tipo: str            # "tiro" | "gol"
    zona: str            # "yellow" | "blue"
    frame_inicio: int
    frame_fin: int
    dur_frames: int
    dur_s: float | None
    xy_cm: tuple[float, float] | None   # posición de referencia (None en ruta px)

@dataclass
class ShotGoalResult:
    eventos: list[ShotGoalEvent]
    resumen: dict        # conteos por tipo y zona, fps, params, ruta usada
```

---

## 3. Ruta cm (autoridad)

Reusa de `event_goal_geometric`: `_ball_by_frame(result)` (posiciones del balón en cm por
frame). Geometría de `field_template`: `GOAL_LINE_X_LEFT_CM=12`, `GOAL_LINE_X_RIGHT_CM=231`,
boca real `y∈[_GOAL_TOP_Y_CM=61, _GOAL_BOTTOM_Y_CM=121]`.

Por frame y por zona, sobre la muestra del balón **más cercana a la portería** (si hay varias
por ID-switch), dos predicados **estrictos** (validados contra el video real):
- **`crossed(xy)`** (gol): **dentro de la boca real** `_GOAL_TOP_Y_CM ≤ y ≤ _GOAL_BOTTOM_Y_CM`
  (sin ensanchar) **y** cruzó la línea real — `x ≤ GOAL_LINE_X_LEFT_CM − goal_margin` (yellow) /
  `x ≥ GOAL_LINE_X_RIGHT_CM + goal_margin` (blue), `goal_margin=0` por defecto.
- **`in_approach(xy)`** (tiro): dentro de la boca **±`side_cm`** (tolerancia para tiros al
  poste) **y** en la banda de `tiro_depth_cm` frente a la línea (o ya pasada) —
  blue: `x ≥ GOAL_LINE_X_RIGHT_CM − tiro_depth_cm`; yellow simétrico.

**Sin gate de dirección.** Un tiro al poste se queda estático; exigir velocidad lo perdía. La
región + el debounce de `_events_from_series` bastan para descartar pases laterales.

**Tolerancia de huecos** (`_fill_gaps`): el balón parpadea (frames sin detección). Se
construye `flags = [(f, present, in_approach)]` y, mientras el balón está **ausente**, se
sostiene el último valor de `in_approach` hasta `gap_frames` frames; pasado el límite, cierra.
Así un balón parado frente a la portería es **un** lance, no muchos.

Construcción de series y segmentación:
1. `near = _fill_gaps(flags, gap_frames)` sobre el timeline contiguo del balón, por zona.
2. `_events_from_series(near, zona, min_frames, exit_frames, cooldown, fps)` segmenta los
   lances → intervalos `[inicio, fin]`.
3. Para cada intervalo, **clasificar**: si algún frame del intervalo cumplió `crossed` ⇒
   `gol` (`xy_cm` = la muestra que cruzó); si no ⇒ `tiro` (`xy_cm` = primera muestra presente
   del intervalo).

> Nota: `_events_from_series` ya da debounce/cierre/cooldown; aquí cambia **cómo se construye
> el booleano** (predicados estrictos + relleno de huecos) y se añade un **post-paso de
> clasificación** del intervalo.

---

## 4. Ruta px (proxy universal)

Sobre el JSON sin cm (`load_frame_objects` / `events_core`): centroide del balón vs bbox de
`yellow_zone`/`blue_zone` del mismo frame (como `event_goals`), refinado:
- **Encoger el bbox** por `margin` (inset) para descartar toques de borde.
- **Eje de profundidad**: la dimensión del bbox a lo largo de la cual el balón "entra" hacia
  la pared. La pared es el lado de la zona **más alejado del centro de la imagen** (se infiere
  comparando el centro x del bbox de la zona con `W/2`). Fracción `three_quarter_frac=0.75`.
- **`crossed_px`** (gol): centroide rebasa `three_quarter_frac` de la profundidad hacia la
  pared, dentro del bbox encogido.
- **`in_zone_px`** (tiro): centroide dentro del bbox encogido pero sin alcanzar 3/4.

Misma maquinaria que la ruta cm: `flags = [(f, present, in_zone_px or crossed_px)]` →
`_fill_gaps(gap_frames)` → `_events_from_series` → clasificar el intervalo por presencia de
`crossed_px`. **Sin** gate de dirección (coherente con cm). `xy_cm=None` (ruta px).
Documentado como **indicativo y conservador** (subdetecta goles; la autoridad es cm).

---

## 5. API pública

```python
def compute_shot_vs_goal(
    source: str | Path | MetricResult,
    *,
    route: str = "cm",            # "cm" | "px"
    tiro_depth_cm: float = 15.0,        # banda de tiro frente a la línea
    side_cm: float = 12.0,              # tolerancia lateral de la boca (postes)
    goal_margin_cm: float = 0.0,        # penetración exigida más allá de la línea (gol)
    three_quarter_frac: float = 0.75,   # ruta px
    margin_px: float = 0.0,             # inset del bbox, ruta px
    gap_frames: int = 20,               # huecos de detección a fusionar
    min_frames: int = 3,
    exit_frames: int = 3,
    cooldown_frames: int = 15,
    fps: float | None = None,
) -> ShotGoalResult: ...

def write_shot_vs_goal_json(result: ShotGoalResult, path: str | Path) -> Path: ...
```

- `route="cm"`: `source` = ruta a tracks_json (llama a `compute_metric_positions`) o un
  `MetricResult` ya calculado.
- `route="px"`: `source` = ruta a tracks_json (usa `load_frame_objects`).
- Imports perezosos (`cv2`/matplotlib solo en el viz del test). Sin GPU.

---

## 6. Test manual

`testing/test_event_shot_vs_goal.py` (script directo, sin pytest, sin GPU), sobre el clip del
gol `IMG_9933_5m30` (JSON ya en `outputs/inference/fase5_clips/...`):
1. Ruta **cm** y **px**: imprime eventos `tiro`/`gol` por zona.
2. **Ground truth** del clip (validado a mano): `#goles == 1` y `#tiros == 3`. Comparación
   informativa con el gol geométrico laxo y T2 (el estricto nunca cuenta MÁS goles).
3. Casos borde: geometría cm (cruce estricto vs banda/poste/corto), orientación px de la
   pared, entrada lateral (px) ⇒ no cuenta, ruta px sobre JSON vacío ⇒ sin eventos.
4. Viz: línea de tiempo tiro-vs-gol (matplotlib) a `events_paths(stem, "shot_vs_goal", "png")`
   + JSON a `events_paths(stem, "shot_vs_goal", "json")`.

---

## 7. Riesgos y mitigación

- **Balón fragmentado por ID-switch** (varias muestras por frame): tomar la muestra más
  cercana a la portería para `crossed` (ya se hace en gol geométrico).
- **Detección intermitente del balón**: `gap_frames` fusiona los huecos; el debounce y el
  cooldown de `_events_from_series` amortiguan el ruido restante.
- **Falsos goles por márgenes laxos**: se eliminó el ensanche de boca y el corrimiento de la
  línea; el gol exige cruce de la línea real dentro de la boca real (validado: 1 gol, no 3).
- **Orientación px heurística**: documentada como proxy conservador; la ruta cm es la autoridad.
- **Compatibilidad**: no se toca la API de `event_goals`/`event_goal_geometric`; este módulo
  los **consume** (el gol geométrico laxo queda como candidato a deprecar, fuera de esta tarea).

---

## 8. Archivos afectados

- **Nuevo:** `src/core/event_shot_goal.py`, `testing/test_event_shot_vs_goal.py`.
- **Sin tocar:** `event_goals.py`, `event_goal_geometric.py`, `metric_positions.py`,
  `field_template.py` (solo se importan).
