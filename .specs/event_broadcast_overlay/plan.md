# Plan — Overlay de espectador (`event_broadcast_overlay`)

- **Tarea atómica:** `event_broadcast_overlay`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Estado:** Define el *cómo*. **No** implica escribir código todavía (eso es el paso 5,
  habilitado por `tasks.md`).
- **Spec de referencia:** `.specs/event_broadcast_overlay/spec.md`.

---

## 1. Enfoque general

Módulo nuevo `src/core/event_broadcast_overlay.py` que **precalcula** todos los eventos/series
(una sola vez) y luego **renderiza el video frame a frame**, compositando el frame del partido
dentro de un lienzo con márgenes + paneles. Reutiliza los módulos de eventos y de minimapa/
heatmap; no reimplementa nada de detección.

Dos fases:
1. **Precómputo** (una pasada): goles/tiros (`goal_source`), posesión/control, violaciones,
   posiciones en cm (para minimapa/heatmap) y las **series por frame** derivadas (marcador
   acumulado, banner activo, lista de eventos vigente, heatmap acumulado).
2. **Render incremental**: por cada frame del video, dibujar paneles según `layout` y escribir
   al mp4 (`open_video_writer`), con `tqdm`.

---

## 2. Precómputo (helpers)

- **Goles** (`goal_source`): `compute_shot_vs_goal(..., route="cm")` (strict) →
  eventos `tipo=="gol"`; o `compute_geometric_goals(...)` (geometric). Se normaliza a una lista
  `goals = [(frame, zona)]` (frame = `frame_inicio` del gol).
- **Tiros**: de `compute_shot_vs_goal` (`tipo=="tiro"`) para la lista de eventos.
- **Posesión/control**: `compute_possession_refine(by_frame)` → `posesion_por_frame`,
  `control_por_frame`, resumen.
- **Violaciones**: `compute_field_violations(tracks_json)` → eventos (fuera/lack/pushing).
- **cm/minimapa/heatmap**: `compute_metric_positions(tracks_json)` (compartido) →
  `MinimapRenderer` (estela) y `metric_heatmap` (acumulado). Si falla ⇒ modo degradado
  (`overlay_degradado=True`, sin minimapa/heatmap).
- **Series por frame** (dict `frame -> estado`):
  - `marcador[f] = (yellow_count, blue_count)` acumulado de `goals` con `frame ≤ f`.
  - `banner[f]` = (texto, x_pos) si `f` cae en `[gol, gol+banner_secs·fps)`, interpolando
    `x_pos` linealmente de 0 a ancho del video.
  - `event_feed[f]` = últimos `max_items` eventos (tiros+violaciones) con `frame_inicio ≤ f`,
    ordenados por recencia.

---

## 3. Estructuras

```python
@dataclass
class BroadcastResult:
    video: Path                 # mp4 generado
    sample_png: Path | None     # frame de muestra (un gol)
    resumen: dict               # marcador final, conteos, layout, goal_source, degradado
```

---

## 4. Componentes de dibujo (cv2, reutilizables entre layouts)

Funciones puras `np.ndarray -> np.ndarray` (o que dibujan sobre un panel dado):

- `_draw_scoreboard(panel, yellow_n, blue_n)` — “🟡 n  –  n 🔵” con los colores de portería.
- `_draw_goal_banner(frame, texto, x_pos)` — texto deslizante sobre el frame del partido.
- `_draw_metrics_panel(panel, posesion, control)` — barras/labels legibles (reusa el estilo de
  `demo_overlay._metrics_bar`).
- `_draw_event_feed(panel, items)` — lista vertical con tope; icono/color por tipo.
- minimapa: `MinimapRenderer` ya entrega un frame del minimapa; heatmap: `render_heatmap` da una
  imagen; ambos se **encajan** (`_fit`, reusado de `demo_overlay`) en su panel.

`_label`/`_fit` se **reutilizan** de `demo_overlay` (import directo) para no duplicar.

---

## 5. Layouts

Una función por layout que recibe (frame del partido, paneles ya dibujados) y devuelve el
lienzo final:

- **`_compose_layout2(...)` (default, paneles laterales):** lienzo con márgenes anchos;
  arriba el marcador; a la izquierda el panel de métricas; a un lado el minimapa y al **opuesto**
  el heatmap; abajo/lateral la lista de eventos; el video del partido al centro. El banner se
  dibuja **sobre** el video.
- **`_compose_layout1(...)` (paneles superpuestos):** el video ocupa casi todo el lienzo;
  marcador arriba, minimapa y heatmap en **esquinas opuestas** (semitransparentes), métricas y
  lista en esquinas restantes; banner sobre el video.

Ambos consumen los **mismos** paneles dibujados; solo cambia la composición geométrica.

---

## 6. API pública

```python
def render_broadcast_overlay(
    tracks_json: str | Path,
    *,
    layout: int = 2,                 # 1 | 2
    goal_source: str = "strict",     # "strict" | "geometric"
    banner_secs: float = 2.5,
    max_items: int = 6,
    margin_px: int = 220,            # ancho de márgenes (layout 2)
    trajectory_window: int = 60,     # estela del minimapa
    bin_cm: float = 5.0,             # heatmap
    sigma_cm: float = 8.0,
    out_fps: float | None = None,    # default = fps del video
    max_frames: int | None = None,   # cap para pruebas
    progress: bool = True,
) -> BroadcastResult: ...
```

- Lee `tracks_json` (cámara superior). Deriva todo el precómputo (sección 2).
- Render incremental con `iter_frames(video, ...)` + `open_video_writer`; `tqdm` si `progress`.
- Salida: `events_paths(stem, "broadcast", "mp4")`; PNG de muestra en
  `events_paths(stem, "broadcast", "png")` (un frame dentro de la ventana de un gol).
- Imports perezosos (`cv2`, matplotlib vía `render_heatmap`); sin GPU.

---

## 7. Modo degradado

Si `compute_metric_positions` falla o no hay cm fiable: `overlay_degradado=True`, se omiten
minimapa y heatmap (sus paneles quedan vacíos o se reacomoda el layout), y se conservan
marcador, banner, métricas y lista (todo derivable en px). Se anota en `resumen`.

---

## 8. Test manual

`testing/test_event_broadcast_overlay.py` (script directo, sin pytest, sin GPU), sobre
`IMG_9933_5m30` **capado** (`max_frames`):
1. Genera el mp4 con `layout=2, goal_source="strict"` y verifica que el archivo existe y abre
   (frame count > 0).
2. Genera también `layout=1` (mismo cap) — corre sin error.
3. Verifica el marcador final del resumen: strict ⇒ azul=1; (opcional) geometric ⇒ azul=3.
4. Exporta un PNG de muestra de un frame con gol/banner.

> El render completo (1799 frames) se corre aparte; el test capa para velocidad.

---

## 9. Riesgos y mitigación

- **Costo de render** (1799 frames + minimapa/heatmap por frame): precomputar todo lo posible;
  el heatmap acumulado se actualiza incremental; `max_frames` para pruebas.
- **Homografía costosa/ruidosa**: `metric_positions` se calcula una sola vez; modo degradado si
  falla.
- **Layouts divergentes**: se comparten los paneles dibujados; solo cambia la composición, para
  no duplicar lógica.
- **Compatibilidad:** no se tocan `demo_overlay`/`track_overlay` ni los módulos de eventos.

---

## 10. Archivos afectados

- **Nuevo:** `src/core/event_broadcast_overlay.py`, `testing/test_event_broadcast_overlay.py`.
- **Sin tocar:** `demo_overlay.py`, `track_overlay.py`, `event_shot_goal.py`,
  `event_goal_geometric.py`, `event_possession_refine.py`, `event_field_violations.py`,
  `metric_positions.py`, `metric_heatmap.py`, `minimap.py` (solo se importan).
