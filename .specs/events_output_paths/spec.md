# Spec — Rutas de salida dedicadas para eventos (`events_output_paths`)

- **Tarea atómica:** `events_output_paths`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Ronda de entregable de eventos (refinar detección de
  eventos + overlay de espectador) sobre la fase_5 ya completa.
- **Depende de:** nada (tarea base de la ronda).
- **Habilita:** las cuatro tareas siguientes de la ronda (`event_shot_vs_goal`,
  `event_possession_refine`, `event_field_violations`, `event_broadcast_overlay`),
  que escribirán todas sus salidas a través de este helper.

---

## 1. Requisito (historia de usuario)

> **Como** persona que corre el análisis de eventos sobre un video,
> **quiero** que **todos** los productos derivados (JSON, mp4, imágenes) se guarden en
> una **carpeta dedicada por video** dentro de `outputs/`,
> **para** que nada quede suelto en la raíz de `outputs/`, cada video tenga sus
> resultados juntos y sea fácil de inspeccionar, archivar o limpiar.

---

## 2. Motivación (por qué)

- **Hoy las salidas de fase_5 caen planas en `outputs/`.** Los módulos no fijan ruta:
  cada test pasa una ruta a mano y termina escribiendo `outputs/goal_geometric_<stem>.json`,
  `outputs/heatmap_ball_<stem>.png`, `outputs/metric_positions_<stem>.{json,png}`,
  `outputs/demo_<stem>.mp4`, etc. — todo mezclado en la raíz, sin agrupar por video.
- **La inferencia ya tiene su esquema; los eventos no.** `inference_schema.inference_paths`
  centraliza `outputs/inference/[<namespace>/]<stem>/<stem>.{json,mp4}`. Los eventos
  carecen de un equivalente, así que la convención vive dispersa en los tests.
- **La ronda nueva multiplica los productos.** Tiro-a-gol/gol, posesión/control,
  violaciones de campo y el overlay de espectador generarán varios JSON, mp4 e imágenes
  por video; sin una convención central colisionan por nombre y ensucian `outputs/`.
- **Reproducibilidad y limpieza.** Una carpeta por video permite borrar/archivar un
  análisis completo (`outputs/eventos/<stem>/`) sin tocar lo demás.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Helper nuevo `events_paths(stem, kind, ext, *, outputs_dir, namespace=None)`** que
  devuelve la ruta de salida de un producto de eventos:
  `outputs_dir/eventos/[<namespace>/]<stem>/<stem>_<kind>.<ext>`.
  - `stem`: nombre base del video (`Path.stem`).
  - `kind`: etiqueta del producto (`goal_geometric`, `shot_vs_goal`, `possession`,
    `heatmap_ball`, `field_zones_mitades`, `demo`, …) — la decide el llamador.
  - `ext`: extensión sin punto (`json`, `mp4`, `png`, …).
  - `namespace`: subcarpeta opcional (clip/variante/config); `None` ⇒ sin subcarpeta.
- **Una carpeta por video**, todos los productos juntos adentro (no subcarpeta por tipo);
  el `kind` en el nombre del archivo los distingue.
- **Devuelve un `Path` absoluto** resuelto contra `PROJECT_ROOT` (vía `outputs_dir`).
- **No crea carpetas**: como `inference_paths`, eso queda para el escritor (que ya hace
  `parent.mkdir(parents=True, exist_ok=True)`).
- **Migración de los módulos fase_5 y sus tests** a usar el helper, reemplazando las
  rutas planas actuales en `outputs/`:
  - `metric_positions`, `event_goal_geometric`, `metric_heatmap`, `metric_field_zones`,
    `metric_kinematics`, `demo_overlay` (y los `testing/test_*` correspondientes).
- **Test manual** `testing/test_events_output_paths.py` (estilo del repo, sin GPU) que
  verifica las rutas generadas (estructura, sufijo `kind`, extensión, namespace).

### 3.2 Fuera de alcance

- **No** cambia el esquema/formato del contenido de los JSON ni de los videos/imágenes:
  solo **dónde** se escriben.
- **No** toca `inference_schema.inference_paths` ni las salidas de inferencia
  (`outputs/inference/...`) — son un esquema independiente que se conserva.
- **No** introduce la lógica de los nuevos eventos ni del overlay (sus tareas propias).
- **No** crea un script de limpieza/archivado de `outputs/eventos/`.

---

## 4. Comportamiento esperado

- `events_paths("IMG_9933_5m30", "goal_geometric", "json", outputs_dir="outputs")`
  → `<PROJECT_ROOT>/outputs/eventos/IMG_9933_5m30/IMG_9933_5m30_goal_geometric.json`.
- `events_paths("IMG_9933_5m30", "heatmap_ball", "png", outputs_dir="outputs")`
  → `.../outputs/eventos/IMG_9933_5m30/IMG_9933_5m30_heatmap_ball.png`.
- Con `namespace="clipA"`:
  → `.../outputs/eventos/clipA/IMG_9933_5m30/IMG_9933_5m30_demo.mp4`.
- La función **no** crea ninguna carpeta; al volver, la ruta puede no existir todavía.
- Los módulos migrados, al recibir `output_path=None`, **derivan su ruta por defecto**
  con `events_paths(...)` en vez de escribir plano en `outputs/`.

---

## 5. Criterios de aceptación

1. Existe `events_paths(...)` en un módulo nuevo `src/core/events_schema.py` con la firma
   y la estructura de ruta descritas, devolviendo un `Path` absoluto.
2. El helper **no** crea carpetas.
3. `metric_positions`, `event_goal_geometric`, `metric_heatmap`, `metric_field_zones`,
   `metric_kinematics` y `demo_overlay` derivan su ruta por defecto vía `events_paths`
   (ya no escriben en la raíz de `outputs/`).
4. `testing/test_events_output_paths.py` corre sin GPU y verifica: estructura
   `outputs/eventos/<stem>/`, sufijo `kind`, extensión y `namespace`.
5. No hay rutas absolutas hardcodeadas: el raíz sale de `outputs_dir` (config) resuelto
   contra `PROJECT_ROOT`.

---

## 6. Notas / decisiones

- **Subdirectorio `eventos`** (español), coherente con la fase. Análogo a `inference`.
- **`namespace` opcional** por simetría con `inference_paths` (clips, variantes); por
  defecto sin subcarpeta para mantener rutas simples.
- **Módulo nuevo** (`events_schema.py`) en vez de ampliar `inference_schema.py`: el
  esquema de inferencia describe el JSON de detección/tracking; las rutas de eventos son
  otro dominio.
- Todo bajo `outputs/` (git-ignored); no se versiona nada nuevo en `assets/`.
