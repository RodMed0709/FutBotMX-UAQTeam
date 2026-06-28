# Documentación por fases — FutBot MX 2026 (UAQ Team)

Esta carpeta documenta **todo el código de `src/`** organizado por **fases**, de lo
fundamental a lo más derivado. Cada documento de fase describe sus módulos, **enlaza al
código** (`archivo.py:línea`), al **notebook** de referencia (cuando existe) y a la
**tarea SDD** (`.specs/<tarea>/`) que lo originó.

> Para **usar** la API desde un notebook, empieza por el recetario:
> [`notebooks/cookbook_pipeline.ipynb`](../notebooks/cookbook_pipeline.ipynb).
> Para la **metodología**, lee la [constitución SDD](../.specs/constitution.md).

## Mapa del sistema

El proyecto se divide en dos grandes mitades:

```
  GPU / pod                                   CPU / local (lee el tracking JSON)
  ─────────────────────────────────          ─────────────────────────────────────
  detección → segmentación → tracking   ──►   homografía → capa métrica → eventos
       (pipeline principal, modular)            (+ Kalman, + overlay narrativo)
```

La **frontera dura** entre ambas mitades es el **tracking JSON** (`Track` /
`TrackObservation`, `obj_id` estable): todo el post-proceso lee de ahí y no sabe qué
detector ni tracker lo generó.

## Índice de fases

| # | Fase | Qué cubre | Documento |
|---|---|---|---|
| 00 | **Fundamentos** | config, rutas, extracción de frames, escritura de video | [00_fundamentos.md](00_fundamentos.md) |
| 01 | **Datos / dataset** | manifiesto `db_metadata.csv`, splits, frames de evaluación | [01_datos.md](01_datos.md) |
| 02 | **Preliminares** | exploración SAM3 + fine-tune de YOLO (la innovación base) | [02_preliminares.md](02_preliminares.md) |
| 03 | **Detección** | detectores intercambiables (`sam3_text`, `yolo_sam3`) | [03_deteccion.md](03_deteccion.md) |
| 04 | **Segmentación** | SAM3 por frame + overlay por clase | [04_segmentacion.md](04_segmentacion.md) |
| 05 | **Tracking** | identidad estable (ByteTrack / BoT-SORT) + overlay por `obj_id` | [05_tracking.md](05_tracking.md) |
| 06 | **Pipeline principal** | fachada `run_inference`, lotes, esquema de salida (config-driven) | [06_pipeline_principal.md](06_pipeline_principal.md) |
| 07 | **Benchmark** | comparación sin ground-truth (eficiencia + consistencia) | [07_benchmark.md](07_benchmark.md) |
| 08 | **Homografía** | proyección campo→cenital, minimapa | [08_homografia.md](08_homografia.md) |
| 09 | **Capa métrica** | posiciones en cm, velocidad/distancia, zonas, heatmap | [09_capa_metrica.md](09_capa_metrica.md) |
| 10 | **Eventos** | goles, posesión, fueras/área, overlay narrativo (el entregable) | [10_eventos.md](10_eventos.md) |
| 11 | **Kalman** | estimación de estado río abajo (oclusión + suavizado) | [11_kalman.md](11_kalman.md) |
| 12 | **Evidencia de métricas (paper)** | scripts de medición que respaldan los números del paper (measurement-only) | [12_evidencia_metricas_paper.md](12_evidencia_metricas_paper.md) |

## Convenciones que cruzan todas las fases

- **Config-driven:** ningún path/parámetro hardcodeado; todo sale del JSON activo en
  `configs/` (seleccionado por `CONFIG_FILENAME` en `.env`) y se resuelve con
  [`get_abs_path`](../src/utils.py).
- **Imports perezosos:** `torch`, `cv2`, `imageio`, `supervision`/`trackers`,
  `matplotlib` se importan *dentro* de las funciones para que `import src` sea barato.
- **`Detection` es la moneda** entre detección/segmentación (ver
  [04_segmentacion.md](04_segmentacion.md)); el **tracking JSON** es la moneda del
  post-proceso (ver [05_tracking.md](05_tracking.md)).
- **Salidas pesadas** (mp4, frames, GT) → `outputs/` o `data/` (git-ignored);
  manifiestos ligeros → `assets/` (versionados).
