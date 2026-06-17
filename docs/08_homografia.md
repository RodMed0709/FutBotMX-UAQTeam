# Fase 08 — Homografía (campo → cenital)

> Primera pieza de la mitad CPU-local. Proyecta lo que ve la cámara a un **campo métrico
> top-down** (243×182 cm) vía homografía por frame, base de toda medición en cm. El
> camino **consolidado** es el **por líneas** (`VideoHomographyLines`, ~9–23 cm); el de
> máscaras queda como *legacy*.
>
> **Entrada:** debe ser la footage **cámara-superior** (campo completo). La vista
> portrait de Meta-Glasses no funciona.

- **Notebooks:** [`fase_4_homografia/`](../notebooks/fase_4_homografia/) — la línea `v2_*`
  es la vigente; [`v2_07_minimap_polish_cenital.ipynb`](../notebooks/fase_4_homografia/v2_07_minimap_polish_cenital.ipynb)
  es la referencia visual del minimapa cenital.
- **Tareas SDD:** [`field_homography`](../.specs/field_homography/),
  [`homografia_v2_robusta`](../.specs/homografia_v2_robusta/),
  [`homography_consolidation`](../.specs/homography_consolidation/)
- **Spec extendida:** [`docs/specs/2026-06-16-homografia-robusta-multifeature.md`](specs/2026-06-16-homografia-robusta-multifeature.md)

---

## Geometría del campo (modelo métrico)

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `field_template.py` | [`field_template.py:103`](../src/core/field_template.py#L103) | Geometría métrica del campo (243×182 cm, rectángulo interior, círculo central, áreas). `render_field(...)`. |
| `field_landmarks.py` | [`field_landmarks.py:84`](../src/core/field_landmarks.py#L84) | Landmarks/líneas nombrados en cm; `points_array`, `static_world_points`, `draw_landmarks`. |

## El camino consolidado: por líneas

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `VideoHomographyLines` | [`homography_multifeature.py:191`](../src/core/homography_multifeature.py#L191) | **La homografía vigente**: ajusta a las líneas blancas del campo + continuidad temporal. ~9–23 cm de error. |
| `solve_lines_masks(...)` | [`homography_multifeature.py:125`](../src/core/homography_multifeature.py#L125) | Resuelve una homografía de un frame (líneas + máscara de carpeta). |
| `field_white_lines(...)` | [`homography_multifeature.py:21`](../src/core/homography_multifeature.py#L21) | Extrae las líneas blancas del campo. |
| `detect_center_circle(...)` | [`homography_multifeature.py:357`](../src/core/homography_multifeature.py#L357) | Detecta el círculo central (feature extra). |

## Caminos legacy (referencia)

| Símbolo | Ubicación | Estado |
|---|---|---|
| `homography.py` (`FrameHomography`, `project_points`) | [`homography.py:144`](../src/core/homography.py#L144) | Camino A: SAM3-only (quad de `green_floor`). `project_points` **sí se reutiliza**. |
| `auto_homography.py` (`VideoHomography`) | [`auto_homography.py:270`](../src/core/auto_homography.py#L270) | Caminos B/C: color-auto y SAM3+YOLO máscaras (legacy). |
| `homography_metrics.py` | [`homography_metrics.py:104`](../src/core/homography_metrics.py#L104) | Métricas de error (reproyección, jitter) para comparar variantes. |

## `src/core/minimap.py` — el minimapa

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `CenitalMinimapRenderer` | [`minimap.py:251`](../src/core/minimap.py#L251) | Estilo cenital pulido (la referencia visual del entregable). |
| `MinimapRenderer` | [`minimap.py:99`](../src/core/minimap.py#L99) | Renderizador genérico de trayectorias. |
| `draw_field_overlay_on_frame(frame, H)` | [`minimap.py:365`](../src/core/minimap.py#L365) | Reproyecta el campo sobre el video (homografía embebida). |

## `src/core/minimap_pipeline.py` — driver standalone (legacy)

Función configurable, pero **aún en el camino de máscaras** (diferido):

```python
render_minimap_video(
    video_path, tracks_json=None, output_path=None, max_frames=None,
    start_frame=0, frame_step=1,
    detector="sam3_text",        # "sam3_text" | "yolo_sam3" | "yolo"
    conf=None, bundle=None, draw_overlay=False, smooth_beta=0.4, progress=True,
) -> dict
```

| Símbolo | Ubicación | Estado |
|---|---|---|
| `render_minimap_video(...)` | [`minimap_pipeline.py:205`](../src/core/minimap_pipeline.py#L205) | Video minimapa standalone. **El entregable NO lo usa** (el broadcast renderiza su propio minimapa cenital desde la métrica por líneas). Migrar a líneas solo si se presentan estos artefactos secundarios. |

---

### Cómo encaja con el resto

La homografía por líneas es la **base** de la [capa métrica](09_capa_metrica.md):
`compute_metric_positions(..., homography="lines")` la usa para pasar las posiciones del
tracking JSON a cm. Sobre esos cm se montan [eventos](10_eventos.md), zonas, heatmap y el
overlay narrativo.
