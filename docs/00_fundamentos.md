# Fase 00 — Fundamentos

> Capa base sobre la que se apoya todo lo demás: resolución de rutas config-driven,
> entrada/salida de video y utilidades de visualización. **No** hace inferencia.

- **Notebook de referencia:** [`cookbook_pipeline.ipynb`](../notebooks/cookbook_pipeline.ipynb)
- **Tareas SDD:** [`env_setup`](../.specs/env_setup/), [`editable_module`](../.specs/editable_module/),
  [`abs_dir_func`](../.specs/abs_dir_func/), [`abs_video_path`](../.specs/abs_video_path/),
  [`data_volume_mounts`](../.specs/data_volume_mounts/), [`frame_extraction`](../.specs/frame_extraction/),
  [`frame_visualization`](../.specs/frame_visualization/), [`source_fps`](../.specs/source_fps/),
  [`video_writer`](../.specs/video_writer/), [`frame_window_sampling`](../.specs/frame_window_sampling/)

---

## `src/utils.py` — rutas y display

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `get_abs_path(relative_path)` | [`utils.py:22`](../src/utils.py#L22) | Convierte una ruta **relativa** del config en absoluta, resuelta contra `PROJECT_ROOT` (estable sin importar el cwd). Lanza `ValueError` (ruta absoluta / inválida) o `FileNotFoundError` (no existe). **Único punto** para traducir rutas del config. |
| `show_frames(frames)` | [`utils.py:64`](../src/utils.py#L64) | Muestra un arreglo `(N,H,W,3)` con matplotlib (solo display). |
| `PROJECT_ROOT` | [`utils.py`](../src/utils.py) | Raíz del proyecto = `Path(__file__).resolve().parents[1]`. |

**Regla de oro:** el código nunca arma rutas desde `Path.cwd()`; siempre pasa por
`get_abs_path`. Así host y contenedor resuelven `data/raw` y `assets/sam3` idénticamente.

## `src/core/frame_extraction.py` — leer frames del video

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `extract_frames(video_path, all_frames=False)` | [`frame_extraction.py:163`](../src/core/frame_extraction.py#L163) | Devuelve `(N,H,W,3)`. En modo cuota toma `preprocess.frame_quota` frames repartidos; con `all_frames=True`, todos. Acepta path **relativo a `PROJECT_ROOT`** o **absoluto**. |
| `get_frame_indices(video_path, all_frames=False)` | [`frame_extraction.py:136`](../src/core/frame_extraction.py#L136) | Los índices de frame **fuente** que `extract_frames` muestrearía (para trazabilidad). |
| `iter_frames(...)` | [`frame_extraction.py:195`](../src/core/frame_extraction.py#L195) | Generador **streaming** frame a frame (lo usa tracking para no cargar el video en RAM). |
| `get_video_fps(video_path)` | [`frame_extraction.py:243`](../src/core/frame_extraction.py#L243) | FPS real del video fuente. |
| `get_frame_count(video_path)` | [`frame_extraction.py:266`](../src/core/frame_extraction.py#L266) | Conteo barato de frames (dimensiona las barras `tqdm`). |

El muestreo por cuota lo decide [`_select_frame_indices`](../src/core/frame_extraction.py#L114);
la resolución del path, [`_resolve_video_path`](../src/core/frame_extraction.py#L76).

## `src/core/video_writer.py` — escribir mp4

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `write_video(frames, path, ...)` | [`video_writer.py:82`](../src/core/video_writer.py#L82) | Escribe un mp4 a partir de un arreglo de frames (modo **batch**). FPS por defecto desde config. |
| `open_video_writer(path, fps=None)` | [`video_writer.py:129`](../src/core/video_writer.py#L129) | Context manager **incremental**: escribe frame a frame (lo usa tracking para no acumular en RAM). |

## `src/core/video_stabilize.py` — estabilización (auxiliar)

| Símbolo | Ubicación | Qué hace |
|---|---|---|
| `estimate_transforms(frames_gray)` | [`video_stabilize.py:28`](../src/core/video_stabilize.py#L28) | Estima la trayectoria de cámara entre frames. |
| `stabilize_frames(frames_rgb, smooth_radius=15)` | [`video_stabilize.py:58`](../src/core/video_stabilize.py#L58) | Suaviza el movimiento de cámara. Auxiliar de homografía; **no** está en la ruta principal. |

---

### Cómo encaja con el resto

`get_abs_path` + `extract_frames`/`iter_frames` + `write_video`/`open_video_writer` son
los ladrillos que consumen **todas** las fases siguientes. La segmentación
([04](04_segmentacion.md)) extrae frames y compone overlays; el tracking
([05](05_tracking.md)) usa el **streaming** (`iter_frames` + `open_video_writer`) para
correr videos largos sin OOM.
