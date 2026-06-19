# Plan técnico — fps real de la fuente en modo completo (`source_fps`)

- **Tarea atómica:** `source_fps`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** no hay draft previo para este plan.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar `get_video_fps(video_path)` (helper que
expone el fps del video vía decord) y cómo cablearlo en `run_pipeline` para que el
modo completo (`all_frames=True`) escriba el mp4 al fps real de la fuente, dejando
intacto `extract_frames` y el modo cuota.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Lectura de video / fps:** `decord` (`decord.VideoReader.get_avg_fps`), ya
  usado e importado en `src/core/frame_extraction.py`.
- **Validación de ruta:** `_resolve_video_path` existente en `frame_extraction.py`.
- Sin nuevas dependencias.

---

## 3. Diseño

### 3.1 `get_video_fps` (en `src/core/frame_extraction.py`)

```python
def get_video_fps(video_path: Path) -> float:
    """Devuelve el fps promedio del video (relativo a PROJECT_ROOT o absoluto)."""
    abs_path = _resolve_video_path(video_path)
    reader = decord.VideoReader(str(abs_path))
    return float(reader.get_avg_fps())
```

- Reutiliza `_resolve_video_path` (acepta ruta relativa/absoluta y valida
  existencia; lanza `ValueError`/`FileNotFoundError`).
- `decord` ya está importado a nivel de módulo; abre el video solo para leer
  metadatos (no decodifica frames).

### 3.2 Exportación

- En `src/core/__init__.py`: añadir `get_video_fps` al import desde
  `frame_extraction` y a `__all__`.

### 3.3 Cableado en `run_pipeline` (`src/core/pipeline.py`)

- Importar `get_video_fps` (junto a `extract_frames`).
- Renombrar la variable de config para distinguirla y resolver el fps por modo:

```python
classes, outputs_dir, config_fps = _load_pipeline_config()
...
fps = get_video_fps(video_path) if all_frames else config_fps
```

- `fps` se sigue pasando a `write_video(np.stack(composed), mp4_path, fps=fps)` y
  se registra en el payload del JSON (campo `fps`). El resto de la orquestación no
  cambia.

### 3.4 Manejo de errores

| Situación | Excepción |
|---|---|
| `video_path` no `Path` / inexistente | `ValueError`/`FileNotFoundError` (vía `_resolve_video_path`) |
| Fallo de decord al abrir | excepción de `decord` propagada |

`extract_frames` no se modifica (firma y comportamiento idénticos).

---

## 4. Cambios de configuración

- **Ninguno.** Usa claves ya existentes.

---

## 5. Validación

### 5.1 `get_video_fps` (agente, local)

- Añadir a `testing/test_frame_extraction.py` una verificación que, sobre el video
  ya localizado, imprima `get_video_fps(video)` y compruebe que es un float > 0.
- **El agente lo ejecuta** en local (no usa modelo; hay videos locales).
- Lint (`ruff`, `black`) e importabilidad (`from src.core import get_video_fps`).

### 5.2 Pipeline modo completo (usuario, RunPod/GPU)

- Se crea un **notebook comparativo** `notebooks/fase_0/07_pipeline_full_video_check.ipynb`
  (lo crea el agente; lo corre el usuario) que ejecuta el pipeline real
  (`run_pipeline(video, all_frames=True)`) y, para inspección visual sencilla,
  muestra en **una celda el video original** y en **la siguiente el video anotado**
  (vía `IPython.display.Video`).
- Correr ese notebook en RunPod y confirmar que el mp4 anotado se reproduce a la
  velocidad del original y que el JSON registra el fps de la fuente. (La corrida con
  inferencia es inviable en CPU.)

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Función presente | §3.1, §3.2 | `get_video_fps` + export |
| AC-2 fps correcto | §3.1 | `get_avg_fps()` |
| AC-3 Entrada flexible | §3.1 | `_resolve_video_path` |
| AC-4 Modo completo usa fps fuente | §3.3 | `get_video_fps` si `all_frames` |
| AC-5 Modo cuota intacto | §3.3 | `config_fps` si no `all_frames` |
| AC-6 `extract_frames` intacto | §3.4 | no se modifica |
| AC-7 Error claro | §3.4 | `_resolve_video_path` |
| AC-8 Validación | §5.1, §5.2 | local (get_video_fps) + RunPod (pipeline) |

---

## 7. Riesgos y consideraciones

- **Doble apertura del video en modo completo:** `run_pipeline` abrirá el video una
  vez para `get_video_fps` (metadatos) y otra en `extract_frames` (frames). El
  costo de leer metadatos es despreciable frente a la inferencia; se acepta por la
  limpieza de mantener `extract_frames` con responsabilidad única.
- **`get_avg_fps` promedio:** decord devuelve el fps **promedio**; para videos con
  fps variable podría no ser exacto al frame, pero es la mejor referencia
  disponible y adecuada para reproducción.
- **fps no entero:** `get_avg_fps` puede devolver valores como 59.7; `imageio`/
  ffmpeg aceptan fps float, así que se pasa tal cual.
