# Tasks — fps real de la fuente en modo completo (`source_fps`)

- **Tarea atómica:** `source_fps`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** `get_video_fps` no usa modelo; el **agente ejecuta** su
> verificación en local. El pipeline completo (`all_frames=True` con inferencia) y
> el notebook comparativo los corre el usuario en **RunPod**.

---

## Fase A — Helper

- [x] **T1 — `get_video_fps` en `src/core/frame_extraction.py`**
  - Definir `get_video_fps(video_path: Path) -> float`: resolver con
    `_resolve_video_path`, abrir `decord.VideoReader` y devolver
    `float(reader.get_avg_fps())`.
  - **Verificación:** devuelve un float > 0 sobre un video real; ruta inválida →
    `ValueError`/`FileNotFoundError`; `extract_frames` no cambia.
  - **Plan:** §3.1. **Spec:** AC-1, AC-2, AC-3, AC-6, AC-7.

---

## Fase B — Exportación

- [x] **T2 — Exportar `get_video_fps` en `src/core/__init__.py`**
  - Añadirla al import desde `frame_extraction` y a `__all__`.
  - **Verificación:** `from src.core import get_video_fps` funciona; `ruff`/`black`
    pasan.
  - **Plan:** §3.2. **Spec:** AC-1.

---

## Fase C — Cableado en el pipeline

- [x] **T3 — Resolver el fps por modo en `run_pipeline`**
  - Importar `get_video_fps`; renombrar la var de config a `config_fps`; resolver
    `fps = get_video_fps(video_path) if all_frames else config_fps`; mantener el
    paso a `write_video` y al JSON.
  - **Verificación:** en modo completo el fps proviene de la fuente; en cuota sigue
    el de config; el JSON refleja el fps usado; `mode` inválido sigue lanzando
    `NotImplementedError`.
  - **Plan:** §3.3. **Spec:** AC-4, AC-5.

---

## Fase D — Validación

- [x] **T4 — Verificación local de `get_video_fps` (crear + ejecutar)**
  - Añadir a `testing/test_frame_extraction.py` una verificación que imprima
    `get_video_fps(video)` y compruebe que es un float > 0.
  - **El agente lo ejecuta** en local.
  - **Verificación:** la verificación corre y reporta un fps plausible; lint OK.
  - **Plan:** §5.1. **Spec:** AC-8.

- [x] **T5 — Crear notebook `notebooks/fase_0/07_pipeline_full_video_check.ipynb`**
  - Notebook que corre el pipeline **real** sobre un video completo
    (`run_pipeline(video, all_frames=True)`) y facilita la inspección visual:
    - localiza un video real y lo ejecuta;
    - **una celda** muestra el **video original** (`IPython.display.Video`);
    - **la siguiente celda** muestra el **video anotado** (el mp4 de salida).
  - **No ejecutar aquí**; solo crearlo (lo corre el usuario en RunPod/GPU).
  - **Verificación:** el notebook existe y es parseable (celdas coherentes); su
    corrida visual queda para T6.
  - **Plan:** §5.2. **Spec:** AC-8.

- [ ] **T6 — Validación del pipeline completo en RunPod (usuario)**
  - Ejecutar el notebook `07_pipeline_full_video_check.ipynb` en RunPod: correr el
    pipeline en modo completo y **comparar visualmente** original vs. anotado en
    las dos celdas.
  - Confirmar que el video anotado se reproduce a la velocidad del original (fps de
    la fuente) y que el JSON registra ese fps.
  - **Verificación:** salida coherente; AC-4 satisfecho end-to-end.
  - **Responsable:** usuario (RunPod).
  - **Plan:** §5.2. **Spec:** AC-4, AC-8.

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 `get_video_fps` | §3.1 | AC-1, AC-2, AC-3, AC-6, AC-7 |
| T2 exportación | §3.2 | AC-1 |
| T3 cableado en `run_pipeline` | §3.3 | AC-4, AC-5 |
| T4 verificación local | §5.1 | AC-8 |
| T5 notebook comparativo (crear) | §5.2 | AC-8 |
| T6 validación RunPod (usuario) | §5.2 | AC-4, AC-8 |
