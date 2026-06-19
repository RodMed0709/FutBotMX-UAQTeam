# Tasks — Escritor de video (`video_writer`)

- **Tarea atómica:** `video_writer`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** el escritor no usa modelo ni GPU, así que el **agente sí
> ejecuta** el script de validación en local; el mp4 de prueba queda en
> `outputs/test_video_maker/` para inspección.

---

## Fase A — Configuración

- [x] **T1 — Añadir `outputs_dir` y `output_fps` a la config**
  - En `configs/00_testing_config.json`: `working_dirs.outputs_dir = "outputs"` y
    `visualization.output_fps = 4` (ediciones aditivas; resto intacto).
  - **Verificación:** el JSON sigue válido y contiene ambas claves;
    `classes`/`overlay_alpha`/`preprocess` intactos.
  - **Plan:** §4. **Spec:** AC-4, AC-5.

---

## Fase B — Módulo y función

- [x] **T2 — Crear `src/core/video_writer.py` con `_load_output_fps` y `write_video`**
  - `_load_output_fps()`: lee `CONFIG_FILENAME` del `.env` con `strip()` →
    `get_abs_path` → `json.load` → `visualization.output_fps`.
  - `write_video(frames, output_path, fps=None) -> Path`: valida `frames`
    (`(N,H,W,3) uint8`, `N>0`); resuelve fps (param > config); crea el dir padre
    (`mkdir(parents, exist_ok)`); escribe el mp4 con `imageio.get_writer`
    (`FFMPEG`, `libx264`, `yuv420p`); devuelve la ruta. `imageio` importado
    **dentro** de la función.
  - **Verificación:** frames inválidos/ vacíos/ no-uint8 → `ValueError`; con frames
    válidos escribe el mp4 y devuelve su `Path`; `import src.core` no carga
    `imageio` hasta invocar la función.
  - **Plan:** §3.2–§3.5, §3.7. **Spec:** AC-2, AC-3, AC-4, AC-6, AC-7.

---

## Fase C — Exportación

- [x] **T3 — Exportar en `src/core/__init__.py`**
  - Añadir `from src.core.video_writer import write_video` y sumarlo a `__all__`.
  - **Verificación:** `from src.core import write_video` funciona; `ruff check .` y
    `black .` pasan sobre el código nuevo.
  - **Plan:** §3.1. **Spec:** AC-1.

---

## Fase D — Validación (local, ejecutada por el agente)

- [x] **T4 — Crear y ejecutar `testing/test_video_writer.py`**
  - Frames sintéticos `(N,H,W,3) uint8`; escribe el mp4 en
    `outputs/test_video_maker/` (crea la carpeta; nombre fijo, se sobrescribe).
  - Verifica: archivo existe y tamaño > 0; se puede **releer** con imageio
    (nº frames/ dimensiones coherentes); la creación del dir funciona; entrada
    inválida → `ValueError`.
  - **El agente lo ejecuta** en local; el mp4 queda para inspección.
  - **Verificación:** el script corre headless y todas las aserciones pasan;
    lint OK.
  - **Plan:** §5.1. **Spec:** AC-8.

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 config (outputs_dir, output_fps) | §4 | AC-4, AC-5 |
| T2 `write_video` + `_load_output_fps` | §3.2–§3.5, §3.7 | AC-2, AC-3, AC-4, AC-6, AC-7 |
| T3 exportación | §3.1 | AC-1 |
| T4 script de validación (crear+ejecutar) | §5.1 | AC-8 |
