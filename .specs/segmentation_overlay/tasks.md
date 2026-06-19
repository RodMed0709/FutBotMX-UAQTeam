# Tasks — Visualización multi-clase de detecciones (`segmentation_overlay`)

- **Tarea atómica:** `segmentation_overlay`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** el agente **no ejecuta** los artefactos de validación
> (ni el script `testing/test_overlay.py` ni el notebook): solo los **crea**. La
> corrida real la hace el usuario en **RunPod**. Durante la implementación el
> agente solo verifica lo ligero (lint, importabilidad, imports perezosos), sin
> ejecutar la validación.

---

## Fase A — Configuración

- [x] **T1 — Añadir `visualization.overlay_alpha` a la config**
  - Agregar `"visualization": { "overlay_alpha": 0.55 }` a
    `configs/00_testing_config.json` (edición aditiva; resto intacto).
  - **Verificación:** el JSON sigue siendo válido y contiene
    `visualization.overlay_alpha`; `working_dirs`/`preprocess`/`classes` intactos.
  - **Plan:** §4. **Spec:** AC-4.

---

## Fase B — Módulo y funciones

- [x] **T2 — Crear `src/core/overlay.py` con `_load_overlay_config`**
  - Helper que lee `CONFIG_FILENAME` del `.env` con `strip()` → `get_abs_path` →
    `json.load`, y devuelve `(classes, default_alpha)` desde `classes` y
    `visualization.overlay_alpha`.
  - **Verificación:** devuelve las clases y el alpha por defecto de la config;
    ausencias relevantes lanzan `ValueError`/`KeyError`/`FileNotFoundError`.
  - **Plan:** §3.1, §3.3. **Spec:** AC-3, AC-4.

- [x] **T3 — `overlay_detections` (composición → uint8)**
  - Validar `frame` (→ `H,W`); resolver `classes`/`alpha`/mapa de color (param con
    prioridad sobre config); mezcla en float sobre **copia**
    (`out[mask] = (1-alpha)*out[mask] + alpha*color01`); retorno
    `(out*255).round().clip(0,255).astype(uint8)`.
  - Chequeo defensivo: `mask.shape != (H,W)` → `warnings.warn` + omitir.
  - **Verificación:** devuelve `uint8 (H,W,3)`; píxeles bajo máscara viran al color
    de clase; no muta la entrada; dict vacío → copia del frame; frame inválido →
    `ValueError`.
  - **Plan:** §3.4, §3.6. **Spec:** AC-2, AC-6, AC-7.

- [x] **T4 — `show_overlay` (display + leyenda, matplotlib perezoso)**
  - Compone con `overlay_detections`; `imshow` + leyenda (`mpatches.Patch` por
    clase, color↔`name`); matplotlib importado **dentro** de la función;
    display-only.
  - **Verificación:** `import src.core` no carga matplotlib hasta invocar
    `show_overlay`; la función no devuelve array ni escribe a disco.
  - **Plan:** §3.5. **Spec:** AC-5, AC-6.

---

## Fase C — Exportación

- [x] **T5 — Exportar en `src/core/__init__.py`**
  - Añadir `from src.core.overlay import overlay_detections, show_overlay` y
    sumarlos a `__all__`.
  - **Verificación:** `from src.core import overlay_detections, show_overlay`
    funciona; `ruff check .` y `black .` pasan sobre el código nuevo.
  - **Plan:** §3.1. **Spec:** AC-1.

---

## Fase D — Validación (artefactos creados por el agente, ejecutados por el usuario)

- [x] **T6 — Crear el script headless `testing/test_overlay.py`**
  - Frame y máscaras **sintéticas**, `classes` explícitas; llama
    `overlay_detections` y verifica forma `(H,W,3)`, `dtype uint8`, viraje de color
    bajo máscara, frame de entrada no mutado. Sin matplotlib ni modelo (corre
    headless).
  - **No ejecutar aquí**; solo crearlo (lo corre el usuario en RunPod).
  - **Verificación:** el archivo existe, pasa lint y es importable/parseable; su
    ejecución real queda para la Fase E.
  - **Plan:** §5.1. **Spec:** AC-8.

- [x] **T7 — Crear el notebook `notebooks/fase_0/06_segmentation_overlay_check.ipynb`**
  - Notebook que extrae un frame real, obtiene detecciones de
    `detect_classes_in_frame` y llama `show_overlay` para inspección visual.
  - **No ejecutar aquí**; solo crearlo (lo corre el usuario en RunPod/GPU).
  - **Verificación:** el notebook existe y es coherente (celdas parseables); su
    corrida visual queda para la Fase E.
  - **Plan:** §5.2. **Spec:** AC-8.

---

## Fase E — Validación manual (a cargo del usuario, en RunPod)

- [ ] **T8 — Ejecutar las validaciones en RunPod**
  - Correr el script headless:
    `python testing/test_overlay.py` (todas las aserciones pasan).
  - Ejecutar el notebook `06_segmentation_overlay_check.ipynb` y confirmar
    visualmente el overlay con leyenda sobre detecciones reales.
  - **Verificación:** salida coherente; criterios AC-1 a AC-8 satisfechos.
  - **Responsable:** usuario (RunPod).

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 config alpha | §4 | AC-4 |
| T2 `_load_overlay_config` | §3.1, §3.3 | AC-3, AC-4 |
| T3 `overlay_detections` | §3.4, §3.6 | AC-2, AC-6, AC-7 |
| T4 `show_overlay` | §3.5 | AC-5, AC-6 |
| T5 exportación | §3.1 | AC-1 |
| T6 script headless (crear) | §5.1 | AC-8 |
| T7 notebook (crear) | §5.2 | AC-8 |
| T8 validación manual (RunPod) | §5 | AC-8 |
