# Tasks — Núcleo de segmentación por texto (`text_segmentation`)

- **Tarea atómica:** `text_segmentation`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.
>
> **Nota de ejecución:** el script de validación **no se ejecuta en local ni por
> el agente**; la corrida real es en **RunPod (GPU)** y es responsabilidad del
> usuario (la inferencia en CPU es inviable). Durante la implementación solo se
> verifica lo ligero (lint, importabilidad, imports perezosos), sin inferencia.

---

## Fase A — Estructura

- [x] **T1 — Crear `src/core/segmentation.py` con la dataclass `Detection`**
  - Crear el archivo y definir
    `@dataclass class Detection: obj_id: int; mask: np.ndarray; score: float`.
  - **Verificación:** el módulo importa sin errores y `Detection` existe con los
    tres campos.
  - **Plan:** §3.1, §3.2. **Spec:** AC-1, AC-2.

---

## Fase B — Funciones

- [x] **T2 — `_mask_from_logits` (upscale bilinear + umbral)**
  - Implementar `_mask_from_logits(logits, W, H)`: `cv2.resize(..., (W, H),
    INTER_LINEAR)` si la forma difiere y luego `> 0.0` → máscara booleana `(H, W)`.
  - **Verificación:** dada una matriz de logits de tamaño arbitrario, devuelve una
    máscara booleana con forma exactamente `(H, W)`.
  - **Plan:** §3.5. **Spec:** AC-4.

- [x] **T3 — `segment_with_text` (inferencia por prompt) + imports perezosos**
  - Validar `frame` (§3.8); `bundle = bundle or load_sam3()`; convertir a PIL;
    sesión SAM3 (`init_video_session` → `add_text_prompt` → `model(...)`) bajo
    `@torch.no_grad()`; recolectar `Detection(obj_id, mask=_mask_from_logits(...),
    score)`.
  - Importar `torch`, `cv2`, `PIL` **dentro** de la función.
  - **Verificación:** devuelve `list[Detection]` (vacía si no hay objetos); cada
    `mask` es booleana `(H, W)`; frame inválido → `ValueError`; `import src.core`
    no carga `torch`/`cv2`/`PIL` hasta invocar la función (vía `sys.modules`).
  - **Plan:** §3.4, §3.5, §3.6, §3.8. **Spec:** AC-3, AC-4, AC-7.

- [x] **T4 — `detect_classes_in_frame` + `_load_classes`**
  - `_load_classes()`: leer `CONFIG_FILENAME` del `.env` con `strip()` →
    `get_abs_path(f"configs/{...}")` → `json.load` → `config["classes"]`.
  - `detect_classes_in_frame(frame, classes=None, bundle=None)`: resolver `bundle`
    **una vez**; `classes` de la config si es `None`; por cada clase usar
    `cls["sam3_prompts"][0]` y armar `{cls["name"]: segment_with_text(frame,
    prompt, bundle)}`.
  - **Verificación:** devuelve dict con claves = `name` de las clases de la config;
    cada valor es `list[Detection]`; el modelo se resuelve una sola vez (no recarga
    por clase).
  - **Plan:** §3.7. **Spec:** AC-5, AC-6, AC-7.

---

## Fase C — Exportación

- [x] **T5 — Exportar en `src/core/__init__.py`**
  - Añadir `from src.core.segmentation import Detection, segment_with_text,
    detect_classes_in_frame` y sumarlos a `__all__`.
  - **Verificación:** `from src.core import segment_with_text,
    detect_classes_in_frame, Detection` funciona desde cualquier cwd; `ruff check
    .` y `black .` pasan sobre el código nuevo.
  - **Plan:** §3.1, §3.6. **Spec:** AC-1.

---

## Fase D — Script de prueba

- [x] **T6 — Crear `testing/test_segmentation.py`**
  - Localizar un `.MOV` real (rglob sobre `dataset_dir`), extraer un frame con
    `extract_frames` (p. ej. el central).
  - `segment_with_text(frame, "robot")` → imprimir nº detecciones, forma de máscara
    y scores.
  - `detect_classes_in_frame(frame)` → por clase, conteo y score medio.
  - Manejar **sin abortar** la ausencia de pesos/datos; acotar a una clase/prompt
    o avisar del costo para no colgarse.
  - **No ejecutar el script aquí**; solo crearlo (la corrida es en RunPod).
  - **Verificación:** el archivo existe, pasa lint y es importable/parseable; su
    ejecución real queda para la Fase E.
  - **Plan:** §5. **Spec:** AC-8.

---

## Fase E — Validación manual (a cargo del usuario, en RunPod/GPU)

- [x] **T7 — Ejecutar y validar manualmente en RunPod (GPU)**
  - Ejecutar el script donde haya pesos + GPU:
    ```bash
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_segmentation.py
    ```
  - Confirmar que `segment_with_text` produce máscaras del tamaño del frame con
    scores razonables y que `detect_classes_in_frame` devuelve detecciones por
    clase.
  - **Verificación:** salida coherente; criterios AC-1 a AC-8 satisfechos.
  - **Plan:** §5, §7. **Spec:** AC-8.
  - **Responsable:** usuario (RunPod).

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 `Detection` + módulo | §3.1, §3.2 | AC-1, AC-2 |
| T2 `_mask_from_logits` | §3.5 | AC-4 |
| T3 `segment_with_text` + imports perezosos | §3.4–§3.6, §3.8 | AC-3, AC-4, AC-7 |
| T4 `detect_classes_in_frame` + `_load_classes` | §3.7 | AC-5, AC-6, AC-7 |
| T5 exportación | §3.1, §3.6 | AC-1 |
| T6 script de prueba | §5 | AC-8 |
| T7 validación manual (RunPod) | §5, §7 | AC-8 |
