# Tasks — Función de extracción de frames de un vídeo

- **Tarea atómica:** `frame_extraction`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Configuración y estructura

- [x] **T1 — Crear el submódulo `src/core/`**
  - Crear el paquete `src/core/` con su `__init__.py`.
  - **Verificación:** `src/core/__init__.py` existe y `src.core` es importable.
  - **Plan:** §3.1. **Spec:** AC-1.

- [x] **T2 — Añadir la clave de cuota a la configuración**
  - Agregar `"frame_quota": 30` (ejemplo) dentro del bloque `preprocess` de
    `configs/00_testing_config.json`.
  - **Verificación:** el JSON sigue siendo válido y contiene
    `preprocess.frame_quota` con un entero positivo.
  - **Plan:** §3.4, §4. **Spec:** AC-3.

---

## Fase B — Módulo y función

- [x] **T3 — Crear `src/core/frame_extraction.py` con la firma**
  - Definir `extract_frames(video_path: Path, all_frames: bool = False) -> np.ndarray`.
  - Importar `pathlib.Path`, `numpy`, `decord` y `get_abs_path` de `src/utils.py`.
  - **Verificación:** la función existe con la firma indicada y el módulo importa
    sin errores.
  - **Plan:** §3.1, §3.2. **Spec:** AC-1, AC-2.

- [x] **T4 — Validación de la ruta del vídeo vía `get_abs_path`**
  - Convertir `video_path` a ruta relativa respecto a `PROJECT_ROOT` y pasarla
    como `str` a `get_abs_path` para obtener la ruta absoluta verificada.
  - **Verificación:** una ruta de vídeo inexistente/ inválida detiene el proceso
    con `FileNotFoundError`/`ValueError`; una válida devuelve su ruta absoluta.
  - **Plan:** §3.3. **Spec:** AC-6.

- [x] **T5 — Lectura de la configuración y de la cuota**
  - Leer `CONFIG_FILENAME` del `.env` con `strip()`; construir
    `configs/<CONFIG_FILENAME>`, resolver con `get_abs_path` y parsear el JSON.
  - Obtener la cuota desde `preprocess.frame_quota`.
  - **Verificación:** la función obtiene la cuota desde la config (no del código);
    si la clave falta o es inválida lanza `KeyError`/`ValueError`.
  - **Plan:** §3.4, §3.6. **Spec:** AC-3.

- [x] **T6 — Lógica de muestreo (modo cuota y modo completo)**
  - Abrir el vídeo con `decord.VideoReader`; obtener `total = len(vr)`.
  - **Modo completo** (`all_frames=True`): índices `range(total)`.
  - **Modo cuota** (`all_frames=False`): si `total <= quota` usar todos los
    frames; si `total > quota`, índices equiespaciados con
    `np.linspace(0, total - 1, quota)` redondeados a `int`.
  - **Verificación:** en modo cuota la cantidad devuelta coincide con la cuota (o
    el total si el vídeo tiene menos) y los índices están repartidos
    uniformemente; en modo completo se cubren todos los frames.
  - **Plan:** §3.5 (pasos 2-4). **Spec:** AC-4, AC-5.

- [x] **T7 — Recuperación de frames y retorno NumPy**
  - Configurar el bridge nativo de decord y recuperar los frames seleccionados
    con `vr.get_batch(indices).asnumpy()`.
  - Devolver un `np.ndarray` con forma `(N, H, W, 3)` (sin escribir a disco).
  - **Verificación:** el valor de retorno es un `np.ndarray` `(N, H, W, 3)`; no se
    generan archivos en disco.
  - **Plan:** §3.5 (pasos 5-6). **Spec:** AC-7.

- [x] **T8 — Manejo de errores**
  - Asegurar el reparto de excepciones de §3.6: ruta inválida
    (`FileNotFoundError`/`ValueError`), `CONFIG_FILENAME` ausente, clave de cuota
    ausente (`KeyError`), cuota no entera o ≤ 0 (`ValueError`).
  - **Verificación:** cada situación de error produce la excepción esperada y
    detiene el proceso.
  - **Plan:** §3.6.

---

## Fase C — Script de prueba

- [x] **T9 — Crear `testing/test_frame_extraction.py`**
  - Verificar la existencia de `dataset_dir` con `get_abs_path`, pero localizar el
    `.MOV` de forma **recursiva** (`rglob`) sobre la ruta **sin resolver el
    symlink** (`PROJECT_ROOT / dataset_dir`), de modo que las rutas queden bajo el
    proyecto (p. ej. `data/raw/.../IMG.MOV`) y `extract_frames` pueda delegarlas a
    `get_abs_path`.
  - Llamar `extract_frames(video, all_frames=False)` y reportar la forma del
    arreglo (cuota o total disponible).
  - Llamar `extract_frames(video, all_frames=True)` y reportar el total.
  - Imprimir formas, `dtype` y conteos; capturar `FileNotFoundError` por ruta y
    reportar **sin abortar** (caso local: symlinks de datos solo válidos en el
    contenedor).
  - **Verificación:** el script existe y, al ejecutarse en el contenedor, imprime
    las formas y conteos de ambos modos sin abortar ante rutas faltantes.
  - **Plan:** §5. **Spec:** AC-8.

---

## Fase D — Validación manual (a cargo del usuario)

- [x] **T10 — Ejecutar y validar manualmente (en el contenedor)**
  - Ejecutar dentro del contenedor:
    ```bash
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_frame_extraction.py
    ```
  - Confirmar que el modo cuota devuelve la cantidad esperada de frames repartidos
    en el tiempo y el modo completo devuelve todos los frames.
  - **Verificación:** salida coherente; criterios AC-1 a AC-8 del spec
    satisfechos.
  - **Plan:** §5, §7. **Spec:** AC-8.
  - **Responsable:** usuario.

---

## Resumen de trazabilidad

| Tarea | Plan | Criterios de aceptación (spec) |
|---|---|---|
| T1 | §3.1 | AC-1 |
| T2 | §3.4, §4 | AC-3 |
| T3 | §3.1, §3.2 | AC-1, AC-2 |
| T4 | §3.3 | AC-6 |
| T5 | §3.4, §3.6 | AC-3 |
| T6 | §3.5 | AC-4, AC-5 |
| T7 | §3.5 | AC-7 |
| T8 | §3.6 | — |
| T9 | §5 | AC-8 |
| T10 | §5, §7 | AC-8 |

---

## Nota de metodología

Este documento cierra el paso 4. La **implementación (paso 5)** de estas tareas
ocurrirá únicamente cuando se indique explícitamente; hasta entonces no se crea
ni modifica código fuente.
