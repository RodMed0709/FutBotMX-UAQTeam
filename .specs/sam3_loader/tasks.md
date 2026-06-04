# Tasks — Carga del modelo SAM3 (`sam3_loader`)

- **Tarea atómica:** `sam3_loader`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan / criterio de aceptación** que la origina. Marcar `- [x]`
> al completar.

---

## Fase A — Estructura

- [ ] **T1 — Crear `src/core/sam3_loader.py` con la dataclass `Sam3Bundle`**
  - Crear el archivo y definir
    `@dataclass class Sam3Bundle: processor; model; device: str`.
  - **Verificación:** el módulo importa sin errores y `Sam3Bundle` existe con los
    tres campos.
  - **Plan:** §3.1, §3.2. **Spec:** AC-3.

---

## Fase B — Módulo y función

- [ ] **T2 — Resolución de la ruta del modelo (`_resolve_sam3_dir`)**
  - Leer `CONFIG_FILENAME` del `.env` con `strip()`; resolver
    `configs/<CONFIG_FILENAME>` con `get_abs_path`; parsear el JSON; leer
    `working_dirs.sam3_dir`; resolverlo con `get_abs_path` → ruta absoluta del
    modelo.
  - **Verificación:** devuelve la ruta absoluta verificada del directorio del
    modelo; con `CONFIG_FILENAME`/clave/ruta ausentes lanza
    `KeyError`/`ValueError`/`FileNotFoundError`.
  - **Plan:** §3.4. **Spec:** AC-2, AC-8.

- [ ] **T3 — Construcción del bundle (`_build_bundle`) con device + imports perezosos**
  - Importar `torch` y `transformers` **dentro** de la función. Resolver device:
    `device or ("cuda" if torch.cuda.is_available() else "cpu")`.
  - Cargar con `AutoProcessor.from_pretrained(str(sam3_dir))` y
    `AutoModel.from_pretrained(str(sam3_dir), dtype=torch.bfloat16,
    low_cpu_mem_usage=True).to(device)`; `model.eval()`.
  - Devolver `Sam3Bundle(processor, model, device)`.
  - **Verificación:** retorna un `Sam3Bundle` con `device` correcto; forzar
    `device="cpu"` se respeta; `import src.core` no carga `torch`/`transformers`
    (no aparecen en `sys.modules` hasta llamar a `load_sam3`).
  - **Plan:** §3.5, §3.6. **Spec:** AC-1, AC-4, AC-5.

- [ ] **T4 — Capa de caché y función pública `load_sam3`**
  - Definir `_cached_load()` decorada con `lru_cache(maxsize=1)` que llama a
    `_build_bundle()`.
  - Definir `load_sam3(*, use_cache: bool = True, device: str | None = None)
    -> Sam3Bundle`: si `use_cache and device is None` → `_cached_load()`; en otro
    caso → `_build_bundle(device=device)`.
  - **Verificación:** 2ª llamada por defecto ⇒ **mismo** objeto
    (`is`); `use_cache=False` ⇒ objeto **distinto**; un `device` forzado no
    contamina el singleton.
  - **Plan:** §3.3, §3.5. **Spec:** AC-6, AC-7.

- [ ] **T5 — Manejo de errores**
  - Asegurar el reparto de excepciones de §3.7: `CONFIG_FILENAME` ausente, config
    inexistente, `working_dirs.sam3_dir` ausente, directorio del modelo
    inexistente, fallo de `from_pretrained` — todas **propagadas** (fallo
    temprano, sin captura silenciosa).
  - **Verificación:** cada situación produce la excepción esperada y detiene el
    proceso.
  - **Plan:** §3.7. **Spec:** AC-9.

- [ ] **T6 — Exportar `load_sam3` en `src/core/__init__.py`**
  - Añadir `from src.core.sam3_loader import load_sam3` y sumar `"load_sam3"`
    (y `"Sam3Bundle"` si se decide exponerla) a `__all__`.
  - **Verificación:** `from src.core import load_sam3` funciona desde cualquier
    cwd; `ruff check .` y `black .` pasan sobre el código nuevo.
  - **Plan:** §3.1, §3.6. **Spec:** AC-1, AC-8.

---

## Fase C — Script de prueba

- [ ] **T7 — Crear `testing/test_sam3_loader.py`**
  - `bundle = load_sam3()` → imprimir `type(model).__name__`, `bundle.device`,
    `dtype` y conteo de parámetros.
  - `bundle2 = load_sam3()` → comprobar `bundle2 is bundle` (caché).
  - `bundle3 = load_sam3(use_cache=False)` → comprobar `bundle3 is not bundle`
    (opt-out fuerza recarga).
  - Capturar el caso de pesos ausentes y reportar **sin abortar** (local sin
    modelo descargado).
  - **Verificación:** el script existe y, donde los pesos estén disponibles,
    imprime la info de carga y confirma caché y opt-out sin abortar.
  - **Plan:** §5. **Spec:** AC-10.

---

## Fase D — Validación manual (a cargo del usuario)

- [ ] **T8 — Ejecutar y validar manualmente (donde existan los pesos)**
  - Ejecutar dentro del contenedor (o pod con GPU):
    ```bash
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_sam3_loader.py
    ```
  - Confirmar que el modelo carga, que la 2ª llamada reutiliza la caché y que
    `use_cache=False` fuerza una recarga; revisar que `device` sea el esperado.
  - **Verificación:** salida coherente; criterios AC-1 a AC-10 del spec
    satisfechos.
  - **Plan:** §5, §7. **Spec:** AC-10.
  - **Responsable:** usuario.

---

## Trazabilidad resumida

| Tarea | Plan | Spec (AC) |
|---|---|---|
| T1 dataclass `Sam3Bundle` | §3.1, §3.2 | AC-3 |
| T2 resolución de ruta | §3.4 | AC-2, AC-8 |
| T3 construcción + device + imports perezosos | §3.5, §3.6 | AC-1, AC-4, AC-5 |
| T4 caché + `load_sam3` | §3.3, §3.5 | AC-6, AC-7 |
| T5 manejo de errores | §3.7 | AC-9 |
| T6 exportación | §3.1, §3.6 | AC-1, AC-8 |
| T7 script de prueba | §5 | AC-10 |
| T8 validación manual | §5, §7 | AC-10 |
