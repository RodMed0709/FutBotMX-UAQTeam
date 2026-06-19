# Plan técnico — Carga del modelo SAM3 (`sam3_loader`)

- **Tarea atómica:** `sam3_loader`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** no hay draft previo para este plan.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar una forma **única y reutilizable** de
cargar SAM3 (processor + model) lista para inferir: resolviendo sola la **ruta**
del modelo desde la configuración (`working_dirs.sam3_dir` vía `get_abs_path`) y
el **dispositivo** (GPU si está disponible, si no CPU, forzable), devolviendo todo
en un objeto único `Sam3Bundle`, con **caché singleton** y opción de desactivarla
(`use_cache`). Esto sustituye el bloque de carga copy-pasteado en los notebooks de
`fase_0/`. Además, definir el script de validación manual.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Carga del modelo:** `transformers` (HuggingFace) — `AutoProcessor` y
  `AutoModel`, patrón `from_pretrained` apuntando a un **directorio local** de
  pesos. Es exactamente la API que ya validan los notebooks `fase_0/` (01–05).
- **Tensores / dispositivo / dtype:** `torch` (`torch.cuda.is_available()`,
  `torch.bfloat16`, `.to(device)`).
- **Configuración:** `json` (estándar) para parsear el `.json`; lectura de
  `CONFIG_FILENAME` desde `.env` con `strip()` (el `.env` la escribe con espacio:
  `CONFIG_FILENAME =...`).
- **Resolución/validación de rutas:** `src/utils.py::get_abs_path` (reutilizada).
- **Caché:** `functools.lru_cache` (estándar).
- **Estructura de salida:** `dataclasses.dataclass` (estándar).

> Nota: `torch` y `transformers` son dependencias **pesadas**; ver §3.6 (imports
> perezosos) para que `import src.core` no las arrastre.

---

## 3. Diseño

### 3.1 Ubicación y módulo

- Archivo nuevo: `src/core/sam3_loader.py`.
- Exportación: añadir en `src/core/__init__.py`
  `from src.core.sam3_loader import load_sam3` y sumar `"load_sam3"` (y, si se
  decide exponerla, `"Sam3Bundle"`) a `__all__`, junto a `extract_frames`.

### 3.2 Estructura de salida — `Sam3Bundle`

```python
@dataclass
class Sam3Bundle:
    processor: "AutoProcessor"
    model: "AutoModel"
    device: str
```

Agrupa todo lo necesario para inferir desde una sola llamada. Los consumidores
leen `bundle.processor`, `bundle.model` y `bundle.device`; ese `device` es el que
luego se pasa a `inference_device` en las sesiones de SAM3 (carga e inferencia
comparten la misma fuente, evitando el desajuste cuda/cpu visto en los notebooks).

### 3.3 Función pública — firma

```python
def load_sam3(*, use_cache: bool = True, device: str | None = None) -> Sam3Bundle:
    ...
```

- Parámetros **keyword-only**.
- `use_cache: bool = True` — caché singleton activa por defecto.
- `device: str | None = None` — `None` ⇒ auto (`cuda` si disponible, si no `cpu`);
  un valor concreto fuerza el dispositivo.
- **Retorno:** `Sam3Bundle`.

### 3.4 Resolución de la ruta del modelo

1. Leer `CONFIG_FILENAME` del `.env` aplicando `strip()` a clave y valor.
2. `get_abs_path(f"configs/{config_filename}")` → ruta absoluta verificada.
3. `json.load(...)` del archivo de configuración.
4. Leer `working_dirs.sam3_dir` (ruta **relativa** a `PROJECT_ROOT`).
5. `get_abs_path(sam3_dir)` → ruta absoluta verificada del directorio del modelo.
6. Pasar `str(...)` a `from_pretrained` (espera `str`).

No se construyen rutas absolutas ni symlinks; toda ruta pasa por `get_abs_path`.

### 3.5 Construcción del modelo y caché

Lógica de construcción única en una función privada, reutilizada por el camino
cacheado y el fresco (sin duplicar):

```python
def _build_bundle(device: str | None = None) -> Sam3Bundle:
    import torch
    from transformers import AutoProcessor, AutoModel

    sam3_dir = _resolve_sam3_dir()              # §3.4
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(str(sam3_dir))
    model = AutoModel.from_pretrained(
        str(sam3_dir),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return Sam3Bundle(processor=processor, model=model, device=device)


@lru_cache(maxsize=1)
def _cached_load() -> Sam3Bundle:
    return _build_bundle()


def load_sam3(*, use_cache: bool = True, device: str | None = None) -> Sam3Bundle:
    if use_cache and device is None:
        return _cached_load()
    return _build_bundle(device=device)
```

- **Camino por defecto** (`use_cache=True`, `device=None`) → `_cached_load()`:
  1ª llamada carga, siguientes reutilizan (singleton, `maxsize=1`).
- **Opt-out** (`use_cache=False`) **o** `device` forzado → `_build_bundle(...)`
  construye una **instancia fresca** sin tocar la cacheada (un device forzado no
  debe contaminar el singleton auto).
- Aislamiento en tests: el camino soportado es `use_cache=False`. Si hiciera falta
  limpiar el singleton, queda `_cached_load.cache_clear()` como recurso interno.

### 3.6 Imports perezosos (dependencias pesadas)

`torch` y `transformers` se importan **dentro** de `_build_bundle` (no a nivel de
módulo). Así `import src.core` —necesario para `extract_frames`, que **no** usa
torch— no obliga a cargar torch/transformers. Sigue el patrón ya usado en
`src/utils.py::show_frames`, que importa `matplotlib` dentro de la función.

### 3.7 Manejo de errores

| Situación | Excepción |
|---|---|
| `CONFIG_FILENAME` ausente en `.env` | `KeyError` / `ValueError` (mensaje claro) |
| Archivo de config inexistente | `FileNotFoundError` (vía `get_abs_path`) |
| Clave `working_dirs.sam3_dir` ausente | `KeyError` |
| Directorio del modelo inexistente | `FileNotFoundError` (vía `get_abs_path`) |
| Fallo al cargar pesos (`from_pretrained`) | excepción de `transformers`/`torch` propagada |

Todas se **propagan** (fallo explícito y temprano), coherente con el estilo de
`get_abs_path` y de `extract_frames`.

---

## 4. Cambios de configuración

- **Ninguno.** La clave `working_dirs.sam3_dir` ya existe en
  `configs/00_testing_config.json`. (La definición de las clases del modelo es de
  la tarea `classes_config`, no de esta.)

---

## 5. Script de validación manual

- Ubicación: `testing/test_sam3_loader.py` (ejecutable manual, no pytest).
- **Flujo:**
  1. `bundle = load_sam3()` → imprimir `type(model).__name__`, `bundle.device`,
     `dtype` de los parámetros y conteo de parámetros (≈ como el bloque de los
     notebooks).
  2. `bundle2 = load_sam3()` → comprobar que es **el mismo** objeto cacheado
     (`bundle2 is bundle`).
  3. `bundle3 = load_sam3(use_cache=False)` → comprobar que es **distinto**
     (`bundle3 is not bundle`) → el opt-out fuerza recarga.
  4. Reportar sin abortar abruptamente si los pesos no están presentes (caso
     local sin el modelo descargado).
- **Ejecución:** donde los pesos de SAM3 existan (contenedor o pod con GPU):
  ```bash
  docker compose --env-file .env -f docker/docker-compose.yml \
    exec futbotmx26 python testing/test_sam3_loader.py
  ```

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Carga centralizada | §3.1, §3.3, §3.5 | `load_sam3` en `src/core/sam3_loader.py` |
| AC-2 Ruta por configuración | §3.4 | `working_dirs.sam3_dir` vía `get_abs_path` |
| AC-3 Salida agrupada | §3.2 | dataclass `Sam3Bundle` |
| AC-4 Dispositivo auto | §3.5 | `cuda if available else cpu` |
| AC-5 Dispositivo forzable | §3.3, §3.5 | parámetro `device` |
| AC-6 Caché por defecto | §3.5 | `lru_cache(maxsize=1)` |
| AC-7 Opt-out de caché | §3.5 | `use_cache=False` → instancia fresca |
| AC-8 Funciona desde cualquier cwd | §3.4 | rutas vía `get_abs_path`/`PROJECT_ROOT` |
| AC-9 Fallo claro | §3.7 | excepciones propagadas |
| AC-10 Validación manual | §5 | `testing/test_sam3_loader.py` |

---

## 7. Riesgos y consideraciones

- **Pesos no presentes en local:** SAM3 es git-ignored y puede no estar en disco
  en el host. La validación manual (AC-10) y cualquier carga real deben correr
  donde los pesos existan (contenedor / pod GPU). Su descarga automática es de la
  futura tarea `bootstrap_data`, fuera de alcance aquí.
- **`lru_cache` y el override de device:** se decide **no** cachear cuando se
  fuerza `device`, para que una llamada puntual a CPU/GPU no fije el singleton; el
  camino cacheado es solo el default (auto). Documentado en §3.5.
- **API de SAM3 vía `transformers`:** se asume estable y disponible (instalada
  según el `requirements.txt`/instalación manual de SAM3). Es la misma API ya
  ejercitada por los notebooks, por lo que el riesgo es bajo.
- **bfloat16 en CPU:** se mantiene `bfloat16` por consistencia con los notebooks;
  si en algún entorno CPU diera problemas de soporte, el ajuste de dtype se
  evaluaría como cambio puntual (no altera el diseño de carga/caché/bundle).
