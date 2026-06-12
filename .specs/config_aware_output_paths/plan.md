# Plan técnico — Salidas con namespace por config (`config_aware_output_paths`)

- **Tarea atómica:** `config_aware_output_paths`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir cómo agregar un **namespace opcional por config** a las rutas de salida, de
forma que cada configuración del benchmark escriba en
`outputs/inference/<run_label>/<stem>/<stem>.{json,mp4}` sin pisar a las demás, y que
el `skip-done` opere **por config**. El mecanismo es:

1. `inference_paths` acepta `namespace` (la pieza que arma la ruta).
2. Un parámetro `run_label` se hilvana por la cadena
   `run_batch → run_inference → run_pipeline | track_video` y, en cada orquestador,
   se pasa a `inference_paths` como `namespace`.

El cambio es **aditivo y retrocompatible**: `run_label=None` (default) reproduce la
ruta plana de hoy.

---

## 2. Stack técnico

- Solo manipulación de `pathlib.Path` (sin dependencias nuevas).
- La derivación de rutas es **lógica pura** → testeable **sin GPU ni SAM3**.

---

## 3. Diseño

### 3.1 Archivos a modificar

| Archivo | Cambio |
|---|---|
| `src/core/inference_schema.py` | `inference_paths` gana `namespace: str \| None = None`. |
| `src/core/pipeline.py` | `run_pipeline` gana `run_label`; lo pasa a `inference_paths`. |
| `src/core/tracking.py` | `track_video` gana `run_label`; lo pasa a `inference_paths`. |
| `src/core/inference.py` | `run_inference` gana `run_label`; lo reenvía a ambas ramas. |
| `src/core/batch.py` | `run_batch` gana `run_label`; lo usa en `skip-done` y lo reenvía a `run_inference`. |
| `testing/test_batch_inference.py` | Smoke (sin GPU) de derivación de rutas + hilvanado. |
| `CLAUDE.md` | Output placement / descripción de `batch` mencionan el namespace. |

### 3.2 `inference_paths` (`inference_schema.py`)

Firma y cuerpo:

```python
def inference_paths(
    video_stem: str, outputs_dir: str, namespace: str | None = None
) -> tuple[Path, Path]:
    base = PROJECT_ROOT / outputs_dir / "inference"
    if namespace:
        base = base / namespace
    base = base / video_stem
    return base / f"{video_stem}.json", base / f"{video_stem}.mp4"
```

- `namespace` va **antes** del `<stem>` (subcarpeta por config).
- `None`/cadena vacía ⇒ ruta actual (retrocompatible).
- Se actualiza el docstring para documentar `namespace`.

### 3.3 Hilvanado de `run_label`

**`run_pipeline` (`pipeline.py`)** y **`track_video` (`tracking.py`)** ganan
`run_label: str | None = None` (al final de la firma). En el bloque que deriva la
ruta por defecto (hoy `pipeline.py:174` / `tracking.py:268`):

```python
# antes
json_path, mp4_path = inference_paths(stem, outputs_dir)
# después
json_path, mp4_path = inference_paths(stem, outputs_dir, namespace=run_label)
```

El camino `output_path is not None` **no se toca** → conserva su prioridad (criterio
#5).

**`run_inference` (`inference.py`)** gana `run_label`; lo reenvía a `run_pipeline`
(rama segmentación) y a `track_video` (rama tracking).

**`run_batch` (`batch.py`)** gana `run_label`; lo usa en dos puntos:

```python
# skip-done (hoy batch.py:234): usar el mismo namespace
json_path, _ = inference_paths(Path(ruta).stem, outputs_dir, namespace=run_label)
# ...
res = run_inference(ruta, ..., run_label=run_label)   # reenviar a la inferencia
```

Así el `skip-done` comprueba la existencia del JSON **bajo esa config** (criterio #6).

### 3.4 Lo que NO cambia (anti-alcance técnico)

- El esquema/contenido del JSON (`build_header`, `frame_record`, `write_inference_json`).
- La precedencia y semántica de `output_path`.
- La estrategia de detector/tracker, los configs, el muestreo.
- `aggregate_config`: sigue leyendo los paths del `summary` (no recalcula rutas).
- Los notebooks de benchmark (entregable aparte).

---

## 4. Cambios de configuración y dependencias

- **Ninguno.** Sin nuevas dependencias ni claves de config.

---

## 5. Validación (`testing/test_batch_inference.py`)

Smoke **sin GPU** (la derivación de rutas es lógica pura):

### 5.1 Derivación de rutas

- `inference_paths("vid", "outputs")` → `.../inference/vid/vid.{json,mp4}` (sin
  namespace).
- `inference_paths("vid", "outputs", namespace="cfgA")` →
  `.../inference/cfgA/vid/vid.{json,mp4}`.
- Dos `namespace` distintos sobre el mismo stem → **rutas distintas** (no colisión).

### 5.2 Hilvanado en `run_batch` (sin invocar SAM3)

- Verificar que `run_batch` acepta `run_label` y que el `skip-done` usa el path
  namespaced: preparar un JSON en `outputs/inference/<run_label>/<stem>/<stem>.json`
  y comprobar que ese video se reporta `skipped` con `run_label` puesto, y **no** se
  saltaría con `run_label=None` (o con otro label). (Reutiliza el patrón de
  `test_batch_inference.py`, que ya fabrica JSON falsos para probar `skip-done` sin
  GPU.)

### 5.3 Calidad

- `ruff check .` y `black .` limpios.

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec §5) | Cubierto por |
|---|---|
| 1. `inference_paths` con/sin `namespace` | §3.2, §5.1 |
| 2. `run_label` propagado por la cadena | §3.3 |
| 3. Retrocompatibilidad con `run_label=None` | §3.2, §3.3, §5.1 |
| 4. JSON+mp4 bajo `<run_label>/<stem>/` | §3.2, §3.3 |
| 5. `output_path` tiene prioridad | §3.3 |
| 6. `skip-done` por config | §3.3, §5.2 |
| 7. No cambia el esquema JSON | §3.4 |
| 8. Smoke sin GPU | §5.1, §5.2 |
| 9. `CLAUDE.md` actualizado | §3.1 |

---

## 7. Riesgos y consideraciones

- **Consistencia del `skip-done`.** El path de comprobación en `batch.py` y el path de
  escritura en `run_pipeline`/`track_video` deben usar **el mismo** `run_label`; como
  todos derivan de `inference_paths(..., namespace=run_label)`, coinciden por
  construcción. El test §5.2 lo confirma.
- **Etiqueta como segmento de ruta.** `run_label` se usa tal cual; los labels del
  benchmark (`sam3_text+bytetrack`) son válidos en Linux. Si en el futuro llegan
  caracteres problemáticos, el saneo sería una mejora aparte (no en esta tarea).
- **Orden de parámetros.** `run_label` se agrega **al final** de cada firma para no
  romper llamadas posicionales existentes.
- **`output_path` + `run_label`.** No se combinan; documentar que `output_path` manda
  evita ambigüedad para quien pase ambos.
