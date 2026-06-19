# Spec — Gestión y organización de metadatos del dataset (`csv_dataset_metadata`)

- **Tarea atómica:** `csv_dataset_metadata`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** un manifiesto tabular (`assets/db_metadata.csv`) con los metadatos de
> los 123 videos de `data/raw` y una partición reproducible en *splits*,
> **para** poder iterar el dataset (en especial el subconjunto de *testing*) de
> forma determinista y consultar resoluciones / fps / duraciones sin abrir cada
> video, montando el pipeline sobre un conjunto controlado.

---

## 2. Motivación (por qué)

- Hoy no existe un **inventario** del dataset: para saber resolución, fps o
  duración de los videos hay que abrirlos uno a uno. Un manifiesto resuelve eso de
  una sola lectura.
- El pipeline necesita poder correr sobre un **subconjunto reproducible** (p. ej.
  *testing*) en vez de los 123 videos; la columna `split` con muestreo *seeded*
  habilita exactamente eso.
- Tener una **vista tabular** (pandas) facilita el análisis exploratorio del
  dataset en notebooks (distribución de resoluciones, fps, duraciones) sin coste de
  decodificación.
- El valor a esta altura es el **manifiesto + splits**, no una optimización de
  rendimiento: las lecturas de solo-metadatos con `decord` ya son baratas. Por eso
  el pipeline **no** se refactoriza en esta tarea (ver §3.2 y §6).

---

## 3. Alcance

### 3.1 Dentro de alcance

- Nuevo submódulo **`src/data/`** (con su `__init__.py`) y un módulo
  **`src/data/metadata.py`** con la lógica de generación/validación del CSV.
- **Descubrir** los videos `.MOV` bajo `dataset_dir` de forma **recursiva** y
  **determinista**.
- **Extraer** por video: `duracion`, `ancho`, `alto`, `fps_average` (vía `decord`,
  solo metadatos).
- **Generar** `assets/db_metadata.csv` con las columnas:
  `id, ruta, nombre, duracion, ancho, alto, fps_average, split`.
- **Particionar** el dataset en *splits* `0/1/2` mediante muestreo aleatorio
  **reproducible** (seed desde la config global).
- **Handler de validación** de esquema independiente: si el CSV no existe o su
  estructura no coincide con la esperada, se **regenera/sobrescribe** por completo.
- **Config:** añadir `working_dirs.metadata_csv` y una sección `seeds` al JSON de
  configuración; el código toma de ahí la ruta y la seed (nunca hardcodeadas).
- **Script de prueba manual** `testing/test_metadata.py` (estilo standalone, como
  los demás `test_*.py`).

### 3.2 Fuera de alcance

- **No** se modifica `extract_frames`, `get_video_fps` ni el pipeline: el refactor
  para que esas etapas lean del CSV queda **pospuesto** (tarea SDD futura, solo si
  aparece necesidad real). Justificación en §6.
- **No** se añade columna de conteo exacto de frames (`n_frames`): no se necesita
  mientras el pipeline no consuma el CSV.
- **No** se paraleliza la extracción (secuencial; 123 videos de solo-metadatos es
  rápido).
- El **cómo técnico** (API exacta de decord/pandas, firmas/tipos, manejo de
  errores, formato concreto del handler): corresponde al `plan.md`.

---

## 4. Comportamiento esperado

### 4.1 Archivo `db_metadata.csv`

- **Ubicación:** ruta relativa tomada de `working_dirs.metadata_csv`
  (`assets/db_metadata.csv`).
- **Columnas (orden fijo):**
  - `id` — entero secuencial `0..N-1`, asignado tras ordenar las rutas de forma
    determinista (alfabético).
  - `ruta` — cadena, ruta **relativa a `PROJECT_ROOT`** (separadores POSIX).
  - `nombre` — cadena, nombre del archivo con extensión.
  - `duracion` — `float`, segundos (`len(reader) / fps_average`).
  - `ancho` — entero (resolución horizontal).
  - `alto` — entero (resolución vertical).
  - `fps_average` — `float` (fps promedio según el contenedor).
  - `split` — entero en `{0, 1, 2}`.

### 4.2 Splits

- `1` = **fine-tuning** → **23** videos (~20 %).
- `2` = **testing** → **20** videos (~20 %).
- `0` = **reserva** → **resto** (~80 videos, ~60 %).
- Muestreo **sin reemplazo y disjunto** (cada video en exactamente un split).
- **Reproducible:** misma seed + mismo conjunto de videos ⇒ misma partición.

### 4.3 Generación / validación (handler)

- Función pública orquestadora (descubrir → extraer → asignar splits → validar →
  escribir), **idempotente**: si el CSV ya existe y es válido, no reescribe salvo
  que se fuerce explícitamente.
- **Handler de validación** independiente: comprueba existencia y conjunto/orden de
  columnas esperadas; ante archivo ausente o esquema incorrecto, **regenera y
  sobrescribe** por completo. Se diseña previendo que el esquema es **mutable** a
  futuro.

### 4.4 Pipeline existente

- Sin cambios: `extract_frames` y el pipeline mantienen firma y comportamiento.

---

## 5. Criterios de aceptación

1. **AC-1 — Módulo presente:** existe `src/data/metadata.py` (con
   `src/data/__init__.py`), importable como paquete editable.
2. **AC-2 — CSV generado:** al ejecutar la función pública se crea
   `assets/db_metadata.csv` con las 8 columnas en el orden definido en §4.1.
3. **AC-3 — Una fila por video:** hay exactamente una fila por cada `.MOV`
   descubierto bajo `dataset_dir` (recursivo), con `id` secuencial `0..N-1`.
4. **AC-4 — Metadatos correctos:** `ancho`, `alto`, `fps_average` y `duracion`
   corresponden a lo que reporta el video real.
5. **AC-5 — Ruta relativa:** `ruta` es relativa a `PROJECT_ROOT` y `get_abs_path`
   la resuelve a un archivo existente.
6. **AC-6 — Splits correctos:** los conteos son 23 (split 1), 20 (split 2) y el
   resto (split 0); los splits son disjuntos y cubren todos los videos.
7. **AC-7 — Reproducibilidad:** con la misma seed (de la config) la partición es
   idéntica entre corridas.
8. **AC-8 — Config:** la ruta del CSV y la seed se leen de la config global
   (`working_dirs.metadata_csv` y `seeds`), sin valores hardcodeados.
9. **AC-9 — Handler:** ante CSV ausente o con esquema incorrecto, se regenera/
   sobrescribe; ante CSV válido, no se reescribe innecesariamente.
10. **AC-10 — Pipeline intacto:** `extract_frames` y el pipeline no cambian.
11. **AC-11 — Validación:** se valida **en local** sobre los videos reales de
    `data/raw` mediante `testing/test_metadata.py` (no usa modelo ni GPU).

---

## 6. Supuestos y notas

- **Refactor pospuesto (decisión explícita):** el objetivo del draft ("evitar
  lecturas directas de video en etapas posteriores") se descarta como motivación
  principal porque (a) el ahorro es marginal —`decord` lee metadatos sin
  decodificar—; (b) `extract_frames` acepta rutas absolutas **fuera** del proyecto,
  que el CSV no cubre, exigiendo un *fallback*; y (c) el modo cuota necesita el
  conteo **exacto** de frames, que no está en el CSV (`duracion × fps` es
  aproximado). El CSV se concibe como **manifiesto + splits**, no como caché de
  metadatos del pipeline.
- **Ubicación `src/data/`:** generar el manifiesto es **preparación de dataset**,
  no lógica de inferencia; por eso vive fuera de `src/core/` (que agrupa
  detección/segmentación/tracking sobre frames).
- **Seeds en config:** coherente con la convención de centralizar todo
  path/parámetro en la config global; queda versionado y accesible desde código.
- **Versionado:** `assets/db_metadata.csv` es un manifiesto ligero y **se versiona
  en git** (no es dato pesado).
- **Dependencias:** reutiliza `decord` (ya usado por `frame_extraction`) y `pandas`
  (ya en `requirements.txt`). No introduce dependencias nuevas.
- Esta especificación **no** define el *cómo* técnico (API exacta de decord/pandas,
  firmas/tipos, formato del handler, manejo de errores ni el detalle del test);
  todo ello corresponde al `plan.md`.
