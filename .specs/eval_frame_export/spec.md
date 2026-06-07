# Spec — Exportar y congelar el set de frames de evaluación (`eval_frame_export`)

- **Tarea atómica:** `eval_frame_export`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.
- **Proceso al que pertenece:** evaluación del pipeline SAM3-only (ver
  `.specs/drafts/evaluation_sam3_only_roadmap.md`, tarea 1). Es el **cimiento** del
  proceso de evaluación.

---

## 1. Requisito (historia de usuario)

> **Como** persona que evalúa el pipeline de análisis de fútbol robótico,
> **quiero** congelar un conjunto **reproducible** de frames extraídos de los 20
> videos del split *testing*, persistidos como imágenes en disco y descritos por un
> **CSV de control**,
> **para** poder (a) anotar ese mismo conjunto como ground-truth, (b) correr el
> pipeline sobre **exactamente los mismos frames**, y (c) que el equipo identifique
> con facilidad de qué video y posición proviene cada frame.

---

## 2. Motivación (por qué)

- La evaluación exige **alineación GT↔predicción**: el frame que se anota y el frame
  que se le da al modelo deben ser **el mismo**. Congelar el conjunto a disco, con
  un identificador estable, es lo que garantiza esa correspondencia.
- El pipeline actual **no persiste frames** (`show_frames` es display-only,
  `write_video` escribe mp4). Hace falta una utilidad explícita que los guarde como
  imágenes individuales para subirlas a la herramienta de anotación (Roboflow).
- Un **CSV de control** hace doble servicio: permite recuperar los frames desde
  código (iterar el set, emparejar GT y predicciones) y da **trazabilidad** humana
  (de qué video y qué posición salió cada imagen), incluido el grupo
  `aleatorio`/`cenital` para el reporte separado.
- El valor a esta altura es **congelar el conjunto + su manifiesto**; no se anota ni
  se predice nada aquí (eso es de las tareas 2 y 4).

---

## 3. Alcance

### 3.1 Dentro de alcance

- Nueva utilidad bajo **`src/data/`** (submódulo de preparación de dataset) que:
  - **Lee** los videos del split *testing* desde `db_metadata.csv` (`split==2`).
  - **Extrae** de cada video los frames de **cuota** reusando
    `src/core/frame_extraction.py::extract_frames(all_frames=False)`.
  - **Persiste** cada frame como una **imagen individual** bajo
    `data/testing_frames/` (carpeta nueva, separada de `data/raw`).
  - **Genera** un **CSV de control** que lista cada frame exportado con su
    procedencia (incluido el **índice del frame original** en el video) y su grupo.
- **Helper aditivo** en `src/core/frame_extraction.py` que **expone los índices de
  frame que el muestreo de cuota selecciona** (hoy `extract_frames` los calcula
  internamente pero no los devuelve). `extract_frames` reusa ese helper; su
  **comportamiento y firma no cambian**. Es lo que permite registrar el frame
  original sin reimplementar/duplicar la lógica de muestreo.
- **Config:** añadir `working_dirs.testing_frames_dir` (= `data/testing_frames`,
  carpeta de imágenes git-ignored) y `working_dirs.testing_frames_csv`
  (= `assets/...csv`, manifiesto **versionado**) al JSON de configuración; el código
  toma ambas rutas de ahí (nunca hardcodeadas). La cuota se sigue leyendo de
  `preprocess.frame_quota`.
- **Idempotencia:** si el set ya existe y es válido, no se regenera salvo que se
  fuerce explícitamente (mismo patrón que `build_metadata_csv`).
- **Script de prueba manual** `testing/test_eval_frame_export.py` (estilo
  standalone, como los demás `test_*.py`).

### 3.2 Fuera de alcance

- **No** anota nada ni sube nada a Roboflow (tarea 2).
- **No** corre el modelo SAM3 ni genera predicciones (tarea 4); no usa GPU.
- **No** modifica el **comportamiento ni la firma** de `extract_frames`,
  `get_video_fps`, el pipeline ni `db_metadata.csv`/su generación. (Sí se **añade**
  un helper aditivo en `frame_extraction.py` que `extract_frames` reusa, sin alterar
  su salida; ver §3.1.)
- **No** cubre frames de tracking (clips densos); solo la muestra dispersa de
  evaluación de segmentación.
- El **cómo técnico** (formato/nombre exacto de las imágenes, API de escritura,
  firmas/tipos, manejo de errores, detalle del test) corresponde al `plan.md`.

---

## 4. Comportamiento esperado

### 4.1 Carpeta de frames

- **Ubicación:** ruta relativa tomada de `working_dirs.testing_frames_dir`
  (`data/testing_frames`), resuelta con `get_abs_path`.
- Contiene una **imagen por frame** exportado de los 20 videos de testing.
- Es **dato pesado → git-ignored** (igual que `data/raw`); se regenera de forma
  determinista, no se versiona.

### 4.2 CSV de control

- **Versionado en git** (manifiesto ligero, como `db_metadata.csv`): vive en
  `assets/` según `working_dirs.testing_frames_csv`, **separado** de las imágenes
  git-ignored. Así el equipo tiene la procedencia de cada frame sin necesidad de
  regenerar las imágenes.
- **Columnas (orden a fijar en el plan, propuesta):**
  - `id` — entero secuencial `0..M-1` del frame dentro del set de evaluación.
  - `video_id` — `id` del video en `db_metadata.csv` (enlace al manifiesto).
  - `video_ruta` — ruta del video de origen relativa a `PROJECT_ROOT` (trazabilidad).
  - `frame_index` — índice **posicional** del frame dentro del muestreo (0..n-1);
    es la clave que alinea GT y predicción.
  - `frame_original` — índice del frame **en el video fuente** (el que el muestreo
    de cuota seleccionó); para trazabilidad exacta hacia el video original.
  - `imagen` — ruta de la imagen exportada relativa a `PROJECT_ROOT`.
  - `grupo` — `aleatorio` o `cenital`.

### 4.3 Identidad y grupo

- Cada frame se identifica de forma estable por **`(video_id, frame_index)`**, lo
  que permite que las tareas 3 (GT) y 4 (predicción) emparejen por la misma clave.
- El **grupo** se deriva de la configuración: los videos en
  `splits.forced_testing` (los 2 de **cámara superior / cenital**) son `cenital`;
  los 18 restantes del split testing son `aleatorio`.

### 4.4 Reproducibilidad e idempotencia

- Misma ejecución (mismos videos + misma cuota) ⇒ **mismos frames y mismo CSV**.
- Si el set ya existe y es válido, **no** se regenera salvo `force` explícito; en
  cualquier otro caso (ausente, inválido, `force=True`) se regenera por completo.

### 4.5 Caso borde

- Si un video tuviera **menos frames que la cuota**, se exportan los que haya, sin
  fallar; el CSV refleja el conteo real de ese video.

---

## 5. Criterios de aceptación

1. **AC-1 — Utilidad presente:** existe el módulo bajo `src/data/` que orquesta la
   exportación, importable como paquete editable.
2. **AC-2 — Origen correcto:** procesa exactamente los videos con `split==2` de
   `db_metadata.csv` (los 20 de testing), ni más ni menos.
3. **AC-3 — Frames en `data/testing_frames/`:** las imágenes se escriben bajo la
   ruta de `working_dirs.testing_frames_dir`, **fuera** de `data/raw` y de `assets/`.
4. **AC-4 — Cuota desde config:** el número de frames/video sale de
   `preprocess.frame_quota`; no hay valores hardcodeados.
5. **AC-5 — CSV de control:** se genera el CSV (versionado, en `assets/` según
   `working_dirs.testing_frames_csv`) con una fila por frame exportado y las
   columnas definidas en §4.2, incluido `grupo`.
6. **AC-6 — Identidad estable:** cada fila tiene `(video_id, frame_index)` único y
   coherente; `video_id` corresponde al `id` de `db_metadata.csv`.
6b. **AC-6b — Frame original:** cada fila registra `frame_original` = índice real
   del frame en el video fuente, consistente con el muestreo de cuota; obtenido del
   helper de `frame_extraction` (no por lógica duplicada).
7. **AC-7 — Grupos correctos:** los frames de los 2 videos de `forced_testing` son
   `cenital`; los de los otros 18 son `aleatorio`.
8. **AC-8 — Rutas relativas:** `video_ruta` e `imagen` son relativas a
   `PROJECT_ROOT` y resuelven a archivos existentes vía `get_abs_path`.
9. **AC-9 — Reproducibilidad:** dos corridas sobre el mismo dataset producen el
   mismo conjunto de imágenes y el mismo CSV.
10. **AC-10 — Idempotencia:** ante set válido no se reescribe; ante ausente/inválido
    o `force`, se regenera por completo.
11. **AC-11 — Caso borde:** un video con menos frames que la cuota no rompe la
    ejecución y queda reflejado en el CSV.
12. **AC-12 — Sin efectos colaterales:** el **comportamiento y la firma** de
    `extract_frames`, el pipeline y `db_metadata.csv` quedan intactos (solo se
    **añade** un helper en `frame_extraction.py` que `extract_frames` reusa); no se
    usa GPU ni el modelo. El helper se cubre con su propia verificación en el test.
13. **AC-13 — Validación local:** `testing/test_eval_frame_export.py` ejercita la
    exportación sobre los videos reales de `data/raw` (split testing).

---

## 6. Supuestos y notas

- **Ubicación `data/testing_frames/` (decisión del responsable):** los frames son
  datos derivados pesados; van en `data/` (no en `assets/`, reservado a manifiestos
  ligeros versionados como `db_metadata.csv`) y en carpeta separada de `raw` para no
  mezclar fuente con derivados.
- **Origen desde el CSV, no `rglob`:** la lista de videos de testing se toma de
  `db_metadata.csv` (`split==2`), no se redescubre, para que el set de evaluación
  quede atado al split reproducible ya fijado.
- **Dos índices por frame:** `frame_index` (posicional, 0..n-1) alinea GT y
  predicción al re-extraer; `frame_original` (índice en el video fuente) da
  trazabilidad exacta hacia el material original. Registrar el segundo exige exponer
  los índices que `extract_frames` muestrea — de ahí el helper aditivo en
  `frame_extraction.py` (§3.1), que evita duplicar la lógica de muestreo.
- **CSV de control versionado (decisión del responsable):** a diferencia de las
  imágenes (dato pesado git-ignored), el CSV es un manifiesto ligero y **se versiona
  en git** en `assets/` (mismo criterio que `db_metadata.csv`), para dar al equipo
  la trazabilidad de los frames sin tener que regenerarlos. Como las imágenes se
  regeneran de forma determinista, el CSV versionado permanece válido.
- **Reúso, no reescritura:** se apoya en `extract_frames` (cuota, determinista) y en
  el patrón idempotente de `build_metadata_csv`; no introduce dependencias nuevas.
- **Sin GPU ni modelo:** solo lectura de video + escritura de imágenes; corre igual
  en local y en contenedor.
- Esta especificación **no** define el *cómo* técnico (formato/nombre de imágenes,
  API de escritura, firmas, manejo de errores, detalle del test); eso es del
  `plan.md`.
