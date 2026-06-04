# Spec — Escritor de video (`video_writer`)

- **Tarea atómica:** `video_writer`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** una función que escriba una secuencia de frames a un archivo de video
> mp4 en disco, con un fps configurable y creando la carpeta de salida si no
> existe,
> **para** poder emitir el entregable del MVP ("video original → proceso → video
> anotado") a partir de los frames compuestos por el overlay.

---

## 2. Motivación (por qué)

- El MVP por-frame produce frames anotados **en memoria** (overlay). Falta la
  pieza que los **persista como video** para tener un entregable reproducible.
- Es la **primera** pieza del pipeline que escribe a disco; conviene aislarla como
  utilidad simple ("tonta") de I/O de video, reutilizable por el `pipeline_runner`.
- La constitución (§5.5) exige un **directorio de outputs configurable** declarado
  en la configuración; esta tarea lo introduce.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Definir en un nuevo módulo `src/core/video_writer.py` una función
  **`write_video(frames, output_path, fps=None)`** que escribe los frames a un
  **mp4**.
- **Crear el directorio de salida** si no existe antes de escribir.
- **fps** con default desde la configuración y override por parámetro.
- Añadir a la configuración la **ruta de outputs** (`working_dirs.outputs_dir`) y
  el **fps de salida por defecto** (`visualization.output_fps`).
- Exportar la función desde `src/core/__init__.py`.
- Un script de validación.

### 3.2 Fuera de alcance

- **Composición del overlay / segmentación / tracking** (otras tareas): el escritor
  recibe los frames ya listos.
- **Orquestación del pipeline** y la **elección del nombre/ruta** concreta del
  archivo de salida (lo decide `pipeline_runner`).
- **Audio**, subtítulos, u otros formatos de contenedor/códec más allá del mp4.
- El **cómo técnico** (librería y API de escritura, códec/contenedor concretos,
  resolución de la ruta de outputs, manejo de dimensiones impares): corresponde al
  `plan.md`.

---

## 4. Comportamiento esperado

- **Entrada:** `frames` como `np.ndarray (N, H, W, 3) uint8` RGB (p. ej. los frames
  compuestos por `overlay_detections`, o los crudos de `extract_frames`), y una
  **ruta de salida** para el mp4.
- **Escritura:** genera un **archivo mp4 reproducible** en la ruta indicada. Si la
  carpeta destino no existe, **la crea**.
- **fps:** si no se indica, se toma el **default de la configuración**
  (`visualization.output_fps`); si se pasa por parámetro, ese **prevalece** (el
  pipeline usará el fps real de la fuente en modo "video completo").
- **Retorno:** la **ruta** del archivo escrito.
- **Validación de entrada:** si `frames` no es `(N, H, W, 3) uint8` o está vacío,
  **falla con un error claro** (no escribe un mp4 vacío/ inválido).
- **Persistencia:** es la **única** pieza del MVP que escribe a disco; no muestra
  nada por pantalla.

---

## 5. Criterios de aceptación

1. **AC-1 — Módulo y función:** existe `src/core/video_writer.py` con
   `write_video`, exportada desde `src/core/__init__.py`.
2. **AC-2 — Escribe mp4:** dado un array de frames válido, `write_video` crea un
   archivo **mp4 reproducible** en la ruta indicada.
3. **AC-3 — Crea el directorio:** si la carpeta de la ruta de salida no existe, se
   crea automáticamente.
4. **AC-4 — fps configurable:** el fps por defecto se lee de
   `visualization.output_fps`; un `fps` por parámetro lo sobreescribe.
5. **AC-5 — Outputs en config:** la configuración declara `working_dirs.outputs_dir`
   (conforme a la constitución §5.5).
6. **AC-6 — Retorno:** la función devuelve la ruta del archivo escrito.
7. **AC-7 — Entrada inválida:** frames con forma incorrecta o vacíos producen un
   error claro, sin generar un archivo inválido.
8. **AC-8 — Validación:** un script (ejecutable en local, sin GPU ni modelo)
   demuestra, con **frames sintéticos**, que se genera un mp4 legible y que la
   carpeta de salida se crea si falta.

---

## 6. Supuestos y notas

- **Dependencias:** no depende funcionalmente de otras tareas (recibe los frames);
  **desbloquea** `pipeline_runner` (6).
- **Librería:** se usará `imageio` (con backend ffmpeg vía `imageio-ffmpeg`, ambos
  en `requirements.txt`), por ser **RGB-nativo** (los frames del proyecto son RGB)
  y de API simple; el detalle es del `plan.md`.
- **Resolución de la ruta de outputs:** `get_abs_path` exige que la ruta exista,
  por lo que **no** sirve para `outputs/` (que puede no existir aún); el escritor
  crea el directorio en su lugar. El detalle de cómo se compone la ruta queda para
  el `plan.md` y el `pipeline_runner`.
- **fps según el modo (contexto):** en modo **cuota** (testeo) los frames son
  muestreados → slideshow con el fps por defecto (4). En modo **video completo**
  (uso real) los frames son contiguos → el pipeline pasa el **fps real de la
  fuente** como override, para reproducción natural.
- **Validación local:** el escritor no usa modelo ni GPU; la validación con frames
  sintéticos corre en local (el agente la ejecuta).
- Esta especificación **no** define el *cómo* técnico (API exacta de `imageio`,
  códec/`pix_fmt`, manejo de dimensiones impares, firma/tipos exactos, ni la
  estructura de las nuevas claves de config); todo ello corresponde al `plan.md`.
