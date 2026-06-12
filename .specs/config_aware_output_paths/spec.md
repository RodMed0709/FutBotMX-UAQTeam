# Spec — Salidas con namespace por config (`config_aware_output_paths`)

- **Tarea atómica:** `config_aware_output_paths`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Rediseño del benchmark sin-GT hacia una composición
  real **detector/segmentador → tracker** (Fase 1: detectores; Fase 2: trackers 2×2).
- **Depende de:** `detector_in_segmentation` (tarea 1, completa), que hizo el detector
  inyectable en ambos modos.
- **Habilita:** los notebooks de benchmark por fases, que corren **varias configs
  sobre los mismos videos** y necesitan que sus salidas **no se pisen**.

---

## 1. Requisito (historia de usuario)

> **Como** persona que corre un benchmark de varias configuraciones sobre el mismo
> conjunto de videos,
> **quiero** que cada config escriba sus salidas (JSON y mp4) en una **subcarpeta
> propia**,
> **para** que las configs no se sobrescriban entre sí, poder inspeccionar/comparar
> los resultados después, y reanudar por config tras un crash.

---

## 2. Motivación (por qué)

- **Hoy las salidas colisionan por nombre.** `inference_paths(stem, outputs_dir)`
  produce `outputs/inference/<stem>/<stem>.{json,mp4}` — **solo** el stem del video,
  sin detector/tracker/config. Correr N configs sobre los mismos videos hace que
  **cada config sobrescriba el JSON de la anterior**.
- **El benchmark depende de un baile frágil.** Por la colisión, el driver actual
  corre "una config a la vez" y **lee el JSON antes de que la siguiente lo pise**. Un
  fallo a mitad pierde el resultado de la config en curso.
- **El `skip-done` queda inútil.** `batch.py` salta un video si su JSON existe, pero
  como el path no distingue config, "ya hecho" significa "hecho por *alguna* config".
  Por eso el benchmark corre con `overwrite=True` (reprocesa siempre) y no puede
  reanudar de forma fina.
- **El rediseño por fases lo exige.** Fase 1 (detectores) y Fase 2 (trackers 2×2)
  corren múltiples configs; necesitan salidas separadas y un `skip-done` por config
  para reanudar.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **`inference_paths` gana `namespace: str | None = None`.** Con valor:
  `outputs/inference/<namespace>/<stem>/<stem>.{json,mp4}` (el `<namespace>` va
  **antes** del `<stem>`, subcarpeta por config). Con `None`: ruta actual, sin
  cambios.
- **Etiqueta explícita del llamador.** El namespace es una etiqueta que se pasa (p.
  ej. el label de config `sam3_text+bytetrack`), **no** auto-derivada de
  `(mode, detector, tracker)`.
- **Propagar `run_label: str | None = None`** por la cadena: `run_pipeline`,
  `track_video`, `run_inference`, `run_batch` lo reciben y lo reenvían; llega a
  `inference_paths` como `namespace`.
- **`skip-done` por config:** `batch.py` calcula la ruta con el mismo `run_label`, de
  modo que un video hecho bajo una config no se salta bajo otra → reanudación fina.
- **Smoke test sin GPU** de la derivación de rutas (lógica pura) y del hilvanado del
  `run_label` en `run_batch`.
- **Actualizar `CLAUDE.md`** (output placement / descripción de `batch`).

### 3.2 Fuera de alcance

- **Auto-derivar el namespace** de `(mode, detector, tracker)`: se opta por etiqueta
  explícita (más simple, retrocompatible, sin el problema de `None` vs nombre
  resuelto).
- **Reestructurar los notebooks** a las dos fases (Fase 1 / Fase 2 2×2) y modificar el
  `01_run_benchmark.ipynb` existente: son entregables aparte. Esta tarea entrega solo
  la **capacidad** (código + test).
- **El esquema/contenido del JSON:** no cambia; solo cambia la **ubicación en disco**.
- **`output_path` (ruta completa explícita):** sigue funcionando igual y **tiene
  prioridad** sobre `run_label`.

---

## 4. Comportamiento esperado

### 4.1 `run_label=None` (default) — retrocompatible

Todo el flujo se comporta **idéntico a hoy**: las salidas caen en
`outputs/inference/<stem>/<stem>.{json,mp4}`. Ningún llamador existente cambia.

### 4.2 `run_label="<config>"` — la capacidad nueva

Las salidas caen en `outputs/inference/<config>/<stem>/<stem>.{json,mp4}`. El mp4
(cuando `render_video=True`) cae en la **misma** subcarpeta, junto al JSON.

### 4.3 Precedencia con `output_path`

Si se pasa `output_path` (ruta completa), **manda** y `run_label` se ignora (no se
combinan). `run_label` solo afecta la ruta **derivada por defecto**.

### 4.4 `skip-done` por config

Con `run_label` puesto y `overwrite=False`, `run_batch` salta solo los videos cuyo
JSON existe **bajo esa config** → reanudación por config tras un crash.

### 4.5 Lectura del benchmark

`aggregate_config` sigue leyendo desde los paths de `json` que devuelve el `summary`
de `run_batch` (no recalcula rutas), así que funciona sin importar el namespace.

---

## 5. Criterios de aceptación

1. `inference_paths(stem, outputs_dir, namespace="X")` →
   `outputs/inference/X/<stem>/<stem>.{json,mp4}`; sin `namespace`, ruta actual.
2. `run_pipeline`, `track_video`, `run_inference`, `run_batch` aceptan
   `run_label: str | None = None` y lo propagan hasta `inference_paths`.
3. Con `run_label=None`, las salidas son idénticas a las de hoy (retrocompatibilidad).
4. Con `run_label` puesto, JSON y mp4 caen en `outputs/inference/<run_label>/<stem>/`.
5. `output_path` explícito tiene prioridad sobre `run_label`.
6. El `skip-done` de `run_batch` opera **por config** (usa el mismo `run_label` para
   derivar la ruta de comprobación).
7. No se modifica el esquema/contenido del JSON.
8. Smoke test (sin GPU) cubre la derivación de rutas y el hilvanado de `run_label`.
9. `CLAUDE.md` refleja el namespace opcional por config.

---

## 6. Supuestos y notas

- Lista completa de supuestos acordada con el usuario (técnicos, funcionales y de
  proceso); **ninguno rechazado**. Decisiones fijadas: namespace **por subcarpeta**,
  **etiqueta explícita** (no auto-derivada), nombre de parámetro **`run_label`**.
- Verificación principal **sin GPU** (la derivación de rutas es lógica pura); la
  corrida real en el pod es opcional.
- Esta tarea es el **segundo eslabón** del rediseño del benchmark: con el detector ya
  desacoplado (tarea 1) y las salidas sin colisión (esta), los notebooks de Fase 1 y
  Fase 2 pueden correr múltiples configs de forma limpia y reanudable.
