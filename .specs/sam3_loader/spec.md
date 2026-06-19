# Spec — Carga del modelo SAM3 (`sam3_loader`)

- **Tarea atómica:** `sam3_loader`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** una forma única y reutilizable de **cargar el modelo SAM3**
> (processor + model) lista para inferir, resolviendo sola la ruta y el
> dispositivo,
> **para** dejar de copiar y pegar el bloque de carga en cada notebook y poder
> obtener un modelo listo desde una sola llamada, igual desde un notebook que
> desde el pipeline.

---

## 2. Motivación (por qué)

- Hoy el bloque de carga de SAM3 está **copy-pasteado** en los notebooks de
  `fase_0/` (01–05), cada uno con su propia variante: ruta del modelo armada a
  mano y **selección de dispositivo inconsistente** (cuda fijo en 01/02/03, auto
  en 04, cpu fijo en 05). No hay una fuente de verdad.
- Esa inconsistencia tiene un riesgo real: la carga del modelo (`.to(device)`) y
  la sesión de inferencia (`inference_device`) piden el dispositivo por separado;
  si se desincronizan, se obtiene un error o una ejecución silenciosa en el
  dispositivo equivocado.
- Centralizar la carga en `src/core/` es el **cimiento** del MVP SAM3-only: todas
  las tareas posteriores (segmentación por texto, tracking, pipeline) consumen el
  modelo cargado. Conviene que lo obtengan de una sola pieza, bien definida y
  reutilizable, alineada con las convenciones del repo (rutas por configuración,
  sin paths absolutos ni symlinks).

---

## 3. Alcance

### 3.1 Dentro de alcance

- Definir una **función de carga** en `src/core/` que entregue SAM3 (processor +
  model) **listo para inferir**.
- **Resolver sola la ruta** del modelo: leer la ruta relativa desde el archivo de
  configuración activo (clave `working_dirs.sam3_dir`) y convertirla a ruta
  absoluta mediante `get_abs_path()`. Nunca rutas hardcoded ni symlinks.
- **Resolver solo el dispositivo**: por defecto usar **GPU si está disponible**,
  si no **CPU**, y exponer el dispositivo elegido junto con el modelo para que
  carga e inferencia usen **la misma** fuente.
- **Cachear el modelo (singleton) con opción de desactivarlo** (`use_cache`): por
  defecto carga una sola vez y reutiliza; con la opción desactivada entrega una
  instancia fresca.
- **Devolver un objeto único** que agrupe processor, model y dispositivo.
- **Fallar de forma clara y temprana** si la ruta del modelo no existe o el modelo
  no puede cargarse.

### 3.2 Fuera de alcance

- **Descargar / aprovisionar los pesos del modelo.** Esta tarea **asume que los
  pesos ya están en disco** (en la ruta que indica la configuración). Automatizar
  su descarga cuando falten es responsabilidad de la futura tarea `bootstrap_data`
  y queda **pendiente / fuera de alcance** aquí.
- La definición del **cómo técnico** (nombres exactos de función/clase, firma y
  tipos concretos, librería y API de carga, dtype, mecanismo de caché concreto,
  forma de exponer el override de dispositivo): corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- **Entrada:** la función **no recibe la ruta del modelo como dato obligatorio**;
  la resuelve sola desde la configuración activa (seleccionada por
  `CONFIG_FILENAME` en `.env`). Acepta, de forma opcional, controles de
  comportamiento: desactivar la caché y forzar un dispositivo concreto.
- **Salida:** un **objeto único** que agrupa el `processor`, el `model` y el
  **dispositivo** efectivamente usado, de modo que quien lo consuma tenga todo lo
  necesario desde una sola llamada.
- **Caché (singleton con opt-out):**
  - Por defecto, la **primera** llamada carga el modelo y las **siguientes**
    reutilizan el ya cargado (sin volver a pagar el costo de carga).
  - Con la caché **desactivada**, la llamada entrega una **instancia fresca**
    (p. ej. para pruebas aisladas), sin afectar la instancia cacheada.
- **Selección de dispositivo:** por defecto **GPU si está disponible, si no CPU**;
  el dispositivo elegido queda reflejado en la salida. Debe poder **forzarse** un
  dispositivo concreto.
- **Caso de fallo:** si la ruta del modelo no existe o el modelo no puede
  cargarse, la función **avisa con un error explícito y temprano**, no de forma
  silenciosa.

---

## 5. Criterios de aceptación

1. **AC-1 — Carga centralizada:** existe una función en `src/core/` que carga
   SAM3 (processor + model) y la entrega lista para inferir.
2. **AC-2 — Ruta por configuración:** la ruta del modelo se obtiene de la clave
   `working_dirs.sam3_dir` del archivo de config activo y se resuelve a absoluta
   con `get_abs_path()`; no hay rutas hardcoded ni symlinks.
3. **AC-3 — Salida agrupada:** la función devuelve un **objeto único** que
   contiene processor, model y el dispositivo usado.
4. **AC-4 — Dispositivo auto:** sin forzar nada, la función usa **GPU si está
   disponible** y **CPU** en caso contrario, y refleja esa elección en la salida.
5. **AC-5 — Dispositivo forzable:** es posible **forzar** un dispositivo concreto
   y la salida lo refleja.
6. **AC-6 — Caché por defecto:** una segunda llamada (con caché activa) **reutiliza**
   el modelo ya cargado en lugar de recargarlo.
7. **AC-7 — Opt-out de caché:** con la caché desactivada, la función entrega una
   **instancia fresca** sin afectar la cacheada.
8. **AC-8 — Funciona desde cualquier cwd:** la carga funciona igual desde un
   notebook o un script, sin depender del directorio de trabajo ni de `sys.path`.
9. **AC-9 — Fallo claro:** ante una ruta de modelo inexistente o un modelo que no
   carga, la función **falla con un error explícito y temprano**.
10. **AC-10 — Validación manual:** se demuestra de forma **exploratoria** (script
    suelto o notebook) que la función carga el modelo, que una segunda llamada
    reutiliza la caché y que el opt-out fuerza una recarga.

---

## 6. Supuestos y notas

- Es el **cimiento** del MVP SAM3-only: **no depende** de ninguna otra tarea y
  puede desarrollarse en paralelo a `classes_config` (tarea 2). Las tareas de
  segmentación, tracking y pipeline consumirán esta carga.
- Esta tarea **solo carga** el modelo: no infiere, no segmenta, no escribe
  outputs.
- Se apoya en convenciones ya existentes del repo (`get_abs_path`, lectura de la
  config seleccionada por `CONFIG_FILENAME`); no introduce mecanismos nuevos de
  rutas.
- Recordatorio de entorno: los pesos de SAM3 son datos pesados, git-ignored, que
  deben estar presentes en disco (modelo de "archivos reales" del repo). Cualquier
  **validación manual** que cargue el modelo real debe ejecutarse donde los pesos
  estén disponibles (contenedor o pod con GPU). Su aprovisionamiento automático lo
  cubrirá `bootstrap_data`.
- Esta especificación **no** define el *cómo* técnico (nombre exacto de la función
  y de la dataclass de salida, firma y tipos, librería/API de carga, dtype,
  mecanismo concreto de caché ni la forma del override de dispositivo); todo ello
  corresponde al `plan.md` de esta misma carpeta.
