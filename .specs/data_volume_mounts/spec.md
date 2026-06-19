# Spec — Montaje correcto de los datos pesados en host y contenedor

- **Tarea atómica:** `data_volume_mounts`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla o reproduce el proyecto,
> **quiero** que las rutas `data/raw` (vídeos) y `assets/sam3` (modelo SAM3)
> resuelvan a los datos reales **tanto al ejecutar en local (venv) como dentro
> del contenedor**,
> **para** trabajar indistintamente en ambos entornos sin symlinks rotos ni
> pasos manuales ocultos, y sin versionar datos pesados.

---

## 2. Motivación (por qué)

- **Bug actual:** el contenedor crea, en su `command` de arranque, los symlinks
  `data/raw → /Meta_Glasses` y `assets/sam3 → /sam3`. Como el repositorio se
  bind-monta completo dentro del contenedor, ese `ln` **reescribe el archivo del
  host**: tras levantar el contenedor, en el host `data/raw` queda como symlink a
  `/Meta_Glasses` (inexistente en local) → **symlink roto en el host**.
- **Causa de fondo:** un symlink que vive dentro del área bind-montada es **el
  mismo inodo** visto desde host y contenedor; no puede apuntar a dos destinos
  distintos según el entorno. Forzarlo provoca el conflicto.
- Se busca un modelo **robusto y reproducible**: que cada entorno haga resolver
  la ruta del proyecto a los datos reales por su cuenta, sin pisarse entre sí.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Corregir `docker/docker-compose.yml` para que los datos pesados se monten
  **directamente** sobre las rutas del proyecto, eliminando los symlinks de
  arranque y los volúmenes intermedios `/Meta_Glasses` y `/sam3`.
- Ajustar el `command` del servicio para que solo mantenga vivo el contenedor.
- Ajustar la semántica y los comentarios de `.env` (`HOST_DATA_DIR`,
  `HOST_SAM3_DIR`) para que sean coherentes con el nuevo montaje.
- Actualizar la documentación afectada (`CLAUDE.md` y, si aplica, los `README.md`
  de `data/` y `assets/`) para reflejar el nuevo modelo y eliminar las notas
  obsoletas sobre "dead symlinks en el host".

### 3.2 Fuera de alcance

- Lógica de los pipelines (detección, segmentación, tracking).
- Cambios en `src/` (incluido `get_abs_path`, que se mantiene tal cual).
- Cambios en los archivos de configuración `configs/*.json`.
- Cambios en el `Dockerfile` (salvo que el `plan.md` detecte algo imprescindible).
- Descarga u organización de los vídeos/modelo (responsabilidad de quien
  reproduce el proyecto).

---

## 4. Modelo de solución (qué debe lograrse)

### 4.1 Contrato de rutas (invariante)

El código accede a los datos **solo** mediante rutas relativas del config
(`data/raw`, `assets/sam3`), resueltas por `get_abs_path` contra `PROJECT_ROOT`.
Por tanto el código **siempre** lee físicamente `<repo>/data/raw` y
`<repo>/assets/sam3`. Hacer que esas rutas resuelvan a los datos reales es
responsabilidad del **setup de cada entorno**, no del código.

### 4.2 Comportamiento en el contenedor

- Los datos del host se montan **directamente** sobre la ruta del proyecto:
  - `${HOST_DATA_DIR}` → `/${CONTAINER_WORKSPACE_DIR}/data/raw`
  - `${HOST_SAM3_DIR}` → `/${CONTAINER_WORKSPACE_DIR}/assets/sam3`
- Estos montajes se aplican **encima** del bind-mount del repositorio,
  **sombreando** lo que haya en esas rutas en el host, sin modificar el
  filesystem del host.
- Ya **no** se crean symlinks en el arranque ni existen los volúmenes
  intermedios `/Meta_Glasses` ni `/sam3`.

### 4.3 Comportamiento en el host (ejecución local en venv)

El contrato es: **`<repo>/data/raw` y `<repo>/assets/sam3` deben *resolver* a los
datos reales.** Se admiten dos formas equivalentes, a elección del usuario:

- **Caso A — datos en el repo:** `data/raw` y `assets/sam3` son **directorios
  reales** con los datos dentro.
- **Caso B — datos en otra ubicación (p. ej. otro disco):** `data/raw` y/o
  `assets/sam3` son **symlinks** a la ruta externa real
  (p. ej. `data/raw → /mnt/otrodisco/videos`).

En ambos casos `get_abs_path` resuelve correctamente en local. El symlink del
Caso B es exclusivo del host, queda **ignorado por Git** y **no entra en
conflicto** con el contenedor, porque el montaje de volumen del contenedor
sombrea/resuelve esa ruta de forma independiente.

### 4.4 Coherencia de rutas host ↔ contenedor

`HOST_DATA_DIR` debe apuntar al directorio cuyo **contenido** es exactamente lo
que debe verse bajo `data/raw` (ídem `HOST_SAM3_DIR` ↔ `assets/sam3`), de modo
que la **estructura de rutas relativas sea idéntica** en host y contenedor
(p. ej. si en host se accede a `data/raw/Meta_Glasses/17Abril/...`, dentro del
contenedor debe verse la misma ruta). Esto corrige el desfase actual, en el que
`HOST_DATA_DIR` apunta a `.../data/raw/Meta_Glasses` y deja un nivel de anidación
distinto entre ambos entornos.

---

## 5. Criterios de aceptación

1. **AC-1 — Sin symlinks de arranque:** el `command` del contenedor ya no ejecuta
   `ln -sfn` ni `mkdir` para `data/raw`/`assets/sam3`; solo mantiene vivo el
   contenedor.
2. **AC-2 — Montaje directo:** `docker/docker-compose.yml` monta `${HOST_DATA_DIR}`
   en `/${CONTAINER_WORKSPACE_DIR}/data/raw` y `${HOST_SAM3_DIR}` en
   `/${CONTAINER_WORKSPACE_DIR}/assets/sam3`; los volúmenes `/Meta_Glasses` y
   `/sam3` ya no existen.
3. **AC-3 — Host intacto:** levantar y operar el contenedor **no** convierte
   `data/raw` ni `assets/sam3` del host en symlinks rotos.
4. **AC-4 — Local funcional (Caso A):** con los datos como directorios reales en
   el repo, `get_abs_path("data/raw")` y `get_abs_path("assets/sam3")` resuelven
   en local sin error.
5. **AC-5 — Local funcional (Caso B):** con `data/raw`/`assets/sam3` como symlinks
   a una ruta externa existente, `get_abs_path(...)` también resuelve en local sin
   error.
6. **AC-6 — Coherencia de rutas:** una misma ruta relativa (p. ej.
   `data/raw/Meta_Glasses/<fecha>/...`) apunta a los mismos datos en host y en
   contenedor.
7. **AC-7 — Contenedor funcional:** dentro del contenedor, las rutas `data/raw` y
   `assets/sam3` muestran los datos del volumen montado.
8. **AC-8 — Sin datos versionados:** `data/raw`, `assets/sam3` y sus contenidos
   pesados siguen excluidos por `.gitignore`; el repo solo conserva la estructura
   (p. ej. vía `.gitkeep`/`README.md`).
9. **AC-9 — Documentación al día:** `CLAUDE.md` (y los README afectados) describen
   el nuevo modelo de montaje directo y ya no mencionan los symlinks de arranque
   ni la trampa de "dead symlinks en el host" como comportamiento vigente.
10. **AC-10 — Reproducibilidad:** otra persona solo completa `HOST_DATA_DIR` y
    `HOST_SAM3_DIR` en su `.env`, coloca sus datos (Caso A o B) y levanta el
    contenedor, sin pasos manuales ocultos.

---

## 6. Supuestos y notas

- La causa raíz es el `ln` de arranque sobre un área bind-montada (sección 2).
- El config (`configs/*.json`) y `get_abs_path` **no se modifican**; el cambio es
  de infraestructura/entorno, no de código de `src/`.
- Con el montaje directo, en el host `data/raw` deja de ser un symlink roto, por
  lo que `testing/test_frame_extraction.py` podrá ejecutarse también en local si
  el usuario tiene los vídeos (Caso A o B); deja de ser exclusivo del contenedor.
- Para el Caso B, si el usuario además levanta el contenedor en la misma copia de
  trabajo, `HOST_DATA_DIR`/`HOST_SAM3_DIR` deben apuntar al **mismo destino** que
  el symlink local, para que la ruta resuelva a los mismos datos en ambos lados.
- Esta especificación **no** define el *cómo* técnico (sintaxis exacta del compose,
  textos finales de `.env`/docs); eso corresponde al `plan.md` de esta carpeta.

---

## 7. Siguientes pasos (metodología)

1. Elaborar `plan.md` con el detalle técnico de implementación.
2. Derivar `tasks.md` con las tareas ejecutables.
3. Implementar (paso 5) únicamente después de aprobar los anteriores.

---

## 8. Adenda / Revisión (2026-06-03) — Modelo "archivos reales"

> Esta adenda **revisa** la solución de esta tarea sin reabrir el ciclo SDD
> completo (es un refinamiento de infraestructura sobre una tarea ya
> implementada). La motivación, el contrato de rutas (§4.1) y los Casos A/B del
> spec siguen siendo válidos; lo que cambia es el **mecanismo de montaje**.

**Por qué:** los **montajes directos** de `${HOST_DATA_DIR}`/`${HOST_SAM3_DIR}`
sobre `data/raw`/`assets/sam3` funcionan, pero arrastran un problema cuando el
host usa **symlinks** (Caso B): el bind-mount del repo lleva el symlink al
contenedor y Docker **resuelve el target del montaje a través de él**, dejando
`data/raw` como symlink (no como dir real) y haciendo que `get_abs_path` resuelva
**fuera de `PROJECT_ROOT`**. El symlink es, de hecho, el único mecanismo de
"environment setup" que rompe Docker.

**Decisión:** adoptar el modelo **"los datos viven como ARCHIVOS REALES en el
repo"**:

- `data/raw` y `assets/sam3` son **siempre directorios reales** (nunca symlinks),
  poblados con los datos por el mecanismo nativo de cada entorno (local: mover/
  colocar/`mount --bind`; RunPod: network volume; futuro: script `bootstrap_data`).
- El **bind-mount del workspace** (`../:/<workspace>`) lleva esos archivos al
  contenedor. **Se eliminan** los volúmenes `${HOST_DATA_DIR}`/`${HOST_SAM3_DIR}`
  del `docker-compose.yml` y las variables correspondientes del `.env`.
- El código y `configs/*.json` no cambian; la convención de rutas relativas se
  mantiene. Ahora `get_abs_path("data/raw")` devuelve `<repo>/data/raw` (dir real
  dentro del proyecto) en host y contenedor.

**Impacto en los criterios de aceptación previos:**
- **Quedan obsoletos** los que dependían del montaje directo de volúmenes de
  datos: AC-2, AC-7 (se cumple ahora vía el bind del workspace) y la regla del
  nivel `Meta_Glasses` de AC-6 (los datos cuelgan directo de `data/raw`).
- **T5** de `tasks.md` (corregir el nivel `Meta_Glasses` en `HOST_DATA_DIR`)
  queda **superado**: ya no existe `HOST_DATA_DIR`.
- Siguen vigentes: AC-3 (host intacto), AC-4/AC-5 (resolución local), AC-8
  (datos no versionados), AC-10 (reproducibilidad — ahora "coloca tus datos
  reales bajo `data/raw`/`assets/sam3`").

**Pendiente relacionado:** el script `bootstrap_data` (descarga idempotente) que
automatiza poblar esos dirs reales (ver la futura tarea `bootstrap_data` y el
TODO en `CLAUDE.md`). La estrategia de RunPod sigue por definir.
