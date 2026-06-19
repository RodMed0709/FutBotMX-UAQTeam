# Tasks — Montaje correcto de los datos pesados en host y contenedor

- **Tarea atómica:** `data_volume_mounts`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan** que la origina. Marcar `- [x]` al completar.

---

## Fase A — `docker-compose.yml`

- [x] **T1 — Montaje directo de los volúmenes de datos**
  - En `docker/docker-compose.yml`, reemplazar los volúmenes
    `${HOST_DATA_DIR}:/Meta_Glasses` y `${HOST_SAM3_DIR}:/sam3` por:
    - `${HOST_DATA_DIR}:/${CONTAINER_WORKSPACE_DIR:-futbot}/data/raw`
    - `${HOST_SAM3_DIR}:/${CONTAINER_WORKSPACE_DIR:-futbot}/assets/sam3`
  - Conservar el bind del repo `../:/${CONTAINER_WORKSPACE_DIR:-futbot}`.
  - **Verificación:** ya no aparecen `/Meta_Glasses` ni `/sam3`; los dos montajes
    apuntan a `…/data/raw` y `…/assets/sam3`.
  - **Plan:** §3.1. **Spec:** AC-2.

- [x] **T2 — Simplificar el `command` a keep-alive**
  - Eliminar `mkdir -p data assets` y los `ln -sfn …`; dejar solo
    `exec tail -f /dev/null` (manteniendo la forma `sh -c`).
  - **Verificación:** el `command` no contiene `ln` ni `mkdir`; el contenedor se
    mantiene vivo tras `up -d`.
  - **Plan:** §3.2. **Spec:** AC-1.

- [x] **T3 — Actualizar comentarios de cabecera del compose**
  - Reescribir el comentario que menciona los symlinks de arranque para describir
    el montaje directo (sombreado del bind-mount).
  - Conservar sin cambios `build/args`, `working_dir`, `env_file`, `ports`,
    `stdin_open`, `tty` y el bloque `gpus` comentado.
  - **Verificación:** los comentarios ya no mencionan symlinks de arranque; el
    resto del servicio queda intacto.
  - **Plan:** §3.3.

---

## Fase B — `.env`

- [x] **T4 — Reescribir semántica y comentarios de `HOST_DATA_DIR`/`HOST_SAM3_DIR`**
  - Documentar que representan la **ruta real de los datos en el host** que se
    monta sobre `data/raw` y `assets/sam3`, con los Casos A (repo) y B (externo).
  - **Verificación:** los comentarios explican la nueva semántica y los dos casos.
  - **Plan:** §4.1, §4.3. **Spec:** AC-10.

- [ ] **T5 — Corregir el nivel `Meta_Glasses` en `HOST_DATA_DIR`**
  - Ajustar el valor para que apunte al directorio **padre** de `Meta_Glasses`
    (contenido = `data/raw`), de modo que las rutas relativas coincidan host↔container.
  - **Verificación:** `HOST_DATA_DIR` termina en `.../data/raw` (no en
    `.../data/raw/Meta_Glasses`); dentro del contenedor se ve
    `data/raw/Meta_Glasses/<fecha>/...`.
  - **Plan:** §4.2. **Spec:** AC-6.

---

## Fase C — Documentación (`CLAUDE.md`)

- [x] **T6 — Actualizar la descripción de Docker/volúmenes**
  - Sustituir la mención de los tres volúmenes (`app`, `/Meta_Glasses`, `/sam3`)
    por el montaje directo de `data/raw` y `assets/sam3`.
  - **Verificación:** `CLAUDE.md` describe el montaje directo; no quedan
    referencias a `/Meta_Glasses` ni `/sam3` como volúmenes vigentes.
  - **Plan:** §6.1. **Spec:** AC-9.

- [x] **T7 — Eliminar/reescribir la nota de "symlinks creados al arranque"**
  - **Verificación:** `CLAUDE.md` ya no afirma que el contenedor crea
    `data/raw → /Meta_Glasses` ni `assets/sam3 → /sam3` al arrancar.
  - **Plan:** §6.2. **Spec:** AC-9.

- [x] **T8 — Reescribir la gotcha "host vs. container / dead symlinks"**
  - Explicar que en el host `data/raw`/`assets/sam3` resuelven si existen los
    datos (Caso A) o vía symlink a ruta externa (Caso B); ya no son symlinks rotos
    por defecto. Documentar el `ln -s` manual del Caso B.
  - **Verificación:** la sección refleja el nuevo modelo y los Casos A/B.
  - **Plan:** §5, §6.3. **Spec:** AC-9.

- [x] **T9 — Matizar "Running the test scripts" y la regla `Meta_Glasses`**
  - Indicar que `test_frame_extraction.py` puede correr también en local si hay
    vídeos (Caso A/B). Mantener `rglob`/symlink sin resolver y añadir la regla del
    nivel `Meta_Glasses` (§4.2).
  - **Verificación:** la sección ya no presenta el contenedor como única vía y
    documenta la regla de `HOST_DATA_DIR`.
  - **Plan:** §6.4, §6.5. **Spec:** AC-9.

---

## Fase D — Validación

- [x] **T10 — Validar la configuración de Compose**
  - Ejecutar
    `docker compose --env-file .env -f docker/docker-compose.yml config`.
  - **Verificación:** resuelve las variables y muestra los montajes sobre
    `…/data/raw` y `…/assets/sam3` sin errores.
  - **Plan:** §8.1. **Spec:** AC-2.

- [x] **T11 — Levantar el contenedor y verificar que el host queda intacto**
  - `... up --build -d`; luego en el host `ls -la data/raw assets/sam3`.
  - **Verificación:** el contenedor queda arriba y en el host `data/raw`/
    `assets/sam3` **no** son symlinks rotos (AC-3).
  - **Plan:** §8.2, §8.3. **Spec:** AC-1, AC-3.

- [x] **T12 — Verificar acceso a datos en el contenedor**
  - `... exec futbotmx26 python testing/test_abs_dir_func.py` y
    `... exec futbotmx26 python testing/test_frame_extraction.py`.
  - **Verificación:** ambos corren sin error y ven los datos montados.
  - **Plan:** §8.4. **Spec:** AC-7.

- [x] **T13 — Verificar resolución en local (Caso A) y coherencia de rutas**
  - En el venv local, `python testing/test_abs_dir_func.py`; comprobar que una
    ruta `data/raw/Meta_Glasses/<fecha>/...` apunta a los mismos datos que en el
    contenedor.
  - **Verificación:** `get_abs_path("data/raw")` y `get_abs_path("assets/sam3")`
    resuelven sin error en local; la ruta coincide host↔container.
  - **Plan:** §8.5, §8.6. **Spec:** AC-4, AC-6.

---

## Fase E — `.gitignore` (solo Caso B)

- [x] **T14 — Ignorar el symlink `assets/sam3` (Caso B)**
  - En `.gitignore`, cambiar `assets/sam3/` por `assets/sam3` (sin barra final),
    para que el patrón ignore también el symlink y no solo el directorio.
    `data/raw` ya estaba sin barra.
  - **Verificación:** `git check-ignore assets/sam3` lo reporta ignorado; el
    symlink ya no aparece como `??` en `git status`.
  - **Plan:** §2 (nota Caso B). **Spec:** AC-8.

---

## Notas de cierre

- Al terminar la implementación, **preguntar** si commitear (constitución §11);
  mensaje sugerido (Conventional Commits, inglés):
  `fix(docker): mount heavy data directly to avoid broken host symlinks`.
- Esta tarea **no** modifica `src/`, `configs/*.json`, `Dockerfile` ni
  `.gitignore`.

> **Revisión 2026-06-03:** ver `spec.md` §8 (adenda). Se adoptó el modelo
> "archivos reales en el repo": se **eliminaron** los montajes de datos del
> `docker-compose.yml` y las variables `HOST_DATA_DIR`/`HOST_SAM3_DIR` del `.env`.
> **T5 queda superado** (ya no existe `HOST_DATA_DIR`).
