# Plan técnico — Montaje correcto de los datos pesados en host y contenedor

- **Tarea atómica:** `data_volume_mounts`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo eliminar el bug del symlink roto y dejar que
`data/raw` (vídeos) y `assets/sam3` (modelo SAM3) resuelvan a los datos reales
tanto en local (venv) como dentro del contenedor, mediante **montaje directo de
volúmenes** sobre las rutas del proyecto, sin symlinks creados en el arranque.
El cambio es de **infraestructura y documentación**; no toca `src/` ni los
`configs/*.json`.

---

## 2. Archivos afectados

| Archivo | Cambio |
|---|---|
| `docker/docker-compose.yml` | Reescribir `volumes` (montaje directo) y `command` (keep-alive). |
| `.env` | Reescribir comentarios y semántica de `HOST_DATA_DIR`/`HOST_SAM3_DIR`; ajustar el valor para corregir el nivel `Meta_Glasses`. |
| `CLAUDE.md` | Actualizar las secciones que describen volúmenes, symlinks de arranque y la gotcha host-vs-container. |
| `.gitignore` | **(Solo si se usa el Caso B)** quitar la barra final del patrón `assets/sam3/` → `assets/sam3`, para que ignore también el **symlink** (un patrón con `/` final solo matchea directorios). `data/raw` ya está sin barra. |

**No se tocan:** `Dockerfile`, `configs/*.json`, `src/**`,
`data/README.md`, `assets/README.md`, `testing/**`.

> **Nota Caso B (gitignore y symlinks):** al convertir `data/raw`/`assets/sam3`
> en symlinks (Caso B), Git deja de verlos como directorios. Un patrón de
> `.gitignore` con barra final (`assets/sam3/`) **no** matchea un symlink, así
> que el symlink aparecería como *untracked* y podría commitearse por accidente.
> Por eso el patrón se deja **sin** barra final. Es un cambio necesario y de bajo
> riesgo, no contemplado en el alcance original del Caso A.

---

## 3. Cambios en `docker/docker-compose.yml`

### 3.1 `volumes` (núcleo del fix)

Estado actual (tres volúmenes; los dos últimos son intermedios):

```yaml
volumes:
  - ../:/${CONTAINER_WORKSPACE_DIR:-futbot}
  - ${HOST_DATA_DIR}:/Meta_Glasses
  - ${HOST_SAM3_DIR}:/sam3
```

Estado objetivo (montaje directo sobre la ruta del proyecto):

```yaml
volumes:
  # Workspace de la aplicación (nombre configurable, no fijo en /app).
  - ../:/${CONTAINER_WORKSPACE_DIR:-futbot}
  # Datos pesados montados DIRECTAMENTE sobre la ruta del proyecto.
  # Estos montajes se aplican ENCIMA del bind-mount del repo (lo sombrean),
  # sin modificar el filesystem del host.
  - ${HOST_DATA_DIR}:/${CONTAINER_WORKSPACE_DIR:-futbot}/data/raw
  - ${HOST_SAM3_DIR}:/${CONTAINER_WORKSPACE_DIR:-futbot}/assets/sam3
```

- La interpolación `${CONTAINER_WORKSPACE_DIR:-futbot}` es válida dentro de la
  cadena del volumen (Compose la resuelve antes de crear el contenedor).
- Desaparecen los puntos de montaje `/Meta_Glasses` y `/sam3`.

### 3.2 `command` (simplificación)

Estado actual (crea symlinks que rompen el host) → objetivo (solo mantener vivo):

```yaml
command:
  - sh
  - -c
  - exec tail -f /dev/null
```

- Se elimina el `mkdir -p data assets` y los `ln -sfn`. Las carpetas `data/` y
  `assets/` ya existen en el repo bind-montado; los datos llegan por los montajes
  de §3.1.

### 3.3 Resto del servicio

Se conservan **sin cambios**: `build` + `args`, `image`, `working_dir`,
`env_file`, `ports: ["8888:8888"]`, `stdin_open`, `tty` y el bloque `gpus: all`
comentado. Se actualiza el comentario de cabecera del archivo para reflejar el
montaje directo (ya no menciona symlinks de arranque).

---

## 4. Cambios en `.env`

### 4.1 Semántica nueva

`HOST_DATA_DIR` y `HOST_SAM3_DIR` pasan a significar **"ruta real de los datos en
el host"**, que Compose monta directamente sobre `data/raw` y `assets/sam3`:

- **Caso A (datos en el repo):** apuntan dentro del repo
  (p. ej. `HOST_DATA_DIR=<repo>/data/raw`).
- **Caso B (datos externos):** apuntan a otro disco
  (p. ej. `HOST_DATA_DIR=/mnt/otrodisco/videos`).

### 4.2 Fix del nivel `Meta_Glasses`

Regla: `HOST_DATA_DIR` debe apuntar al directorio cuyo **contenido** es
exactamente lo que debe verse bajo `data/raw`. Como la convención de rutas usa
`data/raw/Meta_Glasses/<fecha>/...`, el valor debe **incluir** el nivel
`Meta_Glasses`, es decir apuntar al **directorio padre** de `Meta_Glasses`:

```
# Antes (desfasado): .../data/raw/Meta_Glasses  -> container ve data/raw/<fecha>
# Después (correcto): .../data/raw              -> container ve data/raw/Meta_Glasses/<fecha>
```

Así una misma ruta relativa apunta a los mismos datos en host y contenedor
(AC-6).

### 4.3 Comentarios

Se reescriben los comentarios de ambas variables para explicar: (1) la nueva
semántica, (2) la regla del contenido = ruta del proyecto, (3) los Casos A/B, y
(4) que en Caso B el `data/raw`/`assets/sam3` local puede ser un symlink a esa
misma ruta. El resto del `.env` (`CONTAINER_WORKSPACE_DIR`, `CONFIG_FILENAME`) no
cambia.

> `.env` no se versiona; aquí se edita la copia local de trabajo y se documenta
> la convención para quien reproduzca el proyecto.

---

## 5. Setup del host (sin script — manual y documentado)

No se crea ningún script (decisión KISS). El contrato §4.3 del spec se cumple así:

- **Caso A:** colocar los datos como directorios reales en `<repo>/data/raw` y
  `<repo>/assets/sam3`. No requiere ningún paso extra.
- **Caso B:** crear manualmente el symlink hacia la ruta externa, p. ej.:

  ```bash
  ln -sfn /mnt/otrodisco/videos data/raw
  ln -sfn /mnt/otrodisco/sam3   assets/sam3
  ```

  Estos symlinks son locales, quedan ignorados por Git y no afectan al
  contenedor. Para uso combinado (mismo equipo corre local y contenedor),
  `HOST_DATA_DIR`/`HOST_SAM3_DIR` deben apuntar al **mismo destino** que el
  symlink.

Ambos casos se documentan en `CLAUDE.md`.

---

## 6. Cambios en `CLAUDE.md`

Actualizar (sin alterar sus otras secciones) los pasajes que hoy describen el
modelo viejo:

1. **Bloque de Docker / volúmenes:** sustituir la descripción de los tres
   volúmenes (`app`, `/Meta_Glasses`, `/sam3`) por el montaje directo de
   `data/raw` y `assets/sam3`.
2. **"Symlinks de datos pesados creados al arranque":** eliminar/observar que ya
   no existen; los datos llegan por montaje directo que sombrea el bind-mount.
3. **Gotcha "host vs. container / dead symlinks":** reescribir. Ahora en el host
   `data/raw`/`assets/sam3` resuelven si existen los datos (Caso A o symlink
   Caso B); ya no son symlinks rotos por defecto.
4. **"Running the test scripts":** matizar que `test_frame_extraction.py` puede
   correr también en local si hay vídeos disponibles (Caso A/B), no solo en el
   contenedor.
5. **Convención de rutas:** mantener `rglob` y el uso del symlink sin resolver;
   añadir la regla del nivel `Meta_Glasses` (§4.2).

---

## 7. Estructura objetivo (resumen)

```
futbot/
├── .env                       # HOST_DATA_DIR/HOST_SAM3_DIR = ruta real de datos
├── docker/
│   └── docker-compose.yml     # montaje directo, command = keep-alive
├── data/
│   ├── .gitkeep
│   └── raw/                   # Caso A: dir real | Caso B: symlink -> externo
└── assets/
    └── sam3/                  # Caso A: dir real | Caso B: symlink -> externo
```

> `data/raw`, `assets/sam3` y sus contenidos siguen excluidos por `.gitignore`.

---

## 8. Validación

1. **Lint de Compose:**
   `docker compose --env-file .env -f docker/docker-compose.yml config`
   resuelve las variables y muestra los dos montajes sobre
   `…/data/raw` y `…/assets/sam3`.
2. **Levantar:**
   `docker compose --env-file .env -f docker/docker-compose.yml up --build -d`.
3. **Host intacto (AC-3):** tras levantar, en el host `data/raw` y `assets/sam3`
   **no** son symlinks rotos (`readlink`/`ls -la` sin cambios respecto a antes).
4. **Contenedor (AC-7):**
   `... exec futbotmx26 python testing/test_abs_dir_func.py` y
   `... exec futbotmx26 python testing/test_frame_extraction.py` corren sin error
   y ven los datos.
5. **Local Caso A (AC-4):** en el venv, `get_abs_path("data/raw")` y
   `get_abs_path("assets/sam3")` resuelven sin error (vía
   `python testing/test_abs_dir_func.py`).
6. **Coherencia (AC-6):** una ruta `data/raw/Meta_Glasses/<fecha>/...` apunta a
   los mismos datos en host y contenedor.

---

## 9. Riesgos y mitigaciones

- **Montar un volumen sobre una ruta que en el host es un symlink (Caso B):** el
  runtime resuelve el destino del montaje siguiendo el symlink dentro del rootfs
  del contenedor; en la práctica el dato queda accesible vía `data/raw`. Si en
  algún entorno diera problemas, la mitigación es no usar symlink local en ese
  equipo (Caso A) o alinear `HOST_DATA_DIR` con el destino del symlink. En un
  checkout limpio (RunPod/CI) `data/raw` es un dir vacío con `.gitkeep`, así que
  el montaje cae sobre un punto de montaje limpio.
- **`HOST_DATA_DIR` mal apuntado (nivel `Meta_Glasses`):** se mitiga con la regla
  §4.2 y la validación de coherencia (§8.6).
- **Variables `.env` vacías:** si `HOST_DATA_DIR`/`HOST_SAM3_DIR` faltan, Compose
  fallará al expandir; la documentación deja claro que son obligatorias.

---

## 10. Siguientes pasos (metodología)

1. Derivar `tasks.md` con las tareas ejecutables y sus criterios de verificación.
2. Implementar (paso 5) únicamente tras aprobar `tasks.md`.
