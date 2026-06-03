# Plan técnico — Configuración y prueba del entorno de ejecución inicial

- **Tarea atómica:** `env_setup`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo se construye y valida el entorno de ejecución
inicial descrito en el `spec.md`: aislamiento con `venv`, archivos base
(`requirements.txt`, `.env`), dockerización sencilla con volúmenes y symlinks, y
un script de testing que confirme que el entorno es funcional (imports +
versiones + GPU/CUDA), tanto en venv local como dentro del contenedor.

---

## 2. Stack técnico

- **Python:** 3.11 (constitución).
- **Aislamiento:** módulo estándar `venv`.
- **Contenedores:** Docker con `Dockerfile` + `docker-compose.yml` (uso sencillo,
  sin entrypoints elaborados ni registro de imágenes remoto).
- **Imagen base:** `python:3.11-slim`.
- **Dependencias mínimas iniciales:** `numpy`, `matplotlib`,
  `opencv-python-headless`, `jupyter`, `notebook`, `ipykernel`, `torch`.

---

## 3. Estructura de carpetas objetivo

```
sdd-futbot/
├── .env                      # variables de entorno (NO versionado)
├── .gitignore
├── requirements.txt          # dependencias del proyecto
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   └── raw -> /Meta_Glasses  # symlink (creado en el contenedor)
├── assets/
│   └── sam3 -> /sam3         # symlink (creado en el contenedor)
├── testing/
│   └── test_env.py           # script de verificación del entorno
└── .venv/                    # entorno virtual (NO versionado)
```

> `data/raw`, `assets/sam3`, `/Meta_Glasses`, `/sam3` y `.venv/` se excluyen del
> control de versiones.

---

## 4. Entorno virtual (`venv`)

- Se usa el módulo estándar `venv` de Python 3.11.
- El entorno vive **dentro de la carpeta del proyecto** (p. ej. `.venv/`).
- Se excluye del control de versiones.
- Procedimiento conceptual: crear el venv → activarlo → instalar desde
  `requirements.txt`.

---

## 5. Archivos base

### 5.1 `requirements.txt`

- **Si no existe:** se crea con el conjunto mínimo:
  - `numpy`
  - `matplotlib`
  - `opencv-python-headless` (variante **sin GUI**, porque el proyecto se ejecuta
    mayormente en ambientes sin entorno gráfico)
  - dependencias de Jupyter: `jupyter`, `notebook`, `ipykernel`
  - `torch` (necesario para verificar disponibilidad de GPU/CUDA)
- **Si ya existe:** se **respeta tal cual**, no se sobrescribe ni se reduce.

### 5.2 `.env`

- **Si no existe:** se crea y se añaden las variables base.
- Variables requeridas por esta tarea:

  | Variable | Propósito | Valor por defecto / placeholder |
  |---|---|---|
  | `CONTAINER_WORKSPACE_DIR` | Nombre del directorio de trabajo del proyecto dentro del contenedor | `futbot` (por defecto) |
  | `HOST_DATA_DIR` | Ruta en el HOST con los datos; se mapea a `/Meta_Glasses` | *placeholder* (lo completa cada usuario) |
  | `HOST_SAM3_DIR` | Ruta en el HOST con el modelo SAM3; se mapea a `/sam3` | *placeholder* (lo completa cada usuario) |

- El `.env` **no se versiona**; no contiene secretos reales (solo rutas y nombre
  de workspace).

---

## 6. Dockerización

### 6.1 `docker/docker-compose.yml`

- Define un único servicio con nombre exacto **`futbotmx26`**.
- Construye la imagen a partir del `Dockerfile` en `docker/`.
- Lee variables desde el `.env`.
- Declara **3 volúmenes**:

  | Volumen (HOST → CONTAINER) | Origen | Destino |
  |---|---|---|
  | Workspace de la app | `../` | `/${CONTAINER_WORKSPACE_DIR}` |
  | Datos | `${HOST_DATA_DIR}` | `/Meta_Glasses` |
  | Modelo SAM3 | `${HOST_SAM3_DIR}` | `/sam3` |

- El nombre del directorio de la app **no** está fijo en `app`: lo define cada
  desarrollador mediante `CONTAINER_WORKSPACE_DIR`.

### 6.2 `docker/Dockerfile`

Orden de instrucciones (conceptual):

1. **Imagen base:** `python:3.11-slim`.
2. Instalar las **librerías de sistema mínimas** necesarias para
   `opencv-python-headless`.
3. **Crear y activar** el entorno virtual (`venv`).
4. **Copiar `requirements.txt`** e **instalar** las dependencias dentro del venv.
5. **Copiar el resto del proyecto**.
6. Definir el **`WORKDIR`** a partir de `CONTAINER_WORKSPACE_DIR` del `.env`
   (**nunca** asumir `/app`).

> Orientación: copiar primero `requirements.txt` e instalar antes de copiar el
> resto del proyecto para aprovechar la caché de capas de Docker.

### 6.3 Symlinks de datos pesados

- Dentro del proyecto en el contenedor (`/${CONTAINER_WORKSPACE_DIR}`):
  - `data/raw` → `/Meta_Glasses`
  - `assets/sam3` → `/sam3`
- Los symlinks **se crean al construir/levantar el contenedor**, no en el host
  (apuntan a rutas internas del contenedor montadas por los volúmenes).
- Los datos pesados quedan fuera del control de versiones.

---

## 7. Script de testing

- Ubicación: `testing/test_env.py`.
- Responsabilidades:
  1. **Importar** las librerías clave: `numpy`, `cv2` (opencv),
     `matplotlib`, componentes de Jupyter, `torch`.
  2. **Reportar versiones** de cada librería importada.
  3. **Verificar GPU/CUDA** con `torch.cuda.is_available()` (y, si está
     disponible, reportar el dispositivo).
- **Resultado de éxito:** el script termina sin errores. Debe ejecutarse
  correctamente tanto en el **venv local** como **dentro del contenedor**.

---

## 8. Exclusiones de control de versiones (`.gitignore`)

Se asegura que estén ignorados:

- `.venv/`
- `.env`
- `data/raw` y `assets/sam3` (symlinks a datos pesados)
- cualquier ruta de datos/modelos pesados (`/Meta_Glasses`, `/sam3`)
- el contenido de `.specs/drafts/` (constitución)

---

## 9. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por |
|---|---|
| AC-1 Aislamiento | §4 venv |
| AC-2 Archivos base presentes | §5 |
| AC-3 `requirements.txt` respetado | §5.1 |
| AC-4 `.env` con valores por defecto | §5.2 |
| AC-5 Docker + servicio `futbotmx26` | §6 |
| AC-6 Testing (imports/versiones/GPU) | §7 |
| AC-7 Éxito en venv y contenedor | §4, §6, §7 |
| AC-8 Reproducibilidad | §5, §6, §8 |

---

## 10. Riesgos y consideraciones

- **GPU/CUDA:** la verificación con `torch` puede reportar `False` si la máquina
  no tiene GPU; esto es esperado en entornos sin GPU y no debe considerarse fallo
  del script (solo informa el estado). En RunPod se espera GPU disponible.
- **Variante de OpenCV:** usar `opencv-python-headless` evita dependencias de GUI;
  si en el futuro se requiere visualización interactiva, deberá reevaluarse.
- **Rutas del HOST:** `HOST_DATA_DIR` y `HOST_SAM3_DIR` dependen de cada usuario;
  si no se completan en el `.env`, los volúmenes/symlinks no resolverán bien.
- **`WORKDIR` dinámico:** debe mantenerse la coherencia entre
  `CONTAINER_WORKSPACE_DIR` del `.env`, el `WORKDIR` del `Dockerfile` y el
  destino del volumen de la app en `docker-compose.yml`.

---

## 11. Siguiente paso (metodología)

Elaborar `tasks.md` con la descomposición en tareas ejecutables. La
implementación (paso 5) ocurre únicamente después de definir las tareas.
