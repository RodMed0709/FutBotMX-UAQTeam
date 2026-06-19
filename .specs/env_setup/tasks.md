# Tasks — Configuración y prueba del entorno de ejecución inicial

- **Tarea atómica:** `env_setup`
- **Paso de la metodología:** 4 (Descomposición en tareas ejecutables)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Plan de referencia:** [`plan.md`](./plan.md)
- **Estado:** Lista de tareas. La implementación (paso 5) comienza **solo** tras
  aprobar este documento. Aún **no** se crea ni modifica código fuente.

> Convención: cada tarea tiene un **ID**, un **criterio de verificación** y la
> **sección del plan** que la origina. Marcar `- [x]` al completar.

---

## Fase A — Estructura de carpetas

- [x] **T1 — Crear la estructura base de directorios**
  - Crear: `docker/`, `data/`, `assets/`, `testing/`.
  - **Verificación:** las cuatro carpetas existen en la raíz del proyecto.
  - **Plan:** §3.

---

## Fase B — Archivos base

- [x] **T2 — Resolver `requirements.txt`**
  - Comprobar si `requirements.txt` existe.
  - Si **no existe**: crearlo con `numpy`, `matplotlib`,
    `opencv-python-headless`, `jupyter`, `notebook`, `ipykernel`, `torch`.
  - Si **existe**: dejarlo intacto (no sobrescribir ni reducir).
  - **Verificación:** `requirements.txt` está presente; si preexistía, su
    contenido no cambió.
  - **Plan:** §5.1. **Spec:** AC-2, AC-3.

- [x] **T3 — Resolver `.env`**
  - Comprobar si `.env` existe; si no, crearlo.
  - Asegurar las variables base:
    - `CONTAINER_WORKSPACE_DIR` con valor por defecto `futbot`.
    - `HOST_DATA_DIR` como placeholder (a completar por el usuario).
    - `HOST_SAM3_DIR` como placeholder (a completar por el usuario).
  - No incluir secretos reales.
  - **Verificación:** `.env` existe y contiene las tres variables con sus
    valores/placeholders.
  - **Plan:** §5.2. **Spec:** AC-2, AC-4.

---

## Fase C — Dockerización

- [x] **T4 — Crear `docker/Dockerfile`**
  - Imagen base `python:3.11-slim`.
  - Instalar librerías de sistema mínimas para `opencv-python-headless`.
  - Crear y activar el `venv`.
  - Copiar `requirements.txt` e instalar dependencias (antes de copiar el resto,
    para aprovechar la caché de capas).
  - Copiar el resto del proyecto.
  - Definir `WORKDIR` a partir de `CONTAINER_WORKSPACE_DIR` (no asumir `/app`).
  - **Verificación:** la imagen se construye sin errores y el `WORKDIR` resultante
    coincide con `CONTAINER_WORKSPACE_DIR`.
  - **Plan:** §6.2.

- [x] **T5 — Crear `docker/docker-compose.yml`**
  - Definir un único servicio llamado exactamente `futbotmx26`.
  - Construir desde el `Dockerfile` de `docker/` y leer variables del `.env`.
  - Declarar los 3 volúmenes:
    - `../:/${CONTAINER_WORKSPACE_DIR}`
    - `${HOST_DATA_DIR}:/Meta_Glasses`
    - `${HOST_SAM3_DIR}:/sam3`
  - **Verificación:** `docker compose config` resuelve las variables; el servicio
    se llama `futbotmx26` y declara los 3 volúmenes correctos.
  - **Plan:** §6.1. **Spec:** AC-5.

- [x] **T6 — Crear los symlinks de datos pesados (en build/arranque del contenedor)**
  - `data/raw` → `/Meta_Glasses`
  - `assets/sam3` → `/sam3`
  - Modelado como paso del build/arranque del contenedor, no en el host.
  - **Verificación:** dentro del contenedor, `data/raw` y `assets/sam3` resuelven
    a `/Meta_Glasses` y `/sam3` respectivamente.
  - **Plan:** §6.3.

---

## Fase D — Testing

- [x] **T7 — Crear `testing/test_env.py`**
  - Importar `numpy`, `cv2`, `matplotlib`, componentes de Jupyter, `torch`.
  - Reportar la versión de cada librería.
  - Verificar GPU/CUDA con `torch.cuda.is_available()` y, si hay GPU, reportar el
    dispositivo (la ausencia de GPU se informa, no se considera fallo).
  - Terminar sin errores cuando las importaciones tienen éxito.
  - **Verificación:** el script existe y, al ejecutarse, imprime versiones y el
    estado de GPU/CUDA sin lanzar excepciones.
  - **Plan:** §7. **Spec:** AC-6.

---

## Fase E — Exclusiones de control de versiones

- [x] **T8 — Crear/actualizar `.gitignore`**
  - Ignorar: `.venv/`, `.env`, `data/raw`, `assets/sam3`, rutas de datos/modelos
    pesados (`/Meta_Glasses`, `/sam3`) y el contenido de `.specs/drafts/`.
  - **Verificación:** `git status` no lista los artefactos excluidos.
  - **Plan:** §8.

---

## Fase F — Validación end-to-end

- [x] **T9 — Validar el entorno en venv local**
  - Crear el `venv`, activarlo e instalar desde `requirements.txt`.
  - Ejecutar `testing/test_env.py`.
  - **Verificación:** el script corre sin errores y reporta versiones + estado de
    GPU/CUDA.
  - **Plan:** §4, §7. **Spec:** AC-1, AC-7.

- [x] **T10 — Validar el entorno en el contenedor**
  - Levantar el servicio `futbotmx26` con `docker compose`.
  - Ejecutar `testing/test_env.py` dentro del contenedor.
  - **Verificación:** el script corre sin errores dentro del contenedor; los
    volúmenes y symlinks resuelven correctamente.
  - **Plan:** §6, §7. **Spec:** AC-5, AC-6, AC-7.

- [x] **T11 — Confirmar reproducibilidad**
  - Verificar que, partiendo del repositorio y completando los placeholders del
    `.env`, otra persona puede levantar el mismo entorno y obtener un resultado
    de prueba equivalente.
  - **Verificación:** checklist de AC-1 a AC-8 del spec cumplido.
  - **Plan:** §5, §6, §8. **Spec:** AC-8.

---

## Resumen de trazabilidad

| Tarea | Plan       | Criterios de aceptación (spec) |
| ----- | ---------- | ------------------------------ |
| T1    | §3         | —                              |
| T2    | §5.1       | AC-2, AC-3                     |
| T3    | §5.2       | AC-2, AC-4                     |
| T4    | §6.2       | AC-5                           |
| T5    | §6.1       | AC-5                           |
| T6    | §6.3       | AC-5                           |
| T7    | §7         | AC-6                           |
| T8    | §8         | AC-8                           |
| T9    | §4, §7     | AC-1, AC-7                     |
| T10   | §6, §7     | AC-5, AC-6, AC-7               |
| T11   | §5, §6, §8 | AC-8                           |

---

## Nota de metodología

Este documento cierra el paso 4. La **implementación (paso 5)** de estas tareas
ocurrirá únicamente cuando se indique explícitamente; hasta entonces no se crea
ni modifica código fuente.
