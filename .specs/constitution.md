# Constitución del proyecto

> Documento global, único e inamovible. Establece los principios, reglas no
> negociables y aspectos de arquitectura que rigen todo el proyecto. Cualquier
> spec, plan o tarea debe respetar lo aquí definido. Modificar este documento
> requiere una decisión explícita y consciente del responsable del proyecto.

---

## 1. Propósito del proyecto

Proyecto de **machine learning** y **visión por computadora** cuyo fin es analizar
vídeos de partidos de **fútbol robótico** mediante detección, segmentación y
rastreo de objetos.

El proyecto contempla **dos pipelines principales**:

1. **Pipeline base**
   - **Detección:** YOLO
   - **Segmentación:** SAM3
   - **Tracking:** ByteTrack

2. **Pipeline con fine-tuning**
   - Misma estructura (detección → segmentación → tracking), pero el **modelo
     detector YOLO se somete a un proceso de fine-tuning**.
   - La estrategia exacta de fine-tuning **aún no está definida**. Caminos
     posibles bajo evaluación: **Roboflow** y el propio **SAM3** (p. ej. para
     generar/asistir el etiquetado).

Ambos pipelines comparten la misma base de datos, las mismas convenciones de
configuración y la misma metodología de trabajo descritas en este documento.

---

## 2. Reglas no negociables

Estas reglas no se rompen sin modificar antes esta constitución:

1. **Disciplina de la metodología SDD (Spec-Driven Development).** No se escribe
   ni se modifica código del proyecto antes de que exista el `tasks.md` de la
   tarea correspondiente (ver sección 6).
2. **El agente solo produce archivos `.md` durante los pasos 1–4.** Cualquier
   modificación real del proyecto (código, configuración funcional, datos)
   ocurre únicamente en el paso 5 de implementación.
3. **Toda ruta de datos es relativa y se accede mediante configuración.** Está
   prohibido incrustar rutas absolutas o "hardcodeadas" en el código.
4. **Configuración fuera del código.** Las configuraciones globales viven en
   archivos `.json` versionados; los secretos viven en `.env` y nunca se
   versionan.
5. **Reproducibilidad.** El proyecto debe poder ejecutarse en cualquier máquina
   con Docker y prepararse para correr en RunPod, sin pasos manuales ocultos.
6. **Aislamiento de entorno.** Independientemente del contenedor, el trabajo en
   Python se realiza siempre bajo un entorno virtual.
7. **Versión de Python fija:** `3.11`.
8. **Exclusiones obligatorias del repositorio remoto** (ver sección 7): vídeos,
   datos y pesos del modelo SAM3 (y demás modelos), y el contenido de
   `.specs/drafts/`.
9. **Un atómico, una carpeta.** Cada tarea o problema atómico tiene su propia
   subcarpeta dentro de `.specs/` con sus archivos `spec.md`, `plan.md` y
   `tasks.md`.
10. **La constitución es única y global.** Solo existe un archivo de
    constitución y prevalece sobre cualquier otro documento del proyecto.
11. **Commits controlados y estandarizados.** El repositorio está ligado a un
    remoto: el agente **nunca** hace `commit`/`push` por iniciativa propia. Cuando
    lo considere oportuno (p. ej. al cerrar un paso o una tarea atómica)
    **pregunta** si debe commitear, y solo procede con la confirmación del
    responsable. Todos los mensajes de commit siguen el estándar definido en §7.1.

---

## 3. Datos del proyecto

- **Dataset:** colección de **123 vídeos en formato MOV**.
- **Ubicación:** alojados en una unidad en la nube. **Quien reproduzca el
  proyecto es responsable de descargar y organizar los vídeos en crudo.**
- **Resolución:** no fija / no especificada en la documentación.
- **Duración:** no fija; se presume que puede ser **menor a 5 minutos**.
- Los vídeos y los datos/pesos de los modelos **nunca** se incluyen en el
  repositorio remoto.

---

## 4. Stack técnico

- **Lenguaje:** Python **3.11**.
- **Gestión de dependencias:** sencilla, mediante un único archivo
  `requirements.txt`.
- **Entorno virtual:** obligatorio para todo trabajo en Python, dentro o fuera
  del contenedor.
- **Componentes de ML/CV:** YOLO (detección), SAM3 (segmentación), ByteTrack
  (tracking). Para el segundo pipeline, herramientas de fine-tuning aún por
  definir (Roboflow / SAM3 como candidatos).

---

## 5. Arquitectura y configuración

### 5.1 Dockerización

- El proyecto debe ser **dockerizable** para garantizar reproducibilidad y
  ejecución en **RunPod**.
- El uso de Docker se mantiene **sencillo**: basta con los archivos de
  configuración `Dockerfile` y `docker-compose` para levantar un entorno donde el
  proyecto pueda ejecutarse.
- **No** se requieren entrypoints elaborados, **ni** almacenar la imagen en un
  repositorio remoto de imágenes.

### 5.2 Ruta del proyecto en el contenedor

- Ruta por defecto: `/app`.
- Debe ser una **variable de configuración**, para dar margen a la
  administración de RunPod, a cambios de proveedor o a entornos propios.

### 5.3 Archivos de configuración (`.json`)

- Las configuraciones generales y globales se almacenan en archivos `.json`.
- **Nomenclatura obligatoria:** `{NN}_{EXP}.json`
  - `NN` → versión o *trial* del experimento (p. ej. `00`, `01`).
  - `EXP` → descriptivo del propósito del archivo.
  - Ejemplo: `00_testing_env_setup.json`.
- Estos archivos contienen, de forma centralizada, las **rutas relativas** a:
  - directorio de vídeos,
  - datos del modelo SAM3 (y demás modelos),
  - directorio de modelos y sus pesos,
  - directorio de outputs,
  - y cualquier otra ruta de datos del proyecto.
- En el código principal, dichas rutas **siempre** se acceden a través del
  archivo de configuración, nunca de forma directa.

### 5.4 Secretos (`.env`)

- Debe existir un archivo `.env` que almacene secretos y variables que no se
  quieren exponer.
- El `.env` **no se versiona**.

### 5.5 Outputs

- Los resultados del pipeline se almacenan en un **directorio de outputs
  configurable**, separado de los datos de entrada y declarado en el archivo de
  configuración correspondiente.

---

## 6. Metodología de trabajo (5 pasos)

El trabajo se basa en cinco pasos fundamentales y **secuenciales**:

1. **Constitución (este archivo).** Define los principios inamovibles del
   proyecto. Un solo archivo, global.
2. **`spec.md`.** Describe **qué** se quiere construir y **por qué**. Un archivo
   por tarea o proceso. Formato de **historia de usuario / requisito**.
3. **`plan.md`.** Redacción técnica de todo lo necesario para implementar la
   especificación: stack técnico y arquitectura de la solución.
4. **`tasks.md`.** Descompone el plan en una **lista de tareas ejecutables**.
5. **Implementación.** Solo a partir de este punto el agente implementa las
   tareas siguiendo el plan y modifica el proyecto. **Antes de este paso, el
   agente únicamente crea los archivos `.md` correspondientes.**

> **Regla de oro:** no hay implementación sin `tasks.md`, no hay `tasks.md` sin
> `plan.md`, y no hay `plan.md` sin `spec.md`.

### 6.1 Organización en `.specs/`

Por cada tarea o problema atómico a resolver se crea una **subcarpeta** dentro
de `.specs/`. Dentro de esa subcarpeta se almacenan sus archivos `spec.md`,
`plan.md` y `tasks.md`.

```
.specs/
├── constitution.md           # este documento (global)
├── drafts/                   # borradores (excluidos del repo remoto)
└── {tarea-atomica}/
    ├── spec.md
    ├── plan.md
    └── tasks.md
```

---

## 7. Versionado y exclusiones

- El versionado corre a cargo de **Git**, conectado a un repositorio remoto en
  **GitHub**.
- **Excluidos del repositorio remoto** (vía `.gitignore` u equivalente):
  - los **vídeos** (dataset en crudo),
  - los **datos y pesos** del modelo SAM3 y demás modelos,
  - el contenido de **`.specs/drafts/`**,
  - el archivo **`.env`**.

### 7.1 Estándar de commits

- **El agente no commitea ni pushea por su cuenta** (regla no negociable #11).
  Cuando juzgue que un paso está cerrado, **pregunta** antes de ejecutar el commit
  y espera confirmación explícita.
- **Formato obligatorio: Conventional Commits**, con el **mensaje en inglés**:
  `type(scope): short imperative summary`.
  - `type` ∈ `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `style`, `perf`,
    `build`, `ci`.
  - `scope` (opcional pero recomendado): preferentemente el nombre de la **tarea
    atómica** de `.specs/` afectada (p. ej. `config_naming`) o el área
    (`docker`, `sdd`, `config`).
  - El **summary** va en imperativo, sin punto final, ≤ 72 caracteres.
- Un commit por unidad lógica de cambio; el cuerpo (opcional) explica el *por qué*.
- Ejemplos:
  - `feat(docker): add Dockerfile and compose for RunPod`
  - `fix(config): read data_path from the io key`
  - `docs(sdd): move constitution to .specs/`

---

## 8. Protocolo de asunciones para archivos en `.specs/`

Siempre que exista una petición para **crear un archivo dentro de `.specs/`**, el
agente debe asumir la siguiente instrucción:

1. **Antes** de desplegar la lista, preguntar qué asunciones tomar en cuenta:
   *técnicas, no técnicas, funcionales, todas* u *otra opción*, en forma de lista
   para que el usuario elija.
2. Mostrar en un **listado numerado** todas las cosas asumidas (técnicas, no
   técnicas y/o funcionales según lo elegido).
3. El usuario indica los **números** de las asunciones que no le gustaron.
4. Por cada una, el agente hace **preguntas una a una**, mostrando una **barra de
   progreso** (cuántas preguntas lleva y cuántas faltan), ofreciendo **4
   asunciones nuevas** y una **quinta opción "otra"** para respuesta libre.
5. Al final, el agente indica si **ya está listo** para elaborar el documento.

> Recordatorio: en este paso del proceso **aún no se modifica nada del
> proyecto**; solo se elabora el archivo `.md` solicitado.
