# Spec — Configuración y prueba del entorno de ejecución inicial

- **Tarea atómica:** `env_setup`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla o reproduce el proyecto,
> **quiero** un entorno de ejecución inicial aislado, configurado y verificable,
> **para** poder trabajar sin mezclar dependencias con otras instalaciones de
> Python y garantizar que cualquiera pueda replicar el entorno de forma
> confiable.

---

## 2. Motivación (por qué)

- Evitar que el proyecto se mezcle con otras instalaciones de Python del sistema
  (aislamiento).
- Garantizar **reproducibilidad**: que cualquiera pueda levantar el mismo entorno
  en su máquina o en RunPod.
- Establecer una base sólida y validada antes de construir los pipelines
  (YOLO + SAM3 + ByteTrack), de modo que los problemas de entorno no se
  confundan con problemas de modelo.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Asegurar el **aislamiento** del entorno de Python.
- Comprobar la existencia de los archivos base y crearlos si faltan:
  - `requirements.txt`
  - `.env`
- Preparar la **configuración de Docker** en una carpeta dedicada, con un
  servicio de `docker-compose` llamado **`futbotmx26`**.
- Crear una **carpeta de testing** con un script pequeño que valide que el
  entorno es funcional.

### 3.2 Fuera de alcance

- Instalación o configuración de los modelos del pipeline (YOLO, SAM3,
  ByteTrack).
- Descarga u organización de los vídeos del dataset.
- Cualquier lógica de segmentación, detección, tracking o análisis.

---

## 4. Comportamiento esperado (criterios de los archivos base)

### 4.1 `requirements.txt`

- Si **no existe**, se crea con un conjunto **mínimo** de dependencias,
  suficiente para validar que el entorno funciona.
- Si **ya existe**, se **respeta tal cual** (no se sobrescribe ni se reduce).

### 4.2 `.env`

- Se crea con **variables base predefinidas** y **valores por defecto**
  (p. ej. ruta del proyecto en el contenedor, identificadores del entorno).
- No contiene secretos reales; los valores por defecto son seguros para
  compartir como punto de partida.
- El `.env` **no se versiona** (conforme a la constitución).

### 4.3 Configuración de Docker

- Los archivos de Docker viven en una **carpeta dedicada**.
- El `docker-compose` define un servicio cuyo nombre es exactamente
  **`futbotmx26`**.
- El uso de Docker es **sencillo**: basta con `Dockerfile` + `docker-compose`
  para levantar un entorno ejecutable (sin entrypoints elaborados ni registro de
  imágenes remoto).

### 4.4 Carpeta y script de testing

- Existe una **carpeta de testing** dedicada.
- Contiene un **script pequeño** que verifica que el entorno es funcional.
- El script debe:
  1. **Importar** las librerías clave instaladas.
  2. **Reportar sus versiones**.
  3. **Verificar la disponibilidad de GPU/CUDA**.

---

## 5. Criterios de aceptación

1. **AC-1 — Aislamiento:** el entorno de Python está aislado del sistema (venv),
   independientemente de su ejecución dentro del contenedor.
2. **AC-2 — Archivos base presentes:** tras la tarea existen `requirements.txt` y
   `.env`; si no existían, fueron creados con los criterios de la sección 4.
3. **AC-3 — `requirements.txt` respetado:** si el archivo ya existía, su
   contenido no se modificó.
4. **AC-4 — `.env` con valores por defecto:** el `.env` contiene las variables
   base predefinidas con valores por defecto y sin secretos reales.
5. **AC-5 — Docker:** existe la carpeta dedicada de Docker y el servicio de
   `docker-compose` se llama `futbotmx26`; el entorno puede levantarse con esos
   archivos.
6. **AC-6 — Testing:** existe la carpeta de testing con un script que, al
   ejecutarse, importa las librerías clave, reporta versiones y verifica
   GPU/CUDA.
7. **AC-7 — Éxito de la verificación:** el script de prueba **se ejecuta sin
   errores tanto en el venv local como dentro del contenedor**.
8. **AC-8 — Reproducibilidad:** otra persona, partiendo del repositorio, puede
   levantar el mismo entorno y obtener un resultado de prueba equivalente.

---

## 6. Supuestos y notas

- La versión de Python es **3.11** (constitución).
- Las rutas de datos son relativas y se acceden vía configuración; esta tarea
  sienta las bases del entorno, no define aún las rutas del pipeline.
- La línea duplicada del borrador (carpeta de testing mencionada dos veces) se
  interpreta como **un único requisito**.
- Esta especificación **no** define el *cómo* técnico; eso corresponde al
  `plan.md` de esta misma carpeta.

---

## 7. Siguientes pasos (metodología)

1. Elaborar `plan.md` con el detalle técnico de implementación.
2. Derivar `tasks.md` con las tareas ejecutables.
3. Implementar (paso 5) únicamente después de los anteriores.
