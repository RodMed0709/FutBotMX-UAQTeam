# Spec — Aceptar rutas absolutas de vídeo en `extract_frames`

- **Tarea atómica:** `abs_video_path`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** que `extract_frames` acepte una **ruta absoluta** a un vídeo siempre
> que apunte a un **archivo existente y válido**, aunque esté **fuera** de la raíz
> del proyecto,
> **para** poder extraer frames de vídeos ubicados en montajes o ubicaciones
> externas (`data` y `assets/sam3` ya no tienen por qué vivir dentro del
> proyecto), sin que la función rechace rutas absolutas legítimas.

---

## 2. Motivación (por qué)

- El enfoque del proyecto cambió: los datos pesados (`data/raw`, `assets/sam3`)
  pueden **resolverse fuera de `PROJECT_ROOT`** (montajes directos del contenedor,
  enlaces a ubicaciones externas en el host). En consecuencia, una ruta absoluta
  válida que apunte **fuera** del proyecto es ahora un **caso de uso legítimo**.
- Hoy `extract_frames` rechaza con `ValueError` cualquier ruta absoluta que no
  esté bajo `PROJECT_ROOT`, porque delega toda la verificación en `get_abs_path`,
  utilidad pensada **solo** para rutas relativas. Esto impide pasar la ruta
  absoluta de un vídeo externo aunque el archivo exista. Es el **bug** que esta
  tarea corrige.
- Se busca **ampliar** el contrato de la función (aceptar rutas absolutas) sin
  romper el comportamiento actual con rutas relativas ni alterar el resto del
  proyecto.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Permitir que `extract_frames` reciba una **ruta absoluta** a un vídeo y la
  acepte cuando corresponda a un **archivo existente y válido**, sin exigir que
  esté bajo `PROJECT_ROOT`.
- Mantener el soporte de **rutas relativas** tal como funciona hoy (resueltas
  respecto a la raíz del proyecto).
- Conservar las **validaciones** de entrada (tipo de la ruta, existencia) con los
  mismos errores que ya emite la función.

### 3.2 Fuera de alcance

- Modificar `src/utils.py::get_abs_path` o su contrato (sigue aceptando solo
  rutas relativas; su uso para rutas relativas no cambia).
- Cambiar la **firma pública** de `extract_frames` (sus parámetros y su salida).
- Alterar la lógica de los **modos de extracción** (cuota / completo) ni la
  lectura de la cuota desde configuración.
- Procesar directorios o lotes de vídeos, persistir frames a disco, o cualquier
  etapa posterior del pipeline.
- La definición del **cómo técnico** (estructura exacta de la resolución de la
  ruta, tipos, mensajes de error): corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- **Entrada:** la **ruta de un vídeo** como `Path`, que puede ser:
  - **relativa** a `PROJECT_ROOT` (comportamiento actual), o
  - **absoluta**, apuntando a un archivo en cualquier ubicación del sistema
    (dentro o fuera del proyecto).
- **Ruta relativa:** se resuelve respecto a la **raíz del proyecto** y se verifica
  su existencia, igual que hoy (en línea con la constitución §2.3: las rutas del
  proyecto se acceden vía la raíz/configuración).
- **Ruta absoluta:** se acepta **siempre que apunte a un archivo existente y
  válido**, sin imponer que esté bajo `PROJECT_ROOT`. Ya **no** se lanza
  `ValueError` por estar "fuera del proyecto".
- **Validación de "archivo válido":** "válido" significa que la ruta **existe** y
  es un **archivo** (no un directorio). Una ruta a un directorio se rechaza con
  error. La validez del **contenido** como vídeo la sigue determinando la capa de
  lectura al abrirlo.
- **Symlinks:** una ruta absoluta que sea symlink se acepta mientras su **destino
  exista**; no se fuerza una resolución del symlink que rompa el acceso a los
  montajes/ubicaciones externas.
- **Validaciones que se conservan:**
  - Si `video_path` **no es un `Path`**, se lanza `ValueError` (igual que hoy).
  - Si la ruta (absoluta o relativa) **no existe**, se lanza `FileNotFoundError`
    (igual que hoy).
- **Sin cambios:** los modos **cuota** (por defecto) y **completo** (`all_frames`)
  se comportan exactamente como antes; la cuota se sigue leyendo de la
  configuración.

---

## 5. Criterios de aceptación

1. **AC-1 — Ruta absoluta externa aceptada:** dada una ruta **absoluta** a un
   vídeo **existente fuera de `PROJECT_ROOT`**, `extract_frames` la procesa y
   devuelve frames, **sin** lanzar `ValueError`.
2. **AC-2 — Ruta absoluta interna aceptada:** una ruta **absoluta** a un vídeo
   existente **dentro** de `PROJECT_ROOT` también se procesa correctamente.
3. **AC-3 — Ruta relativa intacta:** una ruta **relativa** a un vídeo existente se
   sigue resolviendo respecto a la raíz del proyecto y se procesa como hoy.
4. **AC-4 — Inexistencia:** una ruta (absoluta o relativa) que **no existe** hace
   que la función falle con `FileNotFoundError`.
5. **AC-5 — Directorio rechazado:** una ruta que **existe pero es un directorio**
   se rechaza con el error correspondiente (no se trata como vídeo válido).
6. **AC-6 — Tipo inválido:** un `video_path` que **no es `Path`** falla con
   `ValueError`, igual que hoy.
7. **AC-7 — Firma y modos sin cambios:** la firma pública de `extract_frames` no
   cambia, y los modos cuota/completo conservan su comportamiento (incluida la
   lectura de la cuota desde configuración).
8. **AC-8 — `get_abs_path` intacta:** la utilidad `get_abs_path` no se modifica;
   las rutas relativas se siguen apoyando en ella.
9. **AC-9 — Validación manual:** se demuestra de forma **exploratoria** (script
   suelto o notebook) que, sobre un vídeo real, una ruta **absoluta externa**
   produce frames, y que las rutas relativas siguen funcionando.

---

## 6. Supuestos y notas

- El cambio se limita a la **lógica de resolución/validación de la ruta del
  vídeo** dentro de `frame_extraction.py`; no toca la firma pública ni el resto
  del proyecto.
- La validación manual que use vídeos reales debe ejecutarse donde los datos
  resuelvan (contenedor con montajes, o host con los datos presentes bajo
  `data/raw`), conforme a las notas de entorno del proyecto.
- Esta especificación **no** define el *cómo* técnico (estructura exacta de la
  función auxiliar de resolución, mensajes de error, uso de `Path.is_file()` vs.
  `exists()`, etc.); eso corresponde al `plan.md` de esta misma carpeta.

---

## 7. Siguientes pasos (metodología)

1. Elaborar `plan.md` con el detalle técnico de implementación.
2. Derivar `tasks.md` con las tareas ejecutables.
3. Implementar (paso 5) únicamente después de los anteriores.
