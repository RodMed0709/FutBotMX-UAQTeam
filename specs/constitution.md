# Constitution — FutBotMX UAQ Team

> Reglas fijas del proyecto. **No cambian sin acuerdo de todo el equipo.**
> Antes de escribir un `spec.md`, un `plan.md` o cualquier línea de código, esto ya está decidido.
> Si tienes una duda del tipo "¿qué versión de Python?" o "¿dónde pongo la config?", la respuesta está aquí.

---

## 1. Propósito

Construir el sistema de visión por computadora del reto Copa FutBotMX 2026: detección,
segmentación y seguimiento de los objetos en cancha (RoboCup) a partir de video de las
Meta Glasses y de cámara externa.

La prioridad es un **MVP funcional** antes que cualquier feature avanzada. El pipeline base es:

```
detección (YOLO) → segmentación + propagación en video (SAM 3) → tracking → [eventos: post-MVP]
```

---

## 2. Entorno (fijo)

- **Python 3.11** (única versión soportada).
- **Infra:** RunPod, dos pods compartidos por el equipo:
  - **CPU pod** — desarrollo, preparación de datos, pruebas ligeras.
  - **GPU pod** — RTX 5090 (Blackwell). Inferencia y entrenamiento.
- **PyTorch se instala aparte** según el pod (no va en `requirements.txt` con versión fija):
  - CPU pod: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  - GPU pod: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- **Repo en el pod:** `/workspace/FutBotMX-UAQTeam`
- **SAM 3:** instalación manual (no está en PyPI). Pesos en `assets/sam3/` (ignorados por git).

---

## 3. Stack / librerías

- La lista canónica de dependencias vive en **`requirements.txt`**. Es la única fuente de verdad.
- Si necesitas una librería nueva, **se agrega a `requirements.txt` en un PR** — no se instala "suelta" en el pod sin registrarla.
- Versiones acotadas (pinned ranges) para que los cuatro corramos el mismo entorno. Cero "en mi máquina sí jala".

---

## 4. Estructura del repositorio (no se modifica)

```
FutBotMX-UAQTeam/
├── assets/        # pesos SAM 3, multimedia del README (ignorado en git lo pesado)
├── configs/       # configuración: rutas, hiperparámetros, clases  → JSON
├── data/          # datasets (ignorado en git; solo estructura)
│   └── raw/17Abril/   # videos: Cámaras/ (externa) + *.MOV (Meta Glasses)
├── models/        # checkpoints y pesos entrenados (ignorado en git)
├── notebooks/     # exploración individual por persona
│   └── fase_0/
├── specs/         # ESTE directorio — contratos de trabajo (SDD)
├── src/           # código fuente limpio, modular (producto final)
├── main.py
└── requirements.txt
```

**Regla:** nadie reorganiza carpetas raíz. Si crees que falta una, se discute primero.

---

## 5. Reglas de código

1. **Config desde JSON, cero hardcode.**
   Toda **ruta, hiperparámetro y lista de clases** se lee de un archivo en `configs/` (formato JSON).
   Prohibido clavar paths o números mágicos dentro del código o los notebooks.
   - Las clases del proyecto viven en `configs/classes.json` (una sola definición para todos).
   - Hoy son 3: `orange ball`, `robot`, `green floor`. El reto contempla hasta 5; se amplían ahí, no en cada notebook.

   Ejemplo mínimo de config:
   ```json
   {
     "paths": {
       "data_raw": "data/raw/17Abril",
       "sam3_weights": "assets/sam3"
     },
     "classes": ["orange ball", "robot", "green floor"],
     "hparams": { "conf_threshold": 0.4, "device": "cuda" }
   }
   ```

2. **De notebook a `src/`.**
   Los notebooks son playground (explorar, validar). Todo lo que ya funciona y es parte del
   producto se **refactoriza a módulos limpios en `src/`**. El entregable corre desde `src/`, no desde un notebook.

3. **Un notebook por persona / tarea.**
   Nombrado individual (ej. `nb_<tarea>_<nombre>.ipynb`). **No se editan notebooks ajenos** → evita merge conflicts.

4. **Clean code básico:** funciones cortas, nombres claros, sin código muerto. `black` + `ruff` antes de subir.

---

## 6. Flujo de trabajo: Spec-Driven Development (SDD)

Toda tarea sigue esta secuencia. **Nadie escribe código sin un `spec.md` aprobado.**

```
spec.md  →  plan.md  →  tasks.md  →  código  →  verificar contra el spec
 (QUÉ)       (CÓMO)      (PASOS)
```

- **`spec.md`** — qué hace la pieza, su **contrato** (qué recibe / qué entrega) y los criterios de éxito.
  Define el contrato la coordinación del equipo. **El contrato no se cambia unilateralmente** (rompe a los demás).
- **`plan.md`** — decisiones técnicas: algoritmo, librerías, estructura, riesgos. Lo escribe el responsable de la tarea.
- **`tasks.md`** — checklist de pasos atómicos. Es el avance visible: se marca `[x]` conforme se completa.

Cada subcarpeta de `specs/` es una tarea con esos tres archivos.

**El spec es el juez:** si el código no cumple los criterios de éxito del spec, no está terminado.

---

## 7. Reglas de datos

- `data/` y `models/` están en `.gitignore` (archivos pesados). Solo se versiona la estructura.
- **El dataset original (`data/raw/`) no se modifica.** Las transformaciones generan datos nuevos en `data/interim/` o `data/processed/`.
- Las clases a detectar/segmentar son las definidas en `configs/classes.json`.

---

## 8. Git / colaboración

- GitHub compartido. Se permite conectar Google Colab al repo.
- **Trabajo en ramas**, no directo a `main`. Una rama por tarea (ej. `feat/yolo-detector`).
- Commits pequeños y descriptivos. Merge a `main` vía PR.
- No subir pesos, datos, ni notebooks con outputs gigantes (ya cubierto por `.gitignore`).

---

## 9. Roadmap (orden de prioridad)

1. **MVP** — pipeline base funcionando end-to-end sobre un video real:
   detección → segmentación + propagación → tracking.
2. **Tracking robusto** — Kalman + manejo de oclusión; física del balón y compensación de
   ego-motion (la cámara/lentes se mueven) como segunda iteración.
3. **Detección de eventos** — post-MVP. Sin responsable asignado todavía; se decide al cerrar el MVP.
