# Spec — Núcleo de segmentación por texto (`text_segmentation`)

- **Tarea atómica:** `text_segmentation`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** una función reutilizable que, dado un frame y un prompt de texto,
> segmente con SAM3 y devuelva las detecciones (máscara + score), y otra que
> aplique las clases del proyecto a un frame completo,
> **para** tener el núcleo de inferencia por-frame del MVP fuera de los notebooks,
> bajo las convenciones del repo (modelo vía `load_sam3`, clases vía config).

---

## 2. Motivación (por qué)

- Los notebooks 01/02 ya validan la segmentación por texto (SAM3 zero-shot
  resuelve las 3 clases con buenos scores), pero la lógica vive **suelta y
  duplicada** en celdas, con `processor`/`model` globales y las clases definidas a
  mano.
- El MVP por-frame necesita este núcleo como pieza estable y reutilizable: es lo
  que consumen el overlay (tarea 4) y el `pipeline_runner` (tarea 6).
- Conviene apoyarse en las piezas ya construidas: el modelo se obtiene de
  `load_sam3` (`sam3_loader`) y las clases de la configuración (`classes_config`),
  evitando globals y definiciones duplicadas.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Definir, en un nuevo módulo `src/core/segmentation.py`:
  - una dataclass **`Detection`** (identificador, máscara, score),
  - **`segment_with_text(frame, prompt)`** → lista de `Detection` para un prompt,
  - **`detect_classes_in_frame(frame, classes)`** → detecciones por clase para
    todas las clases del proyecto.
- Las máscaras se devuelven **a tamaño del frame original** `(H, W)`.
- El modelo se obtiene de `load_sam3` y las clases de la configuración.
- Exportar las piezas públicas desde `src/core/__init__.py`.
- Un script de validación manual en `testing/`.

### 3.2 Fuera de alcance

- **Tracking** / identidad estable de objetos entre frames (tarea 5,
  `video_tracking`).
- **Visualización / overlay** de las máscaras (tarea 4, `segmentation_overlay`).
- **Escritura de video (mp4)** o de cualquier salida a disco (pipeline/escritor).
- **Export a COCO** y auto-anotación (fase 1, `coco_autoannotate`).
- **Distinguir robots aliados vs enemigos** (decisión abierta, fuera del MVP base).
- El **cómo técnico** (firmas y tipos exactos, conversión a PIL, API de la sesión
  SAM3, upscale de logits, integración con `kernels`, lectura de la config):
  corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- **Entrada:** un **frame** como `np.ndarray (H, W, 3)` RGB (un elemento de la
  salida de `extract_frames`). La función lo adapta internamente a lo que SAM3
  necesita.
- **`segment_with_text(frame, prompt)`:** ejecuta SAM3 con ese prompt de texto y
  devuelve una **lista de `Detection`**; cada `Detection` tiene un identificador
  de objeto, una **máscara booleana a tamaño del frame** `(H, W)` y un score.
  Si no hay objetos, devuelve lista vacía.
- **`detect_classes_in_frame(frame, classes)`:** aplica **todas las clases del
  proyecto** al frame y devuelve un **diccionario `{nombre_de_clase:
  [Detection, ...]}`**. Usa el **prompt activo** de cada clase (el primero de su
  lista `sam3_prompts`). Si no se le pasan clases, las **lee de la configuración**.
- **Máscaras full-res:** se entregan al **tamaño del frame de entrada**, listas
  para que overlay / pipeline las usen sin reescalar.
- **Post-proceso:** la limpieza de máscara (NMS / relleno de huecos / motas) la
  aporta `kernels` en la inferencia; la función solo lleva la máscara a tamaño
  original y la umbraliza.
- **Modo por-frame independiente:** cada frame y cada clase se segmentan por
  separado; **no** se garantiza identidad estable de objetos entre clases ni entre
  frames (eso es tracking).
- **Origen del modelo:** las funciones operan sobre el modelo cargado por
  `load_sam3`; si no se les proporciona, lo obtienen por defecto.

---

## 5. Criterios de aceptación

1. **AC-1 — Módulo y piezas:** existe `src/core/segmentation.py` con la dataclass
   `Detection`, `segment_with_text` y `detect_classes_in_frame`, exportadas desde
   `src/core/__init__.py`.
2. **AC-2 — `Detection`:** cada detección expone identificador de objeto, máscara
   y score.
3. **AC-3 — Segmentación por prompt:** `segment_with_text(frame, prompt)` devuelve
   una lista de `Detection` (vacía si no hay objetos).
4. **AC-4 — Máscara full-res:** la máscara de cada `Detection` es booleana y tiene
   la forma `(H, W)` del frame de entrada.
5. **AC-5 — Detección por clases:** `detect_classes_in_frame` devuelve un dict
   `{nombre_de_clase: [Detection]}` usando el prompt activo de cada clase de la
   configuración.
6. **AC-6 — Clases desde config:** sin argumentos de clases, la función las toma
   de la configuración (tarea `classes_config`).
7. **AC-7 — Modelo vía `load_sam3`:** las funciones usan el modelo cargado por
   `load_sam3`; no dependen de variables globales.
8. **AC-8 — Validación manual:** se demuestra de forma exploratoria, sobre un
   **frame real**, que `segment_with_text` y `detect_classes_in_frame` producen
   detecciones coherentes (máscara del tamaño correcto, scores razonables).

---

## 6. Supuestos y notas

- **Dependencias:** depende de `sam3_loader` (1) y `classes_config` (2);
  **desbloquea** `segmentation_overlay` (4) y `pipeline_runner` (6).
- **Rendimiento (medido):** la inferencia la domina el forward del modelo
  (~860M params); en **CPU es inviable** (minutos por inferencia, y el modo
  por-frame son varias inferencias por frame). El tamaño del frame casi no influye
  (el processor reescala internamente). Por tanto, la **validación real y las
  corridas se hacen en GPU**; en CPU, a lo sumo un *smoke test* simbólico. Esto
  acota el alcance de la prueba del MVP (clip corto en GPU).
- **`kernels`:** disponible en los tres entornos (local `pip install -r`,
  contenedor automático, pod ya instalado), por lo que la limpieza de máscara se
  delega a la inferencia y **no** se porta el `refine_mask` (cv2) del notebook 04;
  queda como utilidad opcional futura si la calidad lo exigiera.
- **Prompt activo:** se usa `sam3_prompts[0]` de cada clase; los demás prompts de
  la lista quedan disponibles para experimentación (ver `classes_config`).
- Esta especificación **no** define el *cómo* técnico (firmas y tipos exactos,
  conversión a PIL, API concreta de la sesión SAM3, upscale bilinear de logits,
  módulo y exportación, ni el script de validación); todo ello corresponde al
  `plan.md` de esta misma carpeta.
