# Spec — fps real de la fuente en modo completo (`source_fps`)

- **Tarea atómica:** `source_fps`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** que en modo completo (`all_frames=True`) el video anotado se genere a
> la **velocidad real del video original**,
> **para** que el entregable "video original → proceso → video anotado" se
> reproduzca de forma natural y el pipeline quede **100% real** en ese modo.

---

## 2. Motivación (por qué)

- `pipeline_runner` dejó un **placeholder**: en modo completo el mp4 se escribe al
  fps de configuración (pensado para el slideshow de la cuota), no al fps real del
  video. Eso hace que el video completo se reproduzca a velocidad incorrecta.
- `extract_frames` ya abre el video con `decord` (que conoce el fps), pero **no lo
  expone**; falta una forma limpia de obtener el fps de la fuente para pasárselo al
  escritor en modo completo.
- Cerrar esto convierte el modo completo en el **uso real** prometido, sin tocar el
  modo cuota (testeo / generación de frames para fine-tuning).

---

## 3. Alcance

### 3.1 Dentro de alcance

- Definir **`get_video_fps(video_path)`** (en `src/core/frame_extraction.py`) que
  devuelve el fps promedio del video, aceptando path relativo a `PROJECT_ROOT` o
  absoluto.
- **Cablear** en `run_pipeline`: en modo completo usar `get_video_fps(video)`; en
  modo cuota seguir con el fps de configuración.
- Reflejar el fps efectivo en el **mp4** y en el **JSON** de detecciones.
- Exportar `get_video_fps` desde `src/core/__init__.py`.
- Actualizar el test del pipeline / añadir validación de `get_video_fps`.

### 3.2 Fuera de alcance

- **No** se modifica la firma ni el comportamiento de `extract_frames` (opción B:
  helper independiente).
- **Tracking** u otras mejoras del pipeline.
- Cambios en el **modo cuota** (sigue con el fps de config).
- El **cómo técnico** (API de decord, firma/tipos exactos, manejo de errores):
  corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- **`get_video_fps(video_path)`:**
  - **Entrada:** path del video relativo a `PROJECT_ROOT` o absoluto.
  - **Salida:** el **fps promedio** del video (float).
  - **Error:** video inexistente/ inválido → error claro (coherente con
    `extract_frames`).
- **`run_pipeline`:**
  - Modo **completo** (`all_frames=True`): `fps = get_video_fps(video)` → mp4 a la
    velocidad real; el JSON registra ese fps.
  - Modo **cuota** (`all_frames=False`): sin cambios, fps de config.
- **`extract_frames`:** sin cambios (firma y retorno idénticos).

---

## 5. Criterios de aceptación

1. **AC-1 — Función presente:** existe `get_video_fps` en
   `src/core/frame_extraction.py`, exportada desde `src/core/__init__.py`.
2. **AC-2 — fps correcto:** `get_video_fps` devuelve el fps real de un video
   (acorde a lo que reporta el contenedor del video).
3. **AC-3 — Entrada flexible:** acepta path relativo a `PROJECT_ROOT` o absoluto.
4. **AC-4 — Modo completo usa fps fuente:** `run_pipeline(video, all_frames=True)`
   escribe el mp4 a `get_video_fps(video)` y lo refleja en el JSON.
5. **AC-5 — Modo cuota intacto:** `run_pipeline(video)` (cuota) sigue usando el fps
   de config.
6. **AC-6 — `extract_frames` intacto:** su firma y comportamiento no cambian.
7. **AC-7 — Error claro:** un video inexistente/ inválido produce un error claro.
8. **AC-8 — Validación:** `get_video_fps` se valida **en local** sobre un video
   real (no usa modelo); el pipeline completo end-to-end se valida en **RunPod**.

---

## 6. Supuestos y notas

- **Dependencias:** comparte la lectura de video con `frame_extraction` y modifica
  `pipeline_runner`. Cierra el "pipeline 100% real" en modo completo.
- **Opción B (helper independiente):** `get_video_fps` reabre el video solo para
  leer metadatos (barato; no decodifica frames), manteniendo `extract_frames` con
  responsabilidad única y sin retorno *union*.
- **Validación local posible:** a diferencia de la inferencia, obtener el fps no
  usa el modelo ni GPU; hay videos locales, así que el agente puede ejecutar la
  validación de `get_video_fps`.
- Esta especificación **no** define el *cómo* técnico (API exacta de decord,
  firma/tipos, manejo de errores, ni la actualización concreta del test); todo ello
  corresponde al `plan.md`.
