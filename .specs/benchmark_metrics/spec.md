# Spec — Métricas y tabla comparativa del benchmark (`benchmark_metrics`)

- **Tarea atómica:** `benchmark_metrics`
- **Paso de la metodología:** 2 (Especificación)
- **Proceso:** segunda y **última** tarea del benchmark sin-GT de las 6
  configuraciones detector × tracker. Consume los JSON producidos por `run_batch`
  (instrumentado en `batch_timing`) y emite la **tabla comparativa** que decide qué
  configuración es mejor.
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que construye el benchmark del pipeline,
> **quiero** un módulo que, a partir de los JSON de inferencia y el timing de
> `run_batch`, calcule métricas **sin ground-truth** (eficiencia, consistencia de
> trayectoria y estabilidad de máscara) y las resuma en **una tabla de las 6
> configuraciones**,
> **para** decidir con datos qué combinación detector × tracker es la mejor, sin
> esperar a las anotaciones manuales (que siguen pausadas).

---

## 2. Motivación (por qué)

- El benchmark compara 6 configs (detector `sam3_text`/`yolo_sam3` × tracker
  none/`bytetrack`/`botsort`) sobre 5 videos de testing. Ya tenemos cómo
  **producir** los JSON (con timing) vía `run_batch`; falta cómo **leerlos y
  resumirlos** en métricas comparables.
- Sin GT, la decisión se apoya en métricas **sin referencia**: eficiencia (FPS,
  VRAM), consistencia física de trayectorias (longitud de tracklet, fragmentación,
  suavidad) y estabilidad temporal de las máscaras (IoU temporal, jitter). El smoke
  `00_phase2_cost_smoke.ipynb` ya demostró que estas métricas son baratas de calcular
  y que la señal de máscara es informativa.
- Hoy esos cálculos viven a mano en un notebook. Un módulo reutilizable y testeable
  los estandariza, los agrega por config y emite la tabla de decisión.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Módulo lector/agregador puro** (no corre inferencia): consume los JSON ya escritos
  por `run_batch` + el **timing** de su valor de retorno (`elapsed_s`, `fps`,
  `peak_vram_mb`).
- **Métricas de eficiencia** (del retorno de `run_batch`): FPS y VRAM pico, agregadas
  por config.
- **Métricas de trayectoria** (de la sección `tracks` del JSON, por `obj_id`):
  - **Longitud media de tracklet** (frames por `obj_id`).
  - **Tasa de fragmentación proxy** (un `obj_id` termina y otro nuevo arranca en ≤N
    frames dentro de un radio espacial cercano).
  - **Suavidad**: varianza de la aceleración (segunda derivada) de los centroides.
- **Métricas de máscara** (de la sección `frames`, requieren `rle` + `obj_id`
  estable):
  - **IoU temporal** de la máscara del mismo `obj_id` entre frames consecutivos.
  - **Jitter del centro de masa** de los píxeles segmentados.
- **Configs sin tracking** (segmentación): reportan **solo eficiencia**; trayectoria y
  máscara quedan **N/A** (el JSON no tiene `tracks` ni `obj_id` estable).
- **Selector seeded** de los 5 videos de testing (`split=2`, semilla fija),
  reproducible, como función reutilizable del módulo.
- **Agregación por config** (media sobre los 5 videos) → **una fila por config** →
  tabla de hasta 6 filas, que se **imprime** y se **persiste como CSV** bajo
  `outputs/`.
- **Parámetros con defaults sensatos** (ventana de fragmentación, radio espacial),
  como argumentos, no hardcode disperso.
- **Driver de las 6 corridas** (notebook en `notebooks/benchmark_models/`): orquesta
  config por config (corre `run_batch` → calcula métricas con los JSON frescos →
  guarda la fila → siguiente, con `overwrite=True`). Es exploración, **no** código de
  `src/`.

### 3.2 Fuera de alcance

- **Correr la inferencia** o modificar `run_batch`/`run_inference`/el esquema: el
  módulo solo **lee**.
- **Layout de outputs por config**: se evita procesando **una config a la vez** en el
  driver (los JSON se leen frescos antes de sobrescribir). No se versiona la ruta de
  salida ni se toca `inference_paths`.
- **Métricas que requieran GT** (mAP, MOTA, IDF1, mIoU vs humano): es el otro proceso
  (evaluación con GT, pausado).
- **Ranking automático / "ganador"**: el módulo emite la tabla; la interpretación y la
  decisión final son humanas.
- **Visualizaciones** (gráficas de trayectoria, overlays): fuera; solo la tabla.
- El **cómo técnico** (firmas exactas, fórmulas, estructura interna): es del `plan.md`.

---

## 4. Comportamiento esperado (criterios de aceptación)

1. **No corre inferencia**: el módulo solo lee JSON + consume el timing de
   `run_batch`; no importa SAM3 ni carga modelos.
2. **Trayectoria**: dado un JSON de tracking, calcula longitud media de tracklet,
   fragmentación proxy y suavidad por `obj_id`/agregadas por video.
3. **Máscara**: dado un JSON con `rle`, calcula IoU temporal y jitter del centro de
   masa por `obj_id`; si el JSON **no** trae `rle`, esas métricas son `None`/N/A sin
   romper.
4. **Eficiencia**: integra `fps` y `peak_vram_mb` provenientes del resumen de
   `run_batch`.
5. **Configs sin tracking**: producen fila con eficiencia y trayectoria/máscara en
   N/A (no hay `tracks`).
6. **Agregación**: una fila por config (media sobre los videos); la tabla de 6 configs
   se imprime y se escribe a CSV bajo `outputs/`.
7. **Selector seeded**: la selección de 5 videos de `split=2` es **reproducible**
   (misma semilla → mismos videos).
8. **Verificación local sin GPU**: un smoke con JSON sintéticos (frames/tracks con
   `rle` fabricado) ejercita trayectoria + máscara + agregación + tabla, sin SAM3.

---

## 5. Dependencias y relación con otras tareas

- **Depende de:** `batch_timing` (timing en el retorno de `run_batch`),
  `batch_detector_tracker` (selección de configs), el esquema de salida
  (`frames`/`tracks`, `decode_rle`, `num_frames`), y `db_metadata.csv` (selector
  seeded sobre `split=2`).
- **Reusa:** `decode_rle` (de `inference_schema`), la lógica de IoU temporal
  prototipada en `00_phase2_cost_smoke.ipynb`, y opcionalmente `get_trajectories`.
- **Cierra** el benchmark sin-GT: con esta tabla se decide la configuración a usar
  (y, más adelante y de forma opcional, a comparar contra un GT real).
- **No** depende de GT: es la vía sin-referencia mientras la anotación sigue pausada.
