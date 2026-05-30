# Spec — Segmentación y propagación en video con SAM 3

- **Owner:** Rodrigo + Leonardo
- **Estado:** propuesto → (pendiente de aprobación)
- **Posición en el pipeline:** segundo eslabón. Recibe inicialización (cajas de YOLO o prompts de texto), segmenta y propaga máscaras a lo largo del video.

---

## Objetivo

Hacer que SAM 3 corra **sobre video**: dado un video y una inicialización (prompts), segmentar
los objetos y **propagar sus máscaras frame a frame** manteniendo identidad por objeto.
Es el corazón visual del sistema.

## Alcance

- Inferencia de SAM 3 en video (no solo imagen suelta).
- Propagación temporal de máscaras con identidad estable por objeto.
- Soporte a los dos formatos: landscape (cámara externa 1920×1080) y portrait (Meta Glasses ~1456×1936).
- Inicialización vía prompts de texto **y/o** cajas provenientes del detector YOLO (tarea 01).

## Fuera de alcance

- Entrenar el detector → tarea `01`.
- Filtro de Kalman / física / velocidad → tarea `03`.
- LoRA / fine-tuning de SAM 3 → fase posterior (ver roadmap), no en el MVP.

## Contrato

**Entrada**
- Video (ruta) + inicialización:
  - prompts de texto (las clases de `configs/classes.json`), **o**
  - cajas iniciales por objeto en formato de la tarea 01 (`bbox_xyxy` + `class_name`).
- Configuración desde `configs/` (ruta de pesos SAM 3, device, dtype).

**Salida** — por frame, una lista de objetos segmentados:
```json
{
  "frame_idx": 0,
  "objects": [
    {
      "object_id": 1,
      "class_name": "orange ball",
      "mask": "<RLE o polígono>",
      "bbox_xyxy": [x1, y1, x2, y2]
    }
  ]
}
```
- `object_id` **estable a lo largo del video** (el mismo objeto conserva su id entre frames).
- `bbox_xyxy` derivada de la máscara (caja que la encierra) — esto es lo que consume el tracker.
- Formato de máscara (RLE vs polígono) se fija en `plan.md`; debe ser serializable.

> El tracker (tarea 03) consume `object_id` + `bbox_xyxy` (y opcionalmente la máscara).
> Mantener ese formato estable.

## Criterios de éxito

- [ ] SAM 3 corre end-to-end sobre un video real de `data/raw/17Abril/` y produce máscaras por frame.
- [ ] Las máscaras se propagan: un objeto conserva su `object_id` entre frames consecutivos.
- [ ] Funciona en los dos formatos (landscape y portrait).
- [ ] Salida en el formato del contrato, serializable.
- [ ] Función reutilizable importable desde `src/`.

## Riesgos conocidos (atender en plan.md)

- **Regresión de velocidad de SAM 3 vs SAM 2 (~5-6×)** — medir tiempos, considerar submuestreo de frames.
- **Memory leak reportado en SAM 3 video** — vigilar uso de memoria en videos largos; liberar estado.
- Resolución portrait alta (Meta Glasses) → costo de inferencia; evaluar resize.
