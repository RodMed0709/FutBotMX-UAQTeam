# Spec — Detección con YOLO

- **Owner:** Luis Felipe
- **Estado:** propuesto → (pendiente de aprobación)
- **Posición en el pipeline:** primer eslabón. Alimenta a SAM 3 (como prompts de caja) y al tracker.

---

## Objetivo

Dado un frame de video, detectar los objetos del reto y devolver sus cajas (bounding boxes)
con su clase y confianza. Es el detector que arranca el pipeline: provee las posiciones
iniciales que SAM 3 segmenta y que el tracker sigue.

## Alcance

- Detección **por frame** (imagen individual). La propagación temporal NO es de esta tarea.
- Clases: las definidas en `configs/classes.json` (hoy: `orange ball`, `robot`, `green floor`).
- Entrenar/afinar un modelo YOLO sobre el dataset del equipo (Roboflow / `data/raw/17Abril/`).

## Fuera de alcance

- Segmentación de máscaras → tarea `02_segmentacion_sam3_video`.
- Seguimiento entre frames / IDs → tarea `03_tracking`.

## Contrato

**Entrada**
- Un frame: imagen RGB (`np.ndarray` H×W×3) **o** ruta a imagen/video + índice de frame.
- Configuración desde `configs/` (ruta de pesos, umbral de confianza, lista de clases).

**Salida** — lista de detecciones por frame. Cada detección:
```json
{
  "class_id": 0,
  "class_name": "orange ball",
  "bbox_xyxy": [x1, y1, x2, y2],
  "confidence": 0.93
}
```
- `bbox_xyxy` en píxeles, esquina sup-izq y inf-der.
- Coordenadas en el espacio del frame original (sin reescalar).

> **Este formato es el contrato.** SAM 3 y el tracker dependen de él. No se cambia sin avisar a los owners de 02 y 03.

## Criterios de éxito

- [ ] Corre sobre los dos formatos de video (landscape cámara externa y portrait Meta Glasses).
- [ ] Detecta el balón (`orange ball`) de forma consistente en frames con balón visible.
- [ ] Devuelve exactamente el formato del contrato.
- [ ] mAP@0.5 ≥ (umbral a fijar en `plan.md` con el set de validación).
- [ ] Función reutilizable importable desde `src/` (no solo en notebook).

## Notas

- El detector y SAM 3 son complementarios: YOLO da cajas rápidas y estables para inicializar SAM 3.
- Revisar el dataset de Roboflow ya etiquetado del equipo antes de etiquetar de cero.
