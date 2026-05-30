# Plan — Detección con YOLO

> **Lo llena el owner (Luis Felipe).** Aquí van las decisiones de CÓMO, no el QUÉ.
> El QUÉ y el contrato están en `spec.md` y no se cambian.

## Enfoque técnico
- Modelo / versión YOLO a usar: _(ej. YOLOv8/v11, tamaño n/s/m — justificar)_
- Pre-entrenado vs entrenado de cero: _(a definir)_
- Resolución de entrada: _(a definir)_

## Datos
- Fuente de etiquetas: _(Roboflow del equipo / etiquetado nuevo)_
- Split train/val/test: _(a definir)_
- Aumentaciones: _(a definir)_

## Librerías
- _(ej. ultralytics, supervision — ya en requirements.txt)_

## Configuración (JSON en configs/)
- Qué parámetros se exponen: pesos, umbral de confianza, NMS, clases, device.

## Riesgos
- Pocos datos etiquetados → considerar transferencia / data augmentation.
- Balón pequeño y rápido → revisar tamaño de ancla / resolución.
- _(otros)_

## Métrica objetivo
- mAP@0.5 mínimo aceptable: _(fijar número)_
