# Plan — Segmentación y propagación en video con SAM 3

> **Lo llenan los owners (Rodrigo + Leonardo).** Decisiones de CÓMO. El contrato del `spec.md` no se toca.

## Enfoque técnico
- Mecanismo de propagación de SAM 3 en video: _(memory / video predictor — documentar API real)_
- Inicialización: prompts de texto vs cajas de YOLO vs ambos: _(a definir)_
- Formato de máscara de salida: RLE vs polígono: _(decidir; serializable)_
- Manejo de identidad (`object_id`) entre frames: _(a definir)_

## Rendimiento
- Estrategia ante la regresión 5-6× vs SAM 2: _(submuestreo de frames, resize, batch)_
- Manejo del memory leak: liberar estado cada N frames / reiniciar predictor: _(a definir)_

## Integración
- Cómo recibe las cajas de la tarea 01 (formato del contrato de YOLO).
- Qué entrega exactamente al tracker (tarea 03).

## Configuración (JSON en configs/)
- Pesos SAM 3, device, dtype (bfloat16), resolución de inferencia, frecuencia de frames.

## Riesgos
- Memoria en videos largos.
- Pérdida de identidad tras oclusión prolongada.
- _(otros)_
