# Plan — Tracking

> **Lo llena la owner (Victoria).** Decisiones de CÓMO. El contrato del `spec.md` no se toca.

## Enfoque técnico
- Modelo de estado de Kalman: _(posición + velocidad 2D; matrices F, H, Q, R)_
- Asociación detección↔track: _(IoU + Hungarian / distancia)_
- Umbral de oclusión (frames sin medición antes de matar un track): _(fijar número)_
- Gestión de nacimiento/muerte de tracks: _(a definir)_

## Iteración 2 (post-MVP, dejar diseñado)
- Modelo físico del balón: parábola en vuelo, velocidad ~constante en piso.
- Ego-motion compensation: estimar movimiento de la cámara para separar movimiento real vs aparente.
- Considerar IMM (Interacting Multiple Model) si se necesitan dos modelos de movimiento.

## Librerías
- _(ej. filterpy para Kalman, scipy para Hungarian, supervision — ya en requirements.txt)_

## Configuración (JSON en configs/)
- Ruido de proceso/medición, umbral de oclusión, umbral de IoU.

## Riesgos
- Ego-motion: la velocidad en píxeles ≠ velocidad real (cámara/lentes en movimiento).
- Oclusión prolongada → pérdida de ID.
- Múltiples robots parecidos → cambios de ID (ID switches).
