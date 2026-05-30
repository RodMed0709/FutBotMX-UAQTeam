# Tasks — Tracking

> Checklist de pasos atómicos. Marca `[x]` conforme avanzas.

- [ ] Implementar filtro de Kalman 2D (posición + velocidad) por objeto
- [ ] Implementar asociación detección↔track (IoU + Hungarian)
- [ ] Manejar oclusión: predicción de Kalman sin medición + umbral para matar track
- [ ] Estimar velocidad por objeto
- [ ] Probar con detecciones reales (salida de la tarea 01 / 02) sobre un video
- [ ] Medir estabilidad de ID (conteo de ID switches)
- [ ] Refactorizar a módulo en `src/`
- [ ] Verificar salida contra el contrato del `spec.md`
- [ ] (Iteración 2) Diseñar ego-motion compensation y física del balón
