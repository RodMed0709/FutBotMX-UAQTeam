# Tasks — Segmentación y propagación en video con SAM 3

> Checklist de pasos atómicos. Marca `[x]` conforme avanzas.

- [ ] Cargar SAM 3 y validar inferencia sobre 1 frame (ya explorado en notebooks/fase_0)
- [ ] Probar el predictor de video de SAM 3 sobre un clip corto
- [ ] Lograr propagación de máscara con `object_id` estable entre frames
- [ ] Soportar inicialización por cajas (formato del contrato de YOLO)
- [ ] Validar en formato landscape (cámara externa)
- [ ] Validar en formato portrait (Meta Glasses)
- [ ] Medir tiempos de inferencia y memoria; aplicar mitigaciones (submuestreo/resize)
- [ ] Definir y serializar la salida en el formato del contrato
- [ ] Refactorizar a módulo en `src/`
- [ ] Verificar salida contra el contrato del `spec.md`
