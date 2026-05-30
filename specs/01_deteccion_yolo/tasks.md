# Tasks — Detección con YOLO

> Checklist de pasos atómicos. Marca `[x]` conforme avanzas. Este es el avance visible del equipo.

- [ ] Revisar el dataset etiquetado existente (Roboflow del equipo)
- [ ] Definir `configs/classes.json` con las clases canónicas
- [ ] Preparar split train/val/test
- [ ] Entrenar / afinar modelo YOLO
- [ ] Evaluar mAP en validación
- [ ] Implementar función `detect_frame(frame, cfg) -> List[Detection]` con el formato del contrato
- [ ] Probar sobre video landscape (cámara externa) y portrait (Meta Glasses)
- [ ] Refactorizar a módulo en `src/`
- [ ] Verificar salida contra el contrato del `spec.md`
