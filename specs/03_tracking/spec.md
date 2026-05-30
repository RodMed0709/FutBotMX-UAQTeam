# Spec — Tracking (seguimiento de objetos)

- **Owner:** Victoria
- **Estado:** propuesto → (pendiente de aprobación)
- **Posición en el pipeline:** tercer eslabón. Recibe detecciones/máscaras por frame y produce trayectorias con identidad estable.

---

## Objetivo

Seguir los objetos entre frames manteniendo un **ID estable**, robusto ante **oclusión**.
Usar filtro de Kalman para predecir posición cuando la medición falta (objeto tapado) y estimar
velocidad. Es la pieza que convierte detecciones sueltas en trayectorias coherentes.

## Alcance

- Asociación de detecciones entre frames (mismo objeto → mismo `track_id`).
- Filtro de Kalman por objeto: predicción durante oclusión + suavizado.
- Estimación de velocidad por objeto (especialmente el balón).

## Fuera de alcance

- Detección y segmentación → tareas `01` y `02`.
- Clasificación de eventos (gol, pase, etc.) → post-MVP.
- Física avanzada (parábola del balón) y compensación de ego-motion → **segunda iteración**, no el MVP base.

## Contrato

**Entrada** — por frame, detecciones con caja + clase. Proviene del detector (tarea 01) y/o de
las máscaras de SAM 3 (tarea 02, usando su `bbox_xyxy`):
```json
{ "frame_idx": 0, "detections": [ {"class_name": "orange ball", "bbox_xyxy": [x1,y1,x2,y2]} ] }
```

**Salida** — tracks con identidad estable:
```json
{
  "frame_idx": 0,
  "tracks": [
    {
      "track_id": 7,
      "class_name": "orange ball",
      "bbox_xyxy": [x1, y1, x2, y2],
      "velocity": [vx, vy],
      "occluded": false
    }
  ]
}
```
- `track_id` estable a lo largo del video.
- `velocity` en píxeles/frame (la conversión a unidades físicas reales requiere ego-motion → iteración 2).
- `occluded` = `true` cuando la posición viene de la predicción de Kalman (sin medición).

## Criterios de éxito

- [ ] Asigna `track_id` estable mientras el objeto es visible.
- [ ] Mantiene el ID tras una oclusión corta (umbral de frames a fijar en `plan.md`).
- [ ] Estima velocidad del balón de forma razonable.
- [ ] Consume el formato del contrato de entrada (compatible con salidas de 01 y 02).
- [ ] Función reutilizable importable desde `src/`.

## Notas

- Empezar **simple**: Kalman 2D (posición + velocidad) + asociación por IoU/Hungarian. Validar antes de complicar.
- **Iteración 2 (no MVP):** modelo físico del balón (parábola en vuelo, velocidad casi constante en piso)
  y **ego-motion compensation** — la cámara/lentes se mueven, así que la velocidad en píxeles no es la real.
  Tenerlo en el radar desde el diseño, implementarlo después.
