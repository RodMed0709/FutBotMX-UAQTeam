# Arquitectura del proyecto y roadmap a la entrega

> **Referencia viva del estado del proyecto.** Resume el pipeline real en `src/`, qué
> piezas son intercambiables hoy, dónde encaja Kalman (fase_6) y los próximos pasos para
> cerrar la entrega.
>
> - **Fecha de este corte:** 2026-06-17
> - **Entrega objetivo:** 2026-06-19
> - **Entregable principal:** el video de espectador `event_broadcast_overlay.render_broadcast_overlay`.

---

## 1) Pipeline principal — detección/segmentación → tracking (GPU/pod)

```
                          run_inference  (inference.py)  ── facade único por video
                                  │  mode = "segmentation" | "tracking"
              ┌───────────────────┴────────────────────┐
              ▼                                          ▼
        pipeline.py                                 tracking.py
     (per-frame, obj_id                          (streaming, obj_id
      NO estable)                                  ESTABLE)
              │                                          │
              ▼                                          ▼
     ╔═══════════════════╗                     ╔═══════════════════╗   ╔══════════════════╗
     ║  DETECTOR  ⇆      ║  get_detector(...)  ║  DETECTOR  ⇆      ║──▶║  TRACKER  ⇆      ║
     ║  • sam3_text      ║                     ║  • sam3_text      ║   ║  • bytetrack     ║
     ║  • yolo_sam3      ║                     ║  • yolo_sam3      ║   ║  • botsort       ║
     ╚═══════════════════╝                     ╚═══════════════════╝   ╚══════════════════╝
              │                                          │ get_tracker(...)
              ▼                                          ▼
     overlay.py / video                        ┌──────────────────────────────┐
     (mp4 seg + JSON)                          │  CONTRATO: tracking JSON      │  ◀── la "moneda"
                                               │  Track / TrackObservation     │      de todo el post
                                               │  (obj_id estable, [+ masks])  │
                                               └──────────────────────────────┘
```

**Desmontable aquí (limpio, vía registro):**
- `DETECTOR` → `get_detector("sam3_text" | "yolo_sam3")` ✅
- `TRACKER` → `get_tracker("bytetrack" | "botsort")` ✅

El `tracking JSON` es la **frontera dura**: todo el post lee de ahí y no sabe qué detector/tracker
lo generó.

## 2) Post-proceso — homografía → eventos (CPU local, lee el JSON)

```
            ┌──────────────────────────────┐
            │  tracking JSON               │
            └───────────────┬──────────────┘
                            ▼
                ╔════════════════════════════╗
                ║  HOMOGRAFÍA  ⇆ (flag)      ║  metric_positions.compute_metric_positions(
                ║  • "lines"  (v2_07,        ║      homography="lines" | "masks")
                ║     VideoHomographyLines)  ║
                ║  • "masks"  (legacy)       ║   ◀── BASE COMPARTIDA del post
                ╚════════════════════════════╝
                            │  xy_cm por frame/obj_id  +  H_por_frame
            ┌───────────────┼───────────────────────────┬───────────────────┐
            ▼               ▼                            ▼                   ▼
   ╔═════════════════╗  ┌─────────────┐         ┌──────────────────┐  ┌─────────────┐
   ║ ESTIMADOR DE    ║  │ zonas       │         │ EVENTOS          │  │ heatmap     │
   ║ ESTADO/CINEMÁT. ║  │ (field_     │         │ • shot_vs_goal   │  │ (metric_    │
   ║ ⇆ (HOY fijo)    ║  │  zones)     │         │ • goal_geometric │  │  heatmap)   │
   ║ • T4 dif.finitas║  └─────────────┘         │ • possession_ref │  └─────────────┘
   ║   (metric_kinem)║                          │ • field_violat.  │
   ║ • Kalman (f6) ✗ ║                          └──────────────────┘
   ╚═════════════════╝                                   │
            └───────────────┬──────────────────────────────┘
                            ▼
              ┌──────────────────────────────────────┐
              │  event_broadcast_overlay              │  ◀── EL ENTREGABLE
              │  (scoreboard, banner gol, posesión,   │
              │   minimap cenital, heatmap, homog.)   │
              └──────────────────────────────────────┘
```

## 3) ¿Qué tan desmontable está hoy cada pieza?

| Pieza | Mecanismo de cambio | Estado |
|---|---|---|
| Detector | registro `get_detector` | ✅ **limpio** |
| Tracker | registro `get_tracker` | ✅ **limpio** |
| Homografía | flag `homography="lines"\|"masks"` | 🟡 **flag** (no registro, pero funciona) |
| Detectores de eventos | se elige qué módulo llamar en el overlay | 🟡 **call-site** (no hay registro) |
| Fuente de gol (estricto vs geométrico) | qué función llama el overlay | 🟡 **call-site** |
| **Estimador de estado/cinemática** | **hardcodeado** (broadcast no lo parametriza) | 🔴 **NO desmontable** |

## 4) Dónde encaja Kalman (fase_6)

Kalman **no es otra homografía** y **no mejora la homografía**. Es un **estimador de estado**
que corre *río abajo*, sobre la misma `xy_cm` de `metric_positions` (que ya usa la homografía por
líneas de `v2_07`). Hoy `kalman_kinematics.run_kalman_on_track` existe y está validado, pero **el
broadcast NO lo enchufa** (ni `event_broadcast_overlay` ni `metric_positions` importan Kalman).

**Qué aporta** (vs diferencias finitas T4): velocidad más suave/física, **relleno de oclusión**
(predict-only) y rechazo robusto de outliers (gating Mahalanobis en vez del corte duro de 300 cm/s).
**Qué NO toca:** el sesgo absoluto de landmarks (~9–23 cm) sigue igual; `sigma_z` modela el ruido
*temporal* frame-a-frame, no el sesgo absoluto.

Patrón para volverlo desmontable (espejo de `homography=`):

```
metric_positions ─▶ ESTIMADOR DE ESTADO ⇆ (flag nuevo)
                      • kinematics="finite"  (T4, actual)
                      • kinematics="kalman"  (fase_6: velocidad suave + relleno oclusión)
                    ─▶ eventos / overlay
```

---

## 5) Estado actual (resumen)

**Cerrado / entregable:**
- Pipeline base + YOLO+SAM3 + benchmark sin-GT.
- Consolidación de homografía: el broadcast usa el camino **por líneas**. Cerrada para el entregable.
- fase_5 análisis de eventos: goles, posesión, zonas, heatmap, broadcast overlay.

**En curso / experimental (no bloqueante para la entrega):**
- Evaluación SAM3-only (mIoU/Dice vs GT) — **pausada**, espera anotaciones del equipo.
- Kalman (fase_6) — **experimental**, validado, **no integrado** al entregable; NIS tuning fino pendiente.
- `event_goal_geometric` es laxo (falsos positivos); el overlay puede preferir la fuente estricta.

**TODO de infraestructura (cada uno = tarea SDD atómica):**
- Limpiar el parche `sys.path` en `testing/` (rápido).
- `bootstrap_data` (descarga idempotente de `data/raw` + `assets/sam3`; ubicación RunPod sin decidir).
- `minimap_pipeline` → camino por líneas — **opcional/diferido** (el broadcast no lo usa).

---

## 6) Roadmap a la entrega del 19 (propuesta — decisiones pendientes marcadas con ❓)

Orden por prioridad para cerrar un **broadcast overlay pulido y reproducible**. Dos días.

1. **Congelar el clip y la corrida de referencia del broadcast.** Reproducir
   `03_broadcast_overlay_demo.ipynb` sobre el clip de referencia (`IMG_9933_5m30`) con el clip
   **crudo** (no el segmentado) y verificar el render de punta a punta.
2. **❓ Decisión: fuente de gol en el entregable.** Estricta (`event_shot_goal`) vs geométrica
   (`event_goal_geometric`, más falsos positivos). Recomendado: **estricta** por default.
3. **❓ Decisión: ¿Kalman entra al entregable del 19 o se deja como mejora post-entrega?**
   - **(a) No entra (recomendado para el 19):** se entrega lo ya validado; Kalman queda como
     trabajo demostrado en fase_6. Cero riesgo.
   - **(b) Comparativa con/sin Kalman:** notebook en fase_5 que use lo ya empaquetado (sin tocar
     `src/`) para mostrar velocidad/oclusión con y sin Kalman. Valor de presentación, riesgo bajo.
   - **(c) Integrarlo al broadcast:** requiere SDD (estimador de estado desmontable, `kinematics=`).
     Mayor valor arquitectónico pero **riesgo alto en 2 días** → mejor post-entrega.
4. **Pulido de presentación del broadcast** (si sobra tiempo): textos, banner, leyendas del minimap.
5. **Post-entrega (no para el 19):** refactor de arquitectura para subir homografía y estimador de
   estado al nivel de "registro" (como detector/tracker), `minimap_pipeline` por líneas, `bootstrap_data`.

> **Pendiente de confirmar con el equipo:** los puntos ❓ (2) y (3) antes de fijar el alcance final.
