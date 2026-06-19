# Spec — Posesión vs control (`event_possession_refine`)

- **Tarea atómica:** `event_possession_refine`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Ronda de entregable de eventos (refinar detección +
  overlay de espectador) sobre la fase_5 ya completa.
- **Depende de:** `events_output_paths` (helper de rutas, hecho); reutiliza
  `events.compute_possession` (posesión por cercanía + histéresis) y
  `events_core` (`load_frame_objects`, `ball_centroid`).
- **Habilita:** el overlay de espectador (`event_broadcast_overlay`), cuyo panel de
  métricas mostrará posesión y control por separado de forma legible.

---

## 1. Requisito (historia de usuario)

> **Como** persona que analiza el partido,
> **quiero** que el sistema distinga **posesión** (qué robot está con el balón) de
> **control** (qué robot realmente lo conduce) y que la posesión **se actualice ante
> cambios radicales**,
> **para** que las métricas no sean engañosas (un robot junto a un balón muerto no debe
> figurar como dominante) y reflejen lo que ocurre en la cancha.

---

## 2. Motivación (por qué)

- **Una sola métrica es engañosa.** Hoy `compute_possession` reporta una única noción:
  el robot más cercano al balón dentro de un gate, con histéresis. Eso confunde dos cosas
  distintas:
  - un robot **junto a un balón quieto** (posesión nominal, pero no lo está jugando);
  - un robot **conduciendo el balón** (control real).
  El espectador necesita ver ambas, no una mezcla.
- **La posesión se queda "pegada".** La histéresis de N frames consecutivos suaviza el
  parpadeo, pero también **retrasa** cambios obvios: cuando el balón sale disparado a otro
  robot (cambio radical), la posesión debería actualizarse **de inmediato**, no esperar N
  frames.
- **El overlay lo exige.** El panel de métricas del overlay de espectador debe mostrar
  posesión y control de forma separada y creíble; sin esta tarea, solo hay una métrica
  ambigua.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Posesión** (se conserva): robot más cercano al balón dentro del gate
  (`gate_k·diagonal`), con histéresis, tal como la da `compute_possession`.
- **Control** (nuevo): subconjunto **más estricto** de la posesión. Un robot controla en
  un frame si (a) es el poseedor y (b) el balón **se mueve con él** — el balón se desplaza
  ≥ `move_thresh` dentro de una **ventana** de `control_window` frames mientras ese robot
  es el poseedor. Balón muerto junto a un robot ⇒ posesión **sin** control.
- **Actualización por cambio radical**: histéresis **adaptativa** sobre la serie cruda de
  posesión. Un cambio se confirma de inmediato (1 frame) si es **radical** —
  (a) **salto grande del balón** (desplazamiento > `jump_thresh` en 1–2 frames) **o**
  (b) **cambio de robot más cercano con margen claro** (el nuevo mucho más cerca que el
  anterior, factor `clear_factor`)— y conserva `min_frames` para cambios ambiguos.
- **Balón no visible** ⇒ ni posesión ni control (`None`); se distingue de "libre".
- **Resumen no engañoso**: separa `% posesión` por robot, `% control` por robot, `% libre`,
  `% no_visible`, y cuenta cambios de posesión y de control.
- **Salida** vía `events_paths(stem, "possession_refine", …)`: JSON + viz PNG (línea de
  tiempo posesión vs control).

### 3.2 Fuera de alcance

- **No** introduce equipos/bandos (no hay clase equipo): posesión/control por `obj_id`.
- **No** cambia ni deprecia `compute_possession` (`events.py`): esta tarea **construye
  encima** (la consume y re-deriva la serie cruda con el mismo `gate_k`).
- **No** conecta el resultado al overlay (es `event_broadcast_overlay`).
- **No** usa homografía/cm: el movimiento del balón se mide en **píxeles** (Capa A,
  universal). La variante en cm queda fuera.
- **No** corre en GPU ni re-infiere: solo lee el JSON de tracking.

---

## 4. Comportamiento esperado

- Un robot que está junto a un **balón quieto** aparece en **posesión** pero **no** en
  control durante ese tramo.
- Cuando el balón **sale disparado** a otro robot, la **posesión** cambia en ese instante
  (no tras `min_frames`), porque es un cambio radical.
- Un **parpadeo** de cercanía (dos robots casi a la misma distancia, sin movimiento claro
  del balón) **no** cambia la posesión hasta sostenerse `min_frames` (se evita el ruido).
- El `% control` de un robot es **≤** su `% posesión` (el control es más estricto).
- Sobre `IMG_9933_5m30`, el resumen debe ser coherente: la suma de control + posesión-sin-
  control + libre + no_visible cubre todos los frames; ningún robot con balón muerto domina
  el control.

---

## 5. Criterios de aceptación

1. Existe un módulo nuevo `src/core/event_possession_refine.py` que **consume**
   `compute_possession` y produce, por frame, **posesión** y **control** (por `obj_id`),
   más un resumen.
2. **Control** = posesión **y** balón en movimiento (≥ `move_thresh` en una ventana de
   `control_window` frames); balón quieto junto a un robot ⇒ no control.
3. La posesión se **actualiza de inmediato** ante un cambio radical (salto del balón o
   cambio de cercanía claro) y mantiene la histéresis `min_frames` para cambios ambiguos.
4. El resumen separa `% posesión` y `% control` por robot, más `% libre` y `% no_visible`,
   y cuenta cambios de posesión y de control; `% control ≤ % posesión` por robot.
5. Umbrales configurables por parámetro (`control_window`, `move_thresh`, `jump_thresh`,
   `clear_factor`, además de `gate_k`/`min_frames` heredados).
6. Salida vía `events_paths` (`kind="possession_refine"`): JSON + PNG (línea de tiempo
   posesión vs control).
7. Test manual sin GPU sobre `IMG_9933_5m30` que imprime posesión vs control, verifica las
   invariantes (`control ⊆ posesión`, cobertura de frames) y dibuja la línea de tiempo.

---

## 6. Notas / decisiones

- **Parámetros configurables** (defaults a fijar en el plan): `control_window` (~5 frames),
  `move_thresh` (fracción de la diagonal del robot por frame), `jump_thresh` (salto radical
  del balón, ~1 diagonal en 1–2 frames), `clear_factor` (cuánto más cerca debe estar el
  nuevo robot para ser cambio radical), heredados `gate_k` y `min_frames`.
- **Movimiento en píxeles (Capa A).** Universal y sin homografía; suficiente para distinguir
  balón quieto de balón conducido. La medición en cm (cuando haya homografía) se evaluará
  aparte si el overlay lo pide.
- **Compatibilidad:** `events.compute_possession` se conserva; esta tarea lo usa como
  cimiento. Si más adelante el overlay solo consume `event_possession_refine`, se evaluará
  qué exponer, fuera de esta tarea.
- **Sin ground-truth cuantitativo:** la validación es de coherencia (invariantes + viz),
  como el resto de fase_5 sin GT.
