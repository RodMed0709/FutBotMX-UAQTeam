# Spec — Violaciones de campo (`event_field_violations`)

- **Tarea atómica:** `event_field_violations`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Ronda de entregable de eventos (refinar detección +
  overlay de espectador) sobre la fase_5 ya completa.
- **Depende de:** `events_output_paths` (hecho); reutiliza `compute_metric_positions`
  (posiciones en cm, Capa B), `field_template` (geometría de la cancha),
  `events_core`/`events` (`load_frame_objects`, `ball_centroid`, bboxes de robot) y
  `_events_from_series` (segmentación de episodios).
- **Habilita:** el overlay de espectador (`event_broadcast_overlay`), cuya lista dinámica de
  eventos mostrará fueras, entradas a área chica y (con margen) lack-of-progress / pushing.

---

## 1. Requisito (historia de usuario)

> **Como** persona que analiza el partido,
> **quiero** detectar **fueras** (robot que sale del campo o entra al área chica),
> **lack-of-progress** (juego estancado) y **pushing** (empuje dentro del área chica),
> **para** completar la lista de eventos del partido más allá de goles/tiros y posesión.

---

## 2. Motivación (por qué)

- **Faltan eventos del reglamento.** Hoy fase_5 detecta goles/tiros (gol/tiro) y posesión/
  control, pero no las **infracciones de campo**: un robot que sale de la cancha o invade el
  área chica (ambos = *fuera*), un partido **estancado**, o un **empuje** en el área chica.
- **Geometría ya disponible.** La homografía (Capa B) da posiciones en cm y `field_template`
  ya define el rectángulo de líneas blancas y el área chica (forma de D): se puede decidir
  geométricamente si un robot está fuera o dentro del área chica.
- **Algunos eventos no son 100% detectables.** Lack-of-progress (lo fiable sería el audio) y
  pushing (sin clase de contacto) solo se pueden **aproximar**; se entregan como
  **probabilísticos** (confianza heurística) con margen de error explícito.
- **El overlay los necesita.** La lista dinámica de eventos del overlay de espectador se nutre
  de estos eventos.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **`fuera`** (Capa B, cm) — un único tipo con `causa ∈ {"salida_campo", "area_chica"}`:
  - **salida_campo**: el centroide de un robot sale del rectángulo de líneas blancas
    (`x∈[12,231]`, `y∈[12,170]`), **salvo** dentro de la boca de portería (`y∈[61,121]` en
    `x<12`/`x>231`), que no cuenta.
  - **area_chica**: el centroide de un robot **entra** al polígono del área chica (forma de D)
    de cualquier portería — **también cuenta como fuera**.
  - Episodio **sostenido** (debounce); termina cuando el robot vuelve a posición legal (el
    juego no se pausa: el robot reaparece). Uno por episodio por robot.
- **`lack_of_progress`** (Capa A, px, **probabilístico**): el balón se mantiene casi inmóvil
  (desplazamiento < umbral) durante una ventana larga (≥ `min_secs`); se emite con
  **`probabilidad`** que crece con la duración y la quietud. Indicativo.
- **`pushing`** (Capa A, px, **probabilístico**): **solo dentro del área chica** — dos robots
  en contacto sostenido (bboxes solapados / centroides muy cerca), **mientras el contacto
  ocurre en el área chica**. Un empuje fuera del área chica **no** es pushing. `probabilidad`
  crece con la fuerza del contacto (solape/cercanía) y la duración; el **desplazamiento** del
  empujado suma como **bonus, no como requisito** (validado con datos: los empujones tipo
  "sumo" dejan a los robots casi inmóviles, exigir desplazamiento los perdía). Indicativo.
- **Salida de evento**: tipo, `causa` (si aplica), frames inicio/fin, duración, `obj_ids`
  involucrados, portería/zona si aplica, posición de referencia (cm o px), y `probabilidad`
  (1.0 para `fuera`; <1 para `lack_of_progress`/`pushing`).
- **Resumen** con conteos por tipo/causa y duraciones; salida vía
  `events_paths(stem, "field_violations", …)`: JSON + viz PNG.

### 3.2 Fuera de alcance

- **No** introduce equipos/bandos: los actores son `obj_id` de robot.
- **No** introduce la clase humano/brazo ⇒ **interferencia humana** queda fuera (inviable
  ahora).
- **No** conecta el resultado al overlay (es `event_broadcast_overlay`).
- **No** corre en GPU ni re-infiere: lee el JSON de tracking (+ `metric_positions` para cm).
- Fuera/área chica **requieren** cámara superior + homografía fiable; en tomas sin cm fiable
  se **omiten** esos dos (lack-of-progress / pushing siguen disponibles en px).

---

## 4. Comportamiento esperado

- Un robot cuyo centroide sale del rectángulo de líneas blancas ⇒ `fuera` con
  `causa="salida_campo"`; al reingresar, el episodio cierra.
- Un robot cuyo centroide entra al área chica ⇒ `fuera` con `causa="area_chica"`.
- Un robot que cruza `x<12`/`x>231` **dentro de la boca** (yendo al gol) **no** dispara fuera.
- Sobre `IMG_9933_5m30`, el tramo de balón parado (≈ f502-676, ya visto en
  `event_shot_vs_goal`) debe aparecer como **lack_of_progress** con probabilidad alta.
- Pushing solo aparece si el contacto robot-robot ocurre **dentro del área chica**.

---

## 5. Criterios de aceptación

1. Existe un módulo nuevo `src/core/event_field_violations.py` que produce eventos
   `tipo ∈ {"fuera","lack_of_progress","pushing"}`, con `fuera` distinguiendo
   `causa ∈ {"salida_campo","area_chica"}`.
2. **Fuera** se decide en cm con la geometría de `field_template` (rectángulo de líneas +
   polígono del área chica), con la excepción de la boca de portería; segmentado por episodios
   con `_events_from_series`.
3. **Lack-of-progress** y **pushing** se calculan en px y llevan `probabilidad ∈ (0,1]`;
   pushing exige que el contacto ocurra dentro del área chica.
4. Cada evento reporta actores (`obj_ids`), frames, duración, referencia y probabilidad; el
   resumen agrega por tipo/causa.
5. Umbrales configurables (tolerancia de líneas, ventana/umbral de quietud, solape/ventana de
   pushing, `min_frames`/`exit_frames`/`cooldown`/`gap_frames`).
6. Salida vía `events_paths` (`kind="field_violations"`): JSON + PNG.
7. Test manual sin GPU sobre `IMG_9933_5m30` que valida invariantes y la coherencia esperada
   (el balón parado sale como lack-of-progress) y dibuja una viz.

---

## 6. Notas / decisiones

- **`fuera` unifica salida del campo y entrada al área chica** (decisión del usuario: entrar
  al área chica *también* es fuera), conservando `causa` para el análisis.
- **`pushing` es estrictamente dentro del área chica** (decisión del usuario): el contacto en
  cualquier otro contexto no se reporta como pushing.
- **Probabilístico = confianza heurística** acotada **(0, 0.95]** (nunca 1.0, que se reserva a
  los geométricos `fuera`), no una probabilidad calibrada; documentada y configurable.
  Lack-of-progress y pushing se entregan como indicativos.
- **Pushing sin requisito de desplazamiento** (refinamiento por datos): los robots son grandes
  y en un empujón apenas se desplazan respecto a su tamaño; el contacto sostenido dentro del
  área chica es la señal, el movimiento solo sube la confianza.
- **Compatibilidad:** no se tocan los módulos consumidos (`metric_positions`, `field_template`,
  `events`/`events_core`); esta tarea construye encima.
- **Sin ground-truth cuantitativo:** validación de coherencia (invariantes + viz), como el
  resto de fase_5 sin GT.
