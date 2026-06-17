# Spec — Tiro a gol vs gol (`event_shot_vs_goal`)

- **Tarea atómica:** `event_shot_vs_goal`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni modificar
  código aún.
- **Proceso al que pertenece:** Ronda de entregable de eventos (refinar detección +
  overlay de espectador) sobre la fase_5 ya completa.
- **Depende de:** `events_output_paths` (helper de rutas, hecho); reutiliza
  `event_goal_geometric` (gol en cm), `event_goals` (`_events_from_series`),
  `metric_positions` (posiciones en cm) y `field_template` (geometría de la cancha).
- **Habilita:** el overlay de espectador (`event_broadcast_overlay`), cuyo marcador y
  banner distinguirán *tiro* de *gol*.

---

## 1. Requisito (historia de usuario)

> **Como** persona que analiza el partido,
> **quiero** que el sistema distinga un **tiro a gol** de un **gol** real y deje de
> contar como gol cosas que solo rozan el bbox de la portería o pasan por el costado,
> **para** que el marcador y los eventos sean creíbles y no engañosos.

---

## 2. Motivación (por qué)

- **El bbox de la portería es demasiado permisivo.** La ruta px actual
  (`event_goals.compute_goal_zone_events`) cuenta "candidato a gol" cuando el centroide
  del balón entra al **bbox** de `yellow_zone`/`blue_zone`. Ese bbox suele ser grande:
  - un balón que pasa **por el costado** de la portería entra al bbox y se cuenta como gol;
  - un balón que solo **toca el borde** del bbox dispara el evento.
- **No hay matiz tiro vs gol.** Hoy todo es "candidato a gol": no existe la categoría
  intermedia *tiro a gol*, que es la mayoría de los lances reales.
- **La cancha pesa más que la portería (contradicción del draft).** El gol real se define
  por **cruzar la línea de gol** (la línea blanca frente a la portería que la homografía ya
  identifica), no por tocar la caja de la portería. El gol geométrico en cm
  (`event_goal_geometric`) va en esa dirección, pero no separa tiro de gol ni exige
  **dirección** del balón hacia la portería.
- **El overlay necesita la distinción.** El marcador solo debe subir con un **gol**; los
  **tiros** alimentan la lista dinámica de eventos. Sin esta tarea, el overlay no puede
  diferenciarlos.

---

## 3. Alcance

### 3.1 Dentro de alcance

- **Evento clasificado** con `tipo ∈ {"tiro", "gol"}` por lance, por zona
  (`"yellow"`/`"blue"`), con frames inicio/fin, duración y posición de referencia.
- **Definición de gol (cm, cámara superior) — ESTRICTA:** el centroide del balón **cruza la
  línea de gol real** (`x ≥ GOAL_LINE_X_RIGHT_CM` azul / `x ≤ GOAL_LINE_X_LEFT_CM` amarilla,
  con penetración opcional `goal_margin_cm`, default 0) **y** cae **dentro de la boca real**
  `y∈[_GOAL_TOP_Y_CM, _GOAL_BOTTOM_Y_CM]`. **Sin** ensanchar la boca ni correr la línea hacia
  el campo (esos dos "márgenes" eran la causa de los falsos goles).
- **Definición de tiro a gol (cm):** el balón entra a una **banda frente a la portería**
  (profundidad `tiro_depth_cm` antes de la línea, default **15 cm**), dentro de la boca
  **±`side_cm`** (tolerancia para tiros al poste, default 12 cm), **sin** cumplir el test
  estricto de gol. Incluye el balón que pasó la línea pero fuera de la boca (tiro al poste) y
  el que se queda corto.
- **Sin dirección obligatoria:** un tiro al poste se queda **estático** y aun así es un tiro;
  exigir velocidad hacia la portería lo descartaba. La región (banda + boca) y el debounce ya
  filtran los pases laterales.
- **Tolerancia de huecos de detección (`gap_frames`, default 20):** el balón se detecta de
  forma intermitente; un balón parado frente a la portería parpadea y se fragmentaba en muchos
  lances. Los huecos de ausencia ≤ `gap_frames` se **fusionan** en un solo lance.
- **Ruta px (universal, refinada):** para videos sin cm fiable, refinar la clasificación
  sobre el bbox de la zona — **encoger el bbox** por un margen y exigir que el centroide
  rebase **3/4 de la profundidad** del bbox hacia la pared = gol; entrar sin alcanzar 3/4 =
  tiro; tocar solo el borde = nada. Queda como **proxy** (cámaras parciales).
- **Sin doble conteo:** un lance que termina en gol no emite además un tiro suelto
  (debounce/cierre/cooldown reutilizando `_events_from_series`).
- **Resumen** con conteos por tipo y por zona (`tiros`, `goles`), eventos ordenados,
  fps/segundos (estilo de los resúmenes fase_5).
- **Salida** vía `events_paths(stem, "shot_vs_goal", ...)`: JSON + viz PNG.

### 3.2 Fuera de alcance

- **No** conecta el resultado al overlay (es `event_broadcast_overlay`).
- **No** elimina ni cambia la API pública de `event_goals` (T2) ni `event_goal_geometric`:
  esta tarea **construye encima** (puede reusarlos); su deprecación, si procede, es otra
  decisión.
- **No** detecta otros eventos (fuera, área chica, etc. → `event_field_violations`).
- **No** introduce equipos/bandos ni cambia el tracking.
- **No** corre en GPU ni re-infiere: solo lee el JSON (+ T3 cm).

---

## 4. Comportamiento esperado

- Sobre `IMG_9933_5m30` (clip validado a mano por el equipo): el resultado correcto es
  **1 gol + 3 tiros** (todos zona azul). El gol geométrico laxo y T2 contaban 3/2 "goles":
  dos eran un tiro al poste (balón pasado la línea pero fuera de la boca) y un tiro corto
  (balón que no llega a la línea); el módulo estricto los reclasifica como **tiro**.
- Un balón que pasa la línea **fuera de la boca real** (poste) **no** es gol (es tiro). Un
  balón que se **queda corto** de la línea tampoco es gol (es tiro).
- El número de **goles** del módulo estricto debe ser ≤ el del gol geométrico laxo y el de
  los candidatos de T2 (este módulo es más estricto, nunca cuenta más goles).

---

## 5. Criterios de aceptación

1. Existe un módulo nuevo `src/core/event_shot_goal.py` que produce eventos clasificados
   `tipo ∈ {"tiro","gol"}` por zona, reutilizando `_events_from_series`,
   `metric_positions` y la geometría de `field_template` (sin duplicar el motor de estados).
2. **Gol** = cruce de la línea de gol real en cm **dentro de la boca real** (estricto, sin
   ensanchar boca ni correr línea); **tiro** = banda `tiro_depth_cm` frente a la línea (boca
   ±`side_cm`) sin cumplir el test de gol. **No** se exige dirección.
3. Los **huecos de detección** ≤ `gap_frames` se fusionan en un solo lance (balón parpadeante).
4. La ruta **px** clasifica tiro/gol con bbox encogido + regla de 3/4 (proxy universal,
   conservador: subdetecta goles).
5. No hay doble conteo de un mismo lance (gol no emite además tiro).
6. El resumen reporta `tiros`/`goles` por zona + duraciones; salida vía `events_paths`
   (`kind="shot_vs_goal"`), JSON + PNG.
7. Test manual sin GPU sobre `IMG_9933_5m30` que verifica el ground truth **1 gol + 3 tiros**,
   compara (informativo) con el gol geométrico laxo y visualiza la línea de tiempo tiro-vs-gol.

---

## 6. Notas / decisiones

- **Parámetros configurables** (defaults): `tiro_depth_cm` (=15), `side_cm` (=12),
  `goal_margin_cm` (=0), `gap_frames` (=20), `three_quarter_frac` (=0.75, ruta px),
  `margin_px` (=0, ruta px), `min_frames`, `exit_frames`, `cooldown_frames`.
- **Refinamiento basado en datos (`IMG_9933_5m30`):** la regla original (dirección
  obligatoria + márgenes que ensanchaban boca y línea) daba 3 falsos goles. La validación a
  mano del equipo (1 gol + 3 tiros) fijó la semántica estricta + tolerancia de huecos de
  arriba. El gol geométrico laxo (`event_goal_geometric`) conserva su regla previa (queda como
  candidato a deprecar fuera de esta tarea).
- **Cámara superior = cm (autoridad); px = proxy.** El gol "real" vive en cm; la ruta px se
  conserva para tomas parciales pero su clasificación es indicativa.
- **Compatibilidad:** `event_goals` y `event_goal_geometric` se conservan; esta tarea los usa
  como cimiento. Si más adelante el overlay solo consume `event_shot_goal`, se evaluará
  deprecar la salida cruda de T2, fuera de esta tarea.
