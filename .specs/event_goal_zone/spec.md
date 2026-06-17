# Spec — `event_goal_zone`: balón en zona de gol (fase_5, Capa A)

## Contexto

Segunda tarea de **fase_5 (análisis de eventos)**, **Capa A — relacional (en píxeles),
universal**: compara la posición del balón con el bbox de las zonas de gol **en el mismo
frame**, así que funciona sobre el JSON de **cualquier** video sin homografía ni GPU.

Esta tarea **promueve** la carga compartida del JSON (`FrameObject`/`load_frame_objects`)
de `events.py` a un módulo común `events_core.py`, que ya reusan T1 (posesión) y T2.

## Objetivo

Detectar **candidatos a gol**: instantes en que el **balón entra en una zona de gol**
(`yellow_zone`/`blue_zone`), agregándolos en **eventos** discretos (con debounce y
cooldown para no contar parpadeos ni repetir un mismo lance).

## Entrada

- JSON de tracking de `run_inference(mode="tracking", ...)` (vía `load_frame_objects`).
  De **cualquier** config 2×2. No re-infiere modelos.
- Parámetros (defaults razonables):
  - `margin`: holgura (px) al test punto-en-bbox de la zona (default 0).
  - `min_frames`: frames consecutivos del balón dentro para **abrir** un evento.
  - `exit_frames`: frames consecutivos fuera para **cerrar** el evento.
  - `cooldown_frames`: refractario tras cerrar, antes de admitir un nuevo evento en la
    misma zona (evita doble conteo del mismo lance).
  - `fps`: para reportar duraciones en segundos (del JSON si está).

## Salida

Estructura Python (dict/dataclass) con:
- `eventos`: lista de `{zona: "yellow" | "blue", frame_inicio, frame_fin, dur_frames,
  dur_s}` (un evento = una entrada sostenida del balón en la zona).
- `resumen`: nº de eventos por zona y total.
- Opcionalmente se **escribe a un JSON**.

## Método

Por cada `frame_index`:
1. **Balón**: centroide del `orange_ball` de mayor `score` (`None` si no hay → fuera de zona).
2. **Zonas presentes**: bboxes de `yellow_zone` y `blue_zone` ese frame (puede faltar una;
   solo se procesan las presentes). Si hay **varios tracks** de una zona, el balón está
   "dentro" si cae en **cualquiera**.
3. **Dentro/fuera** por zona: el centroide del balón cae dentro del bbox (± `margin`).
4. **Máquina de estados por zona** sobre la serie dentro/fuera:
   - fuera→dentro sostenido `min_frames` ⇒ **abre** evento (inicio = primer frame dentro).
   - dentro→fuera sostenido `exit_frames` ⇒ **cierra** evento (fin = último frame dentro).
   - tras cerrar, `cooldown_frames` de refractario antes de admitir otro evento en esa zona.
5. **Agregación**: lista de eventos + conteos por zona.

Todo es numpy/CPU; reusa `events_core` (carga por-frame + balón).

## No-objetivos

- **Gol geométrico** real (balón cruzando la línea de gol en cm) — refinamiento de Capa B
  (cámara superior + homografía), no aquí. Por eso se llama **"candidato a gol"**.
- **Atribución** (qué robot marcó, posesión+gol) — después / T7.
- **Video overlay** — T7. (Aquí solo viz de validación en el test.)

## Verificación

- **Smoke funcional** sobre un JSON existente en `outputs/` (reusa el harness de T1 que
  imprime video/duración), en local sin GPU:
  - corre sin error; con `IMG_9780` (solo `yellow_zone`) detecta eventos solo de esa zona,
    sin fallar por `blue_zone` ausente.
  - casos borde: balón nunca en zona (0 eventos); balón dentro de forma intermitente
    (debounce evita múltiples eventos); zona ausente (se omite).
- **Visualización de validación**: línea de tiempo marcando los intervalos balón-en-zona
  (color por zona) + resumen impreso. Sin overlay sobre el video (eso es T7).
- Lint limpio (`ruff`). T1 (`event_possession`) sigue funcionando tras mover la base a
  `events_core`.
