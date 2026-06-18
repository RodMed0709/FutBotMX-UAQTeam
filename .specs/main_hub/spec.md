# spec.md — `main_hub`

> Paso 2 del SDD. Describe **qué** se construye y **por qué**. No define implementación
> (eso es `plan.md`). Documento en español.

## 1. Resumen

Crear `main.py` en la raíz del proyecto: un **hub de consola** que ejecuta el pipeline
**end-to-end** sobre **un video** (no un clip; el `main` solo **lee** el video, no lo
recorta) y produce, como entregable principal, el **video de espectador** (broadcast).

El hub es **interactivo** (pregunta qué piezas usar) o **automático** (`--default`), es
**idempotente** (reanuda desde lo ya corrido sin rehacer lo caro) y **centraliza** todas
sus salidas en una carpeta dedicada por video.

## 2. Motivación / por qué

- Hoy el pipeline completo se ejecuta encadenando notebooks/llamadas sueltas
  (`run_inference` → homografía → eventos → `render_broadcast_overlay`). Para la
  convocatoria se necesita **un punto de entrada único y reproducible** que cualquiera
  pueda correr con un comando.
- La inferencia (SAM3/GPU) es **cara**: si una corrida se interrumpe, rehacerla desde
  cero es costoso. El hub debe **reaprovechar** el trabajo ya hecho.
- Las salidas hoy quedan **dispersas** (`outputs/inference/…`, `outputs/eventos/…`) y
  cuesta saber qué se generó. El hub debe **reportar con claridad** dónde quedó cada
  artefacto, **sin moverlos** (mover rompería el skip-done nativo y, con él, la
  idempotencia: la siguiente corrida re-haría la inferencia).

## 3. Alcance

### 3.1 Dentro del alcance
- Un archivo nuevo **`main.py` en la raíz**.
- Orquestación de las fachadas **existentes** de `src/` (no se reimplementa lógica de
  inferencia/eventos).
- Interfaz de **consola bonita** (menús, color, barra de progreso).
- Idempotencia "A" (ver §6) y **reporte claro** de la ubicación de salidas (ver §7).
- Añadir a `requirements.txt` la(s) librería(s) de consola.

### 3.2 Fuera del alcance
- **No** se modifican módulos de `src/` (el hub solo orquesta fachadas).
- **No** mueve ni reubica artefactos: cada fachada escribe en su ruta nativa; el hub
  solo **reporta** dónde quedaron (mover rompería la idempotencia).
- **No** genera ni recorta clips: usa el video de entrada tal cual.
- **No** añade idempotencia granular del post-proceso CPU (homografía/eventos se
  recalculan; ver §6, "Idempotencia A").
- **No** entrena ni hace fine-tuning; **no** descarga datos (eso es `bootstrap_data`).

## 4. Historias de usuario

- **HU-1 (corrida guiada).** Como integrante del equipo, quiero ejecutar
  `python main.py <video>` y que el programa me **pregunte** qué detector/segmentador y
  qué tracker usar y si quiero los overlays individuales, para obtener el video de
  espectador sin recordar las llamadas internas.
- **HU-2 (corrida por defecto).** Como integrante del equipo, quiero
  `python main.py <video> --default` para correr la **configuración por defecto del
  proyecto** sin que me pregunte nada.
- **HU-3 (reanudar).** Como integrante del equipo, si una corrida se interrumpió, quiero
  volver a lanzar el mismo comando y que **no rehaga la inferencia** ya completada ni el
  broadcast ya generado, ahorrando tiempo/GPU.
- **HU-4 (entrada robusta).** Como integrante del equipo, quiero que el programa
  **valide** que la ruta es un video usable y me dé un **error claro** si no lo es,
  antes de cargar nada pesado.
- **HU-5 (saber dónde quedó todo).** Como integrante del equipo, quiero que al terminar
  el programa me **muestre en pantalla la ruta exacta** de cada artefacto generado o
  reusado, para localizarlos y compartirlos sin adivinar.
- **HU-6 (vista de cámara correcta).** Como integrante del equipo, quiero declarar si el
  video es de **cámara superior** (campo completo) o **genérico**, porque homografía,
  eventos y broadcast **solo** tienen sentido sobre cámara superior; en un clip genérico
  quiero que el `main` corra el **pipeline base** (detección/segmentación/tracking) y
  **omita** el post-proceso métrico con un aviso claro, en vez de producir resultados sin
  sentido.

## 5. Requisitos funcionales

### 5.1 Invocación y parámetros
- **RF-1.** Firma: `python main.py <ruta_video> [--default] [--overwrite]
  [--vista {superior,generica}]`.
- **RF-2.** `<ruta_video>` es **posicional y obligatoria** (acepta ruta relativa a la
  raíz del proyecto o absoluta).
- **RF-3.** `--default`: corre **sin preguntar nada** la inferencia por defecto del
  proyecto (detector del config activo, fallback `sam3_text`; tracker `bytetrack`),
  overlays individuales **OFF**, **vista `superior`** (RF-23), y los fijos de RF-9.
- **RF-3b.** `--vista {superior,generica}`: declara el ángulo de cámara (RF-23). Si se
  pasa, **prevalece** y no se pregunta; si se omite, se decide por modo (interactivo
  pregunta; `--default`/no-TTY ⇒ `superior`).
- **RF-4.** `--overwrite`: fuerza re-correr **todo** aunque existan artefactos previos
  (desactiva la idempotencia de §6).

### 5.2 Validación de la entrada (antes de cargar nada pesado)
- **RF-5.** Verificar, en orden, que la ruta: **existe**, es **archivo**, tiene
  **extensión de video** (`.MOV/.mov/.mp4`…), y **se abre con cv2** con **≥ 1 frame**.
  Si algo falla → mensaje claro y **salida con código ≠ 0**.
- **RF-6.** Si el video es **largo** (supera ~1 min / un umbral de frames configurable),
  **advertir** que el pipeline es costoso y que se prefieren clips, pero **continuar**
  (el `main` no recorta).

### 5.3 Selección de piezas
- **RF-7.** En modo interactivo, preguntar **en orden**: (a) **detector/segmentador**
  (`sam3_text` | `yolo_sam3`); (b) **tracker** (`bytetrack` | `botsort`); (c) **vista de
  cámara** (`superior` | `generica`), salvo que venga por `--vista` (RF-3b/RF-23); (d) si
  se generan los **overlays individuales** de segmentación y tracking (sí/no).
- **RF-8.** Las opciones ofrecidas se derivan de los **registros existentes**
  (`get_detector`/`get_tracker`), de modo que añadir una pieza no rompe el `main`.
- **RF-9.** **Fijos por defecto** (no se preguntan en ningún modo): homografía por
  **líneas**, **Kalman ON**, **gol estricto**, **broadcast layout 2**.

### 5.4 Ejecución del pipeline
- **RF-10.** El hub corre **siempre en modo `tracking`** para el flujo principal
  (homografía/eventos requieren `obj_id` estables).
- **RF-11.** Secuencia de etapas: **inferencia (tracking)** → [**overlays individuales**
  opcionales] → **homografía/métrica** → **eventos** → **broadcast (video espectador)**.
  Las etapas de **homografía/métrica/eventos/broadcast** están **condicionadas** a la
  vista de cámara (§5.6): solo corren si la vista es `superior`.
- **RF-12.** Overlay individual de **tracking** vía `render_obj_id_overlay`; overlay de
  **segmentación** vía una corrida `mode="segmentation"` adicional. Solo si el usuario
  los pidió (RF-7c).
- **RF-13.** La **salida final destacada** es el **video de espectador** (broadcast);
  el programa imprime su ruta al terminar.

### 5.5 Experiencia de consola
- **RF-14.** Interfaz **amigable**: menús de selección, texto con color/paneles, y
  **barra de progreso** durante las etapas pesadas.
- **RF-15.** Resumen final claro: qué se reusó, qué se generó y **dónde** quedó cada
  artefacto.

### 5.6 Vista de cámara (homografía/eventos solo en cámara superior)
- **RF-23.** El pipeline base (detección/segmentación/tracking) aplica a **cualquier**
  clip, pero **homografía, eventos y broadcast** requieren **cámara superior** (campo
  completo). El `main` maneja una **vista declarada**: `superior` | `generica`.
- **RF-24.** Si la vista es **`generica`**: el `main` corre el pipeline base (+overlays si
  se pidieron) y **omite** homografía/eventos/broadcast, marcándolos `omitido` con el
  motivo "vista genérica" en el reporte.
- **RF-25.** Si la vista es **`superior`**: el `main` intenta el post-proceso. **Valida**
  la homografía: si sale **degradada** (no se reconoce el campo → el clip no parece
  cámara superior), **avisa** y **omite** eventos/broadcast (no produce un broadcast
  degradado). Si la homografía es válida, genera el broadcast normalmente.

## 6. Idempotencia (modelo "A")

- **RF-16.** El hub reaprovecha lo **caro** ya hecho, evaluándolo contra las **rutas
  nativas** de cada fachada (las mismas que usa el skip-done existente; ver §7):
  - Si **existe el tracking JSON** del video en su ruta nativa
    (`outputs/inference/<run_label>/<stem>/<stem>.json`) → **no re-infiere**; lo reusa.
  - Si **existe el broadcast `.mp4`** en su ruta nativa
    (`outputs/eventos/<stem>/<stem>_broadcast.mp4`) → **lo salta**.
- **RF-17.** El post-proceso **CPU** (homografía/métrica/eventos) **se recalcula** en
  cada corrida (es barato); **no** se persiste/saltea de forma granular.
- **RF-18.** `--overwrite` ignora RF-16 y rehace todo. Una corrida nunca **borra**
  artefactos válidos salvo que se reemplacen por `--overwrite`.

## 7. Ubicación y reporte de salidas (sin mover)

- **RF-19.** El hub **no mueve ni reubica** artefactos: cada fachada escribe en su **ruta
  de dominio nativa** (`outputs/inference/<run_label>/<stem>/…`, `outputs/eventos/<stem>/…`).
  Esto **preserva el skip-done nativo** y, con él, la idempotencia A (§6).
- **RF-20.** Al terminar —y también cuando una etapa se **reusa** por idempotencia— el
  hub **imprime en pantalla** un resumen con la **ruta exacta** de cada artefacto
  (tracking JSON, video de tracking, overlays individuales, broadcast `.mp4` + PNG de
  muestra), indicando si fue **generado** o **reusado**.
- **RF-21.** **Clip crudo para el broadcast:** el broadcast debe usar el video **sin
  máscaras**, pero por defecto resuelve el sibling `<stem>.mp4` del JSON, que es el
  **segmentado**. **Mecanismo decidido:** se añade a `render_broadcast_overlay` un
  parámetro **opcional `clip=`** (retro-compatible: `None` ⇒ comportamiento actual)
  que fija el clip del lienzo y se **reenvía** a `compute_metric_positions(video=clip)`.
  El `main` le pasa el **video de entrada** como `clip`. Es el **único** cambio permitido
  en `src/` (ver RNF-1). No se mueven ni copian salidas de inferencia.
- **RF-22.** `outputs/` es git-ignored (convención del repo); estas salidas no se versionan.

## 8. Requisitos no funcionales

- **RNF-1.** **Cambios en `src/` acotados a uno**: el único cambio permitido es añadir el
  parámetro opcional `clip=` a `render_broadcast_overlay` (RF-21), **retro-compatible**.
  Fuera de eso, el hub solo **orquesta** fachadas (no reimplementa lógica) y **no mueve**
  salidas.
- **RNF-2.** **Lazy imports** (estilo del repo): `main.py` importa barato; torch/cv2/SAM3
  se cargan solo al ejecutar la etapa que los necesita.
- **RNF-3.** **Config-driven**: rutas y parámetros vienen del config activo / `get_abs_path`;
  cero rutas absolutas hardcodeadas.
- **RNF-4.** **Manejo de errores por etapa**: cada etapa en `try/except` con mensaje
  claro; salida con código ≠ 0 si una etapa falla; respeta la idempotencia (no destruye
  lo ya válido).
- **RNF-5.** Mensajes de consola en **español**.

## 9. Criterios de aceptación

- **CA-1.** `python main.py <video_valido>` (interactivo) pregunta detector→tracker→
  overlays, corre el pipeline y deja el broadcast en `outputs/main/<stem>/`, imprimiendo
  su ruta.
- **CA-2.** `python main.py <video_valido> --default` corre sin preguntar nada y produce
  el mismo entregable con la config por defecto.
- **CA-3.** Relanzar el mismo comando tras una corrida completa **no re-infiere** ni
  re-renderiza el broadcast (idempotencia A); con `--overwrite` sí rehace todo.
- **CA-4.** `python main.py <ruta_invalida_o_no_video>` falla con **mensaje claro** y
  **código ≠ 0**, sin cargar SAM3.
- **CA-5.** Tras una corrida, el programa **muestra en pantalla la ruta exacta** de cada
  artefacto (generado o reusado), que vive en su **ruta nativa**
  (`outputs/inference/…`, `outputs/eventos/…`). Relanzar el comando no rehace lo caro
  (los chequeos de idempotencia aciertan contra esas rutas).
- **CA-6.** El video de espectador usa el **clip crudo** (sin máscaras quemadas).
- **CA-7.** Con `--vista generica`, el `main` corre solo el pipeline base y marca
  homografía/eventos/broadcast como `omitido` (motivo "vista genérica"); con
  `--vista superior` sobre un clip que no es superior, la homografía degradada hace que
  eventos/broadcast se **omitan con aviso** (no se genera broadcast degradado).

## 10. Supuestos y decisiones (congelados)

Derivados del protocolo de asunciones (constitución §8). Todos **aceptados** por el
responsable:

1. SDD completo (`spec`→`plan`→`tasks`).
2. `main.py` en la **raíz**.
3. Entregable: `main.py` + `requirements.txt` + doc breve + **un** cambio mínimo
   retro-compatible en `src/` (param `clip=` en `render_broadcast_overlay`, RF-21).
4. Docs y consola en español; commits en inglés.
5. `python main.py <video> [--default]` (ruta posicional obligatoria).
6. Validación: existe + archivo + extensión video + abre con cv2 + ≥1 frame.
7. El `main` **no** genera clips; advierte si el video es largo y continúa.
8. El flujo principal es **tracking** (segmentación solo como overlay opcional).
9. Preguntas: detector → tracker → ¿overlays individuales?
10. Fijos: homografía líneas, Kalman ON, gol estricto, broadcast layout 2.
11. Overlays: tracking (`render_obj_id_overlay`) y segmentación (corrida `mode="segmentation"`).
12. `--default`: detector del config (fallback `sam3_text`) + `bytetrack`, overlays OFF.
13. Etapas en orden inferencia→[overlays]→homografía→eventos→broadcast; salida = broadcast.
14. Idempotencia A + `--overwrite`.
15. Consola: librería de prompts + librería de estilo/progreso (se fija en `plan.md`).
16. Validación de video reusando `frame_extraction` (cv2).
17. Solo orquesta fachadas (`run_inference`, `render_obj_id_overlay`, `render_broadcast_overlay`).
18. Lazy imports.
19. **No mover artefactos**: quedan en sus rutas nativas; el clip crudo se provee al
    broadcast sin alterar las salidas de inferencia (mecanismo en `plan.md`).
20. Manejo de errores por etapa; salida ≠0 al fallar; no borra lo válido.
21. `run_label`/namespace derivado para no pisar otras configuraciones.
22. **Reporte en pantalla** de la ruta exacta de cada artefacto (generado/reusado); el
    hub **no centraliza por movimiento** (mover rompería la idempotencia).
23. **Vista de cámara** `superior|generica` (RF-23): homografía/eventos/broadcast solo en
    `superior`. Mecanismo: **declaración + validación** (flag `--vista` o pregunta
    interactiva; si declara `superior` pero la homografía sale degradada, se omite con
    aviso).
24. **Default de vista**: `--default`/no-TTY ⇒ `superior` (corre todo, coherente con que
    el entregable es el broadcast).
25. La **validación** de "es superior" se hace con la **homografía/métrica existente**
    (modo degradado de `compute_metric_positions`), sin heurística de imagen nueva.

## 11. Dependencias y riesgos

- **Dep.** La inferencia pesada (SAM3/`yolo_sam3`) requiere **GPU/pod**; en local solo
  se ejercita la validación y, si hay un tracking JSON previo, el post-proceso CPU.
- **Riesgo.** `render_broadcast_overlay` resuelve el clip como el sibling `<stem>.mp4`
  del JSON, que normalmente es el **segmentado** → se mitiga con RF-21 (proveer el clip
  crudo al broadcast sin mover las salidas de inferencia; mecanismo en `plan.md`).
- **Riesgo.** Idempotencia y ubicación: como **no se mueve** nada (RF-19), los chequeos de
  "ya existe" (RF-16) aciertan contra las rutas nativas y el skip-done queda intacto; el
  diseño debe **no** introducir copias que dejen artefactos "huérfanos" fuera de esas rutas.
