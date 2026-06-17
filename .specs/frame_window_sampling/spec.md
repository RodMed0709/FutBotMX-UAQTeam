# Spec — Ventana y submuestreo de frames (`start_frame` / `frame_step`)

> SDD **retroactiva**: la capacidad ya está implementada y verificada (ver `tasks.md`).
> Documenta lo hecho para mantener la metodología consistente.

## Objetivo

Permitir recorrer un **tramo arbitrario** de un video y/o tomar **1 de cada N frames**,
sin procesar el video completo desde el inicio. Es una capacidad **general de
`frame_extraction`** (no específica de homografía): habilita renderizar clips de momentos
concretos (p. ej. una jugada o un gol) y abaratar el costo de forma proporcional.

## Motivación

`iter_frames` solo recorría **desde el frame 0** y de forma **contigua**. Esto obligaba a:
- procesar todo el prefijo del video para llegar a un momento de interés;
- pagar el costo de **todos** los frames aunque bastara una muestra (1 de cada N).

El detonante concreto fue reproducir un clip de la demo de fase_4
(`IMG_9933_c` = `start=15000`, `every=2`) con `render_minimap_video`, que no tenía forma
de arrancar en un offset ni de submuestrear. Pero la capacidad sirve a **cualquier**
consumidor de `iter_frames` y será clave en **fase_5** (extraer clips de eventos).

## Entrada

`iter_frames(video_path, max_frames=None, start_frame=0, frame_step=1)`:
- `start_frame` (int ≥ 0): índice del frame **fuente** donde empezar.
- `frame_step` (int ≥ 1): paso de muestreo (`1` = todos, `2` = 1 de cada 2, …).
- `max_frames` (int | None): **cantidad** de frames entregados (no un rango); `None` = hasta el final.

## Salida

Generador de `(frame_index, frame_rgb)` donde **`frame_index` es el índice en el video
fuente** (refleja `start_frame`/`frame_step`), de modo que sigue **casando con los
`frame_index` de un `tracks_json`** (los tracks se indexan por frame fuente).

## Método

- `iter_frames` itera `range(start_frame, total, frame_step)` y corta cuando ha entregado
  `max_frames` frames. Valida `start_frame ≥ 0` y `frame_step ≥ 1` (`ValueError` si no).
- `render_minimap_video` **expone** `start_frame`/`frame_step` y:
  - dimensiona la barra de progreso con el conteo *strided*
    (`ceil((total - start_frame) / frame_step)`, acotado por `max_frames`);
  - escribe el mp4 de salida a **`fps_fuente / frame_step`** para conservar la velocidad
    real (con `frame_step=2`, 150 frames a 15 fps duran 10 s, no 5 s a 30 fps);
  - aplica el default de `max_frames` derivado de `tracks_json` **solo** en el recorrido
    completo (`start_frame=0` y `frame_step=1`), porque con recorte `max_frames` es una
    cuenta, no un rango.

## No-objetivos

- Muestreo por **tiempo** (segundos) en vez de por índice de frame — fuera de alcance.
- Selección de frames **no uniforme** (p. ej. por keyframes/movimiento) — fuera de alcance.
- Cambiar el contrato de `extract_frames` (modo lote por cuota) — intacto.

## Compatibilidad

Los parámetros nuevos tienen **defaults** (`start_frame=0`, `frame_step=1`) → todos los
consumidores existentes de `iter_frames` (p. ej. `track_video`) siguen igual sin cambios.

## Verificación

- Índices correctos: `iter_frames(start_frame=1800, frame_step=2, max_frames=5)` →
  `[1800, 1802, 1804, 1806, 1808]` (verificado en local sobre `IMG_9933.MOV`).
- Validaciones: `start_frame<0` y `frame_step<1` levantan `ValueError`.
- Integración: `render_minimap_video(start_frame=15000, frame_step=2, max_frames=150)`
  reproduce el tramo de `IMG_9933_c` con duración real (10 s), verificado en pod.
