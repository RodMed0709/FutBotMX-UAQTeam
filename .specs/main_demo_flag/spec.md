# spec.md — `main_demo_flag`

## Contexto

El hub `main.py` corre el pipeline end-to-end **sobre un video** que hoy se pasa como
argumento posicional (ruta relativa o absoluta). Para una **demostración guiada y
reproducible** queremos que el usuario no tenga que conocer ni teclear rutas: que
elija de una lista curada de clips demo (provistos por el bootstrap) y el hub corra
sobre el elegido.

Los clips demo y su catálogo viven en el **manifiesto versionado**
`assets/bootstrap_manifest.json` (definido por la tarea `bootstrap_data`), que es la
**fuente de verdad única** del conjunto demo.

## Objetivo

Agregar a `main.py` una flag **`--demo`** que convierte la corrida en una demo
guiada: su **primer paso** es un **selector interactivo** del archivo demo; el resto
del flujo continúa normal.

## Alcance

- Nueva flag `--demo` en `main.py`.
- `--demo` **sobreescribe la selección de entrada**: ignora `--default` y `--vista`, y
  vuelve **opcional** el argumento posicional `video` (`python main.py --demo`). Si se
  pasan juntos, `--demo` manda sobre ellos.
- `--demo` es **combinable con `--overwrite`**: `python main.py --demo --overwrite`
  elige el demo por menú **y** fuerza rehacer todo de cero (habilita la validación de
  reproducibilidad de forma guiada).
- Primer prompt bajo `--demo`: `questionary.select` con los demos **presentes en
  local**, leídos del manifiesto (ítems cuyo `paquetes` incluye `"demo"` y cuyo clip
  existe en disco). El elegido se vuelve el video de entrada.
- Cada entrada demo lleva su `vista`; al elegirla, el hub **autoselecciona** esa vista
  (no se pregunta), respetando el gate de cámara superior para homografía/eventos.
- Tras elegir el demo, el flujo **continúa interactivo normal** (detector / tracker /
  overlays). `--demo` solo reemplaza el paso "qué video" (y fija la vista).

## Fuera de alcance

- Definir o poblar el manifiesto (lo hace `bootstrap_data`).
- Descargar demos: `main --demo` **no** auto-descarga; si no hay demos presentes,
  termina con un error claro que sugiere correr `python -m src.bootstrap_data`.
- Cambiar el comportamiento del hub sin `--demo` (ruta posicional, `--default`,
  `--vista`, `--overwrite` siguen igual).

## Comportamiento esperado

```
$ python main.py --demo
Elige un clip demo:
  > IMG_9933_5m30   (cámara superior)
    IMG_9938_5m00   (cámara superior)
    video-597_singular_display   (genérica)
    video-836_singular_display   (genérica)
# … continúa: detector / tracker / overlays … luego corre el pipeline
```

- `python main.py --demo --overwrite` → igual, pero rehace todas las etapas (re-corre
  SAM3, regenera overlays y broadcast) para validar reproducibilidad.
- `python main.py --demo` sin demos presentes → error:
  "No hay clips demo en local. Corre `python -m src.bootstrap_data` (opción demos)."

## Consideraciones

- **Terminal no interactiva + `--demo`:** como `--demo` exige al menos el prompt de
  selección, en no-TTY termina con mensaje claro (igual que el flujo interactivo
  actual exige TTY).
- **Precedencia de flags:** la resolución de piezas debe aplicar `--demo` **antes** de
  validar/usar `video`/`--default`/`--vista`, dejando `--overwrite` intacto.
- **Fuente de verdad compartida:** el selector lee el **mismo** manifiesto que el
  bootstrap; agregar un demo nuevo (en el manifiesto) lo hace aparecer aquí sin tocar
  código.
