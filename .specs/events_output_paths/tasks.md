# Tasks — Rutas de salida dedicadas para eventos (`events_output_paths`)

- **Tarea atómica:** `events_output_paths`
- **Paso de la metodología:** 4 (Descomposición en tareas) → habilita el paso 5
  (implementación).
- **Spec/plan de referencia:** `.specs/events_output_paths/{spec,plan}.md`.

> A partir de aquí (y **solo** aquí) se autoriza escribir/modificar código.

---

## Aclaración respecto al plan

Las funciones `write_*`/`render_*` de los módulos métricos reciben la ruta como
**argumento explícito** (hoy la arma el test), no derivan un default interno y **no
reciben el `stem`**. Por eso la convención se aplica en los **call sites** que hoy
construyen la ruta (los `testing/test_*`) y en el **único** default interno existente
(`demo_overlay.compose_demo`, `output_path=None`). No se añade un `stem` a funciones que
no lo necesitan: se mantienen agnósticas de ruta.

---

## T1 · Crear el helper `events_paths`

- [ ] Crear `src/core/events_schema.py` con `events_paths(stem, kind, ext, *,
      outputs_dir="outputs", namespace=None) -> Path` según el plan §2.
- [ ] Función **pura** (sin `mkdir`); ruta absoluta resuelta contra `PROJECT_ROOT`.
- [ ] Docstring en español, estilo del repo.

**Verificación:** `python -c "from src.core.events_schema import events_paths; print(events_paths('x','demo','mp4'))"`
imprime `<PROJECT_ROOT>/outputs/eventos/x/x_demo.mp4`.

---

## T2 · Test manual del helper

- [ ] Crear `testing/test_events_output_paths.py` (script directo, sin pytest, sin GPU).
- [ ] Comprobar: estructura `outputs/eventos/<stem>/<stem>_<kind>.<ext>`; `kind`/`ext`
      correctos; `namespace` insertado antes del `<stem>`; ruta absoluta bajo
      `PROJECT_ROOT`; el helper **no** crea carpetas.

**Verificación:** `python testing/test_events_output_paths.py` termina OK.

---

## T3 · Migrar el default de `demo_overlay`

- [ ] En `compose_demo`, cuando `output_path is None`, derivar la ruta con
      `events_paths(tracks_json.stem, "demo", "mp4")` en vez de
      `tracks_json.parent / f"{stem}_demo.mp4"`.
- [ ] No cambiar la firma pública (sigue aceptando `output_path` explícito).

**Verificación:** revisar que un `compose_demo(tracks_json)` sin ruta apunta a
`outputs/eventos/<stem>/<stem>_demo.mp4` (inspección de la ruta; no requiere correr el
render completo).

---

## T4 · Migrar los call sites de los tests métricos

Para cada test que hoy escribe plano en `outputs/`, construir la ruta con `events_paths`
(mismo `kind` de la tabla del plan §3) antes de llamar al `write_*`/`render_*`:

- [ ] `testing/test_metric_positions*` → `kind="metric_positions"` (json/png).
- [ ] `testing/test_*goal_geometric*` → `kind="goal_geometric"` (json/png).
- [ ] `testing/test_*kinematics*` / speed_distance → `kind="metric_speed_distance"`.
- [ ] `testing/test_*heatmap*` → `kind="heatmap_ball"` / `"heatmap_robot"`.
- [ ] `testing/test_*field_zones*` → `kind=f"field_zones_{esquema}"`.

(Localizar los nombres reales de los tests con `ls testing/`; ajustar los que existan.)

**Verificación:** cada test migrado corre (local, sin GPU donde aplique) y deja sus
salidas bajo `outputs/eventos/<stem>/`.

---

## T5 · Cierre

- [ ] `ruff check src/core/events_schema.py testing/test_events_output_paths.py` limpio.
- [ ] `black` aplicado a los archivos nuevos/modificados.
- [ ] Confirmar que `inference_schema.py` y `outputs/inference/...` quedaron intactos.
- [ ] Confirmar con el usuario antes de cualquier commit (constitución §7.1/§11).

---

## Orden sugerido

T1 → T2 (helper validado) → T3 → T4 → T5. T3 y T4 son independientes entre sí una vez
existe el helper.

---

## Fuera de alcance (recordatorio del spec)

- No cambia el formato/contenido de JSON/imágenes/videos, solo su ubicación.
- No toca el esquema de inferencia.
- No borra las salidas viejas en la raíz de `outputs/` (git-ignored).
