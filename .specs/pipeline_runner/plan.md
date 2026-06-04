# Plan técnico — Primer pipeline ejecutable (`pipeline_runner`)

- **Tarea atómica:** `pipeline_runner`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Borrador de referencia:** no hay draft previo para este plan.
- **Estado:** Diseño técnico. **No** implica crear ni modificar código fuente aún.

---

## 1. Objetivo del plan

Definir, a nivel técnico, cómo implementar `run_pipeline(...)`: orquestar
`video → extract_frames → (por frame: detect_classes_in_frame → overlay_detections)
→ write_video`, escribir además un JSON de detecciones, cargar el modelo una sola
vez, leer la config una sola vez y auto-nombrar las salidas bajo `outputs/`.

---

## 2. Stack técnico

- **Python:** 3.11.
- **Orquestación:** reutiliza piezas ya construidas de `src.core`
  (`extract_frames`, `load_sam3`, `detect_classes_in_frame`, `overlay_detections`,
  `write_video`).
- **Arrays / serialización / rutas:** `numpy` (`np.stack`), `json`, `pathlib`.
- **Configuración:** `json` + `src.utils.get_abs_path` para leer **una vez**
  `classes`, `working_dirs.outputs_dir` y `visualization.output_fps`.

> No se importan piezas pesadas directamente: `torch`/`imageio` quedan perezosos en
> sus módulos, así `import src.core` (que ahora también importa `pipeline`) sigue
> ligero.

---

## 3. Diseño

### 3.1 Ubicación y módulo

- Archivo nuevo: `src/core/pipeline.py`.
- Exportación en `src/core/__init__.py`: `from src.core.pipeline import
  run_pipeline` y sumarlo a `__all__`.

### 3.2 Firma

```python
def run_pipeline(
    video_path: Path | str,
    output_path: Path | None = None,
    all_frames: bool = False,
    mode: str = "per_frame",
) -> dict[str, Path]: ...   # {"video": <mp4>, "detections": <json>}
```

### 3.3 Lectura única de configuración

`_load_pipeline_config()` lee la config (leer `CONFIG_FILENAME` del `.env` con
`strip()` → `get_abs_path` → `json.load`) y devuelve `classes`, `outputs_dir` y
`output_fps` en **una sola** lectura. Estos valores se pasan **explícitos** a las
piezas, para que no re-lean la config por frame.

### 3.4 Composición de rutas de salida

```python
stem = Path(video_path).stem
if output_path is not None:
    mp4_path = Path(output_path)
    json_path = mp4_path.with_name(f"{mp4_path.stem}_detections.json")
else:
    base = PROJECT_ROOT / outputs_dir
    mp4_path = base / f"{stem}_annotated.mp4"
    json_path = base / f"{stem}_detections.json"
json_path.parent.mkdir(parents=True, exist_ok=True)  # write_video crea el dir del mp4
```

### 3.5 Flujo de orquestación

```python
if mode != "per_frame":
    raise NotImplementedError(f"mode '{mode}' no soportado (solo 'per_frame').")

classes, outputs_dir, fps = _load_pipeline_config()
bundle = load_sam3()
frames = extract_frames(video_path, all_frames=all_frames)

composed = []
records = []
for i, frame in enumerate(frames):
    print(f"  frame {i + 1}/{len(frames)}")
    dets = detect_classes_in_frame(frame, classes=classes, bundle=bundle)
    composed.append(overlay_detections(frame, dets, classes=classes))
    records.append({
        "index": i,
        "detections": {
            name: [{"obj_id": d.obj_id, "score": d.score} for d in cdets]
            for name, cdets in dets.items()
        },
    })

mp4_path = write_video(np.stack(composed), mp4_path, fps=fps)
<escribir JSON>   # §3.6
return {"video": mp4_path, "detections": json_path}
```

- El `Sam3Bundle` se carga **una vez** y se reutiliza en `detect_classes_in_frame`.
- `classes` se resuelven una vez y se pasan a detección y overlay.

### 3.6 Esquema y escritura del JSON

```python
payload = {
    "video": str(video_path),
    "mode": mode,
    "all_frames": all_frames,
    "fps": fps,
    "num_frames": len(frames),
    "classes": [c["name"] for c in classes],
    "frames": records,
}
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

- Por frame y clase: lista de `{obj_id, score}` (sin máscaras; la cuenta se deriva
  del largo). El metadato `fps` es el mismo que se pasó a `write_video`.

### 3.7 fps

`fps = output_fps` (de config), resuelto una vez y usado **a la vez** para
`write_video` y para el metadato del JSON. En modo completo hoy es el mismo
placeholder; el fps real de la fuente se cableará en la tarea siguiente.

### 3.8 Manejo de errores

| Situación | Excepción |
|---|---|
| `mode` distinto de `per_frame` | `NotImplementedError` (antes de cargar el modelo) |
| Video inexistente / inválido | `FileNotFoundError`/`ValueError` (vía `extract_frames`) |
| Config/clave ausente | `ValueError`/`KeyError`/`FileNotFoundError` (propagados) |
| Falla de inferencia o escritura | excepción propagada de la pieza subyacente |

---

## 4. Cambios de configuración

- **Ninguno.** Usa claves ya existentes: `classes`, `working_dirs.outputs_dir`,
  `visualization.output_fps`.

---

## 5. Validación

### 5.1 Verificación ligera (agente, local)

- Lint (`ruff`, `black`) e importabilidad (`from src.core import run_pipeline`).
- `run_pipeline(<algo>, mode="tracking")` lanza `NotImplementedError` **antes** de
  cargar el modelo (no requiere GPU).

### 5.2 Script — `testing/test_pipeline.py` (usuario, RunPod/GPU)

- Localiza un `.MOV` real (rglob sobre `dataset_dir`), corre `run_pipeline(video)`
  (modo cuota por defecto) y verifica:
  - se genera el **mp4** y el **JSON** bajo `outputs/`;
  - el JSON es **parseable** y contiene `frames`/`classes`/metadatos coherentes.
- Lo **ejecuta el usuario en RunPod** (usa modelo/GPU; en CPU es inviable):
  ```bash
  docker compose --env-file .env -f docker/docker-compose.yml \
    exec futbotmx26 python testing/test_pipeline.py
  ```

---

## 6. Trazabilidad con los criterios de aceptación del spec

| Criterio (spec) | Cubierto por | Observación |
|---|---|---|
| AC-1 Módulo y función | §3.1, §3.2 | `run_pipeline` + export |
| AC-2 Orquestación end-to-end | §3.5 | extract → detect → overlay → write |
| AC-3 JSON de detecciones | §3.6 | `{obj_id, score}` por frame/clase |
| AC-4 Modelo una sola vez | §3.5 | `load_sam3()` + bundle reusado |
| AC-5 Modos cuota/completo | §3.5 | `all_frames` |
| AC-6 `mode` preparado | §3.5, §3.8 | `per_frame` único; resto `NotImplementedError` |
| AC-7 Salidas en `outputs/` | §3.4 | auto-nombre + override |
| AC-8 Entrada flexible | §3.5 | delega en `extract_frames` |
| AC-9 Validación | §5 | ligera (agente) + script (RunPod) |

---

## 7. Riesgos y consideraciones

- **Memoria de los frames compuestos:** en modo **completo** se acumulan todos los
  frames `(N,H,W,3) uint8` + los compuestos en memoria antes de escribir; para
  videos largos puede ser pesado. Para el MVP (cuota / clips) es aceptable; si
  escala, la tarea de fps-real podría también streamear la escritura frame a frame.
- **Costo en GPU:** varias inferencias por frame × N frames; el modo completo de un
  video real puede tardar. La corrida es en RunPod.
- **fps placeholder en modo completo:** hasta la tarea siguiente, el modo completo
  reproduce al fps de config (no al de la fuente). Documentado en el spec.
- **`np.stack` exige forma homogénea:** todos los frames compuestos comparten
  `(H,W,3)` del video; el overlay preserva el tamaño, así que `np.stack` es seguro.
