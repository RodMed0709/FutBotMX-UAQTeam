"""Prueba manual de la segmentacion por caja con SAM3 (tarea sam3_box_prompt).

Estrategia (sin depender de YOLO, que aun no existe): usar el detector SAM3-text ya
existente para producir **cajas reales** y re-segmentarlas por **box-prompt**, como
prueba cruzada del camino nuevo.

Flujo:
  1. Localiza un .MOV real (rglob sobre dataset_dir) y extrae un frame.
  2. detect_classes_in_frame(frame) -> elige una clase con detecciones y deriva sus
     cajas xyxy con mask_to_bbox_centroid.
  3. boxes_to_masks(frame, boxes) -> mascaras box-prompt; verifica 1:1, no-vacio,
     formas y dtype.
  4. Compone un overlay y lo guarda bajo outputs/ para inspeccion visual.

IMPORTANTE: la inferencia de SAM3 en CPU es inviable (minutos por llamada). Este
script debe ejecutarse en RunPod (GPU), donde estan los pesos y el dataset. En
ausencia de pesos/datos se reporta sin abortar.

Uso (en RunPod / contenedor con GPU):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_box_prompt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import detect_classes_in_frame, extract_frames, load_sam3  # noqa: E402
from src.core.detectors import boxes_to_masks  # noqa: E402
from src.core.inference_schema import mask_to_bbox_centroid  # noqa: E402
from src.core.overlay import overlay_detections  # noqa: E402
from src.utils import get_abs_path  # noqa: E402


def load_env(env_path: Path) -> dict[str, str]:
    """Parseo simple de un archivo .env (KEY = value), aplicando strip()."""
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def find_video() -> Path | None:
    """Resuelve dataset_dir y devuelve el primer .MOV encontrado (recursivo)."""
    env = load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        print("[FALLO] No se encontro CONFIG_FILENAME en el .env.")
        return None

    try:
        config_path = get_abs_path(f"configs/{config_filename}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"[FALLO] No se pudo resolver la configuracion: {exc}")
        return None

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_rel = config.get("working_dirs", {}).get("dataset_dir", "")
    if not dataset_rel:
        print("[FALLO] No hay dataset_dir en working_dirs.")
        return None

    try:
        get_abs_path(dataset_rel)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[FALTA] dataset_dir '{dataset_rel}': {type(exc).__name__}: {exc}")
        print("        (ejecuta esta prueba en RunPod, donde estan los datos)")
        return None

    dataset_dir = PROJECT_ROOT / dataset_rel
    movs = sorted({*dataset_dir.rglob("*.MOV"), *dataset_dir.rglob("*.mov")})
    if not movs:
        print(f"  Sin archivos .MOV en {dataset_dir} (recursivo).")
        return None
    print(f"  Encontrados {len(movs)} archivos .MOV (recursivo).")
    return movs[0].relative_to(PROJECT_ROOT)


def pick_boxes(by_class: dict) -> tuple[str, list[list[float]], list[float]]:
    """Elige una clase con detecciones y devuelve (clase, cajas xyxy, scores)."""
    for name, cdets in by_class.items():
        boxes, scores = [], []
        for det in cdets:
            geom = mask_to_bbox_centroid(det.mask)
            if geom is None:
                continue
            (x, y, w, h), _ = geom
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            scores.append(float(det.score))
        if boxes:
            return name, boxes, scores
    return "", [], []


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    print("== Localizacion de video ==")
    video = find_video()
    if video is None:
        print("\n[FALLO] No hay video para probar el box-prompt.")
        return 1

    frames = extract_frames(video)
    frame = frames[len(frames) // 2]
    h, w = frame.shape[:2]
    print(f"  Frame de prueba: {w}x{h}\n")

    bundle = load_sam3()

    print("== detect_classes_in_frame -> cajas reales ==")
    by_class = detect_classes_in_frame(frame, bundle=bundle)
    cls_name, boxes, scores = pick_boxes(by_class)
    if not boxes:
        print("[FALLO] Ninguna clase produjo cajas (sin detecciones).")
        return 1
    print(f"  clase '{cls_name}': {len(boxes)} cajas para box-prompt\n")

    print("== boxes_to_masks (box-prompt) ==")
    dets = boxes_to_masks(frame, boxes, bundle=bundle, scores=scores)

    # Aserciones (AC-3/4/5/6).
    assert len(dets) == len(boxes), f"esperado {len(boxes)} dets, hubo {len(dets)}"
    shapes_ok = all(d.mask.shape == (h, w) and d.mask.dtype == bool for d in dets)
    nonempty = sum(int(d.mask.any()) for d in dets)
    print(f"  dets={len(dets)} (1:1 OK)  full-res&bool={shapes_ok}")
    print(f"  mascaras no vacias: {nonempty}/{len(dets)}")
    print(f"  pixeles por mascara: {[int(d.mask.sum()) for d in dets]}")
    assert shapes_ok, "alguna mascara no es full-res (H,W) bool"
    assert nonempty > 0, "todas las mascaras salieron vacias"

    print("\n== caso vacio ==")
    assert boxes_to_masks(frame, [], bundle=bundle) == [], "caso vacio debe ser []"
    print("  boxes_to_masks(frame, []) == []  OK")

    print("\n== overlay de inspeccion ==")
    by_class_out = {cls_name: dets}
    composed = overlay_detections(frame, by_class_out)
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "test_box_prompt_overlay.png"
    try:
        import imageio.v3 as iio

        iio.imwrite(out_png, composed)
        print(f"  overlay guardado en {out_png.relative_to(PROJECT_ROOT)}")
    except Exception as exc:  # noqa: BLE001 - inspeccion opcional, no debe abortar
        print(f"  (no se pudo guardar overlay: {type(exc).__name__}: {exc})")

    print("\n== Resultado ==")
    print("  OK: demostracion de sam3_box_prompt completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
