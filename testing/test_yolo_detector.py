"""Prueba manual del detector de cajas YOLO (tarea yolo_detector).

Flujo:
  1. Localiza un .MOV real (rglob sobre dataset_dir) y extrae un frame.
  2. load_yolo() -> carga best.pt desde working_dirs.yolo_weights.
  3. detect_boxes(frame) -> cajas por clase del repo; verifica claves, formas y scores.
  4. Opcional: dibuja las cajas y guarda un PNG bajo outputs/ para inspeccion.

IMPORTANTE: ``best.pt`` vive en el POD (artefacto de fase_1), NO en el repo, y
``ultralytics`` solo esta instalado alli. Por eso esta prueba se ejecuta en el pod;
en local fallara con un FileNotFoundError claro (peso ausente) o por falta de
ultralytics. En ausencia de datos/peso se reporta sin abortar.

Uso (en RunPod / contenedor con GPU):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_yolo_detector.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import extract_frames  # noqa: E402
from src.core.detectors import BoxDetection, detect_boxes, load_yolo  # noqa: E402
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


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    print("== Localizacion de video ==")
    video = find_video()
    if video is None:
        print("\n[FALLO] No hay video para probar el detector YOLO.")
        return 1

    frames = extract_frames(video)
    frame = frames[len(frames) // 2]
    h, w = frame.shape[:2]
    print(f"  Frame de prueba: {w}x{h}\n")

    print("== load_yolo ==")
    try:
        model = load_yolo()
    except FileNotFoundError as exc:
        print(f"[FALTA] {exc}")
        return 1
    print("  modelo YOLO cargado.\n")

    print("== detect_boxes ==")
    out = detect_boxes(frame, model=model)

    # Aserciones (AC-3/4/5/6).
    assert isinstance(out, dict), "la salida debe ser un dict"
    assert "green_floor" not in out, "green_floor no debe aparecer (sin yolo_id)"
    for name, dets in out.items():
        for d in dets:
            assert isinstance(d, BoxDetection), "elementos deben ser BoxDetection"
            assert len(d.bbox) == 4, "bbox debe tener 4 valores xyxy"
            assert 0.0 <= d.score <= 1.0, "score fuera de [0,1]"
    total = sum(len(v) for v in out.values())
    for name, dets in out.items():
        sc = [round(d.score, 3) for d in dets]
        print(f"  {name:12s}  cajas={len(dets):3d}  scores={sc}")
    print(f"  total cajas: {total}\n")

    print("== overlay de inspeccion (opcional) ==")
    try:
        import cv2

        vis = frame[..., ::-1].copy()  # RGB->BGR para cv2
        for name, dets in out.items():
            for d in dets:
                x0, y0, x1, y1 = (int(v) for v in d.bbox)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis, name, (x0, max(0, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_dir = PROJECT_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / "test_yolo_detector_overlay.png"
        cv2.imwrite(str(out_png), vis)
        print(f"  overlay guardado en {out_png.relative_to(PROJECT_ROOT)}")
    except Exception as exc:  # noqa: BLE001 - inspeccion opcional, no debe abortar
        print(f"  (no se pudo guardar overlay: {type(exc).__name__}: {exc})")

    print("\n== Resultado ==")
    print("  OK: demostracion de yolo_detector completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
