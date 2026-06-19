"""Prueba manual de la segmentacion por texto (tarea text_segmentation).

Flujo:
  1. Localiza un .MOV real (rglob sobre dataset_dir) y extrae un frame.
  2. segment_with_text(frame, "robot") -> reporta nº detecciones, forma de mascara
     y scores.
  3. detect_classes_in_frame(frame) -> por clase, conteo y score medio.

IMPORTANTE: la inferencia de SAM3 en CPU es inviable (minutos por llamada). Este
script debe ejecutarse en RunPod (GPU), donde estan los pesos y el dataset. En
ausencia de pesos/datos se reporta sin abortar.

Uso (en RunPod / contenedor con GPU):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_segmentation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.core import (  # noqa: E402
    detect_classes_in_frame,
    extract_frames,
    segment_with_text,
)
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


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    print("== Localizacion de video ==")
    video = find_video()
    if video is None:
        print("\n[FALLO] No hay video para probar la segmentacion.")
        return 1

    frames = extract_frames(video)
    frame = frames[len(frames) // 2]
    h, w = frame.shape[:2]
    print(f"  Frame de prueba: {w}x{h}\n")

    print("== segment_with_text(frame, 'robot') ==")
    dets = segment_with_text(frame, "robot")
    print(f"  detecciones: {len(dets)}")
    if dets:
        d0 = dets[0]
        ok_shape = d0.mask.shape == (h, w) and d0.mask.dtype == bool
        print(
            f"  mask[0] forma={d0.mask.shape} dtype={d0.mask.dtype} (full-res={ok_shape})"
        )
        print(f"  scores: {[round(d.score, 3) for d in dets]}\n")

    print("== detect_classes_in_frame(frame) ==")
    by_class = detect_classes_in_frame(frame)
    for name, cdets in by_class.items():
        avg = float(np.mean([d.score for d in cdets])) if cdets else 0.0
        print(f"  {name:12s}  dets={len(cdets):3d}  score_medio={avg:.3f}")

    print("\n== Resultado ==")
    print("  OK: demostracion de text_segmentation completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
