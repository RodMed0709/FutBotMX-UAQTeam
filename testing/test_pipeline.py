"""Prueba del pipeline ejecutable (tarea pipeline_runner).

Flujo:
  1. Localiza un .MOV real (rglob sobre dataset_dir).
  2. run_pipeline(video) en modo cuota (por defecto).
  3. Verifica que se generan el mp4 anotado y el JSON de detecciones, y que el
     JSON es parseable y coherente.

IMPORTANTE: el pipeline corre SAM3 (varias inferencias por frame); en CPU es
inviable. Ejecutar en RunPod (GPU), donde estan los pesos y el dataset. En
ausencia de pesos/datos se reporta sin abortar.

Uso (en RunPod / contenedor con GPU):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import run_pipeline  # noqa: E402
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


def check_invalid_detector() -> bool:
    """T8 (sin GPU): un detector inexistente levanta ValueError ANTES de cargar SAM3.

    La resolucion del detector en run_pipeline ocurre tras leer la config y antes de
    load_sam3 / extract_frames, asi que basta un nombre basura para forzar el fallo
    barato. No requiere GPU, pesos ni un video real.
    """
    bogus = "__definitivamente_no_es_un_detector__"
    try:
        run_pipeline("data/raw/__dummy__.MOV", detector=bogus)
    except ValueError as exc:
        print(f"  ValueError esperado (fail-cheap): {exc}  OK")
        return True
    except Exception as exc:  # cualquier otra excepcion = no es fail-cheap por nombre
        print(f"  [FALLO] se esperaba ValueError por detector invalido, vino: {exc!r}")
        return False
    print("  [FALLO] no se levanto ValueError con un detector invalido.")
    return False


def check_detector_modes(video: Path) -> bool:
    """T9/T10 (pod/GPU): yolo_sam3 corre en segmentacion y casa con sam3_text.

    Corre la misma cuota de frames con ambos detectores y compara la **estructura**
    del JSON (claves de cabecera y de cada frame), no los valores. Confirma que:
      - yolo_sam3 funciona como segmentador suelto (mismo esquema), y
      - sam3_text (== default) sigue produciendo el esquema de siempre.
    """
    payloads = {}
    for det in ("yolo_sam3", "sam3_text"):
        print(f"  run_pipeline(detector={det!r}) ...")
        res = run_pipeline(video, detector=det, render_video=False)
        payloads[det] = json.loads(Path(res["json"]).read_text(encoding="utf-8"))

    yolo, sam3 = payloads["yolo_sam3"], payloads["sam3_text"]
    if set(yolo) != set(sam3):
        print(f"  [FALLO] claves de cabecera distintas: {set(yolo) ^ set(sam3)}")
        return False
    if yolo["frames"] and sam3["frames"]:
        ky = set(yolo["frames"][0]) if yolo["frames"][0] else set()
        ks = set(sam3["frames"][0]) if sam3["frames"][0] else set()
        # Ambos frames pueden no tener detecciones; comparamos solo si hay contenido.
        if ky and ks and ky != ks:
            print(f"  [FALLO] claves de frame distintas: {ky ^ ks}")
            return False
    print(f"  esquema yolo_sam3 == sam3_text  OK (claves: {sorted(yolo)})")
    return True


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    print("== T8: detector invalido => ValueError (sin GPU) ==")
    t8_ok = check_invalid_detector()
    if not t8_ok:
        return 1
    print()

    print("== Localizacion de video ==")
    video = find_video()
    if video is None:
        print("\n[FALLO] No hay video para probar el pipeline.")
        return 1
    print(f"  Video objetivo: {video}\n")

    print("== run_pipeline (modo cuota) ==")
    result = run_pipeline(video)
    mp4_path, json_path = result["video"], result["json"]
    print(f"  mp4 : {mp4_path}")
    print(f"  json: {json_path}\n")

    print("== Verificaciones ==")
    assert mp4_path.is_file() and mp4_path.stat().st_size > 0, "mp4 ausente/vacio"
    print(f"  mp4 existe, tamano={mp4_path.stat().st_size} bytes  OK")

    assert json_path.is_file(), "json ausente"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "frames" in payload and "classes" in payload, "json sin frames/classes"
    assert len(payload["frames"]) == payload["num_frames"], "num_frames incoherente"
    print(
        f"  json OK: {payload['num_frames']} frames, clases={payload['classes']}, "
        f"fps={payload['fps']}"
    )

    print("\n== T9/T10: yolo_sam3 vs sam3_text en segmentacion (pod/GPU) ==")
    if not check_detector_modes(video):
        print("\n[FALLO] el detector inyectable no produjo el esquema esperado.")
        return 1

    print("\n== Resultado ==")
    print("  OK: demostracion de run_pipeline + detector inyectable completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
