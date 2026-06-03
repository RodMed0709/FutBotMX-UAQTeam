"""Prueba manual de la extraccion de frames (tarea frame_extraction).

Flujo:
  1. Lee CONFIG_FILENAME del .env y resuelve el archivo de configuracion.
  2. Resuelve dataset_dir y localiza un .MOV de forma recursiva (rglob).
  3. Llama extract_frames(video, all_frames=False) -> modo cuota, reporta forma.
  4. Llama extract_frames(video, all_frames=True) -> modo completo, reporta total.
  5. Imprime formas, dtype y conteos.

Las rutas que no existen (caso local: data/raw es un symlink que solo existe
dentro del contenedor) se reportan sin abortar la demostracion. Por eso esta
prueba debe ejecutarse DENTRO del contenedor, donde los videos estan montados.

Uso (en el contenedor):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_frame_extraction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import extract_frames  # noqa: E402
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

    # Verificar que dataset_dir exista (esto resuelve el symlink internamente).
    try:
        get_abs_path(dataset_rel)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[FALTA] dataset_dir '{dataset_rel}': {type(exc).__name__}: {exc}")
        print("        (esperado en el host; ejecuta esta prueba en el contenedor)")
        return None

    # Buscar sobre la ruta SIN resolver el symlink, para conservar rutas bajo el
    # proyecto (p. ej. data/raw/.../IMG.MOV). Asi extract_frames puede convertirlas
    # a relativas y delegar la resolucion/verificacion en get_abs_path.
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
        print("\n[FALLO] No hay video para probar la extraccion de frames.")
        return 1
    print(f"  Video objetivo: {video}\n")

    print("== Modo cuota (all_frames=False) ==")
    frames_quota = extract_frames(video, all_frames=False)
    print(
        f"  forma: {frames_quota.shape}  dtype: {frames_quota.dtype}  "
        f"frames: {frames_quota.shape[0]}\n"
    )

    print("== Modo completo (all_frames=True) ==")
    frames_all = extract_frames(video, all_frames=True)
    print(
        f"  forma: {frames_all.shape}  dtype: {frames_all.dtype}  "
        f"frames: {frames_all.shape[0]}\n"
    )

    print("== Resultado ==")
    print("  OK: demostracion de extract_frames completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
