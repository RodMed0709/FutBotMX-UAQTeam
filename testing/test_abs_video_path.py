"""Prueba manual: rutas absolutas de video en extract_frames (tarea abs_video_path).

Verifica el contrato ampliado de extract_frames / _resolve_video_path:

  1. Ruta ABSOLUTA EXTERNA: la ruta absoluta (.resolve()) de un video real produce
     frames sin lanzar ValueError, aunque no este bajo PROJECT_ROOT.
  2. Ruta RELATIVA (regresion): una ruta relativa sigue funcionando como antes.
  3. Ruta INEXISTENTE: una ruta absoluta falsa lanza FileNotFoundError.
  4. DIRECTORIO: una ruta absoluta a un directorio lanza FileNotFoundError.

Las rutas que no existen (caso local: data/raw puede ser un symlink que solo
resuelve dentro del contenedor) se reportan sin abortar la demostracion. Por eso
los escenarios 1-2 conviene ejecutarlos donde los videos esten disponibles.

Uso (en el contenedor):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_abs_video_path.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core import extract_frames
from src.utils import get_abs_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def find_video_relative() -> Path | None:
    """Resuelve dataset_dir y devuelve el primer .MOV como ruta relativa al proyecto."""
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
        print("        (esperado en el host; ejecuta esta prueba en el contenedor)")
        return None

    # Buscar sobre la ruta SIN resolver el symlink para conservar rutas relativas.
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
    video_rel = find_video_relative()
    have_video = video_rel is not None
    if have_video:
        print(f"  Video objetivo (relativa): {video_rel}\n")
    else:
        print("  Sin video real; se omiten los escenarios 1-2.\n")

    # --- Escenario 1: ruta ABSOLUTA EXTERNA -------------------------------------
    print("== Escenario 1: ruta absoluta (.resolve()) ==")
    if have_video:
        abs_path = (PROJECT_ROOT / video_rel).resolve()
        print(f"  Ruta absoluta: {abs_path}")
        try:
            frames = extract_frames(abs_path, all_frames=False)
            print(
                f"  OK -> forma: {frames.shape}  dtype: {frames.dtype}  "
                f"frames: {frames.shape[0]}"
            )
        except Exception as exc:  # noqa: BLE001 (demostracion)
            print(f"  [FALLO] excepcion inesperada: {type(exc).__name__}: {exc}")
    else:
        print("  [OMITIDO] sin video real.")
    print()

    # --- Escenario 2: ruta RELATIVA (regresion) ---------------------------------
    print("== Escenario 2: ruta relativa (regresion) ==")
    if have_video:
        try:
            frames = extract_frames(video_rel, all_frames=False)
            print(f"  OK -> forma: {frames.shape}  frames: {frames.shape[0]}")
        except Exception as exc:  # noqa: BLE001 (demostracion)
            print(f"  [FALLO] excepcion inesperada: {type(exc).__name__}: {exc}")
    else:
        print("  [OMITIDO] sin video real.")
    print()

    # --- Escenario 3: ruta INEXISTENTE ------------------------------------------
    print("== Escenario 3: ruta absoluta inexistente ==")
    fake = (PROJECT_ROOT / "no_existe" / "fantasma.MOV").resolve()
    try:
        extract_frames(fake)
        print("  [FALLO] se esperaba FileNotFoundError y no se lanzo.")
    except FileNotFoundError as exc:
        print(f"  OK -> FileNotFoundError: {exc}")
    except Exception as exc:  # noqa: BLE001 (demostracion)
        print(f"  [FALLO] excepcion inesperada: {type(exc).__name__}: {exc}")
    print()

    # --- Escenario 4: DIRECTORIO ------------------------------------------------
    print("== Escenario 4: ruta absoluta a un directorio ==")
    a_dir = PROJECT_ROOT.resolve()
    try:
        extract_frames(a_dir)
        print("  [FALLO] se esperaba FileNotFoundError y no se lanzo.")
    except FileNotFoundError as exc:
        print(f"  OK -> FileNotFoundError: {exc}")
    except Exception as exc:  # noqa: BLE001 (demostracion)
        print(f"  [FALLO] excepcion inesperada: {type(exc).__name__}: {exc}")
    print()

    print("== Resultado ==")
    print("  Demostracion de rutas absolutas en extract_frames completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
