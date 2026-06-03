"""Prueba manual de la utilidad get_abs_path (tarea abs_dir_func).

Flujo:
  1. Lee CONFIG_FILENAME del archivo .env (en la raiz del proyecto).
  2. Resuelve configs/<CONFIG_FILENAME> a ruta absoluta con get_abs_path y lee el JSON.
  3. Resuelve las rutas relativas de working_dirs (dataset_dir, sam3_dir).
  4. Intenta abrir un archivo .MOV dentro de dataset_dir con OpenCV y reporta.
  5. Imprime en consola las rutas absolutas y su estado de existencia.

Las rutas que no existen (caso local: data/raw y assets/sam3 son symlinks que
solo existen dentro del contenedor) se reportan sin abortar la demostracion.

Uso:
    python testing/test_abs_dir_func.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_abs_path  # noqa: E402


def load_env(env_path: Path) -> dict[str, str]:
    """Parseo simple de un archivo .env (KEY = value), aplicando strip()."""
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def try_resolve(label: str, relative: str) -> Path | None:
    """Resuelve una ruta relativa reportando el resultado sin abortar."""
    try:
        resolved = get_abs_path(relative)
    except (ValueError, FileNotFoundError) as exc:
        print(f"  [FALTA] {label}: '{relative}' -> {type(exc).__name__}: {exc}")
        return None
    print(f"  [ok]    {label}: '{relative}' -> {resolved}  (existe: {resolved.exists()})")
    return resolved


def try_read_mov(dataset_dir: Path) -> None:
    """Intenta abrir el primer archivo .MOV de dataset_dir con OpenCV."""
    print("\n== Lectura de video .MOV ==")
    # Busqueda recursiva: los videos suelen estar en subcarpetas (p.ej. por fecha).
    movs = sorted({*dataset_dir.rglob("*.MOV"), *dataset_dir.rglob("*.mov")})
    if not movs:
        print(f"  Sin archivos .MOV en {dataset_dir} (recursivo); se omite la lectura de video.")
        return

    print(f"  Encontrados {len(movs)} archivos .MOV (recursivo).")
    target = movs[0]
    print(f"  Intentando leer: {target}")
    try:
        import cv2  # noqa: PLC0415 - import local; solo si hay video que leer
    except ImportError as exc:
        print(f"  OpenCV no disponible ({exc}); se omite la lectura de video.")
        return

    cap = cv2.VideoCapture(str(target))
    try:
        if not cap.isOpened():
            print("  [FALLO] No se pudo abrir el video.")
            return
        ok, _frame = cap.read()
        print("  [ok]    Frame leido correctamente." if ok else "  [FALLO] No se pudo leer un frame.")
    finally:
        cap.release()


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    env = load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        print("[FALLO] No se encontro CONFIG_FILENAME en el .env.")
        return 1

    print("== Archivo de configuracion ==")
    config_rel = f"configs/{config_filename}"
    config_path = try_resolve("config", config_rel)
    if config_path is None:
        print("\n[FALLO] No se pudo resolver el archivo de configuracion.")
        return 1

    config = json.loads(config_path.read_text(encoding="utf-8"))
    working_dirs = config.get("working_dirs", {})

    print("\n== Rutas de working_dirs ==")
    dataset_rel = working_dirs.get("dataset_dir", "")
    sam3_rel = working_dirs.get("sam3_dir", "")
    dataset_dir = try_resolve("dataset_dir", dataset_rel) if dataset_rel else None
    try_resolve("sam3_dir", sam3_rel) if sam3_rel else None

    if dataset_dir is not None:
        try_read_mov(dataset_dir)

    print("\n== Resultado ==")
    print("  OK: demostracion de get_abs_path completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
