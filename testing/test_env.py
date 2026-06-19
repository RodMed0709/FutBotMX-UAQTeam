"""Prueba de humo del entorno de ejecucion (env_setup).

Verifica que el entorno es funcional:
  1. Importa las librerias clave.
  2. Reporta la version de cada una.
  3. Verifica la disponibilidad de GPU/CUDA (su ausencia se informa, no falla).

Uso:
    python testing/test_env.py

Codigo de salida 0 si todas las importaciones requeridas tienen exito; 1 si
alguna libreria requerida no se pudo importar.
"""

from __future__ import annotations

import importlib
import sys

# (modulo_a_importar, nombre_legible) — librerias clave del entorno base.
REQUIRED = [
    ("numpy", "NumPy"),
    ("cv2", "OpenCV"),
    ("matplotlib", "Matplotlib"),
    ("notebook", "Jupyter Notebook"),
    ("ipykernel", "IPyKernel"),
    ("torch", "PyTorch"),
]


def _version(module) -> str:
    return getattr(module, "__version__", "desconocida")


def check_imports() -> tuple[dict[str, object], list[str]]:
    """Importa cada libreria requerida y reporta su version."""
    imported: dict[str, object] = {}
    failed: list[str] = []

    print("== Importaciones y versiones ==")
    for module_name, label in REQUIRED:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - queremos reportar cualquier fallo
            failed.append(module_name)
            print(f"  [FALLO] {label} ({module_name}): {exc}")
            continue
        imported[module_name] = module
        print(f"  [ok]    {label}: {_version(module)}")

    return imported, failed


def check_gpu(torch_module) -> None:
    """Reporta el estado de GPU/CUDA sin considerar su ausencia como fallo."""
    print("\n== GPU / CUDA ==")
    if torch_module is None:
        print("  torch no disponible; se omite la verificacion de GPU.")
        return

    available = torch_module.cuda.is_available()
    print(f"  CUDA disponible: {available}")
    if available:
        count = torch_module.cuda.device_count()
        print(f"  Dispositivos CUDA: {count}")
        for i in range(count):
            print(f"    [{i}] {torch_module.cuda.get_device_name(i)}")
        print(f"  Version CUDA (torch): {torch_module.version.cuda}")
    else:
        print("  Sin GPU CUDA detectada (esperado en entornos CPU; en RunPod deberia haber GPU).")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}\n")

    imported, failed = check_imports()
    check_gpu(imported.get("torch"))

    print("\n== Resultado ==")
    if failed:
        print(f"  FALLIDO: no se pudieron importar: {', '.join(failed)}")
        return 1

    print("  OK: el entorno es funcional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
