"""Prueba manual de la carga del modelo SAM3 (tarea sam3_loader).

Flujo:
  1. load_sam3() -> carga el modelo; reporta clase, device, dtype y num. params.
  2. load_sam3() de nuevo -> debe devolver el MISMO objeto (cache singleton).
  3. load_sam3(use_cache=False) -> debe devolver un objeto DISTINTO (recarga).

Requiere que los pesos de SAM3 esten presentes en disco (assets/sam3). En el
host sin los pesos descargados, la carga fallara de forma controlada y se
reportara sin abortar abruptamente. Por eso esta prueba debe ejecutarse donde los
pesos esten disponibles (contenedor o pod con GPU).

Uso (en el contenedor):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_sam3_loader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import load_sam3  # noqa: E402


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    print("== Carga inicial (load_sam3) ==")
    try:
        bundle = load_sam3()
    except FileNotFoundError as exc:
        print(f"[FALTA] No se encontraron los pesos del modelo: {exc}")
        print(
            "        (esperado en el host; ejecuta esta prueba donde esten los pesos)"
        )
        return 1
    except (ValueError, KeyError) as exc:
        print(f"[FALLO] Configuracion invalida: {type(exc).__name__}: {exc}")
        return 1

    # Conteo de parametros (import perezoso de torch ya ocurrio dentro de load_sam3).
    n_params = sum(p.numel() for p in bundle.model.parameters())
    dtype = next(bundle.model.parameters()).dtype
    print(f"  clase    : {type(bundle.model).__name__}")
    print(f"  device   : {bundle.device}")
    print(f"  dtype    : {dtype}")
    print(f"  params   : {n_params / 1e6:.1f}M\n")

    print("== Reuso de cache (segunda llamada por defecto) ==")
    bundle2 = load_sam3()
    if bundle2 is bundle:
        print("  OK: la segunda llamada reutiliza el mismo objeto (singleton).\n")
    else:
        print("  [FALLO] la segunda llamada NO reutilizo la cache.\n")
        return 1

    print("== Opt-out de cache (use_cache=False) ==")
    bundle3 = load_sam3(use_cache=False)
    if bundle3 is not bundle:
        print("  OK: use_cache=False entrego una instancia fresca.\n")
    else:
        print("  [FALLO] use_cache=False devolvio el objeto cacheado.\n")
        return 1

    print("== Resultado ==")
    print("  OK: demostracion de load_sam3 completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
