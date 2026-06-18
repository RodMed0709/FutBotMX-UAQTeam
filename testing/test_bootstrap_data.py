"""Prueba manual del bootstrap de datos (tarea bootstrap_data).

Smoke **sin red ni GPU**: valida la lógica pura y la clasificación de recursos sin
descargar nada de Drive.

Cubre:
  1. load_manifest / select_package sobre el manifiesto real (paquetes demo/all).
  2. normalize_drive_id (URL de archivo, de carpeta, open?id=, ID pelón).
  3. is_present (archivo / dir con archivos) e is_manual.
  4. ensure_env en un tmp: crea desde plantilla, respeta uno existente, detecta llaves
     faltantes (no-destructivo).
  5. run_bootstrap(dry_run=True) sobre un manifiesto sintético en un project_root tmp:
     clasifica presente / pendiente / manual sin tocar la red.

`src` es un paquete editable (`pip install -e .`), así que NO se parchea sys.path.

Uso:
    python testing/test_bootstrap_data.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.bootstrap_data import (
    ensure_env,
    is_manual,
    is_present,
    load_manifest,
    normalize_drive_id,
    run_bootstrap,
    select_package,
)

_fails: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'[PASS]' if cond else '[FAIL]'} {label}")
    if not cond:
        _fails.append(label)


def test_load_and_select() -> None:
    print("1) load_manifest / select_package (manifiesto real)")
    m = load_manifest()
    check(isinstance(m.get("items"), list) and len(m["items"]) > 0, "items no vacío")
    demo = select_package(m["items"], "demo")
    allp = select_package(m["items"], "all")
    check(len(demo) >= 1, f"paquete demo tiene ítems ({len(demo)})")
    check(len(allp) >= 1, f"paquete all tiene ítems ({len(allp)})")
    try:
        select_package(m["items"], "xxx")
        check(False, "paquete inválido levanta ValueError")
    except ValueError:
        check(True, "paquete inválido levanta ValueError")


def test_normalize_drive_id() -> None:
    print("2) normalize_drive_id")
    cases = {
        "https://drive.google.com/file/d/ABC123/view?usp=drive_link": "ABC123",
        "https://drive.google.com/drive/folders/FOLD9?usp=sharing": "FOLD9",
        "https://drive.google.com/open?id=XYZ&foo=bar": "XYZ",
        "PELON_42": "PELON_42",
    }
    for raw, expected in cases.items():
        check(normalize_drive_id(raw) == expected, f"{raw[:40]} -> {expected}")


def test_presence() -> None:
    print("3) is_present / is_manual")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "data").mkdir()
        (root / "data" / "x.mp4").write_text("x")
        (root / "empty").mkdir()
        check(
            is_present({"tipo": "clip", "destino": "data/x.mp4"}, root),
            "archivo presente",
        )
        check(
            not is_present({"tipo": "clip", "destino": "data/no.mp4"}, root),
            "archivo ausente",
        )
        check(is_present({"tipo": "dir", "destino": "data"}, root), "dir con archivo")
        check(not is_present({"tipo": "dir", "destino": "empty"}, root), "dir vacío")
    check(is_manual({"manual": True}) and not is_manual({}), "is_manual")


def test_ensure_env() -> None:
    print("4) ensure_env (tmp, no-destructivo)")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / ".env.example").write_text("A=1\nB=2\n")
        r1 = ensure_env(root)
        check(r1["status"] == "creado" and (root / ".env").exists(), "crea .env")
        r2 = ensure_env(root)
        check(r2["status"] == "presente", "respeta .env existente")
        (root / ".env").write_text("A=1\n")  # falta B
        r3 = ensure_env(root)
        check(r3["missing_keys"] == ["B"], "detecta llave faltante B")


def test_run_bootstrap_dry() -> None:
    print("5) run_bootstrap(dry_run) clasifica presente/pendiente/manual")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "assets").mkdir()
        (root / ".env.example").write_text("CONFIG_FILENAME=x\n")
        # presente: creamos el archivo del item 'pesos'
        (root / "assets" / "w.pt").parent.mkdir(exist_ok=True)
        (root / "assets" / "w.pt").write_text("w")
        manifest = {
            "schema_version": 1,
            "items": [
                {
                    "nombre": "clip_ausente",
                    "paquetes": ["demo"],
                    "recursos": [
                        {"tipo": "clip", "drive_id": "ID1", "destino": "data/a.mp4"}
                    ],
                },
                {
                    "nombre": "pesos",
                    "paquetes": ["demo"],
                    "recursos": [
                        {"tipo": "file", "drive_id": "ID2", "destino": "assets/w.pt"}
                    ],
                },
                {
                    "nombre": "dataset",
                    "paquetes": ["demo"],
                    "recursos": [
                        {
                            "tipo": "dir",
                            "manual": True,
                            "drive_id": "FOLD",
                            "destino": "data/raw/17Abril",
                        }
                    ],
                },
            ],
        }
        (root / "assets" / "bootstrap_manifest.json").write_text(json.dumps(manifest))
        rep = run_bootstrap("demo", root, dry_run=True)
        estados = {o.nombre: o.estado for o in rep.outcomes}
        check(rep.env["status"] == "creado", ".env creado en el run")
        check(estados.get("clip_ausente") == "pendiente", "ausente -> pendiente")
        check(estados.get("pesos") == "presente", "existente -> presente")
        check(estados.get("dataset") == "manual", "manual ausente -> manual")


def main() -> int:
    for t in (
        test_load_and_select,
        test_normalize_drive_id,
        test_presence,
        test_ensure_env,
        test_run_bootstrap_dry,
    ):
        t()
    print()
    if _fails:
        print(f"FALLARON {len(_fails)} checks: {_fails}")
        return 1
    print("Todos los checks pasaron ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
