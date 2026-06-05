"""Prueba manual del manifiesto de metadatos del dataset (tarea csv_dataset_metadata).

Flujo:
  1. Genera el CSV con build_metadata_csv(force=True).
  2. Comprueba esquema (columnas/orden), id secuencial, una fila por video, tipos y
     rangos, y que la columna 'ruta' resuelve con get_abs_path.
  3. Comprueba los conteos de splits (23 / 20 / resto), disjuntos y cubrientes.
  4. Reproducibilidad: dos corridas force=True producen la misma columna 'split'.
  5. Idempotencia: force=False no reescribe (mtime estable) y el handler valida; un
     header corrupto invalida el esquema y fuerza regeneracion.

No usa modelo ni GPU: lee solo metadatos de los videos reales bajo data/raw.

Uso:
    python testing/test_metadata.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_metadata_csv, validate_metadata_schema  # noqa: E402
from src.data.metadata import (  # noqa: E402
    COLUMNS,
    SPLIT_FINETUNING,
    SPLIT_RESERVE,
    SPLIT_SIZES,
    SPLIT_TESTING,
    _load_metadata_config,
)
from src.utils import get_abs_path  # noqa: E402


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    _, metadata_csv, _, forced_testing = _load_metadata_config()
    csv_path = PROJECT_ROOT / metadata_csv

    print("== Generacion (force=True) ==")
    df = build_metadata_csv(force=True)
    assert csv_path.exists(), f"No se creo el CSV: {csv_path}"
    print(f"  CSV creado: {csv_path}  filas: {len(df)}\n")

    print("== Esquema y contenido ==")
    assert list(df.columns) == COLUMNS, f"Columnas inesperadas: {list(df.columns)}"
    assert list(df["id"]) == list(range(len(df))), "id no es secuencial 0..N-1"
    assert df["nombre"].notna().all(), "Hay nombres vacios"
    assert (df["ancho"] > 0).all() and (df["alto"] > 0).all(), "ancho/alto invalidos"
    assert (df["fps_average"] > 0).all(), "fps_average invalido"
    assert (df["duracion"] > 0).all(), "duracion invalida"
    # 'ruta' relativa debe resolver a un archivo existente.
    for ruta in df["ruta"]:
        get_abs_path(ruta)
    print(
        "  columnas/orden OK, id secuencial OK, tipos/rangos OK, rutas resueltas OK\n"
    )

    print("== Splits ==")
    counts = df["split"].value_counts().to_dict()
    print(f"  conteos: {counts}")
    assert counts.get(SPLIT_FINETUNING, 0) == SPLIT_SIZES[SPLIT_FINETUNING]
    assert counts.get(SPLIT_TESTING, 0) == SPLIT_SIZES[SPLIT_TESTING]
    reserva = len(df) - sum(SPLIT_SIZES.values())
    assert counts.get(SPLIT_RESERVE, 0) == reserva
    assert set(df["split"].unique()) <= {SPLIT_RESERVE, SPLIT_FINETUNING, SPLIT_TESTING}
    print(f"  23/20/{reserva} disjuntos y cubrientes  OK\n")

    print("== Videos fijados a testing ==")
    print(f"  forzados: {forced_testing}")
    for ruta in forced_testing:
        fila = df[df["ruta"] == ruta]
        assert not fila.empty, f"Ruta fijada ausente del CSV: {ruta}"
        assert int(fila["split"].iloc[0]) == SPLIT_TESTING, f"{ruta} no esta en testing"
    print("  todos los fijados en testing (split=2)  OK\n")

    print("== Reproducibilidad ==")
    df2 = build_metadata_csv(force=True)
    assert list(df["split"]) == list(df2["split"]), "Los splits no son reproducibles"
    print("  misma seed -> misma particion  OK\n")

    print("== Idempotencia (force=False no reescribe) ==")
    mtime_before = csv_path.stat().st_mtime_ns
    time.sleep(0.01)
    build_metadata_csv(force=False)
    assert csv_path.stat().st_mtime_ns == mtime_before, "force=False reescribio el CSV"
    assert validate_metadata_schema(csv_path) is True, "Handler invalida un CSV valido"
    print("  CSV no reescrito y handler valida  OK\n")

    print("== Handler ante esquema corrupto ==")
    backup = csv_path.read_text(encoding="utf-8")
    csv_path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
    assert validate_metadata_schema(csv_path) is False, "Handler valido un esquema malo"
    # Restaurar via regeneracion (debe sobrescribir el header corrupto).
    df3 = build_metadata_csv(force=False)
    assert list(df3.columns) == COLUMNS, "No se regenero tras esquema corrupto"
    assert validate_metadata_schema(csv_path) is True
    _ = backup  # contenido previo ya regenerado de forma determinista
    print("  esquema corrupto -> False -> regeneracion  OK\n")

    print("== Resultado ==")
    print("  OK: todas las comprobaciones del manifiesto pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
