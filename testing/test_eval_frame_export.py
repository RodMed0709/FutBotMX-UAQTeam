"""Prueba manual del set de frames de evaluacion (tarea eval_frame_export).

Flujo:
  1. Genera el set con export_testing_frames(force=True).
  2. Comprueba esquema (columnas/orden), id secuencial, video_id ⊆ ids split==2,
     20 videos distintos y grupo ∈ {aleatorio, cenital}.
  3. Verifica que cada 'imagen' existe en disco y abre como PNG valido.
  4. frame_original: para un video de muestra, coincide con get_frame_indices
     alineado por posicion con frame_index; extract_frames conserva su salida.
  5. Los 2 videos de forced_testing quedan marcados 'cenital'; el resto 'aleatorio'.
  6. Idempotencia: force=False no reescribe (mtime estable) y el handler valida; un
     header corrupto invalida el esquema y fuerza regeneracion.

No usa modelo ni GPU: lee los videos reales bajo data/raw y escribe PNGs bajo
data/testing_frames. Pensado para ejecutarse en el pod (volumen compartido), aunque
corre igual en local si data/raw esta poblado.

Uso:
    python testing/test_eval_frame_export.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from src.core.frame_extraction import extract_frames, get_frame_indices  # noqa: E402
from src.data import (  # noqa: E402
    export_testing_frames,
    validate_testing_frames_schema,
)
from src.data.eval_frames import (  # noqa: E402
    COLUMNS,
    GROUP_CENITAL,
    GROUP_RANDOM,
    _load_eval_frames_config,
    _load_testing_videos,
)
from src.utils import get_abs_path  # noqa: E402


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    metadata_csv, _, frames_csv, forced_testing = _load_eval_frames_config()
    csv_path = PROJECT_ROOT / frames_csv

    print("== Generacion (force=True) ==")
    df = export_testing_frames(force=True)
    assert csv_path.exists(), f"No se creo el CSV: {csv_path}"
    print(f"  CSV creado: {csv_path}  filas: {len(df)}\n")

    print("== Esquema y contenido ==")
    assert list(df.columns) == COLUMNS, f"Columnas inesperadas: {list(df.columns)}"
    assert list(df["id"]) == list(range(len(df))), "id no es secuencial 0..M-1"
    testing_videos = _load_testing_videos(metadata_csv)
    testing_ids = set(testing_videos["id"].astype(int))
    assert set(df["video_id"]) <= testing_ids, "Hay video_id fuera de split==2"
    assert df["video_id"].nunique() == 20, "Se esperaban 20 videos de testing"
    assert set(df["grupo"]).issubset({GROUP_RANDOM, GROUP_CENITAL}), "grupo invalido"
    print(
        f"  columnas OK, id secuencial OK, {df['video_id'].nunique()} videos, grupos OK\n"
    )

    print("== Imagenes en disco ==")
    for ruta_img in df["imagen"]:
        abs_img = get_abs_path(ruta_img)  # resuelve y verifica existencia
        img = cv2.imread(str(abs_img))
        assert img is not None, f"PNG ilegible: {abs_img}"
    print(f"  {len(df)} imagenes existen y abren como PNG  OK\n")

    print("== frame_original (sin logica duplicada) ==")
    sample_video_id = int(df["video_id"].iloc[0])
    sample = df[df["video_id"] == sample_video_id].sort_values("frame_index")
    ruta = sample["video_ruta"].iloc[0]
    indices = get_frame_indices(Path(ruta), all_frames=False)
    assert list(sample["frame_original"]) == [
        int(i) for i in indices
    ], "frame_original no coincide con get_frame_indices"
    frames = extract_frames(Path(ruta), all_frames=False)
    assert frames.ndim == 4 and frames.shape[-1] == 3, "extract_frames cambio su salida"
    assert len(frames) == len(indices), "frames e indices desalineados"
    print(f"  video {sample_video_id}: frame_original alineado con indices  OK\n")

    print("== Grupos (cenital vs aleatorio) ==")
    forced_set = set(forced_testing)
    print(f"  forzados (cenital): {forced_testing}")
    for ruta_v in df["video_ruta"].unique():
        esperado = GROUP_CENITAL if ruta_v in forced_set else GROUP_RANDOM
        grupos = set(df[df["video_ruta"] == ruta_v]["grupo"])
        assert grupos == {esperado}, f"{ruta_v}: grupo {grupos} != {esperado}"
    print("  todos los videos con el grupo correcto  OK\n")

    print("== Idempotencia (force=False no reescribe) ==")
    mtime_before = csv_path.stat().st_mtime_ns
    time.sleep(0.01)
    export_testing_frames(force=False)
    assert csv_path.stat().st_mtime_ns == mtime_before, "force=False reescribio el CSV"
    assert (
        validate_testing_frames_schema(csv_path) is True
    ), "Handler invalida un CSV valido"
    print("  CSV no reescrito y handler valida  OK\n")

    print("== Handler ante esquema corrupto ==")
    csv_path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
    assert (
        validate_testing_frames_schema(csv_path) is False
    ), "Handler valido un esquema malo"
    df2 = export_testing_frames(force=False)  # debe regenerar por esquema invalido
    assert list(df2.columns) == COLUMNS, "No se regenero tras esquema corrupto"
    assert validate_testing_frames_schema(csv_path) is True
    print("  esquema corrupto -> False -> regeneracion  OK\n")

    print("== Resultado ==")
    print("  OK: todas las comprobaciones del set de evaluacion pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
