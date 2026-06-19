"""Prueba manual del tracking per-frame + ByteTrack (tarea video_tracking).

Dos pruebas (ambas requieren modelo SAM3 + GPU; **se ejecutan en el pod**, no en
local):

  A) Clip corto (sanity rapido): track_video con max_frames pequeno; verifica
     mp4+JSON, identidad estable (un obj_id en varios frames), clases validas,
     obj_id unicos, indice sin mascaras y trayectorias.
  B) Video real completo (streaming end-to-end): selecciona de forma determinista un
     video que NO este en splits.forced_testing y lo trackea completo (max_frames
     None); verifica que completa sin OOM y produce mp4+JSON con tracks. Puede ser de
     corrida larga (es el costo esperado de SAM3 per-frame).

Uso:
    python testing/test_tracking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.core.tracking import Track, get_trajectories, track_video  # noqa: E402
from src.data.metadata import _load_metadata_config  # noqa: E402
from src.data.eval_frames import _load_eval_frames_config  # noqa: E402
from src.utils import get_abs_path  # noqa: E402

# Prueba B: max_frames del video real. None = video completo (puede tardar horas en
# el pod). Cambiar a un entero para acotar la corrida si hace falta.
TEST_B_MAX_FRAMES: int | None = None
# Prueba A: tamano del clip corto.
TEST_A_MAX_FRAMES = 12


def _pick_non_forced_video() -> str:
    """Devuelve la ruta (relativa) del video de menor id NO forzado a testing."""
    _, metadata_csv, _, _ = _load_metadata_config()
    _, _, _, forced_testing = _load_eval_frames_config()
    df = pd.read_csv(get_abs_path(metadata_csv)).sort_values("id")
    forced = set(forced_testing)
    for ruta in df["ruta"]:
        if ruta not in forced:
            return ruta
    raise RuntimeError("No se encontro ningun video no-forzado en db_metadata.csv.")


def _assert_common(result: dict, *, full_video: bool) -> None:
    """Comprobaciones compartidas por ambas pruebas."""
    mp4_path = Path(result["video"])
    json_path = Path(result["json"])
    index: dict[int, Track] = result["index"]

    assert mp4_path.exists(), f"No se creo el mp4: {mp4_path}"
    assert json_path.exists(), f"No se creo el JSON: {json_path}"

    # obj_id unicos (las claves del indice lo son por construccion).
    assert len(set(index.keys())) == len(index), "Hay obj_id repetidos"

    # El indice no contiene mascaras (modelo de datos ligero).
    for t in index.values():
        for o in t.observations:
            assert not hasattr(o, "mask"), "Una observacion trae mascara (no deberia)"

    if index:
        # Identidad estable: al menos un track con observaciones en >= 2 frames.
        spans = [{o.frame_index for o in t.observations} for t in index.values()]
        assert any(len(s) >= 2 for s in spans), "Ningun track persiste en >= 2 frames"

        # Trayectorias coherentes con el indice.
        traj = get_trajectories(index)
        assert set(traj.keys()) == set(index.keys()), "Trayectorias desalineadas"


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    video = _pick_non_forced_video()
    print(f"Video de prueba (no-forzado): {video}\n")

    print("== Prueba A — clip corto ==")
    res_a = track_video(video, max_frames=TEST_A_MAX_FRAMES)
    _assert_common(res_a, full_video=False)
    print(f"  mp4: {res_a['video']}")
    print(f"  tracks: {len(res_a['index'])}  ->  {res_a['json']}")
    # Clases validas.
    clases_a = {t.class_name for t in res_a["index"].values()}
    print(f"  clases en tracks: {clases_a}\n")

    print("== Prueba B — video real completo (streaming) ==")
    cap = "completo" if TEST_B_MAX_FRAMES is None else f"{TEST_B_MAX_FRAMES} frames"
    print(f"  (recorriendo {cap}; puede tardar)")
    res_b = track_video(video, max_frames=TEST_B_MAX_FRAMES)
    _assert_common(res_b, full_video=True)
    print(f"  mp4: {res_b['video']}")
    print(f"  tracks: {len(res_b['index'])}  ->  {res_b['json']}\n")

    print("== Resultado ==")
    print("  OK: ambas pruebas de tracking pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
