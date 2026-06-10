"""Prueba A/B del tracker BoT-SORT vs ByteTrack (tarea botsort_tracker).

Corre el tracking con el detector ``yolo_sam3`` y compara el tracker ``botsort``
(ultralytics + GMC) contra ``bytetrack`` (actual) sobre el MISMO video del smoke
anterior (``IMG_9871.MOV``), full frames. Con GMC se espera **menos fragmentacion**
(<= nº de tracks por clase). Es reporte comparativo + aserciones de sanidad.

IMPORTANTE: requiere ultralytics (BoT-SORT + YOLO best.pt) + SAM3 + GPU -> se ejecuta
en el POD. En local falla por falta de pesos/ultralytics.

Uso (en RunPod / contenedor con GPU):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_botsort_tracker.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.track_overlay import render_obj_id_overlay  # noqa: E402
from src.core.tracking import track_video  # noqa: E402
from src.core.trackers import get_tracker  # noqa: E402
from src.utils import get_abs_path  # noqa: E402

# Mismo video canonico del smoke anterior (detector_strategy).
PINNED_VIDEO = "data/raw/17Abril/Cámaras/IMG_9871.MOV"


def _tracks_by_class(json_path: Path) -> dict[str, int]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    tracks = data.get("tracks", [])
    by_class: dict[str, int] = {}
    for t in tracks:
        by_class[t["class"]] = by_class.get(t["class"], 0) + 1
    # sanidad: obj_id unicos
    obj_ids = [t["obj_id"] for t in tracks]
    assert len(obj_ids) == len(set(obj_ids)), "obj_id no son unicos"
    assert tracks, "la seccion 'tracks' esta vacia"
    return by_class


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    print("== validacion del factory ==")
    get_tracker("bytetrack", frame_rate=30.0)
    try:
        get_tracker("inexistente", frame_rate=30.0)
        print("[FALLO] get_tracker no valido el nombre desconocido.")
        return 1
    except ValueError as exc:
        print(f"  OK ValueError: {exc}")

    print("\n== video pineado ==")
    try:
        get_abs_path(PINNED_VIDEO)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[FALTA] {PINNED_VIDEO}: {type(exc).__name__}: {exc}")
        print("        (ejecuta esta prueba en el pod, donde estan los datos)")
        return 1
    print(f"  {PINNED_VIDEO}")

    # Salidas DISTINTAS por tracker, fuera de outputs/inference/ (que usa el stem del
    # video) para no pisar artefactos de smokes previos.
    ab_dir = PROJECT_ROOT / "outputs" / "botsort_ab"
    ab_dir.mkdir(parents=True, exist_ok=True)

    def run(tracker_name: str):
        # include_masks=True -> el JSON lleva RLE para que el overlay pinte mascaras.
        # render_video=False -> se omite el overlay "vivo" (solo mascaras por clase, no
        # muestra IDs); el video util para el A/B es el post-pass por obj_id de abajo.
        res = track_video(
            PINNED_VIDEO,
            output_path=ab_dir / f"IMG_9871_{tracker_name}.mp4",
            detector="yolo_sam3",
            tracker=tracker_name,
            include_masks=True,
            render_video=False,
        )
        assert set(res) == {"json", "video", "index"}, res.keys()
        by_class = _tracks_by_class(res["json"])
        # Post-pass: overlay por obj_id (caja + 'nombre #id' + estela + mascara RLE).
        ov = render_obj_id_overlay(
            res["json"],
            PINNED_VIDEO,
            output_path=ab_dir / f"IMG_9871_{tracker_name}_objid.mp4",
            draw_masks=True,
        )
        print(f"  {tracker_name:9s} tracks/clase: {by_class}")
        print(f"    JSON   : {Path(res['json']).relative_to(PROJECT_ROOT)}")
        print(f"    overlay: {Path(ov).relative_to(PROJECT_ROOT)}")
        return res, by_class

    print("\n== botsort (detector='yolo_sam3') — full frames + overlay obj_id ==")
    res_bs, bs_by_class = run("botsort")

    # green_floor presente (via text-prompt).
    data = json.loads(Path(res_bs["json"]).read_text(encoding="utf-8"))
    green_seen = any(
        det for fr in data.get("frames", [])
        for cls_, det in fr.get("detections", {}).items()
        if cls_ == "green_floor" and det
    )
    print(f"  green_floor presente en frames: {green_seen}")

    print("\n== A/B: bytetrack (mismo video) — full frames + overlay obj_id ==")
    res_bt, bt_by_class = run("bytetrack")

    print("\n== comparativa (menos = menos fragmentacion) ==")
    for name in sorted(set(bs_by_class) | set(bt_by_class)):
        b = bs_by_class.get(name, 0)
        y = bt_by_class.get(name, 0)
        flag = "OK" if b <= y else "↑"
        print(f"  {name:12s}  botsort={b:3d}  bytetrack={y:3d}  [{flag}]")

    print("\n== Resultado ==")
    print("  OK: demostracion de botsort_tracker completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
