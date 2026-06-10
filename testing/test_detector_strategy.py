"""Prueba A/B del detector inyectable en el tracking (tarea detector_strategy).

Corre el tracking con el detector **yolo_sam3** (YOLO -> SAM3 box-prompt + green_floor
por texto) sobre el MISMO video canonico de ``notebooks/fase_2_YOLO_SAM3``
(``IMG_9871.MOV``), **full frames**, para comparar lado a lado con
``demo_hybrid_IMG_9871.mp4``.

Que comparar:
  - Mascaras (YOLO->box-prompt) y green_floor: deben verse equivalentes al demo.
  - obj_id / colores: NO deben empatar -> aqui se usa ByteTrack (obj_id estable) en
    vez del tracker IoU casero del notebook. Esa es la mejora.

Incluye una guarda rapida de NO-REGRESION con el detector ``sam3_text``.

IMPORTANTE: requiere YOLO (best.pt) + SAM3 + GPU -> se ejecuta en el POD. En local
falla por falta del peso/ultralytics.

Uso (en RunPod / contenedor con GPU):
    docker compose --env-file .env -f docker/docker-compose.yml \
      exec futbotmx26 python testing/test_detector_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detectors import get_detector  # noqa: E402
from src.core.tracking import track_video  # noqa: E402
from src.utils import get_abs_path  # noqa: E402

# Video canonico de fase_2 (mismo que demo_hybrid_IMG_9871.mp4).
PINNED_VIDEO = "data/raw/17Abril/Cámaras/IMG_9871.MOV"


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    # Validacion temprana del registro (no requiere modelos).
    print("== registro de detectores ==")
    get_detector("sam3_text")
    get_detector("yolo_sam3")
    try:
        get_detector("inexistente")
        print("[FALLO] get_detector no valido el nombre desconocido.")
        return 1
    except ValueError as exc:
        print(f"  OK ValueError para nombre desconocido: {exc}")

    print("\n== video pineado ==")
    try:
        get_abs_path(PINNED_VIDEO)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[FALTA] {PINNED_VIDEO}: {type(exc).__name__}: {exc}")
        print("        (ejecuta esta prueba en el pod, donde estan los datos)")
        return 1
    print(f"  {PINNED_VIDEO}")

    print("\n== track_video(detector='yolo_sam3') — full frames ==")
    res = track_video(PINNED_VIDEO, detector="yolo_sam3", render_video=True)
    assert set(res) == {"json", "video", "index"}, res.keys()
    json_path = Path(res["json"])
    assert json_path.exists(), "el JSON no se escribio"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tracks = data.get("tracks", [])
    assert tracks, "la seccion 'tracks' esta vacia"

    # obj_id unicos y estabilidad (nº tracks << nº observaciones).
    obj_ids = [t["obj_id"] for t in tracks]
    assert len(obj_ids) == len(set(obj_ids)), "obj_id no son unicos"
    n_obs = sum(len(t["observations"]) for t in tracks)
    by_class: dict[str, int] = {}
    for t in tracks:
        by_class[t["class"]] = by_class.get(t["class"], 0) + 1
    print(f"  tracks={len(tracks)}  observaciones={n_obs}")
    print(f"  tracks por clase: {by_class}")
    print(f"  JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"  mp4 : {Path(res['video']).relative_to(PROJECT_ROOT)}")

    # green_floor debe aparecer en algun frame (via text-prompt).
    green_seen = any(
        "green_floor" in fr.get("detections", {}) and fr["detections"]["green_floor"]
        for fr in data.get("frames", [])
    )
    print(f"  green_floor presente en frames: {green_seen}")

    print("\n== overlay por obj_id (A/B opcional) ==")
    try:
        from src.core.track_overlay import render_obj_id_overlay

        ov = render_obj_id_overlay(json_path, PINNED_VIDEO)
        print(f"  overlay obj_id: {Path(ov).relative_to(PROJECT_ROOT)}")
    except Exception as exc:  # noqa: BLE001 - inspeccion opcional, no debe abortar
        print(f"  (no se genero overlay obj_id: {type(exc).__name__}: {exc})")

    print("\n== no-regresion: detector='sam3_text' (clip corto) ==")
    res2 = track_video(
        PINNED_VIDEO, detector="sam3_text", max_frames=12, render_video=False
    )
    assert Path(res2["json"]).exists(), "no-regresion: JSON sam3_text no se escribio"
    print("  OK: sam3_text sigue produciendo JSON.")

    print("\n== Resultado ==")
    print("  OK: demostracion de detector_strategy completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
