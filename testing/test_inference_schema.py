"""Prueba manual del esquema comun del entregable de inferencia (tarea inference_schema).

Dos partes:

  A) LOCAL (sin GPU): ejercita los helpers de ``src.core.inference_schema`` con
     mascaras sinteticas: round-trip RLE sin perdida, geometria (bbox/centroide),
     registro de deteccion con/sin mascara, ensamblado + escritura/relectura del
     JSON, y la RECONSTRUCCION SIN MODELO (decodificar RLE + overlay sobre un frame
     dummy). No requiere SAM3 ni GPU.

  B) POD (GPU): corre los orquestadores reales (run_pipeline y track_video) con
     include_masks; verifica el JSON unificado en outputs/inference/<stem>/, los
     frame_index reales, la fusion frames+tracks en un solo archivo y el
     comportamiento de include_masks ON/OFF. Requiere modelo SAM3 + GPU.

Uso:
    python testing/test_inference_schema.py          # solo Parte A (local)
    python testing/test_inference_schema.py pod       # Parte A + Parte B (en el pod)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.core.inference_schema import (  # noqa: E402
    SCHEMA_VERSION,
    build_header,
    decode_rle,
    detection_record,
    encode_rle,
    frame_record,
    mask_to_bbox_centroid,
    write_inference_json,
)
from src.core.overlay import overlay_detections  # noqa: E402
from src.core.segmentation import Detection  # noqa: E402

_CLASSES = [{"name": "robot", "color": [255, 0, 0]}]


def part_a_local() -> None:
    """Parte A — helpers del modulo con datos sinteticos (sin GPU)."""
    print("== Parte A — local (sin GPU) ==")
    h, w = 40, 60
    mask = np.zeros((h, w), dtype=bool)
    mask[10:20, 5:25] = True  # y=10..19 (h=10), x=5..24 (w=20)

    # 1) RLE round-trip sin perdida.
    rle = encode_rle(mask)
    assert rle["size"] == [h, w], f"size RLE inesperado: {rle['size']}"
    assert np.array_equal(decode_rle(rle), mask), "RLE no es sin perdida"
    print("  [ok] RLE ida-vuelta sin perdida")

    # 2) Geometria.
    geom = mask_to_bbox_centroid(mask)
    assert geom is not None
    bbox, centroid = geom
    assert bbox == [5, 10, 20, 10], f"bbox inesperado: {bbox}"
    assert centroid == [15.0, 15.0], f"centroide inesperado: {centroid}"
    assert mask_to_bbox_centroid(np.zeros((h, w), bool)) is None, "vacia debe dar None"
    print("  [ok] bbox/centroide y mascara vacia -> None")

    # 3) detection_record con / sin mascara.
    det = Detection(obj_id=7, mask=mask, score=0.9)
    rec_on = detection_record(det, include_masks=True)
    rec_off = detection_record(det, include_masks=False)
    assert rec_on is not None and rec_off is not None
    assert "rle" in rec_on and "rle" not in rec_off, "include_masks no respetado"
    assert rec_on["obj_id"] == 7 and rec_on["bbox"] == [5, 10, 20, 10]
    print("  [ok] detection_record respeta include_masks")

    # 4) Ensamblado + escritura + relectura.
    fr = frame_record(3, {"robot": [det]}, include_masks=True)
    assert fr["frame_index"] == 3
    config = {"working_dirs": {"sam3_dir": "assets/sam3"}, "dummy": True}
    header = build_header(
        video="x.MOV",
        mode="segmentation",
        fps=30.0,
        resolution=(h, w),
        num_frames=1,
        classes=_CLASSES,
        include_masks=True,
        config=config,
    )
    assert header["schema_version"] == SCHEMA_VERSION
    assert header["resolution"] == {"height": h, "width": w}
    assert header["config"] == config, "el snapshot de config no se embebio"

    with tempfile.TemporaryDirectory() as d:
        jp = Path(d) / "out.json"
        write_inference_json(header, [fr], jp)
        loaded = json.loads(jp.read_text(encoding="utf-8"))
    det_loaded = loaded["frames"][0]["detections"]["robot"][0]
    assert det_loaded["rle"]["size"] == [h, w]
    assert "config" in loaded and "classes" in loaded
    print("  [ok] cabecera + frames se escriben y releen")

    # 5) Reconstruccion SIN modelo: decodificar RLE + overlay sobre frame dummy.
    recon = decode_rle(det_loaded["rle"])
    assert np.array_equal(recon, mask), "la mascara reconstruida difiere"
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    det2 = Detection(obj_id=7, mask=recon, score=0.9)
    composed = overlay_detections(frame, {"robot": [det2]}, classes=_CLASSES, alpha=0.5)
    assert composed.shape == (h, w, 3) and composed.dtype == np.uint8
    assert composed[mask].sum() > 0, "el overlay no pinto la mascara reconstruida"
    print("  [ok] reconstruccion visual sin invocar el modelo\n")


def _pick_non_forced_video() -> str:
    """Ruta (relativa) del video de menor id NO forzado a testing."""
    import pandas as pd

    from src.data.eval_frames import _load_eval_frames_config
    from src.data.metadata import _load_metadata_config
    from src.utils import get_abs_path

    _, metadata_csv, _, _ = _load_metadata_config()
    _, _, _, forced_testing = _load_eval_frames_config()
    df = pd.read_csv(get_abs_path(metadata_csv)).sort_values("id")
    forced = set(forced_testing)
    for ruta in df["ruta"]:
        if ruta not in forced:
            return ruta
    raise RuntimeError("No se encontro ningun video no-forzado en db_metadata.csv.")


def _iter_dets(data: dict):
    """Itera todas las detecciones de la vista frame-indexed de un payload."""
    for f in data["frames"]:
        for dets in f["detections"].values():
            yield from dets


def part_b_pod() -> None:
    """Parte B — orquestadores reales en el pod (GPU)."""
    print("== Parte B — pod (GPU) ==")
    from src.core.frame_extraction import get_frame_indices
    from src.core.pipeline import run_pipeline
    from src.core.tracking import track_video

    video = _pick_non_forced_video()
    print(f"  video: {video}")

    # 1) Seg-only con mascaras.
    res = run_pipeline(video, all_frames=False, include_masks=True)
    jp = Path(res["json"])
    assert jp.exists(), f"no se creo el JSON: {jp}"
    assert jp.parent.name == Path(video).stem, "el JSON no esta en carpeta por video"
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert data["mode"] == "segmentation" and data["include_masks"] is True
    src_idx = [int(i) for i in get_frame_indices(Path(video), all_frames=False)]
    got_idx = [f["frame_index"] for f in data["frames"]]
    assert got_idx == src_idx, "frame_index no coinciden con get_frame_indices"
    assert any("rle" in d for d in _iter_dets(data)), "no hay rle pese a include_masks"
    print(f"  [ok] seg-only -> {jp}")

    # 2) Tracking: un UNICO JSON con frames Y tracks (sin _tracks.json aparte).
    res_t = track_video(video, max_frames=8, include_masks=True)
    jp_t = Path(res_t["json"])
    assert jp_t.exists()
    legacy = jp_t.with_name(f"{jp_t.stem}_tracks.json")
    assert not legacy.exists(), f"existe un _tracks.json aparte: {legacy}"
    data_t = json.loads(jp_t.read_text(encoding="utf-8"))
    assert "frames" in data_t and "tracks" in data_t, "no se fundieron frames+tracks"
    assert data_t["mode"] == "tracking"
    print(f"  [ok] tracking unificado -> {jp_t}")

    # 3) include_masks=False -> sin rle.
    res_off = track_video(video, max_frames=4, include_masks=False)
    data_off = json.loads(Path(res_off["json"]).read_text(encoding="utf-8"))
    assert all("rle" not in d for d in _iter_dets(data_off)), "hay rle con masks OFF"
    print("  [ok] include_masks=False -> sin rle\n")


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
    part_a_local()

    if len(sys.argv) > 1 and sys.argv[1] == "pod":
        part_b_pod()
    else:
        print("(Parte B omitida: pasa 'pod' como argumento para correrla en GPU)\n")

    print("== Resultado ==")
    print("  OK: las pruebas del esquema de inferencia pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
