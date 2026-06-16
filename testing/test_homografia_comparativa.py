"""Comparativa de homografía/minimap: notebook vs paquete de src, MISMO clip.

Corre los DOS pipelines sobre el **mismo** tramo (``IMG_9933_c``: start=15000,
every=2 → 150 frames) y guarda ambos mp4 en ``outputs/homografia_comparativa/``,
para comparar sin ambigüedad de "¿es el clip correcto?":

1. ``notebooks/fase_4_homografia/pod_minimap_sam3.py`` (script suelto, YOLO + SAM3).
2. ``src.core.minimap_pipeline.render_minimap_video`` (paquete, ``detector="yolo_sam3"``).

Ambos usan los MISMOS índices de frame y ``conf=0.25``. El script imprime además qué
``best.pt`` usa cada uno (rutas distintas = posible causa de diferencias).

Corre en el POD (GPU + SAM3 + best.pt). NO en local.

    python testing/test_homografia_comparativa.py
"""

import sys

from src.utils import PROJECT_ROOT

# --- Clip de comparación: IMG_9933_c (mismo que el video de referencia del equipo) ---
VIDEO = PROJECT_ROOT / "data/raw/18abril/Camara_superior/IMG_9933.MOV"
OUT_DIR = PROJECT_ROOT / "outputs" / "homografia_comparativa"
START, N_SRC, EVERY = 15000, 300, 2
MAX_FRAMES = len(range(START, START + N_SRC, EVERY))  # 150
CONF = 0.25  # el de la demo (pod_minimap_sam3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not VIDEO.exists():
        raise FileNotFoundError(f"No se encontró el video: {VIDEO}")

    print(f"clip: start={START} every={EVERY} -> {MAX_FRAMES} frames | conf={CONF}")
    print(f"salida: {OUT_DIR}\n")

    # ------------------------------------------------------------------ #
    # 1) Notebook: pod_minimap_sam3.run()                                 #
    # ------------------------------------------------------------------ #
    f4 = PROJECT_ROOT / "notebooks" / "fase_4_homografia"
    sys.path.insert(0, str(f4))  # el módulo añade src/core y fase_2 a sys.path al importarse
    import pod_minimap_sam3 as pod  # noqa: E402

    print(f"[notebook] YOLO best.pt: {pod.YOLO_PT}")
    pod_models = pod.p2.load_models(pod.SAM3_PATH, pod.YOLO_PT, pod.DEVICE)
    out_pod = OUT_DIR / "1_notebook_pod_minimap_sam3.mp4"
    r_pod = pod.run(str(VIDEO), str(out_pod), pod_models,
                    start=START, n_src=N_SRC, every=EVERY, conf=CONF)
    print(f"[notebook] {r_pod}\n")

    # Liberar VRAM antes de cargar los modelos del paquete (evita OOM).
    del pod_models
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # 2) Paquete: src.core.minimap_pipeline.render_minimap_video()        #
    # ------------------------------------------------------------------ #
    from src.core.detectors.yolo_boxes import _resolve_weights
    from src.core.minimap_pipeline import render_minimap_video

    try:
        print(f"[src] YOLO best.pt: {_resolve_weights(None)}")
    except FileNotFoundError as exc:
        print(f"[src] YOLO best.pt: NO RESUELTO -> {exc}")

    out_src = OUT_DIR / "2_src_render_minimap_video_yolo.mp4"
    r_src = render_minimap_video(
        VIDEO, detector="yolo", conf=CONF,  # camino rápido (cajas YOLO + green SAM3) = demo
        start_frame=START, frame_step=EVERY, max_frames=MAX_FRAMES,
        draw_overlay=True, output_path=out_src,
    )
    print(f"[src] {r_src}\n")

    print("LISTO. Compara los dos videos del mismo clip:")
    print(f"  {out_pod}")
    print(f"  {out_src}")
    print("Referencia del equipo (si existe):")
    print(f"  {f4 / 'outputs' / 'IMG_9933_c_minimap.mp4'}")


if __name__ == "__main__":
    main()
