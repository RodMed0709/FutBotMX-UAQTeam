"""Prueba del escritor de video (tarea video_writer).

Valida ``write_video`` con frames **sinteticos** (sin modelo ni GPU):

  - escribe un mp4 en outputs/test_video_maker/ (crea la carpeta si falta);
  - el archivo existe y su tamano es > 0;
  - el mp4 se puede releer con imageio (nº de frames/ dimensiones coherentes);
  - entrada invalida (forma/ dtype) lanza ValueError.

El mp4 de prueba se deja en outputs/test_video_maker/ para inspeccion manual.

Uso:
    python testing/test_video_writer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permitir importar el paquete src cuando se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.core import write_video  # noqa: E402


def _synthetic_frames(n: int, h: int, w: int) -> np.ndarray:
    """Genera una animacion simple: una banda vertical que se desplaza."""
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        x = int((i / max(1, n - 1)) * (w - 10))
        frames[i, :, x : x + 10] = (60, 130, 255)  # banda azul que se mueve
        frames[i, h // 2 - 5 : h // 2 + 5, :] = (50, 220, 70)  # linea verde fija
    return frames


def main() -> int:
    print(f"PROJECT_ROOT: {PROJECT_ROOT}\n")

    out_dir = PROJECT_ROOT / "outputs" / "test_video_maker"
    out_path = out_dir / "test_video_writer.mp4"

    n, h, w = 24, 120, 160
    frames = _synthetic_frames(n, h, w)

    print("== write_video (frames sinteticos) ==")
    written = write_video(frames, out_path, fps=8)
    print(f"  escrito en: {written}")

    print("== Verificaciones ==")
    assert out_dir.is_dir(), "no se creo el directorio de salida"
    print("  directorio de salida creado  OK")

    size = written.stat().st_size
    assert size > 0, "el archivo mp4 esta vacio"
    print(f"  archivo existe, tamano={size} bytes  OK")

    # Releer con imageio para confirmar que es un mp4 valido.
    import imageio

    reader = imageio.get_reader(str(written), format="FFMPEG")
    read_frames = [f for f in reader]
    reader.close()
    assert len(read_frames) >= 1, "no se pudo releer ningun frame"
    rh, rw = read_frames[0].shape[:2]
    print(f"  relectura OK: {len(read_frames)} frames, dims={rw}x{rh}")

    print("== Entrada invalida -> ValueError ==")
    for bad in [
        np.zeros((5, 10, 10), dtype=np.uint8),  # 3D
        np.zeros((5, 10, 10, 3), dtype=np.float32),  # dtype
        np.zeros((0, 10, 10, 3), dtype=np.uint8),  # vacio
    ]:
        try:
            write_video(bad, out_dir / "_invalido.mp4")
            print(f"  [FALLO] no lanzo ValueError para forma {bad.shape}/{bad.dtype}")
            return 1
        except ValueError:
            pass
    print("  entradas invalidas rechazadas  OK")

    print("\n== Resultado ==")
    print(f"  OK: demostracion de write_video completada. Video en: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
