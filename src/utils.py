"""Funciones generales y utilitarias del proyecto FutBotMX.

Utilidades:
- ``get_abs_path``: resolucion de rutas relativas a rutas absolutas respecto a la
  raiz del proyecto.
- ``show_frames``: visualizacion de un conjunto de frames en una cuadricula.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np

# Raiz del proyecto: src/ cuelga directamente de la raiz, por lo que parents[1]
# apunta a la raiz sin importar el directorio de trabajo (cwd).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_abs_path(relative_path: str) -> Path:
    """Convierte una ruta relativa (str) en su ruta absoluta (Path).

    La ruta se resuelve respecto a la raiz del proyecto, de modo que el resultado
    es estable independientemente del directorio de trabajo. Verifica que la ruta
    resuelta exista.

    Args:
        relative_path: ruta relativa, en forma de cadena, respecto a la raiz del
            proyecto.

    Returns:
        La ruta absoluta correspondiente como objeto ``pathlib.Path``.

    Raises:
        ValueError: si ``relative_path`` no es ``str``, esta vacio/solo espacios,
            o es una ruta absoluta en lugar de relativa.
        FileNotFoundError: si la ruta resuelta no existe.
    """
    if not isinstance(relative_path, str):
        raise ValueError(
            f"Se esperaba una ruta relativa de tipo str, se recibio: {type(relative_path).__name__}"
        )

    cleaned = relative_path.strip()
    if not cleaned:
        raise ValueError("La ruta relativa esta vacia.")

    candidate = Path(cleaned)
    if candidate.is_absolute():
        raise ValueError(
            f"Se esperaba una ruta relativa, pero se recibio una ruta absoluta: {relative_path!r}"
        )

    abs_path = (PROJECT_ROOT / candidate).resolve()

    if not abs_path.exists():
        raise FileNotFoundError(f"La ruta resuelta no existe: {abs_path}")

    return abs_path


def show_frames(frames: np.ndarray) -> None:
    """Muestra un conjunto de frames en una cuadricula (solo visualiza).

    Funcion general de inspeccion visual. Recibe un arreglo NumPy 4D con forma
    ``(N, H, W, 3)`` (frames RGB en memoria, p. ej. la salida de
    ``src.core.frame_extraction.extract_frames``) y los dibuja en una cuadricula
    con un maximo de 3 columnas:

    - Si ``N >= 6`` se muestran 6 frames repartidos uniformemente en el conjunto.
    - Si ``0 < N < 6`` se muestran todos los frames disponibles.
    - Si ``N == 0`` no se muestra nada y se emite un aviso (no lanza excepcion).

    El orden de los frames se preserva. La funcion solo muestra: no escribe a
    disco ni devuelve los frames.

    Args:
        frames: arreglo ``numpy.ndarray`` 4D con forma ``(N, H, W, 3)``.

    Returns:
        ``None``. Su efecto es el render de la cuadricula via matplotlib.

    Raises:
        ValueError: si ``frames`` no es un ``numpy.ndarray`` 4D ``(N, H, W, 3)``.
    """
    # matplotlib se importa aqui (no a nivel de modulo) porque solo esta funcion
    # lo necesita; asi importar src.utils para get_abs_path no obliga a cargarlo.
    import matplotlib.pyplot as plt

    if not isinstance(frames, np.ndarray):
        raise ValueError(
            f"Se esperaba un numpy.ndarray, se recibio: {type(frames).__name__}"
        )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            "Se esperaba un arreglo 4D con forma (N, H, W, 3), se recibio forma: "
            f"{frames.shape}"
        )

    total = frames.shape[0]
    if total == 0:
        warnings.warn("No hay frames que visualizar (arreglo vacio).")
        return

    # Seleccion de indices: todos si hay 6 o menos; 6 equiespaciados si hay mas.
    if total <= 6:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, 6).round().astype(int).tolist()

    n = len(indices)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    # Normalizar a una lista plana de ejes para iterar de forma uniforme.
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, indices):
        ax.imshow(frames[idx])
        ax.set_title(f"frame {idx}")
        ax.axis("off")

    # Ocultar ejes sobrantes cuando nrows * ncols > n.
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()
