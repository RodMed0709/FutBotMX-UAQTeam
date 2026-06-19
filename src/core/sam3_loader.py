"""Carga del modelo SAM3 (tarea sam3_loader).

Define ``load_sam3``, la forma unica y reutilizable de cargar SAM3 (processor +
model) lista para inferir. Sustituye el bloque de carga copy-pasteado en los
notebooks de ``notebooks/fase_0/``.

- La **ruta** del modelo se resuelve sola desde la configuracion del proyecto
  (clave ``working_dirs.sam3_dir``) via ``src.utils.get_abs_path``; nunca se
  incrustan rutas absolutas ni se usan symlinks.
- El **dispositivo** se elige automaticamente (GPU si esta disponible, si no CPU)
  y puede forzarse. El device elegido viaja en el ``Sam3Bundle`` para que carga e
  inferencia usen la misma fuente.
- El modelo se **cachea** (singleton) por defecto; ``use_cache=False`` entrega una
  instancia fresca (util para pruebas aisladas).

``torch`` y ``transformers`` se importan de forma **perezosa** dentro de la
funcion de construccion, para que ``import src.core`` (p. ej. para
``extract_frames``, que no usa torch) no obligue a cargarlos.

Esta tarea solo **carga** el modelo: no infiere, no segmenta y no descarga los
pesos (eso ultimo es responsabilidad de la futura tarea ``bootstrap_data``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils import PROJECT_ROOT, get_abs_path


@dataclass
class Sam3Bundle:
    """Agrupa todo lo necesario para inferir con SAM3 desde una sola llamada.

    Attributes:
        processor: el processor de SAM3 (pre/post-procesado y sesiones de video).
        model: el modelo SAM3 cargado, en modo evaluacion.
        device: el dispositivo en el que se cargo el modelo ("cuda" o "cpu"). Es
            la misma fuente que debe usarse al abrir sesiones de inferencia.
        tracker_processor: processor de la **2a cara** de SAM3 (box-prompt,
            ``Sam3TrackerProcessor``). ``None`` hasta que ``ensure_tracker_loaded``
            lo carga **bajo demanda** (carga perezosa/opt-in).
        tracker_model: modelo de la 2a cara (``Sam3TrackerModel``, segmentacion por
            caja). ``None`` hasta que ``ensure_tracker_loaded`` lo carga. La carga
            por defecto (``load_sam3()``) **no** lo toca.
    """

    processor: Any
    model: Any
    device: str
    tracker_processor: Any = None
    tracker_model: Any = None


def _load_env(env_path: Path) -> dict[str, str]:
    """Parseo simple de un archivo .env (KEY = value), aplicando strip()."""
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _resolve_sam3_dir() -> Path:
    """Resuelve la ruta absoluta del directorio del modelo SAM3.

    El nombre del archivo de configuracion se toma de CONFIG_FILENAME en el .env;
    la ruta (relativa a PROJECT_ROOT) se lee de working_dirs.sam3_dir y se resuelve
    y verifica con get_abs_path.

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si falta la clave working_dirs.sam3_dir en la configuracion.
        FileNotFoundError: si el archivo de configuracion o el directorio del
            modelo no existen (via get_abs_path).
    """
    env = _load_env(PROJECT_ROOT / ".env")
    config_filename = env.get("CONFIG_FILENAME")
    if not config_filename:
        raise ValueError("No se encontro CONFIG_FILENAME en el archivo .env.")

    config_path = get_abs_path(f"configs/{config_filename}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    working_dirs = config.get("working_dirs", {})
    if "sam3_dir" not in working_dirs:
        raise KeyError(
            "Falta la clave 'sam3_dir' en 'working_dirs' del archivo de configuracion."
        )

    return get_abs_path(working_dirs["sam3_dir"])


def _build_bundle(device: str | None = None) -> Sam3Bundle:
    """Construye un Sam3Bundle desde cero (logica unica de carga).

    Importa torch/transformers de forma perezosa. Resuelve el dispositivo (auto si
    device es None) y carga processor + model en bfloat16, en modo evaluacion.

    Args:
        device: dispositivo a usar ("cuda"/"cpu"); None elige automaticamente.

    Returns:
        Sam3Bundle con processor, model y el device efectivamente usado.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    sam3_dir = _resolve_sam3_dir()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(str(sam3_dir))
    model = AutoModel.from_pretrained(
        str(sam3_dir),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    return Sam3Bundle(processor=processor, model=model, device=device)


@lru_cache(maxsize=1)
def _cached_load() -> Sam3Bundle:
    """Carga cacheada (singleton) del bundle con device automatico."""
    return _build_bundle()


def load_sam3(*, use_cache: bool = True, device: str | None = None) -> Sam3Bundle:
    """Carga SAM3 (processor + model) listo para inferir.

    Por defecto el modelo se carga una sola vez y se reutiliza (singleton). El
    dispositivo se elige automaticamente (GPU si esta disponible, si no CPU) y
    puede forzarse.

    Args:
        use_cache: si True (por defecto) y no se fuerza device, devuelve la
            instancia cacheada (la primera llamada la crea, las siguientes la
            reutilizan). Si False, devuelve una instancia fresca sin afectar la
            cacheada.
        device: fuerza el dispositivo ("cuda"/"cpu"). Si se indica, se construye
            una instancia fresca (no se usa ni se contamina la cache), para que una
            llamada puntual a un device concreto no fije el singleton automatico.

    Returns:
        Sam3Bundle con processor, model y el device usado.

    Raises:
        ValueError: si CONFIG_FILENAME no esta en el .env.
        KeyError: si falta working_dirs.sam3_dir en la configuracion.
        FileNotFoundError: si la configuracion o el directorio del modelo no
            existen.
    """
    if use_cache and device is None:
        return _cached_load()
    return _build_bundle(device=device)


def ensure_tracker_loaded(bundle: Sam3Bundle) -> Sam3Bundle:
    """Carga **perezosa** de la 2a cara de SAM3 (box-prompt) sobre un bundle.

    SAM3 expone dos caras del **mismo** checkpoint: la de video/texto (``model`` /
    ``processor``, ya cargada por ``load_sam3``) y la de **caja** (``Sam3TrackerModel``
    / ``Sam3TrackerProcessor``), que segmenta a partir de cajas (box-prompt). Esta
    funcion carga la segunda cara **bajo demanda** y la deja en el bundle, para no
    encarecer la carga por defecto ni a los llamadores que solo usan text-prompt.

    Es **idempotente**: si la cara tracker ya esta cargada, devuelve el bundle sin
    recargar. Sobre el bundle cacheado (singleton), la cara tracker queda cargada una
    sola vez para toda la sesion.

    Se carga con las **mismas convenciones** que la cara principal: ``bfloat16``,
    ``low_cpu_mem_usage=True``, modo evaluacion y el **mismo** ``bundle.device`` (para
    que carga e inferencia compartan dispositivo).

    Nota: la carga de ``Sam3TrackerModel`` emite el aviso benigno
    ``sam3_video -> sam3_tracker``. Es **esperado** (los pesos del tracker si estan en
    el checkpoint; verificado que produce mascaras precisas) y **no** se silencia, para
    no ocultar otros avisos.

    Args:
        bundle: ``Sam3Bundle`` ya cargado (de ``load_sam3``) sobre el que se rellenan
            ``tracker_processor`` y ``tracker_model``.

    Returns:
        El **mismo** ``Sam3Bundle``, con la cara tracker garantizada.
    """
    if bundle.tracker_model is not None:
        return bundle

    import torch
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    sam3_dir = _resolve_sam3_dir()
    bundle.tracker_processor = Sam3TrackerProcessor.from_pretrained(str(sam3_dir))
    bundle.tracker_model = (
        Sam3TrackerModel.from_pretrained(
            str(sam3_dir),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to(bundle.device)
        .eval()
    )
    return bundle
