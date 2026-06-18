"""Provisión idempotente de datos no versionados (tarea ``bootstrap_data``).

Script de paquete (`python -m src.bootstrap_data`) que, declarativo sobre el
**manifiesto versionado** ``assets/bootstrap_manifest.json``, verifica qué insumos
pesados están presentes y descarga (con ``gdown``) los que falten del paquete elegido,
dejando ``data/raw``/``assets/sam3``/``assets/yolo`` como dirs reales. También genera
el ``.env`` desde ``.env.example`` si falta.

Paquetes: ``all`` (dataset completo + pesos) y ``demo`` (clips + JSON con ``rle`` +
pesos; autocontenido para correr la Capa B en local sin GPU). El dataset de la
convocatoria se marca ``manual`` (excede el tope de ``gdown.download_folder``): solo se
verifica presencia y se imprimen instrucciones.

Este módulo expone la **lógica pura** (tarea T3): lectura/filtrado del manifiesto,
verificación de presencia, normalización de IDs de Drive y generación del ``.env``. La
descarga (gdown), el menú interactivo y el reporte se añaden en T4/T5. Imports pesados
(``gdown``/``questionary``) son perezosos, dentro de las funciones que los usan.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.utils import PROJECT_ROOT

# Rutas/llaves de convención (sin valores host-específicos).
MANIFEST_REL = "assets/bootstrap_manifest.json"
ENV_EXAMPLE = ".env.example"
ENV_FILE = ".env"
VALID_PACKAGES = ("all", "demo")


# --- manifiesto ---------------------------------------------------------------


def load_manifest(path: Path | str | None = None) -> dict:
    """Lee y valida (mínimamente) el manifiesto de bootstrap.

    Args:
        path: ruta del manifiesto; ``None`` ⇒ ``<PROJECT_ROOT>/assets/bootstrap_manifest.json``.

    Returns:
        El dict del manifiesto (con ``schema_version`` e ``items``).

    Raises:
        FileNotFoundError: si el manifiesto no existe.
        ValueError: si el JSON no trae ``items`` (lista).
    """
    manifest_path = Path(path) if path is not None else PROJECT_ROOT / MANIFEST_REL
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No se encontró el manifiesto: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("Manifiesto inválido: falta la lista 'items'.")
    return data


def select_package(items: list[dict], package: str) -> list[dict]:
    """Filtra los ítems cuyo ``paquetes`` incluye ``package`` (``"all"``/``"demo"``).

    Raises:
        ValueError: si ``package`` no es uno de ``VALID_PACKAGES``.
    """
    if package not in VALID_PACKAGES:
        raise ValueError(
            f"paquete '{package}' no soportado (usa uno de {VALID_PACKAGES})."
        )
    return [it for it in items if package in it.get("paquetes", [])]


def iter_resources(items: list[dict]) -> list[tuple[dict, dict]]:
    """Aplana ``items`` a una lista de ``(item, recurso)`` para recorrer recursos."""
    out: list[tuple[dict, dict]] = []
    for it in items:
        for rec in it.get("recursos", []):
            out.append((it, rec))
    return out


# --- presencia ----------------------------------------------------------------


def is_present(recurso: dict, project_root: Path = PROJECT_ROOT) -> bool:
    """¿El recurso ya está en su ``destino`` (relativo a ``project_root``)?

    - ``dir``: el directorio existe y contiene **≥1** archivo (recursivo).
    - ``file``/``clip``/``tracking_json``: el archivo existe.
    """
    destino = project_root / recurso["destino"]
    if recurso.get("tipo") == "dir":
        return destino.is_dir() and any(p.is_file() for p in destino.rglob("*"))
    return destino.is_file()


def is_manual(recurso: dict) -> bool:
    """¿El recurso es de descarga manual (no se intenta descargar)?"""
    return bool(recurso.get("manual", False))


# --- IDs de Drive -------------------------------------------------------------


def normalize_drive_id(drive_id: str) -> str:
    """Devuelve el ID pelón de Drive a partir de un ID o una URL de compartir.

    Acepta ``.../file/d/<ID>/...``, ``.../folders/<ID>...``, ``...open?id=<ID>`` o un
    ID ya pelón. No valida que el ID exista; solo lo extrae.
    """
    s = drive_id.strip()
    if "drive.google.com" not in s and "/" not in s and "?" not in s:
        return s  # ya es un ID pelón
    for marker in ("/file/d/", "/folders/"):
        if marker in s:
            rest = s.split(marker, 1)[1]
            return rest.split("/", 1)[0].split("?", 1)[0]
    if "id=" in s:
        return s.split("id=", 1)[1].split("&", 1)[0]
    return s


# --- generación del .env ------------------------------------------------------


def _parse_env_keys(text: str) -> set[str]:
    """Llaves (con ``strip()``) presentes en un texto estilo ``.env``."""
    keys: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.partition("=")[0].strip())
    return keys


def ensure_env(project_root: Path = PROJECT_ROOT) -> dict:
    """Asegura un ``.env`` válido a partir de ``.env.example`` (no-destructivo).

    - Si ``.env`` no existe: lo crea copiando ``.env.example``.
    - Si ``.env`` existe: no lo toca; reporta las llaves de la plantilla que falten.

    Returns:
        ``{"status": "creado"|"presente"|"sin_plantilla", "missing_keys": [...]}``.
    """
    env_path = project_root / ENV_FILE
    example_path = project_root / ENV_EXAMPLE

    if not env_path.exists():
        if not example_path.is_file():
            return {"status": "sin_plantilla", "missing_keys": []}
        env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
        return {"status": "creado", "missing_keys": []}

    missing: list[str] = []
    if example_path.is_file():
        template_keys = _parse_env_keys(example_path.read_text(encoding="utf-8"))
        present_keys = _parse_env_keys(env_path.read_text(encoding="utf-8"))
        missing = sorted(template_keys - present_keys)
    return {"status": "presente", "missing_keys": missing}


# --- descarga (T4) ------------------------------------------------------------


def download_resource(recurso: dict, project_root: Path = PROJECT_ROOT) -> Path:
    """Descarga un recurso con ``gdown`` a su ``destino`` (crea el dir padre).

    ``dir`` ⇒ ``gdown.download_folder``; ``file``/``clip``/``tracking_json`` ⇒
    ``gdown.download``. El ``drive_id`` puede ser URL o ID (se normaliza). No debe
    llamarse sobre recursos ``manual``.

    Returns:
        La ruta absoluta del ``destino``.

    Raises:
        RuntimeError: si gdown no devuelve nada (descarga fallida).
    """
    import gdown  # perezoso

    destino = project_root / recurso["destino"]
    drive_id = normalize_drive_id(recurso["drive_id"])

    if recurso.get("tipo") == "dir":
        destino.mkdir(parents=True, exist_ok=True)
        ok = gdown.download_folder(
            id=drive_id, output=str(destino), quiet=False, use_cookies=False
        )
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)
        ok = gdown.download(id=drive_id, output=str(destino), quiet=False)

    if not ok:
        raise RuntimeError(
            f"gdown no pudo descargar '{recurso['destino']}' (id={drive_id}). "
            "Revisa que el enlace sea público ('cualquiera con el enlace')."
        )
    return destino


# --- orquestación (T4) --------------------------------------------------------


@dataclass
class ResourceOutcome:
    """Resultado por recurso: ``estado`` ∈ presente|descargado|manual|pendiente|error."""

    nombre: str
    destino: str
    estado: str
    detalle: str = ""


@dataclass
class BootstrapReport:
    env: dict = field(default_factory=dict)
    outcomes: list[ResourceOutcome] = field(default_factory=list)


def run_bootstrap(
    package: str, project_root: Path = PROJECT_ROOT, *, dry_run: bool = False
) -> BootstrapReport:
    """Provisiona el paquete elegido: asegura ``.env``, verifica y descarga lo faltante.

    Idempotente (salta lo presente) y no-destructivo. Aísla errores por recurso (un
    fallo no aborta el resto del paquete). ``dry_run=True`` no descarga: marca lo
    faltante como ``pendiente``.

    Args:
        package: ``"all"`` o ``"demo"``.
        dry_run: si ``True``, no descarga (útil para validar el plan sin red).

    Returns:
        ``BootstrapReport`` con el estado del ``.env`` y un ``ResourceOutcome`` por recurso.
    """
    report = BootstrapReport(env=ensure_env(project_root))
    manifest = load_manifest(project_root / MANIFEST_REL)
    items = select_package(manifest["items"], package)

    for item, rec in iter_resources(items):
        nombre, destino = item["nombre"], rec["destino"]
        if is_present(rec, project_root):
            report.outcomes.append(ResourceOutcome(nombre, destino, "presente"))
        elif is_manual(rec):
            report.outcomes.append(
                ResourceOutcome(
                    nombre, destino, "manual", f"descarga manual: {rec['drive_id']}"
                )
            )
        elif dry_run:
            report.outcomes.append(
                ResourceOutcome(nombre, destino, "pendiente", "se descargaría")
            )
        else:
            try:
                download_resource(rec, project_root)
                report.outcomes.append(ResourceOutcome(nombre, destino, "descargado"))
            except Exception as exc:  # noqa: BLE001 — aislar fallo por recurso
                report.outcomes.append(
                    ResourceOutcome(nombre, destino, "error", str(exc))
                )
    return report


# --- menú interactivo + reporte (T5) ------------------------------------------

_ESTADO_ESTILO = {
    "presente": "cyan",
    "descargado": "green",
    "manual": "yellow",
    "pendiente": "blue",
    "error": "red",
}


def prompt_package() -> str | None:
    """Menú interactivo de paquete. Devuelve ``"demo"``/``"all"`` o ``None`` (salir)."""
    import questionary

    eleccion = questionary.select(
        "¿Qué deseas provisionar?",
        choices=[
            questionary.Choice("Solo demos (recomendado)", value="demo"),
            questionary.Choice("Todos (dataset completo + pesos)", value="all"),
            questionary.Choice("Salir", value=None),
        ],
    ).ask()
    return eleccion


def print_report(report: BootstrapReport, console) -> None:
    """Imprime el reporte (estado del .env + tabla de recursos) con rich."""
    from rich.table import Table

    env = report.env
    console.print(f"[bold].env:[/] {env.get('status')}", end="")
    if env.get("missing_keys"):
        console.print(f"  [yellow]llaves faltantes: {env['missing_keys']}[/]")
    else:
        console.print("")

    table = Table(title="Provisión de datos")
    table.add_column("Ítem")
    table.add_column("Estado")
    table.add_column("Destino / detalle")
    for o in report.outcomes:
        style = _ESTADO_ESTILO.get(o.estado, "white")
        table.add_row(o.nombre, f"[{style}]{o.estado}[/]", o.detalle or o.destino)
    console.print(table)

    manuales = [o for o in report.outcomes if o.estado == "manual"]
    if manuales:
        console.print(
            "\n[yellow]Descarga manual pendiente[/] (excede el tope de gdown). "
            "Baja estas carpetas del Drive de la convocatoria a su destino:"
        )
        for o in manuales:
            console.print(f"  • {o.destino}  ←  {o.detalle.split(': ', 1)[-1]}")

    errores = [o for o in report.outcomes if o.estado == "error"]
    if errores:
        console.print(f"\n[red]{len(errores)} recurso(s) con error.[/]")


def main() -> int:
    from rich.console import Console

    console = Console()
    if not sys.stdin.isatty():
        console.print(
            "[bold red]Terminal no interactiva:[/] el bootstrap usa un menú; "
            "córrelo en una terminal interactiva."
        )
        return 2

    package = prompt_package()
    if package is None:
        console.print("Saliendo sin cambios.")
        return 0

    console.rule(f"[bold]Bootstrap — paquete '{package}'")
    report = run_bootstrap(package)
    print_report(report, console)
    return 0 if not any(o.estado == "error" for o in report.outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
