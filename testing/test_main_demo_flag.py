"""Prueba manual del flag --demo del hub (tarea main_demo_flag).

Smoke **sin GPU ni red**: valida el parseo de flags, la lectura de demos del manifiesto
(solo los presentes), el selector y la precedencia con --overwrite. Usa monkeypatch del
manifiesto (no descarga ni infiere).

Cubre:
  1. parse_args: --demo vuelve opcional el video y es combinable con --overwrite.
  2. load_demo_choices: lista solo los demos cuyo clip existe en disco.
  3. choose_demo: sin demos presentes -> SystemExit(2) (sugiere el bootstrap).
  4. choose_demo: camino feliz (questionary mockeado) -> devuelve el DemoChoice.

`src` es paquete editable (`pip install -e .`). `main.py` es un script de la raíz (no
parte del paquete), así que se añade PROJECT_ROOT a sys.path solo para importarlo.

Uso:
    python testing/test_main_demo_flag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import questionary  # noqa: E402

import main as hub  # noqa: E402
import src.bootstrap_data as boot  # noqa: E402

_fails: list[str] = []

# Clip real bajo PROJECT_ROOT para simular un demo "presente".
_PRESENT_CLIP = "outputs/fase5_clips/IMG_9933_5m30.mp4"


def check(cond: bool, label: str) -> None:
    print(f"  {'[PASS]' if cond else '[FAIL]'} {label}")
    if not cond:
        _fails.append(label)


class _Console:
    def print(self, *a, **k):  # noqa: D401 - dummy
        pass


def _manifest(present: bool, absent: bool) -> dict:
    items = []
    if present:
        items.append(
            {
                "nombre": "demo_presente",
                "paquetes": ["demo"],
                "vista": "generica",
                "recursos": [
                    {"tipo": "clip", "drive_id": "X", "destino": _PRESENT_CLIP}
                ],
            }
        )
    if absent:
        items.append(
            {
                "nombre": "demo_ausente",
                "paquetes": ["demo"],
                "vista": "superior",
                "recursos": [
                    {
                        "tipo": "clip",
                        "drive_id": "Y",
                        "destino": "data/raw/demos/no.mp4",
                    }
                ],
            }
        )
    return {"schema_version": 1, "items": items}


def test_parse_args() -> None:
    print("1) parse_args (--demo / --overwrite / video opcional)")
    a = hub.parse_args(["--demo", "--overwrite"])
    check(a.demo and a.overwrite and a.video is None, "--demo --overwrite, video None")
    b = hub.parse_args(["clip.mp4"])
    check(b.video == "clip.mp4" and not b.demo, "video posicional sin --demo")


def test_load_demo_choices() -> None:
    print("2) load_demo_choices (solo presentes)")
    orig = boot.load_manifest
    boot.load_manifest = lambda *a, **k: _manifest(present=True, absent=True)
    try:
        choices = hub.load_demo_choices()
    finally:
        boot.load_manifest = orig
    check(len(choices) == 1, f"lista solo el presente ({len(choices)})")
    if choices:
        c = choices[0]
        check(c.nombre == "demo_presente", "nombre correcto")
        check(c.vista == "generica", "vista del manifiesto")
        check(str(c.clip_path).endswith(_PRESENT_CLIP), "clip_path resuelto")


def test_choose_demo_none() -> None:
    print("3) choose_demo sin demos -> SystemExit(2)")
    orig = boot.load_manifest
    boot.load_manifest = lambda *a, **k: _manifest(present=False, absent=True)
    try:
        hub.choose_demo(_Console())
        check(False, "debió levantar SystemExit")
    except SystemExit as e:
        check(e.code == 2, "SystemExit(2)")
    finally:
        boot.load_manifest = orig


def test_choose_demo_happy() -> None:
    print("4) choose_demo camino feliz (questionary mockeado)")
    orig_lm, orig_sel, orig_stdin = boot.load_manifest, questionary.select, sys.stdin
    boot.load_manifest = lambda *a, **k: _manifest(present=True, absent=False)

    class _FakeStdin:
        def isatty(self):
            return True

    class _FakeSelect:
        def __init__(self, value):
            self._value = value

        def ask(self):
            return self._value

    # questionary.select(...).ask() -> el primer DemoChoice de las choices.
    def _fake_select(_msg, choices):
        return _FakeSelect(choices[0].value)

    questionary.select = _fake_select
    sys.stdin = _FakeStdin()
    try:
        chosen = hub.choose_demo(_Console())
    finally:
        boot.load_manifest, questionary.select, sys.stdin = (
            orig_lm,
            orig_sel,
            orig_stdin,
        )
    check(isinstance(chosen, hub.DemoChoice), "devuelve un DemoChoice")
    check(chosen.nombre == "demo_presente", "el demo elegido es el presente")


def main() -> int:
    for t in (
        test_parse_args,
        test_load_demo_choices,
        test_choose_demo_none,
        test_choose_demo_happy,
    ):
        t()
    print()
    if _fails:
        print(f"FALLARON {len(_fails)} checks: {_fails}")
        return 1
    print("Todos los checks pasaron ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
