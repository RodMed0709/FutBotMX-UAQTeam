"""Empaquetado del proyecto FutBotMX para instalacion en modo editable.

Expone el paquete ``src`` (y sus submodulos, p. ej. ``src.core``, ``src.utils``)
de forma importable desde cualquier ubicacion tras ``pip install -e .``.

Las dependencias del proyecto NO se declaran aqui: se gestionan en
``requirements.txt`` (torch y SAM3 se instalan aparte, segun documenta ese
archivo).
"""

from setuptools import find_packages, setup

setup(
    name="futbotmx",
    version="0.0.1",
    description="Copa FutBotMX (UAQ Team) - deteccion, segmentacion y tracking de futbol robotico.",
    python_requires=">=3.11",
    packages=find_packages(include=["src", "src.*"]),
)
