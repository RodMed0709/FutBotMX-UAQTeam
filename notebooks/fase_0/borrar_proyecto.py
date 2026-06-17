# -*- coding: utf-8 -*-
"""Borra un proyecto de Supervisely por ID. Uso: python borrar_proyecto.py <PID>"""
import os
import sys
from pathlib import Path

import supervisely as sly

for line in Path("/workspace/.env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

pid = int(sys.argv[1])
api = sly.Api(os.environ.get("SERVER_ADDRESS", "https://app.supervisely.com"),
              os.environ["SUPERVISELY_API_TOKEN"])
info = api.project.get_info_by_id(pid)
if info is None:
    print(f"proyecto {pid} no existe (ya borrado?)")
else:
    api.project.remove(pid)
    print(f"proyecto {pid} ('{info.name}') BORRADO")
