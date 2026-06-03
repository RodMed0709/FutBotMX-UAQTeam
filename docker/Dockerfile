# ===================================================
# Dockerfile - Proyecto Copa FutBotMX (env_setup)
# Entorno reproducible para el stack SAM3 / YOLO / ByteTrack.
# ===================================================

FROM python:3.11-slim

# Directorio de trabajo del proyecto dentro del contenedor (NO se asume /app).
# Coincide con CONTAINER_WORKSPACE_DIR del .env.
ARG CONTAINER_WORKSPACE_DIR=futbot
ENV CONTAINER_WORKSPACE_DIR=${CONTAINER_WORKSPACE_DIR}

# Indice de wheels de torch. Por defecto CPU; para pods con GPU sobreescribir, p.ej:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Librerias de sistema minimas para opencv-python-headless y la IO de video
# (ffmpeg para av/moviepy/imageio-ffmpeg). git queda disponible para instalar SAM3.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# Crear y "activar" el entorno virtual (anteponiendo su bin al PATH).
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# WORKDIR dinamico a partir de la variable de entorno.
WORKDIR /${CONTAINER_WORKSPACE_DIR}

# Instalar dependencias primero para aprovechar la cache de capas.
# torch se instala aparte (el requirements.txt lo documenta asi).
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto.
COPY . .

# Instalar el proyecto en modo editable: expone el paquete `src` de forma
# importable desde cualquier parte (notebooks, scripts) sin tocar sys.path.
# El enlace editable se escribe en el venv (/opt/venv, fuera del bind-mount) y
# apunta a /${CONTAINER_WORKSPACE_DIR}; en runtime ese path es el codigo del host
# montado, por lo que el import resuelve contra el codigo real y editable.
RUN pip install --no-cache-dir -e .
