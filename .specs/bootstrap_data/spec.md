# spec.md — `bootstrap_data`

## Contexto

El proyecto separa **datos versionados** (manifiestos ligeros en `assets/`) de
**datos pesados no versionados** (videos en `data/raw`, pesos SAM3 en `assets/sam3`,
pesos YOLO en `assets/yolo`). Hoy cada quien consigue y coloca esos archivos a mano.
Falta un mecanismo **idempotente y reproducible** que provea esos insumos, dejando
`data/raw`, `assets/sam3` y `assets/yolo` como **directorios reales con archivos
reales** (nunca symlinks) en cualquier entorno (local, contenedor, RunPod).

Principio de raíz: la convención de **rutas relativas vía config** no cambia; el
código de pipeline no cambia. Lo único que provee el bootstrap es *llenar* esas rutas
con datos reales descargados de **Google Drive**.

## Objetivo

Un **script interactivo** que provea los insumos no versionados en dos paquetes:

- **Todos:** el dataset completo (123 videos) + SAM3 + YOLO `best.pt`. Para correr y
  reproducir el proyecto entero desde cero. El **dataset** vive en el Drive **público
  de la convocatoria** (carpetas `17Abril`/`18abril`); como `17Abril` (88 videos)
  excede el tope de `gdown.download_folder`, es **descarga manual** (`manual: true` en
  el manifiesto): el bootstrap solo **verifica presencia** y, si falta, imprime el
  enlace e instrucciones. La reproducibilidad la cubre el **paquete demo**. Los pesos
  (SAM3/YOLO) y los recursos del demo sí se descargan automáticamente con **gdown**.
- **Solo demos:** un paquete **autocontenido y reproducible** — los clips demo + **sus
  JSON de tracking (con `rle`)** + SAM3 + YOLO `best.pt`. Los JSON permiten correr la
  **Capa B (segmentación-overlay + eventos/broadcast) en local sin GPU al instante**; los
  pesos permiten que `main --overwrite` rehaga **todo de cero** para **validar la
  reproducibilidad**.

El script además **genera el `.env`** si falta (valores fijos, sin rutas
host-específicas), de modo que tras el bootstrap el proyecto queda listo para correr.

## Alcance

- **Manifiesto versionado** `assets/bootstrap_manifest.json` = **fuente de verdad
  única** de qué bajar. Cada ítem declara: `nombre`, `paquetes` (`["all"]`,
  `["demo"]` o ambos), `vista` (opcional, para demos), `destinos` (rutas relativas de
  convención) y el/los **ID(s) de Google Drive** (opción A: los IDs viven en el
  manifiesto; son IDs de carpeta/archivo compartidos, no credenciales). El mismo
  manifiesto lo consume la tarea `main_demo_flag`.
- **Script** `src/bootstrap_data.py`, ejecutable como `python -m src.bootstrap_data`.
  Menú interactivo: `[1] Todos · [2] Solo demos · [3] Salir` (demos resaltado como
  recomendado).
- **Descarga** con `gdown` (añadido a `requirements.txt`) solo de lo que falte del
  paquete elegido, a su `destino` de convención vía `get_abs_path`.
- **Generación de `.env`** desde una plantilla versionada `.env.example`
  (`CONFIG_FILENAME=01_yolo_sam3_config.json`, `CONTAINER_WORKSPACE_DIR=futbot`), solo
  si `.env` no existe; no-destructivo.
- **Documentación de reproducibilidad** (entregable): sección en `README.md` + `docs/`
  explicando el flujo "demos → Capa B local sin GPU; `main … --overwrite` → rehace de
  cero y valida reproducibilidad".

## Fuera de alcance

- Estrategia concreta de RunPod (network volume vs efímero) y su ruta de montaje.
- Verificación por checksum/tamaño (v1 verifica solo **presencia**).
- Reconciliación de faltantes uno a uno dentro de un dir ya poblado (v1 trata cada
  ítem como presente/ausente).
- Alojar realmente los archivos en Drive y obtener sus IDs (insumo humano; el
  manifiesto se llena cuando existan).

## Comportamiento esperado

```
$ python -m src.bootstrap_data
¿Qué deseas provisionar?
  > [2] Solo demos (recomendado)   ← clips + JSON con rle + SAM3 + YOLO
    [1] Todos                       ← 123 videos + SAM3 + YOLO
    [3] Salir
```

1. Asegura `.env` (lo crea desde `.env.example` si falta; si existe, lo respeta y solo
   reporta llaves faltantes).
2. Lee `assets/bootstrap_manifest.json` y filtra los ítems del paquete elegido.
3. Para cada ítem ausente en su `destino`, descarga de Drive con `gdown`; salta los
   presentes (idempotente, no-destructivo).
4. **Reporta**: qué encontró, qué descargó y dónde quedó.

Destinos de convención (rutas relativas, resueltas con `get_abs_path`):

| Recurso | Destino |
| --- | --- |
| Clip demo | `data/raw/demos/<stem>.mp4` |
| JSON de tracking (con `rle`) | `outputs/inference/<run_label>/<stem>/<stem>.json` |
| Pesos SAM3 | `assets/sam3/` |
| Pesos YOLO | `assets/yolo/best.pt` |
| Dataset completo | `data/raw/` |

## Consideraciones

- **Idempotencia:** correr dos veces no re-descarga lo presente.
- **No-destructivo:** nunca borra ni sobreescribe datos ni `.env` existentes.
- **Sin secretos host-específicos:** el `.env` no tiene rutas absolutas (modelo de
  dirs reales); los IDs de Drive viven en el manifiesto versionado (opción A).
- **Invocación manual:** no es entrypoint automático del contenedor (Docker simple).
