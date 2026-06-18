# FutBot MX 2026 — UAQ Team

**Detección, segmentación, seguimiento y análisis de eventos** de partidos de fútbol
robótico (Copa FutBotMX, categoría Profesional). El sistema toma el video de un partido y
produce, por cada robot y por la pelota, **máscaras**, **cajas** y **trayectorias con
identidad estable**, y sobre cámara superior además **proyecta el juego a un campo métrico
(cm)** para detectar **goles, posesión y faltas** y componer un **video narrativo de
espectador**.

![Broadcast — video de espectador](assets/readme/gifs/broadcast_IMG_9933_5m30.gif)

> Documentación de ingeniería por fases en [`docs/`](docs/README.md) y por tarea en
> `.specs/<tarea>/`. Para **usar** la API desde tus notebooks, abre el recetario:
> [`notebooks/cookbook_pipeline.ipynb`](notebooks/cookbook_pipeline.ipynb).

---

## 1. Descripción y arquitectura de la solución

El sistema se organiza en **dos capas**:

- **Capa A — percepción (GPU):** `detector/segmentador → tracker`. Aplica a **cualquier**
  clip y produce máscaras + `obj_id` estables.
- **Capa B — análisis métrico (CPU):** sobre **cámara superior**, proyecta a cm reales y
  deriva métricas y eventos. Lee el JSON de tracking; no re-infiere.

### Enfoque técnico e innovación

**1) Segmentación con SAM 3 + detector afinado (innovación sobre el modelo base).**
`sam3_text` segmenta por *prompt de texto* ("robot", "orange ball"). La innovación es
`yolo_sam3`: un **YOLO afinado** con auto-etiquetas generadas por SAM 3 (sobre los 103
videos NO-testing) localiza cajas rápido y SAM 3 segmenta **dentro** de cada caja
(box-prompt), acoplando dos arquitecturas complementarias.

**2) Tracking con identidad estable.** Las máscaras se reducen a cajas y se asocian frame a
frame con **ByteTrack** o **BoT-SORT** (un tracker por clase) hacia `obj_id` **globalmente
únicos**, en streaming (sin OOM sobre el video completo).

**3) Homografía al campo métrico.** Se estima una homografía por frame que lleva la imagen
al campo cenital oficial (**243 × 182 cm**, líneas con *inset* de 12 cm, círculo central de
radio 30 cm). Con correspondencias mundo←imagen (líneas blancas + esquinas + porterías) se
resuelve la matriz `H ∈ ℝ³ˣ³` por mínimos cuadrados robustos (RANSAC) y la proyección de un
punto es

```
[x' y' w']ᵀ = H · [u v 1]ᵀ ,   (x, y)_cm = (x'/w', y'/w')
```

Para reducir *jitter* entre frames se aplica un suavizado EMA sobre `H` normalizada
(`H₃₃ = 1`). El camino consolidado (ajuste a líneas, `homography_multifeature.py`) alcanza
**~9–23 cm** de error.

**4) Cinemática y filtro de Kalman propio (innovación).** La velocidad base es por
**diferencias finitas**, `v = ‖pₜ − pₜ₋₁‖ / Δt` con `Δt = (f₂−f₁)/fps`, con corte de
outliers y suavizado. Encima corre un **filtro de Kalman de velocidad constante 2D, escrito
desde cero** (estado `x = [pₓ, p_y, vₓ, v_y]`), que aporta velocidad físicamente suave,
**relleno de oclusión** (predict-only) y rechazo robusto de outliers:

```
Predicción:  x⁻ = F(Δt) x ,           P⁻ = F P Fᵀ + Q(Δt)
Innovación:  y  = z − H x⁻ ,          S  = H P⁻ Hᵀ + R
Ganancia:    K  = P⁻ Hᵀ S⁻¹
Corrección:  x  = x⁻ + K y ,          P  = (I − K H) P⁻
Oclusión:    x  = x⁻ , P = P⁻         (sin medición; la incertidumbre crece)
```

con `H = [[1,0,0,0],[0,1,0,0]]`, `Q` derivada de ruido blanco de aceleración (`σ_a`),
`R = σ_z² I` calibrada del error de homografía (`σ_z ≈ 15 cm`), y **gating de Mahalanobis**
`NIS = yᵀ S⁻¹ y` contra `χ²₂(0.99) = 9.21` (reemplaza el corte duro de velocidad sin tirar
el track). Esto reduce la **varianza de aceleración un 98–100 %** frente a diferencias
finitas y elimina picos de velocidad imposibles (p. ej. v_max de balón 196.3 → 116.4 cm/s).

**5) Detección algorítmica de eventos.** Sobre las posiciones en cm: **gol estricto** vs
tiro (geometría de cruce de línea de portería), **posesión vs control** (proximidad
balón-robot con histéresis temporal), **faltas de campo** (fuera, área, *pushing*), zonas
(mitades/tercios) y **mapa de calor** de ocupación. El entregable es el **video de
espectador** (`event_broadcast_overlay`): marcador, banner de gol, panel de posesión,
*feed* de eventos, minimapa cenital + heatmap y la homografía embebida.

---

## 2. Metodología

### Arquitectura del pipeline (flujo de datos)

```
                         CAPA A (GPU)                         CAPA B (CPU, solo cámara superior)
 video ─► [ detector/segmentador ] ─► [ tracker ] ─► JSON ─► [ homografía ] ─► [ métrica cm ] ─► [ eventos ] ─► broadcast
              sam3_text | yolo_sam3     bytetrack        (líneas→H)      posiciones/velocidad     gol/posesión/    (video
                                        | botsort                        zonas/heatmap/Kalman      faltas          espectador)
```

Una **única puerta de entrada** ([`run_inference`](src/core/inference.py)) resuelve el
muestreo de frames, el render y el esquema de salida según el modo
(`segmentation` | `tracking`). Detector y tracker son **piezas intercambiables**
([`src/core/detectors/`](src/core/detectors/), [`src/core/trackers/`](src/core/trackers/)),
y añadir una clase es **solo configuración**. El hub [`main.py`](main.py) orquesta el flujo
completo end-to-end (ver §7).

#### Capa A — percepción (GPU): detección/segmentación → tracking

```
                     run_inference (inference.py) — fachada única por video
                             │  mode = "segmentation" | "tracking"
           ┌─────────────────┴──────────────────┐
           ▼                                     ▼
      pipeline.py                            tracking.py
 (per-frame, obj_id NO estable)        (streaming, obj_id ESTABLE)
           │                                     │
           ▼                                     ▼
 ┌───────────────────┐  get_detector  ┌───────────────────┐  get_tracker ┌──────────────────┐
 │  DETECTOR  ⇆      │ ─────────────▶ │  DETECTOR  ⇆      │ ───────────▶ │  TRACKER  ⇆      │
 │  • sam3_text      │                │  • sam3_text      │              │  • bytetrack     │
 │  • yolo_sam3      │                │  • yolo_sam3      │              │  • botsort       │
 └───────────────────┘                └───────────────────┘              └──────────────────┘
           │                                     │
           ▼                                     ▼
 overlay + mp4 + JSON                 ┌──────────────────────────────┐
 (segmentación)                       │  tracking JSON  ◀── la "moneda"
                                      │  Track / TrackObservation     │   del post-proceso
                                      │  (obj_id estable, [+ máscaras])│
                                      └──────────────────────────────┘
```

El `tracking JSON` es la **frontera dura**: todo el post-proceso lee de ahí y no sabe qué
detector/tracker lo generó. Piezas desmontables vía registro: `get_detector` y `get_tracker`.

#### Capa B — post-proceso (CPU, cámara superior): homografía → eventos

```
 tracking JSON
      │
      ▼
 ┌────────────────────────────┐  compute_metric_positions(homography="lines" | "masks")
 │  HOMOGRAFÍA  ⇆ (flag)      │
 │  • "lines"  (consolidada)  │  ◀── base compartida del post-proceso
 │  • "masks"  (legacy)       │
 └────────────────────────────┘
      │  xy_cm por frame/obj_id  +  H por frame
 ┌────┼─────────────────┬────────────────────┬───────────────┐
 ▼    ▼                 ▼                    ▼               ▼
 ESTIMADOR DE        zonas               EVENTOS           heatmap
 ESTADO/CINEMÁTICA  (mitades/tercios)   • shot_vs_goal    (ocupación cm)
 • dif. finitas (T4)                    • goal_geometric
 • Kalman CV (f6) ✔                     • possession
   suaviza + oclusión                   • field_violations
      └─────────────────┬────────────────────┘
                        ▼
            event_broadcast_overlay  ◀── EL ENTREGABLE
            (marcador, banner de gol, posesión,
             minimapa cenital, heatmap, homografía embebida)
```

### Metodología de desarrollo — Spec-Driven Development (SDD)

El repositorio sigue **SDD** ([`.specs/constitution.md`](.specs/constitution.md)): por cada
tarea atómica se escribe `spec.md → plan.md → tasks.md` **antes** de tocar código, y cada
tarea vive en su propia carpeta `.specs/<tarea>/`. El trabajo y la documentación están en
**español**; lint/formato con `ruff check .` / `black .`. La documentación por fases del
sistema está en [`docs/`](docs/README.md).

---

## 3. Resultados obtenidos y métricas

### 3.1 Demostración visual — pipeline base (cualquier clip)

Detección + segmentación por clase y tracking con identidad estable, sobre clips genéricos
(< 1 min, no cámara superior):

| Segmentación (SAM 3) | Tracking `obj_id` |
|---|---|
| ![segmentación](assets/readme/png/segmentacion_video-714_singular_display.png) | ![tracking](assets/readme/gifs/tracking_video-714_singular_display.gif) |

### 3.2 Análisis avanzado — cámara superior (el entregable)

Video narrativo, mapas de calor dinámicos, posesión con métricas temporales y posiciones en
cm. Ejemplos sobre dos partidos distintos (`IMG_9933`, `IMG_9938`):

| Broadcast (espectador) | Mapa de calor (ocupación) | Zonas / presencia |
|---|---|---|
| ![broadcast](assets/readme/png/broadcast_IMG_9933_8m00.png) | ![heatmap](assets/readme/png/heatmap_ball_IMG_9933_5m30.png) | ![zonas](assets/readme/png/zonas_tercios_IMG_9933_5m30.png) |

**Velocidad: crudo vs Kalman** (suavizado físico + relleno de oclusión):

![velocidad kalman](assets/readme/png/velocidad_kalman_IMG_9933_5m30.png)

Tablas Kalman completas en [`assets/fase6/tables/`](assets/fase6/tables/): la varianza de
aceleración baja **98–100 %** y se rellenan los huecos de oclusión sin falsos goles.

### 3.3 Desempeño cuantitativo — benchmark sin ground-truth

Como aún no hay anotación manual, **no** se mide exactitud (mAP/MOTA/mIoU) sino
**eficiencia y consistencia**, sobre 5 videos de testing (seed=36), tracking acotado a 2500
frames. Drivers en [`notebooks/fase_3_benchmark_models/`](notebooks/fase_3_benchmark_models/).
Honesto: el YOLO se entrenó solo con videos NO-testing, así que el split de testing está
intocado para ambos detectores.

**Fase 1 — eficiencia del detector**

![Fase 1](assets/benchmark/fase1_detectores.png)

| Detector | FPS (↑) | VRAM pico MB (↓) |
|---|---|---|
| `sam3_text` | **1.82** | **2157** |
| `yolo_sam3` | 1.71 | 3151 |

**Fase 2 — trackers (2×2)**

![Fase 2](assets/benchmark/fase2_trackers.png)

| Config | FPS (↑) | VRAM MB (↓) | frag_rate (↓) | tracklet_len (↑) |
|---|---|---|---|---|
| `sam3_text+bytetrack` | 2.15 | **2157** | 0.035 | **192.7** |
| `sam3_text+botsort` | 1.83 | **2157** | 0.061 | 134.5 |
| `yolo_sam3+bytetrack` | **2.25** | 3151 | 0.035 | 186.3 |
| `yolo_sam3+botsort` | 1.95 | 3151 | **0.011** | 146.5 |

![Fase 2 — ejes](assets/benchmark/fase2_ejes.png)

`bytetrack` rinde más que `botsort` (que paga la compensación de cámara); `yolo_sam3+botsort`
tiene la **menor fragmentación** (0.011) y `sam3_text+bytetrack` los **tracklets más largos**
(193) con el menor consumo. BoT-SORT solo ayuda emparejado con `yolo_sam3` (interacción → por
eso el 2×2). `mask_iou` ~0.92 en las 4 configs **apenas discrimina**
([métricas débiles](assets/benchmark/fase2_metricas_debiles.png), suplementarias). La
exactitud llegará con el ground-truth (ver §«Lo que falta»).

---

## 4. Material audiovisual

- 🎥 **Video demo (máx. 2 min):** _[enlace pendiente]_ — muestra la vista original junto al
  resultado segmentado/narrativo (el broadcast superpone segmentación, tracking y
  analítica con explicación visual).
- 📱 **Reel de Instagram (≥ 30 s):** _[enlace pendiente]_

---

## 5. Requisitos de hardware y software

- **Hardware:** GPU NVIDIA para la inferencia (probado en **RTX 5090 / Blackwell**; el
  detector `yolo_sam3` usa ~3.1 GB de VRAM, `sam3_text` ~2.1 GB). La **Capa B**
  (homografía, métrica, eventos, broadcast) corre en **CPU**.
- **Software:** **Python 3.11**, PyTorch (CUDA cu128 o CPU), **SAM 3** (Meta), HF
  Transformers, OpenCV, `supervision`/`trackers` (ByteTrack/BoT-SORT), Ultralytics YOLO,
  `questionary` + `rich` (consola del hub). Lista completa en
  [`requirements.txt`](requirements.txt).

---

## 6. Instalación y reproducción

```bash
git clone <url_del_repositorio>
cd futbot

# Entorno aislado (venv local o conda futbot26), Python 3.11
# Torch va aparte según el destino:
#   GPU (Blackwell): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   CPU:             pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam3.git   # SAM 3 (no está en PyPI)
pip install -e .                                                 # src/ como paquete editable
```

**Configuración:** el config activo se elige con `CONFIG_FILENAME` en `.env`
(`configs/01_yolo_sam3_config.json`). Las rutas se resuelven con
[`src.utils.get_abs_path`](src/utils.py) contra la raíz — nunca se hardcodean.

**Docker / RunPod:**
`docker compose --env-file .env -f docker/docker-compose.yml up --build -d`.

**Insumos no versionados requeridos** (git-ignored; cada quien los provee):

| Insumo | Ruta (config) | Para qué |
|---|---|---|
| Videos `.MOV` | `data/raw/` | dataset |
| Modelo SAM 3 (`sam3.pt` + HF) | `assets/sam3/` | segmentación |
| YOLO afinado `best.pt` | `assets/yolo/best.pt` | detector `yolo_sam3` |
| `.env` | raíz | `CONFIG_FILENAME`, secretos |

> La inferencia requiere GPU. Todo lo que no llama a SAM 3 (rutas, conteo de frames,
> selección del dataset, **post-proceso CPU** si hay un tracking JSON) corre en cualquier
> entorno.

---

## 7. Ejecución del flujo de procesamiento

El punto de entrada es el hub [`main.py`](main.py): corre el pipeline **end-to-end sobre un
video** (solo lo lee; por costo se recomiendan clips < 1 min), es **idempotente** (reusa lo
ya generado sin rehacer la inferencia) y **reporta** dónde quedó cada artefacto.

```bash
# Interactivo: pregunta detector / tracker / vista de cámara / overlays
python main.py data/raw/.../clip.mp4

# Sin preguntar (config por defecto = yolo_sam3 + bytetrack)
python main.py data/raw/.../clip.mp4 --default

# Forzar re-correr todo / declarar vista no superior
python main.py data/raw/.../clip.mp4 --default --overwrite
python main.py data/raw/.../clip_lateral.mp4 --vista generica
```

Fijos del entregable (no se preguntan): homografía por líneas, **Kalman ON**, gol estricto,
broadcast layout 2. La salida destacada es el **video de espectador** en
`outputs/eventos/<clip>/<clip>_broadcast.mp4`. Las etapas de homografía/eventos/broadcast
solo corren en **cámara superior** (`--vista superior`, por defecto); en un clip genérico el
hub corre solo el pipeline base.

**Regenerar los visuales de este README:**

```bash
# 1) (pod) tracking de los clips curados de cámara superior
python notebooks/fase_5_event_analysis/00_prepare_clips.py
# 2) GIFs/PNGs/gráficas -> assets/readme/  (RUN_HEAVY=1 añade la segmentación SAM 3)
RUN_HEAVY=1 jupyter nbconvert --to notebook --execute --inplace \
  notebooks/fase_7_visuales/00_generar_visuales.ipynb
```

---

## Lo que falta / en curso

- **Evaluación con ground-truth — PAUSADA:** a la espera de la anotación manual del equipo
  (Roboflow). Con el COCO GT se medirá mIoU / Boundary IoU / Dice; la evaluación de tracking
  queda diferida.
- **Estrategia de fine-tuning de YOLO — abierta** (Roboflow vs. SAM3-assisted).
- **`bootstrap_data`** — script idempotente para descargar/colocar videos y pesos (hoy
  manual).

---

## 8. Licencia y créditos

- **Licencia:** [Apache License 2.0](LICENSE).
- **Créditos y atribuciones:**
  - **Meta AI** — [SAM 3](https://github.com/facebookresearch/sam3) (segmentación).
  - **ByteTrack / BoT-SORT** vía `trackers` y **Roboflow Supervision** (asociación y
    utilidades de tracking).
  - **Ultralytics YOLO** (detector afinado), **Hugging Face Transformers** (carga de SAM 3).
  - **Anthropic — Claude (Opus 4.8)** como asistente de programación durante el desarrollo.
  - **UAQ Team — Copa FutBotMX 2026.** Filtro de Kalman, homografía multi-feature, capa
    métrica y overlay narrativo: implementación propia.

---

### Más documentación

- [`docs/README.md`](docs/README.md) — documentación por fases (00 fundamentos … 11 Kalman).
- [`notebooks/cookbook_pipeline.ipynb`](notebooks/cookbook_pipeline.ipynb) — recetario de la API.
- [`CLAUDE.md`](CLAUDE.md) — guía de arquitectura para contribuir.
- [`.specs/`](.specs/) — Spec-Driven Development (una carpeta por tarea).
