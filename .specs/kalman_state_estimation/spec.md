# Spec — `kalman_state_estimation` (Fase 6) — estimación de estado por filtro de Kalman

## Contexto

ByteTrack/BoT-SORT ya corren un filtro de Kalman de velocidad constante, pero opera en
**espacio de píxeles** y está enterrado dentro de la *asociación de datos* (solo predice cajas
para el matching IoU del siguiente frame); su salida nunca se expone y no tiene sentido físico
para análisis de juego. Fase 6 introduce un **filtro de Kalman explícito en espacio métrico
(cm) como capa de análisis post-asociación**, que corre **encima** de los tracks ya asociados
(T3 `metric_positions` / `cm_positions_lines`). No re-hace detección, asociación ni IDs.

Tres funciones, todas del **mismo** filtro: (i) estado **posición+velocidad** en cm/cm·s⁻¹;
(ii) **puenteo de oclusión** — sin detección durante *k* frames, corre pasos *predict-only* y
la trayectoria/velocidad continúan ("estimar dónde está el objeto aunque no se vea"); (iii)
velocidad **físicamente más suave** que las diferencias finitas de T4. Corre en **CPU local**
(solo del JSON; sin SAM3/YOLO).

> Honestidad / no-overclaim: el aporte NO es "le pusimos Kalman al tracking" (el tracker ya lo
> tiene), sino **un estimador de estado interpretable en cm, desacoplado de la asociación,
> calibrado al presupuesto de error de la homografía, que recupera posiciones ocluidas y
> de-ruidiza la velocidad**. Se cita el KF interno del tracker en el paper.

## Modelo de estado (velocidad constante, CV)

2D desacoplado por eje, ruido de aceleración blanco. **Estado** (por `obj_id`):

```
x = [ px , py , vx , vy ]ᵀ          (cm, cm/s)
```

**Transición** para un paso `dt = (f₂−f₁)/fps` (se usa el gap REAL entre muestras):

```
        | 1 0 dt 0 |
F(dt) = | 0 1 0 dt |          x⁻ = F x
        | 0 0 1  0 |
        | 0 0 0  1 |
```

**Ruido de proceso** (discrete white-noise-acceleration, σ_a = raíz de PSD, cm/s²):

```
            | dt⁴/4   0     dt³/2   0    |
Q(dt) = σ_a²| 0      dt⁴/4  0      dt³/2 |
            | dt³/2   0     dt²     0    |
            | 0      dt³/2  0      dt²   |
```

**Medición** (T3 da posición, nunca velocidad): `z = [px,py]ᵀ`,  `H = [[1,0,0,0],[0,1,0,0]]`.

**Ruido de medición** calibrado del error de homografía (9–23 cm), isotrópico:

```
R = σ_z² · I₂ ,  σ_z = 15 cm   (R = 900·I₂ cuando status_H == "propagated": homografía estancada)
```

**Predict (cada paso):**
```
x⁻ = F(dt) x ;   P⁻ = F(dt) P F(dt)ᵀ + Q(dt)
```

**Update (solo si hay detección):**
```
y = z − H x⁻                  (innovación)
S = H P⁻ Hᵀ + R               (covarianza de innovación)
K = P⁻ Hᵀ S⁻¹                 (ganancia de Kalman)
x = x⁻ + K y ;   P = (I₄ − K H) P⁻
```

**Oclusión (sin medición):**  `x = x⁻ ;  P = P⁻`  → la trayectoria extrapola por la velocidad y
la incertidumbre `σ_pos = √(P₀₀+P₁₁)` CRECE (base de la elipse de confianza).

**Gating (rechazo de outliers, reemplaza el corte duro de 300 cm/s de T4):** distancia de
Mahalanobis al cuadrado `d² = yᵀ S⁻¹ y`. Si `d² > χ²₂(0.99) = 9.21` → outlier: se SALTA el update
(predict-only este frame) pero **NO se tira el track**. Es adaptativo (escala con la incertidumbre).

**Inicialización:** `x₀ = [z_px, z_py, 0, 0]ᵀ`, `P₀ = diag(σ_z², σ_z², v₀², v₀²)`, `v₀ = 200 cm/s`.

**Cap de oclusión:** si el hueco supera `max_gap_frames`, se TERMINA el segmento (no se alucina
indefinidamente) y se re-inicializa en la próxima detección.

## Parámetros por clase

| Param | `orange_ball` | `robot_a`/`robot_b` |
|---|---|---|
| Modelo | CV | CV |
| σ_a (cm/s²) | 800 (acelera más) | 250 |
| σ_z (cm) | 15 | 15 |
| max_gap_frames | 15 | 30 |
| Gate χ² | 9.21 | 9.21 |

Zonas/alfombra (`green_floor`, `yellow_zone`, `blue_zone`) se EXCLUYEN (son anclas, no móviles).

## Homografía → cm (cómo se obtienen las posiciones)

T3 viejo (`auto_homography`) truena en los clips re-encodeados (size mismatch carpet-RLE vs
.mp4). **`cm_positions_lines.py`** usa `homography_multifeature.VideoHomographyLines` (la buena,
por líneas, de Rodrigo / nb07) + **redimensiona el frame a la resolución de la máscara** →
`x̃' ∼ H x̃` proyecta los foot points a la cancha canónica de 243×182 cm. CPU local, sin SAM3.

## Experimentos y métricas

**Sin ground-truth denso** → protocolo de **oclusión sintética (hold-out)**: se ESCONDE un hueco
de `g` frames de un track con detecciones reales; cada método predice las posiciones ocultas y se
compara contra la **detección verdadera escondida** (error euclidiano en cm). Métodos: `hold`
(congela última pos), `linear` (extrapola por diferencia finita), `kalman` (predict-only). `g ∈
{1,2,3,5,8,12}`, ventanas cada 8 frames, miles de instancias.

- T6.1 recuperación de oclusión (error cm vs g, por método).
- T6.2 suavidad: Var de la aceleración (= Δvelocidad/dt); KF vs finite-diff.
- T6.3 continuidad/cobertura (frames con estimación; oclusión rellenada).
- T6.4 headline v_max balón (T4 vs KF).
- T6.5 consistencia NIS (`d²` medio; bien calibrado ≈ 2).
- T6.6 métricas alimentadas por Kalman (zonas/goles/distancia: T3 crudo vs KF).

## Ablación NIS (calibración de R, `03_kalman_ablation.py`)

Se barre σ_z (R) y σ_a (Q) y se mide el NIS medio (consistente ≈ 2). NIS sube al bajar σ_z. Las
**innovaciones reales son ~2 cm**: el ruido TEMPORAL (frame-a-frame) de la homografía es chico,
aunque el sesgo absoluto de landmarks sea 9–23 cm. Calibrado (IMG_9933_5m30): **σ_z = 2 cm**,
σ_a = 400 (balón, NIS **1.24**), σ_a = 250 (robot, NIS **2.54**). Ambos consistentes (NIS ~ 2 = dof).

## Resultados (IMG_9933_5m30, cenital fiable, params NIS-calibrados)

### DESCUBRIMIENTO: el balón gana, el robot no (dicotomía balístico vs maniobra)

- **Balón (movimiento suave/balístico) — KF VENCE a la extrapolación lineal:**
  g=12 → KF **2.65 cm** vs linear 4.57 (hold 1.93); g=5 → KF 1.61 vs linear 2.40. El KF recupera
  el balón a ~2.6 cm en huecos de 12 frames (~0.4 s) y de-ruidiza la velocidad. NIS 1.24 (calibrado).
- **Robot (maniobra impredecible) — ninguna extrapolación con velocidad vence a "hold":**
  g=12 → KF 9.68, linear 8.53, **hold 4.59 (mejor)**. El modelo de **velocidad constante (CV) NO
  modela un robot que gira/acelera**; extrapolar su velocidad (KF o lineal) es peor que congelar la
  posición. NIS 2.54. **Hallazgo honesto y útil:** motiva un modelo de aceleración-constante (CA)
  o IMM para robots — trabajo futuro.
- **Implicación:** el balón es justo el objeto crítico (goles/posesión/velocidad de tiro), así que
  el KF aporta donde más importa; en robots se reporta el límite del modelo CV.
- **T6.2 — suavidad −99.5%** de Var(aceleración) vs finite-diff (balón y robot).
- **T6.4 — v_max balón:** T4 163.9 → **KF 106.4 cm/s** (suavizado, físicamente plausible ~1.1 m/s).
- **T6.6 — integración a fase_5:** goles **3 → 3** (consistente, no rompe), balón mitad azul
  86.9% → 87.0%, **352 frames de oclusión rellenados**. Minimap con trayectoria Kalman + elipse de
  incertidumbre: `assets/fase6/figures/IMG_9933_5m30_kalman_minimap.png`.

## Limitaciones (honestas)

- **CV limitado para robots que maniobran** (el descubrimiento de arriba) → CA/IMM a futuro.
- `IMG_9938_min1` ruidoso (NIS alto, ~60 cm): su homografía por líneas es peor ahí → reportar **9933**.
- "hold" es fuerte en cenital lento; el KF gana en el balón (suave) y aporta velocidad + incertidumbre
  + cobertura en todos.
- 2 clips cenitales → resultados por-track, sin generalización poblacional.

## Validación

`testing/test_kalman_state.py` PASA 3/3 sobre trayectoria CV sintética: recuperación de velocidad,
predicción de oclusión (sigma crece), **NIS = 1.96 ≈ 2**.

## Archivos

- `src/core/kalman_state.py` — `KalmanCV` (predict/update/predict-only/gating), `run_kalman_on_track`.
- `src/core/kalman_kinematics.py` — driver T3→KF (`compute_kalman_states`, `CLASS_PARAMS`).
- `notebooks/fase_6_kalman/cm_positions_lines.py` — cm vía `VideoHomographyLines` (fix del mismatch).
- `notebooks/fase_6_kalman/01_kalman_experiment.py` — experimentos T6.1–T6.5.
- `notebooks/fase_6_kalman/02_kalman_integration.py` — Kalman→métricas fase_5 + minimap (T6.6).
- `testing/test_kalman_state.py` — validación. Tablas/figuras: `assets/fase6/`.

## Próximos pasos

1. Ablación σ_a/σ_z por NIS (por-clip) → NIS ≈ 2. 2. Modelo aceleración-constante (CA) como
ablación. 3. Paper: T6.1 (KF>linear) + T6.4 (velocidad plausible) + figura trayectoria/elipse;
citar el KF interno del tracker.
