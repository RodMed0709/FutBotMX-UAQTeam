# Fase 6 — Kalman state estimation (bitácora honesta)

## Qué se construyó
- `src/core/kalman_state.py` — KF velocidad-constante 2D from-scratch (numpy), estado
  `[px,py,vx,vy]`. predict/update + **predict-only en oclusión** + gating Mahalanobis
  (reemplaza el corte duro de 300 cm/s de T4). `run_kalman_on_track` con cap `max_gap_frames`.
- `src/core/kalman_kinematics.py` — driver que consume T3 (`MetricResult`) por obj_id, serie
  densa, resumen comparable a T4. `CLASS_PARAMS` por clase.
- `notebooks/fase_6_kalman/01_kalman_experiment.py` — experimentos: T6.1 oclusión sintética,
  T6.2 suavidad, T6.3 continuidad, T6.4 headline v_max, T6.5 NIS → CSVs en `assets/fase6/tables/`.
- `testing/test_kalman_state.py` — **PASA 3/3** (CV sintético): vel err 23.8 cm/s, oclusión
  predice bien + sigma crece, **NIS=1.96≈2**. La matemática del KF está validada.

## Homografía en cm — RESUELTO con VideoHomographyLines (nb07 de Rodrigo)
- El `auto_homography` viejo (que usa T3) truena: `cv2` size mismatch (carpet-RLE a res de
  inferencia vs `.mp4` del demo a otra res).
- **FIX (`cm_positions_lines.py`):** usar `homography_multifeature.VideoHomographyLines` (la
  buena, por líneas) + **redimensionar el frame a la resolución de la carpet-mask** antes de
  `solve_lines_masks`. Así todo alinea (foot points del JSON ya en esa res). CPU local, sin SAM3.
  Los 3 clips corren sin crash. Params del KF de vuelta a **cm** (sigma_z=15 del error 9-23 cm).

## Resultados en cm (2026-06-16) — clip fiable IMG_9933_5m30
- **T6.1 oclusión (cm):** balón gap=12 → hold 1.93 / **kalman 2.33** / linear 4.57; robot gap=12
  → hold 4.59 / **kalman 6.46** / linear 8.53. **El KF VENCE a la extrapolación lineal** (el
  baseline con velocidad), recupera a ~2cm (balón)/~6cm (robot) en 12 frames. "Hold" gana en
  cenital lento (objetos casi no se mueven) — matiz honesto, reportarlo.
- **T6.4 v_max balón:** T4=164 → KF=145 cm/s (suavizado, físicamente plausible ~1.5 m/s).
- **T6.2 suavidad:** −99.7 a −100% varianza de aceleración.

## Caveats honestos
- **IMG_9938_min1 ruidoso** (NIS 14-52, errores ~60cm): su homografía por líneas es peor ahí.
  El cenital fiable es **9933**. Reportar 9933 como principal.
- **NIS aún no ≈2** (sigma_z=15 sobre-suaviza en 5m30: NIS 0.02). Refinamiento: tunear R por-clip
  (ablación 4f). No bloquea el hallazgo (KF > linear, velocidad plausible).

## Próximos pasos
1. Ablación `sigma_a/sigma_z` por NIS (por-clip) → afinar a NIS≈2.
2. Para el paper: tabla T6.1 (KF vs linear vs hold, 9933) + T6.4 (velocidad plausible) +
   framing honesto (hold fuerte en lento; KF gana entre métodos con velocidad + da incertidumbre).
   Citar el KF interno del tracker (no over-claim).
3. Figura: trayectoria con elipse de incertidumbre creciendo en el hueco.
