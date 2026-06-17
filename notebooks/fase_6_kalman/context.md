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

## Bloqueo y decisión
- **T3 (`metric_positions`, cm) está ROTO para estos clips:** `cv2` size mismatch en
  `auto_homography._white_in_carpet` — las máscaras RLE del tracking JSON están a otra
  resolución que los `.mp4` re-encodeados del demo (`fase5_clips/*`). Falla en los 3 clips
  (5m30, 9933_min1, 9938_min1). Necesitaría el **video original full-res** alineado por frame.
- **Workaround:** el experimento corre en **espacio de imagen (px)** usando los centroides del
  tracking JSON directo (`_load_tracks_from_json`), sin homografía/video.

## Resultados honestos (px, 2026-06-16)
- **T6.2 suavidad: KF reduce Var(aceleración) 91–99.9% vs finite-diff.** Real, pero el baseline
  finite-diff es ruidosísimo (Var ~1e8) → cualquier filtro lo baja mucho; interpretar con cuidado.
- **T6.1 oclusión: el KF NO vence a "hold"/lineal en huecos cortos.** En cenital los objetos se
  mueven lento en px; la velocidad del KF se contamina con ruido de centroide y la extrapolación
  predict-only sobre-dispara. Lineal (1 paso finite-diff) ≈ mejor; hold competitivo. NEGATIVO honesto.
- **T6.5 NIS** divergente por clase (balón sobre-suaviza, robots sobre-confían) → CV en px no
  modela bien a los robots maniobrando; params px son stopgap.

## Qué NO se hizo (a propósito)
- NO se metió al paper una tabla "Kalman gana oclusión" — no lo respaldan los datos.
- El aporte defendible del KF aquí: estado posición+velocidad principiado, **suavizado de
  velocidad**, y estimación+incertidumbre **donde T4 deja hueco** (cobertura) — NO "vence baselines".

## Próximos pasos (para que SÍ rinda)
1. **Arreglar T3 → correr en cm** (homografía sobre video original full-res). El beneficio del KF
   (especialmente velocidad cm/s física y oclusión) se evalúa bien en cm, no en px.
2. Mostrar el beneficio de oclusión en **huecos largos** o sobre el **balón en jugadas rápidas**
   (donde la inercia importa), no en cenital lento de huecos cortos.
3. Ablación de `sigma_a/sigma_z` guiada por NIS (4f del plan) una vez en cm.
4. Reframe paper: KF como estado interpretable + suavizado + cobertura, citando el KF interno del
   tracker (no over-claim).
