# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Machine learning / computer vision project (Copa FutBotMX, UAQ Team) whose goal is
to **detect, segment, track and analyze robot-soccer match videos**. It contemplates
two pipelines that share the same dataset, config conventions and methodology:

1. **Base pipeline:** YOLO (detection) → SAM3 (segmentation) → ByteTrack (tracking).
2. **Fine-tuning pipeline:** same structure but the YOLO detector is fine-tuned;
   the fine-tuning strategy is not yet decided (candidates: Roboflow, SAM3-assisted labeling).

### Current state

The **SAM3-only pipeline is built**. The constitution's YOLO→SAM3→ByteTrack "base
pipeline" and the fine-tuning pipeline are **not** (YOLO is unexplored — there is a
stray `notebooks/fase_0/yolov8n.pt` but no YOLO code). A single inference facade
fronts two implementations, all config-driven and reusing the same building blocks:

- **Facade** (`src/core/inference.py::run_inference`): the **single entry point per
  video**, with `mode="segmentation" | "tracking"`. Thin router — validates
  `mode`/`sampling` (raises before loading SAM3), resolves frame sampling per mode
  (`sampling="auto"` → segmentation=quota, tracking=contiguous-prefix; rejects
  `quota`+tracking and `contiguous`+segmentation), unifies the return to
  `{"json", "video", "index"}` (`index` is `None` in segmentation), and propagates a
  preloaded `bundle` to both modes. Does **not** reimplement the inference loop.
- **Segmentation** (`src/core/pipeline.py::run_pipeline`, `mode="per_frame"`):
  `video → extract_frames → detect_classes_in_frame (per frame, per class) →
  overlay → mp4 + JSON`. `obj_id`s are **not** stable across frames here. Now also
  accepts `classes`/`bundle` (like `track_video`) for per-call class filtering and
  model reuse.
- **Tracking** (`src/core/tracking.py::track_video`): streams the video frame by
  frame, reuses the per-frame detector, derives boxes from masks and associates
  them with **ByteTrack** (`trackers.ByteTrackTracker`, one tracker per class) into
  **stable, globally-unique `obj_id`s**. Handles full video without OOM. Output:
  incremental mp4 + a tracker-agnostic track index (`Track`/`TrackObservation`) as
  JSON (no masks).

Atomic tasks done (each with `.specs/<task>/{spec,plan,tasks}.md`): env_setup,
editable_module, abs_dir_func, frame_extraction, abs_video_path, data_volume_mounts,
frame_visualization, sam3_loader, classes_config, text_segmentation,
segmentation_overlay, video_writer, pipeline_runner, source_fps,
csv_dataset_metadata, forced_testing_split, eval_frame_export, gt_annotation
(human/process task — no code), video_tracking, inference_schema, optional_render,
unified_inference, batch_inference.

`notebooks/fase_0/` holds numbered exploration notebooks (SAM3 spikes) — exploratory
reference, **not** pipeline code; the production code lives under `src/`.

### Code architecture (the big picture)

Everything is **config-driven** (no hardcoded paths/params — see Configuration
conventions) and the modules compose into the two pipelines above:

- **`src/core/` — pipeline logic over frames:**
  - `sam3_loader.py::load_sam3()` → `Sam3Bundle(processor, model, device)`; loads
    SAM3 once (HF transformers, bf16, cuda-if-available).
  - `segmentation.py::detect_classes_in_frame(frame, classes, bundle)` →
    `{class_name: [Detection(obj_id, mask, score)]}`, running one SAM3 *single-frame*
    video session per class via text prompts. `Detection` is the shared currency
    across the codebase.
  - `overlay.py::overlay_detections(frame, dets_by_class)` → composited RGB frame
    (colors **by class**, from config); `show_overlay` is display-only.
  - `frame_extraction.py` → `extract_frames` (quota/all, returns `(N,H,W,3)`),
    `get_frame_indices` (the sampled source indices), `iter_frames` (streaming
    generator for tracking), `get_video_fps`. All accept a path relative to
    `PROJECT_ROOT` **or** an absolute path.
  - `video_writer.py` → `write_video` (batch) and `open_video_writer` (incremental
    context manager, used by tracking).
  - `inference.py::run_inference` (single facade) routing to `pipeline.py`
    (per-frame orchestrator) and `tracking.py` (tracking orchestrator).
  - `batch.py::run_batch` — sequential batch layer over `run_inference`: selects
    videos from `db_metadata.csv` (by `split` or explicit list), loads SAM3 **once**,
    skips already-done videos (output JSON exists), isolates per-video errors, render
    OFF by default, and returns a per-video summary (`done`/`skipped`/`failed`).
- **`src/data/` — dataset preparation (not inference):**
  - `metadata.py::build_metadata_csv` → `assets/db_metadata.csv` manifest with a
    reproducible `split` column (0=reserve, 1=fine-tuning [23], 2=testing [20];
    seeded; `splits.forced_testing` pins specific videos to testing).
  - `eval_frames.py::export_testing_frames` → freezes the evaluation frame set
    (testing videos' quota frames) under `data/testing_frames/` (git-ignored) plus a
    versioned control CSV `assets/testing_frames.csv`.
- **`src/utils.py`** → `get_abs_path`, `PROJECT_ROOT`, `show_frames`.

Cross-cutting facts worth knowing before editing:
- **Classes are config data**: each has `name`, `sam3_prompts`, `color`, `coco_id`
  under the config `classes` key; code iterates whatever is there, so **adding a
  class is config-only**, no code change.
- **`Detection.obj_id` is mode-dependent**: per-frame (unstable) in per-frame mode,
  **stable** in tracking mode.
- **Output placement**: heavy outputs (mp4 / extracted frames / GT) go under
  `outputs/` or `data/` (git-ignored); lightweight manifests (`db_metadata.csv`,
  `testing_frames.csv`) live in `assets/` (versioned).
- **Lazy imports**: `torch`, `cv2`, `imageio`, `supervision`/`trackers`, matplotlib
  are imported *inside* functions so `import src.core` stays cheap — keep this style.

**Ongoing processes (check the corresponding draft before resuming):**
- **Evaluation of the SAM3-only pipeline — ONGOING / PAUSED.** Multi-task process
  (segmentation metrics: mIoU/Boundary IoU/Dice vs a manual ground-truth). Roadmap
  and live status log: `.specs/drafts/evaluation_sam3_only_roadmap.md` (read its
  "Estado del proceso (bitácora)" section). Done: `eval_frame_export` (600 testing
  frames + versioned `assets/testing_frames.csv`) and `gt_annotation` SDD docs.
  **Paused** waiting on the team's manual annotations in Roboflow; resume with
  `gt_loader` once the COCO ground-truth lands. Tracking evaluation stays deferred.
- **`video_tracking` — implemented, follow-ups open.** `track_video` is done and
  ran on the pod. The output mp4 *looks like plain segmentation* because the overlay
  colors **by class, not by `obj_id`** (the track quality is visible in the JSON, not
  the video). Open follow-ups (see `.specs/drafts/mvp_sam3_only_roadmap.md` task 5):
  (a) tune the config `tracking` section to reduce fragmentation
  (`lost_track_buffer` ↑, `minimum_consecutive_frames` ↑); (b) add a per-`obj_id`
  overlay so tracking is visible; (c) make the tracked classes configurable to
  exclude the static `green_floor`.

**Pending / TODO (not yet done):**
- **Clean up `testing/` scripts to drop the `sys.path` patch.** `src` is now an
  editable package, so `testing/test_abs_dir_func.py` and
  `testing/test_frame_extraction.py` no longer need
  `sys.path.insert(0, str(PROJECT_ROOT))` — remove it and keep the direct
  `import src...`. Should be done as its own atomic SDD task. See
  `.specs/editable_module/tasks.md` → "Trabajo futuro".
- **`bootstrap_data` (data provisioning script).** The real-files model is now
  adopted (data mounts removed; `data/raw`/`assets/sam3` are real dirs carried by
  the app bind-mount — see `.specs/data_volume_mounts/spec.md` §8). What's still
  pending is **automating** how those dirs get populated: an idempotent script
  that checks whether the videos / `sam3.pt` exist and **downloads them if
  missing** (candidate host: Google Drive), skips when already present (so
  contributors who already have the data don't re-download), keeps URLs/credentials
  in `.env`, and on RunPod targets a **persistent network volume** (download once).
  Should be its own atomic SDD task; draft in `.specs/drafts/bootstrap_data.md`.
  RunPod data-placement strategy still undecided.

## Spec-Driven Development workflow (mandatory)

This repo follows a strict 5-step methodology defined in `.specs/constitution.md`.
**Read the constitution before doing anything.** The non-negotiable rule:

> Do not write or modify any source code until a `tasks.md` exists for that task.
> Steps 1–4 produce only `.md` files; real code changes happen only in step 5.

The sequence per atomic task is: **constitution → spec.md → plan.md → tasks.md → implementation**.
Each atomic task gets its own subfolder under `.specs/` holding its `spec.md`,
`plan.md`, `tasks.md` (e.g. `.specs/env_setup/`). `.specs/drafts/` holds raw,
human-authored inputs and is git-ignored.

### Assumption protocol when creating any file in `.specs/`

Before writing any `.specs/` document you MUST:

1. Ask which assumptions to surface: técnicas / no técnicas / funcionales / todas / otra.
2. Present a **numbered list** of every assumption made.
3. The user replies with the numbers they reject.
4. For each rejected item, ask **one question at a time**, showing a progress bar
   (e.g. `Pregunta 1 de N`) and offering 4 alternative assumptions + a 5th "otra".
5. State explicitly when you are ready, then write the document.

This is the literal protocol from constitution §8. Work and documents are in **Spanish**.

### Commits (constitution §7.1 / §11)

- **Never commit or push on your own initiative** — when a step/task looks done,
  **ask** and wait for explicit confirmation.
- **Conventional Commits, message in English:** `type(scope): short imperative
  summary` (≤72 chars, no trailing period). `scope` is preferably the affected
  `.specs/` atomic task (e.g. `data_volume_mounts`) or area (`docker`, `sdd`,
  `config`). One commit per logical change.

## Configuration conventions (from the constitution)

- Global configs live in JSON files named `{NN}_{EXP}.json` (`NN` = version/trial,
  `EXP` = descriptive). They centralize all **relative** data paths. The active
  example is `configs/00_testing_config.json` (holds `working_dirs.dataset_dir` =
  `data/raw`, `working_dirs.sam3_dir` = `assets/sam3`, and `preprocess.frame_quota`
  = `30`, the frame count read by `extract_frames` in quota mode).
- `.env` key `CONFIG_FILENAME` selects which config file under `configs/` to load.
  Note the current `.env` writes it as `CONFIG_FILENAME =...` (space) — parse with `strip()`.
- Code must access paths only through the config file — never hardcode absolute paths.
  Use `src/utils.py::get_abs_path(relative_path: str) -> Path` to turn a config's
  relative path into an absolute one. It resolves against `PROJECT_ROOT`
  (`Path(__file__).resolve().parents[1]`, stable regardless of cwd) and **raises**
  `ValueError` (bad input / absolute path) or `FileNotFoundError` (resolved path missing).
- Secrets and host-specific paths go in `.env` (not versioned).
- The container workspace path is configurable via `CONTAINER_WORKSPACE_DIR` (default
  `futbot`); never assume `/app`.

## Environment & key commands

- **Python 3.11**, isolated in a `venv` inside the project (`.venv/`).
- **Torch is installed separately** from `requirements.txt` (the file documents this):
  - CPU pod: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  - GPU pod (RTX 5090 / Blackwell): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- **SAM3 is not on PyPI** — install manually:
  `pip install git+https://github.com/facebookresearch/sam3.git`.
- **The project itself installs as an editable package** (`setup.py`): run
  `pip install -e .` once in the local venv so `import src` / `from src.core import
  extract_frames` work from any cwd (notebooks included) without `sys.path` hacks.
  In the container this is done at build time (`pip install -e .` in the
  `Dockerfile`), so no manual step is needed there — just rebuild.
- Dev tooling pinned in `requirements.txt` (no dedicated config in `pyproject`/
  `setup.cfg`, so defaults apply): `ruff check .` (lint), `black .` (format).
- **There is no pytest suite yet.** Despite the `test_*.py` names, the files in
  `testing/` are **standalone manual scripts** run directly with `python` (see
  below), not collected by `pytest`.
- Locally the dev environment observed is a **conda env** (`futbot26`), not the
  `.venv/` the constitution mandates — if `python`/`jupyter` aren't on PATH, use
  that interpreter (e.g. `~/miniconda3/envs/futbot26/bin/python`).
- Docker is intentionally simple: `docker/Dockerfile` + `docker/docker-compose.yml`,
  service name **`futbotmx26`**, prepared to run on RunPod. No custom entrypoints,
  no remote image registry. There is a **single** volume: the app workspace bind
  mount `../` → `/${CONTAINER_WORKSPACE_DIR}`. The `command` only keeps the
  container alive.
- **Heavy data model (real files):** `data/raw` (videos) and `assets/sam3`
  (model) must be **real directories with real files** in the repo (git-ignored).
  The app bind-mount carries them into the container — there are **no separate
  data volumes** and **no `HOST_DATA_DIR`/`HOST_SAM3_DIR`** anymore. This avoids
  the symlink trap: a symlink in the bind-mounted tree is the only mechanism that
  breaks Docker (it resolves the mount target *through* the symlink), so we never
  use one. See `.specs/data_volume_mounts/spec.md` §8 (adenda) and the future
  `bootstrap_data` script for how those dirs get populated.

### Running the test scripts

`testing/` holds standalone manual scripts (run directly, not via pytest) — there is
roughly one `test_*.py` per module (`test_env`, `test_abs_dir_func`,
`test_frame_extraction`, `test_metadata`, `test_eval_frame_export`, `test_sam3_loader`,
`test_segmentation`, `test_overlay`, `test_video_writer`, `test_pipeline`,
`test_tracking`, `test_optional_render`, `test_unified_inference`,
`test_batch_inference`):
```bash
python testing/test_env.py             # imports + versions + torch.cuda check
python testing/test_abs_dir_func.py    # exercises get_abs_path against the configs
python testing/test_frame_extraction.py  # extract_frames on a real .MOV
```
**Model/GPU-dependent tests run on the pod, not locally** — anything that calls SAM3
(`test_sam3_loader`, `test_segmentation`, `test_pipeline`, `test_tracking`) needs the
`assets/sam3` model + GPU. `test_tracking.py` has two checks (short clip + a full real
video that is *not* one of the `splits.forced_testing` videos); cap its
`TEST_B_MAX_FRAMES` for a quick run. `test_frame_extraction.py`/`test_metadata.py`/
`test_eval_frame_export.py` only need the real videos under `data/raw`.
In the container, run them after `up`:
```bash
docker compose --env-file .env -f docker/docker-compose.yml up --build -d
docker compose --env-file .env -f docker/docker-compose.yml exec futbotmx26 python testing/test_abs_dir_func.py
```

**Host vs. container — how `data/raw`/`assets/sam3` resolve:** the config keeps
**project-relative** paths (`data/raw`, `assets/sam3`); `get_abs_path` resolves
them against `PROJECT_ROOT`, so code always reads `<repo>/data/raw` and
`<repo>/assets/sam3`. Because the data are **real files in the repo**, this
resolves identically on host and container: `get_abs_path("data/raw")` returns
`<repo>/data/raw` (a real dir **inside** `PROJECT_ROOT`) in both. Populating
those dirs with the actual data is **environment setup**, not code (today:
place/move the files there; future: the `bootstrap_data` script).

So `get_abs_path("data/raw")` works everywhere with no symlinks involved. Videos
live in dated subfolders, so search them recursively (`rglob`), not `glob`.

**Passing video paths to pipeline code:** `extract_frames` accepts either a path
**relative to `PROJECT_ROOT`** (resolved via `get_abs_path`) or an **absolute**
path to an existing file. With real files under `data/raw`, both work — e.g.
`extract_frames(Path("data/raw/.../x.MOV"))` or the absolute resolved path. When
discovering videos, `rglob` over `PROJECT_ROOT / dataset_dir` and return a
project-relative path (`abs_path.relative_to(PROJECT_ROOT)`); avoid building
paths from `Path.cwd()`, which makes results depend on the kernel's working
directory.

## Data & version control

- Dataset: 123 `.MOV` robot-soccer videos hosted in the cloud (variable resolution
  and duration, typically < 5 min). Whoever reproduces the project downloads and
  organizes the raw videos themselves.
- `.gitignore` already excludes videos, model weights/checkpoints, `.env`, venvs,
  outputs/experiment tracking, and heavy data dirs. Keep videos and model data out
  of the remote repo.
