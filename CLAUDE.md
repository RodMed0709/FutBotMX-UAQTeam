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

Early implementation. Four atomic tasks are done (see `.specs/<task>/tasks.md`):
- **`env_setup`** — venv/Docker environment, `docker/`, `.env`, `testing/test_env.py`.
- **`abs_dir_func`** — `src/utils.py::get_abs_path`, `testing/test_abs_dir_func.py`.
- **`frame_extraction`** — `src/core/frame_extraction.py::extract_frames`,
  `testing/test_frame_extraction.py` (first piece of `src/core/`, the pipeline's
  core logic submodule).
- **`frame_visualization`** — `src/utils.py::show_frames`, validated by
  `notebooks/02_frame_visualization_demo.ipynb`. A *display-only* utility: takes a
  4D `(N,H,W,3)` NumPy array (e.g. `extract_frames`' output) and shows up to 6
  frames (uniformly sampled if more) in a matplotlib grid; it never writes to disk.

Scaffolding dirs: `src/ assets/ configs/ data/ docker/ models/ notebooks/ outputs/ testing/`.
The detection/segmentation/tracking pipelines themselves are **not built yet**.

**Pending / TODO (not yet done):**
- **Clean up `testing/` scripts to drop the `sys.path` patch.** `src` is now an
  editable package, so `testing/test_abs_dir_func.py` and
  `testing/test_frame_extraction.py` no longer need
  `sys.path.insert(0, str(PROJECT_ROOT))` — remove it and keep the direct
  `import src...`. Should be done as its own atomic SDD task. See
  `.specs/editable_module/tasks.md` → "Trabajo futuro".

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
- Dev tooling pinned in `requirements.txt`: `ruff` (lint), `black` (format), `pytest` (tests).
- Docker is intentionally simple: `docker/Dockerfile` + `docker/docker-compose.yml`,
  service name **`futbotmx26`**, prepared to run on RunPod. No custom entrypoints,
  no remote image registry. Three volumes map host paths from `.env`:
  app → `/${CONTAINER_WORKSPACE_DIR}`, `${HOST_DATA_DIR}` → `/Meta_Glasses`,
  `${HOST_SAM3_DIR}` → `/sam3`.
- Heavy data is reached via symlinks created at container **startup** (not build —
  the app bind-mount would hide build-time symlinks): `data/raw → /Meta_Glasses`,
  `assets/sam3 → /sam3`.

### Running the test scripts

`testing/` holds standalone manual scripts (run directly, not via pytest):
```bash
python testing/test_env.py             # imports + versions + torch.cuda check
python testing/test_abs_dir_func.py    # exercises get_abs_path against the configs
python testing/test_frame_extraction.py  # extract_frames on a real .MOV (container only)
```
`test_frame_extraction.py` needs the mounted videos, so run it **inside the
container** (the host symlinks are dead).
In the container, run them after `up`:
```bash
docker compose --env-file .env -f docker/docker-compose.yml up --build -d
docker compose --env-file .env -f docker/docker-compose.yml exec futbotmx26 python testing/test_abs_dir_func.py
```

**Important gotcha — host vs. container:** `data/raw` and `assets/sam3` resolve only
**inside the container**, where `/Meta_Glasses` and `/sam3` are mounted from `HOST_DATA_DIR`/
`HOST_SAM3_DIR`. On the host they are **dead symlinks**, so `get_abs_path("data/raw")`
raises `FileNotFoundError` — this is expected. Anything touching the videos/model
data must run in the container. Videos live in dated subfolders, so search them
recursively (`rglob`), not `glob`.

**Passing video paths to pipeline code:** `get_abs_path` only accepts paths
**relative to `PROJECT_ROOT`** (it rejects absolute paths). But `data/raw` is a
symlink pointing *outside* the project (`/Meta_Glasses`), so a resolved absolute
video path is **not** under `PROJECT_ROOT`. Therefore pass paths like
`data/raw/<date>/.../IMG.MOV` (project-relative, symlink **unresolved**) — e.g.
`extract_frames(Path("data/raw/.../x.MOV"))`. When discovering videos, `rglob`
over `PROJECT_ROOT / dataset_dir` (the unresolved symlink), not over
`get_abs_path(dataset_dir)` (which resolves to `/Meta_Glasses` and yields paths
the pipeline functions will reject).

## Data & version control

- Dataset: 123 `.MOV` robot-soccer videos hosted in the cloud (variable resolution
  and duration, typically < 5 min). Whoever reproduces the project downloads and
  organizes the raw videos themselves.
- `.gitignore` already excludes videos, model weights/checkpoints, `.env`, venvs,
  outputs/experiment tracking, and heavy data dirs. Keep videos and model data out
  of the remote repo.
