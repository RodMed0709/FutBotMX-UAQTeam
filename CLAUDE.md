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

Early implementation. Several atomic tasks are done (each has its own
`.specs/<task>/{spec,plan,tasks}.md`):
- **`env_setup`** — venv/Docker environment, `docker/`, `.env`, `testing/test_env.py`.
- **`editable_module`** — `setup.py`; `src` installs as an editable package so
  `import src` works from any cwd (notebooks included) without `sys.path` hacks.
- **`abs_dir_func`** — `src/utils.py::get_abs_path`, `testing/test_abs_dir_func.py`.
- **`frame_extraction`** — `src/core/frame_extraction.py::extract_frames`,
  `testing/test_frame_extraction.py` (first piece of `src/core/`, the pipeline's
  core logic submodule).
- **`abs_video_path`** — extends `extract_frames` to accept an **absolute** path
  to an existing file (not only PROJECT_ROOT-relative); `testing/test_abs_video_path.py`.
- **`data_volume_mounts`** — Docker data model; **revised** (see its `spec.md` §8)
  to the "real files in the repo" model: no separate data volumes.
- **`frame_visualization`** — `src/utils.py::show_frames`, a *display-only*
  utility: takes a 4D `(N,H,W,3)` NumPy array (e.g. `extract_frames`' output) and
  shows up to 6 frames (uniformly sampled if more) in a matplotlib grid; never
  writes to disk.

Scaffolding dirs: `src/ assets/ configs/ data/ docker/ models/ notebooks/ outputs/ testing/`.
`notebooks/fase_0/` holds numbered exploration notebooks (`00_env_check` …
`05_pipline_testing`), several of them SAM3 spikes — exploratory, not pipeline code.
The detection/segmentation/tracking pipelines themselves are **not built yet**.

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

`testing/` holds standalone manual scripts (run directly, not via pytest):
```bash
python testing/test_env.py             # imports + versions + torch.cuda check
python testing/test_abs_dir_func.py    # exercises get_abs_path against the configs
python testing/test_frame_extraction.py  # extract_frames on a real .MOV
```
`test_frame_extraction.py` needs the videos. With the real-files model it runs
the same **in the container** and **locally** (both read the real `data/raw`).
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
