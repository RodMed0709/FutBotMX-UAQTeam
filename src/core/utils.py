from pathlib import Path

# Root dir for the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_abs_path(relative_path: str):
    """Turns a path relative to the repository into an absolute path."""
    return ROOT_DIR / relative_path
