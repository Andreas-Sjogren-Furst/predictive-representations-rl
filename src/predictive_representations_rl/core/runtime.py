from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` to the directory that holds pyproject.toml and third_party/."""
    start = (start or Path.cwd()).resolve()

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "third_party").is_dir():
            return candidate

    raise FileNotFoundError(f"No project root (pyproject.toml + third_party/) above {start}")


@dataclass(frozen=True)
class Venv:
    """A Python virtual environment, given relative to the project root."""

    path: str

    def python(self, root: Path) -> Path:
        return root / self.path / "bin" / "python"

    def problems(self, root: Path) -> list[str]:
        if not self.python(root).exists():
            return [f"venv interpreter not found: {self.python(root)}"]
        return []


@dataclass(frozen=True)
class Container:
    """A Singularity/Apptainer image, given relative to the project root or absolute."""

    image: str

    def image_path(self, root: Path) -> Path:
        return (root / self.image).expanduser()

    def problems(self, root: Path) -> list[str]:
        if not self.image_path(root).exists():
            return [f"container image not found: {self.image_path(root)}"]
        return []


Runtime = Venv | Container
