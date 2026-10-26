"""Identify the code that produced a run, for the manifest's provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path

import aeris

PACKAGE_DIR = Path(aeris.__file__).resolve().parent


def code_revision(path: Path = PACKAGE_DIR) -> dict[str, str | bool | None]:
    """The git commit of the checkout containing ``path`` and whether it has uncommitted changes.

    Defaults to the directory AERIS itself is imported from, not the current
    working directory. Both values are ``None`` when ``path`` is not inside a
    git checkout (for example an installed wheel) or git is unavailable; a
    commit with ``git_dirty=True`` does not fully identify the code that ran.
    """
    commit = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain") if commit is not None else None
    return {"git_commit": commit, "git_dirty": None if status is None else status != ""}


def _git(path: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()
