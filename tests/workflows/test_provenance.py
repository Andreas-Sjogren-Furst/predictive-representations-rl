import subprocess

import pytest

from aeris.workflows.provenance import PACKAGE_DIR, code_revision

GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]


def git(repo, *args):
    return subprocess.run(
        [*GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "module.py").write_text("x = 1\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


def test_clean_checkout_records_its_commit(repo):
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    assert code_revision(repo) == {"git_commit": head, "git_dirty": False}


def test_uncommitted_changes_are_flagged(repo):
    (repo / "module.py").write_text("x = 2\n")
    assert code_revision(repo)["git_dirty"] is True


def test_outside_a_checkout_nothing_is_guessed(tmp_path):
    assert code_revision(tmp_path) == {"git_commit": None, "git_dirty": None}


def test_defaults_to_the_aeris_package_not_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a git checkout
    assert code_revision() == code_revision(PACKAGE_DIR)
