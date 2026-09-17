"""Nothing that holds a secret reaches git or a Docker build context.

The two ignore files are the whole defence. A key in `.env` is safe because one
line in `.gitignore` says so, and that line has no test anywhere else: a rename,
a reordered negation or a stray `!` would take it out silently, and the first
evidence would be a published key.

`.gitignore` is checked through `git check-ignore` rather than by reading the
file, because the file is a program — order, negation and directory rules
interact — and the only correct interpreter of it is git.
"""

from __future__ import annotations

import fnmatch
import subprocess

import pytest

from src.utils.paths import PROJECT_ROOT

SECRET_BEARING: tuple[str, ...] = (
    ".env",
    ".env.local",
    ".env.production",
    ".streamlit/secrets.toml",
)
"""Paths that hold, or are named in `.env.example` as holding, a credential."""

NOT_PUBLIC: tuple[str, ...] = (
    "data/raw/e0.csv",
    "data/processed/matches.parquet",
    "data/features/features.parquet",
    "data/reports/backtest.parquet",
    "data/dashboard/profiles.json",
    "models/servable.joblib",
    "FINAL_AUDIT_REPORT.md",
    ".kilo/worktrees/agent/README.md",
)
"""Generated artefacts, reader state and local notes: not source, not public."""


def ignored_by_git(path: str) -> bool:
    """What git itself would do with ``path``, whether or not it exists."""
    done = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    if done.returncode not in (0, 1):
        pytest.skip(f"git could not answer: {done.stderr.decode().strip()}")
    return done.returncode == 0


def dockerignore_rules() -> list[str]:
    text = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def ignored_by_docker(path: str) -> bool:
    """Close enough to the daemon's matcher for the patterns this file uses.

    Only prefix and glob rules are written here — no exceptions, no `!` — so
    the simple reading and the daemon's agree.
    """
    return any(
        fnmatch.fnmatch(path, rule.rstrip("/"))
        or fnmatch.fnmatch(path, rule.rstrip("/") + "/*")
        or path.startswith(rule.rstrip("/") + "/")
        or fnmatch.fnmatch(path.rsplit("/", 1)[-1], rule.rstrip("/"))
        for rule in dockerignore_rules()
    )


@pytest.mark.parametrize("path", SECRET_BEARING)
def test_no_secret_file_can_be_committed(path: str) -> None:
    assert ignored_by_git(path), f"{path} would be committed"


@pytest.mark.parametrize("path", SECRET_BEARING)
def test_no_secret_file_can_reach_a_build_context(path: str) -> None:
    assert ignored_by_docker(path), f"{path} would be uploaded to the daemon"


@pytest.mark.parametrize("path", NOT_PUBLIC)
def test_generated_and_local_material_is_not_committed(path: str) -> None:
    assert ignored_by_git(path), f"{path} would be committed"


def test_the_example_env_file_is_the_one_that_is_committed() -> None:
    """`.env.*` is ignored and `.env.example` is negated back in; the negation
    is the fragile half, and without it the documented configuration surface
    disappears from the repository."""
    assert not ignored_by_git(".env.example")


def test_the_example_env_file_holds_no_value_that_looks_like_a_key() -> None:
    """It is the one committed file whose whole purpose is to name secrets."""
    suspicious = [
        line
        for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#") and "=" in line
        for value in [line.split("=", 1)[1].strip().strip("\"'")]
        if len(value) >= 16 and value.replace("-", "").replace("_", "").isalnum()
    ]
    assert not suspicious, suspicious
