"""Every path in the project resolves from one anchor. These tests hold it there."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import PROJECT_ROOT, resolve


def test_project_root_is_the_repository_root() -> None:
    """The parents[2] arithmetic in paths.py is silent when it is wrong.

    Asserting on landmark files means moving the module without fixing the
    index fails here, rather than producing a root two directories too high
    that every later path then hangs off.
    """
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src" / "utils" / "paths.py").is_file()


def test_relative_paths_resolve_against_the_root() -> None:
    assert resolve("data") == PROJECT_ROOT / "data"
    assert resolve(Path("data/raw")) == PROJECT_ROOT / "data" / "raw"


def test_absolute_paths_are_returned_unchanged() -> None:
    """A deployment points DATA_DIR at a mounted volume; that must survive."""
    absolute = Path("/mnt/volume/data")
    assert resolve(absolute) == absolute


def test_resolution_does_not_depend_on_the_working_directory(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The failure this module exists to prevent: a script that works from the
    repo root and breaks when run from scripts/."""
    monkeypatch.chdir(tmp_path)
    assert resolve("data") == PROJECT_ROOT / "data"
