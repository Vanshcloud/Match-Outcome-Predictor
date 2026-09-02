"""Manifests are this project's dataset versioning, in place of DVC.

The behaviour worth protecting is that verification *reports* rather than
raises: the provider does revise history, and the caller is better placed than
this module to decide whether a change is corruption or a correction.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.manifest import (
    build_entries,
    checksum,
    read_manifest,
    verify_manifest,
    write_manifest,
)


def make_tree(root: Path) -> list[Path]:
    (root / "sub").mkdir(parents=True, exist_ok=True)
    a = root / "a.csv"
    b = root / "sub" / "b.csv"
    a.write_text("Div,Date\nE0,x\n", encoding="utf-8")
    b.write_text("Div,Date\nE1,y\n", encoding="utf-8")
    return [a, b]


def test_checksum_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("one", encoding="utf-8")
    first = checksum(path)
    assert first == checksum(path)
    path.write_text("two", encoding="utf-8")
    assert checksum(path) != first


def test_paths_are_recorded_relative_with_forward_slashes(tmp_path: Path) -> None:
    """So a manifest written on one platform verifies on another."""
    files = make_tree(tmp_path)
    entries = build_entries(tmp_path, files)
    assert {entry.path for entry in entries} == {"a.csv", "sub/b.csv"}


def test_entries_are_sorted_so_manifests_diff_cleanly(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    assert [e.path for e in build_entries(tmp_path, reversed(files))] == ["a.csv", "sub/b.csv"]


def test_manifest_records_provenance(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    manifest = write_manifest(
        tmp_path / "manifest.json", tmp_path, files, extra={"provider": "football-data"}
    )
    assert manifest["file_count"] == 2
    assert manifest["provider"] == "football-data"
    assert manifest["total_bytes"] == sum(f.stat().st_size for f in files)


def test_manifest_round_trips(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    destination = tmp_path / "manifest.json"
    written = write_manifest(destination, tmp_path, files)
    assert read_manifest(destination) == written


def test_manifest_is_valid_json_with_a_trailing_newline(tmp_path: Path) -> None:
    """A file without one is a permanent one-line diff in every review."""
    destination = tmp_path / "manifest.json"
    write_manifest(destination, tmp_path, make_tree(tmp_path))
    text = destination.read_text(encoding="utf-8")
    assert text.endswith("\n")
    json.loads(text)


def test_verification_passes_on_untouched_files(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    result = verify_manifest(write_manifest(tmp_path / "m.json", tmp_path, files), tmp_path)
    assert result.ok
    assert len(result.unchanged) == 2


def test_a_changed_file_is_reported_not_raised(tmp_path: Path) -> None:
    """The provider revises files — a corrected scoreline, a late referee.
    That is information for the caller, not an exception."""
    files = make_tree(tmp_path)
    manifest = write_manifest(tmp_path / "m.json", tmp_path, files)
    files[0].write_text("Div,Date\nE0,CHANGED\n", encoding="utf-8")

    result = verify_manifest(manifest, tmp_path)
    assert not result.ok
    assert result.changed == ("a.csv",)
    assert result.missing == ()


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    files = make_tree(tmp_path)
    manifest = write_manifest(tmp_path / "m.json", tmp_path, files)
    files[1].unlink()

    result = verify_manifest(manifest, tmp_path)
    assert result.missing == ("sub/b.csv",)
    assert "1 missing" in result.summary()


def test_truncation_is_detected(tmp_path: Path) -> None:
    """The failure mode the whole manifest exists for: a short file still
    parses, and a season quietly missing its last weeks corrupts every rolling
    feature computed from it without raising anything."""
    files = make_tree(tmp_path)
    manifest = write_manifest(tmp_path / "m.json", tmp_path, files)
    files[0].write_text("Div,Date\n", encoding="utf-8")
    assert verify_manifest(manifest, tmp_path).changed == ("a.csv",)
