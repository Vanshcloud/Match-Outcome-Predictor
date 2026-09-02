"""The fetch cache is an optimisation, and optimisations must fail softly.

A corrupt or unreadable cache costs one slow run; a corrupt cache that raises
costs the pipeline. Every load path here returns an empty cache rather than an
error, and that is the property most of these tests exist to hold.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.ingestion.cache import CACHE_VERSION, CacheEntry, FetchCache


def make(tmp_path: Path) -> FetchCache:
    return FetchCache.load(tmp_path / "fetch_cache.json")


def test_a_missing_cache_starts_empty(tmp_path: Path) -> None:
    cache = make(tmp_path)
    assert len(cache) == 0
    assert cache.get("anything") is None


def test_entries_round_trip(tmp_path: Path) -> None:
    cache = make(tmp_path)
    cache.record_fetch(
        "u", etag='"a"', last_modified="Thu, 01 Jan 2026 00:00:00 GMT", sha256="abc", size=12
    )
    cache.save()

    reloaded = make(tmp_path)
    entry = reloaded.get("u")
    assert entry is not None
    assert (entry.etag, entry.sha256, entry.bytes) == ('"a"', "abc", 12)
    assert "u" in reloaded


def test_a_corrupt_cache_starts_fresh_rather_than_raising(tmp_path: Path) -> None:
    """An optimisation that can halt the pipeline is a liability."""
    path = tmp_path / "fetch_cache.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(FetchCache.load(path)) == 0


def test_a_cache_with_unexpected_fields_starts_fresh(tmp_path: Path) -> None:
    """A record shaped for a different version must not crash the constructor."""
    path = tmp_path / "fetch_cache.json"
    path.write_text(
        json.dumps({"cache_version": CACHE_VERSION, "entries": {"u": {"nonsense": 1}}}),
        encoding="utf-8",
    )
    assert len(FetchCache.load(path)) == 0


def test_a_version_change_starts_fresh(tmp_path: Path) -> None:
    """The recorded shape may change between releases. Re-fetching is always
    safe; interpreting an old record with new code may not be."""
    path = tmp_path / "fetch_cache.json"
    path.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION + 1,
                "entries": {"u": {"url": "u", "fetched_at": "2026-01-01T00:00:00+00:00"}},
            }
        ),
        encoding="utf-8",
    )
    assert len(FetchCache.load(path)) == 0


def test_saving_is_atomic(tmp_path: Path) -> None:
    """Written via a temporary file and replaced, so an interrupted run leaves
    the previous cache intact rather than a half-written one."""
    cache = make(tmp_path)
    cache.record_fetch("u", etag=None, last_modified=None, sha256=None, size=None)
    cache.save()
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "fetch_cache.json").is_file()


def test_misses_expire(tmp_path: Path) -> None:
    cache = make(tmp_path)
    cache.record_miss("u")
    assert cache.is_known_missing("u")
    assert not cache.is_known_missing("u", now=datetime.now(tz=UTC) + timedelta(days=30))


def test_a_fetched_url_is_not_treated_as_missing(tmp_path: Path) -> None:
    cache = make(tmp_path)
    cache.record_fetch("u", etag=None, last_modified=None, sha256=None, size=None)
    assert not cache.is_known_missing("u")


def test_an_unknown_url_is_not_treated_as_missing(tmp_path: Path) -> None:
    assert not make(tmp_path).is_known_missing("never-seen")


def test_forget_forces_an_unconditional_fetch(tmp_path: Path) -> None:
    cache = make(tmp_path)
    cache.record_fetch("u", etag='"a"', last_modified=None, sha256=None, size=None)
    cache.forget("u")
    assert cache.get("u") is None
    cache.forget("u")  # forgetting twice is not an error


def test_age_is_measured_from_the_recorded_check() -> None:
    entry = CacheEntry(url="u", fetched_at=(datetime.now(tz=UTC) - timedelta(days=3)).isoformat())
    assert timedelta(days=2) < entry.age() < timedelta(days=4)
