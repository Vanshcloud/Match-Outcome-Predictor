"""Tests against real downloaded provider files.

Every test here SKIPS when the data is absent, so a clean checkout stays green
and CI needs no network. Run `python scripts/fetch_data.py` first to enable
them, then `pytest -m integration`.

**What is deliberately not here any more.** Milestone 2 spelled out its data
assertions in this file — the result agrees with the score, the table is
chronological, home advantage is plausible. Those moved into
:mod:`src.validation.matches` in Milestone 3 and are now run by the pipeline
itself on every ingest. Restating them here would be the same rule in two
places, which is the kind of duplication that ends with the two disagreeing and
nobody noticing. What is left is the part a unit test genuinely cannot reach:
the real files, the real store, and the incremental state on disk.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.cache import CACHE_FILENAME, FetchCache
from src.ingestion.csv_reader import read_provider_csv
from src.ingestion.manifest import read_manifest, verify_manifest
from src.ingestion.registry import load_registry
from src.pipelines.ingest import MATCHES_FILENAME, RAW_MANIFEST_FILENAME
from src.storage.duckdb_store import DuckDBStore
from src.utils.config import load_settings
from src.validation.card import build_card, competition_coverage
from src.validation.matches import match_checks
from src.validation.report import Outcome, run_checks

pytestmark = pytest.mark.integration

SETTINGS = load_settings()
MATCHES = SETTINGS.paths.processed_dir / MATCHES_FILENAME
RAW = SETTINGS.paths.raw_dir / "football-data"
RAW_MANIFEST = SETTINGS.paths.raw_dir / RAW_MANIFEST_FILENAME


@pytest.fixture(scope="module")
def store() -> DuckDBStore:
    if not MATCHES.is_file():
        pytest.skip(f"no ingested data at {MATCHES}; run scripts/fetch_data.py")
    with DuckDBStore.open_matches(MATCHES) as opened:
        yield opened


@pytest.fixture(scope="module")
def matches(store: DuckDBStore) -> pd.DataFrame:
    return store.read_matches()


# ---- the validation suite, against the real table ---------------------------


def test_the_real_table_passes_every_blocking_check(matches: pd.DataFrame) -> None:
    """The whole suite in one assertion, naming what failed rather than which
    line of a test file did."""
    report = run_checks(matches, match_checks(load_registry()))
    assert report.ok, "\n".join(f"{r.name}: {r.message}" for r in report.blocking)


def test_no_check_is_silently_skipped_on_the_full_ingest(matches: pd.DataFrame) -> None:
    """A skipped check is an unasked question. Over three hundred thousand rows
    there is no excuse for one, and a suite quietly skipping half of itself is
    exactly how a validation gate stops being a gate."""
    report = run_checks(matches, match_checks(load_registry()))
    assert report.of(Outcome.SKIPPED) == ()


def test_the_only_known_warning_is_the_one_we_documented(matches: pd.DataFrame) -> None:
    """One row in 303,517 — an Argentinian match played 2015-01-29 and filed
    under season 2013-14 in the provider's own file. Pinned so a *second*
    upstream defect cannot hide behind the first."""
    report = run_checks(matches, match_checks(load_registry()))
    failed = {result.name for result in report.of(Outcome.FAILED)}
    assert failed <= {"matches fall inside the season they are labelled with"}


# ---- the store over the real file -------------------------------------------


def test_a_store_read_equals_a_direct_read(matches: pd.DataFrame) -> None:
    """Ten million values through DuckDB and back. If this ever diverges,
    every aggregate in the dataset card is describing a different table from
    the one the models train on."""
    assert matches.equals(pd.read_parquet(MATCHES))


def test_a_point_in_time_read_excludes_the_future(store: DuckDBStore) -> None:
    """The operation every temporal split performs. A store that leaks one row
    past the cutoff leaks it into every backtest."""
    cutoff = "2015-06-30"
    past = store.read_matches(until=cutoff)
    assert past["date"].max() <= pd.Timestamp(cutoff)
    assert len(past) < store.count()


def test_the_dataset_card_can_be_generated_from_the_real_table(store: DuckDBStore) -> None:
    report = run_checks(store.read_matches(), match_checks(load_registry()))
    card = build_card(store, report, source=MATCHES)
    coverage = competition_coverage(store)
    assert len(coverage) == len(set(coverage["competition_id"]))
    for competition_id in coverage["competition_id"]:
        assert f"`{competition_id}`" in card


def test_every_registered_competition_that_has_data_is_in_the_card(store: DuckDBStore) -> None:
    registered = {competition.id for competition in load_registry().competitions}
    assert set(competition_coverage(store)["competition_id"]) <= registered


def test_team_ids_are_stable_across_divisions(store: DuckDBStore) -> None:
    """A club promoted from the Championship must keep its id, or every rolling
    feature restarts its history at the promotion. Not expressible as a
    validation check — it is a property of the *ingest as a whole*, and a
    single-competition run would legitimately fail it."""
    result = store.query("""
        SELECT count(*) AS n FROM (
            SELECT home_team_id FROM matches
            WHERE competition_id LIKE 'ENG\\_%' ESCAPE '\\'
            GROUP BY home_team_id
            HAVING count(DISTINCT competition_id) > 1)
        """)
    assert int(result.iloc[0]["n"]) > 0, "expected clubs appearing in two English divisions"


# ---- the real files on disk -------------------------------------------------


def test_every_cached_raw_file_still_parses() -> None:
    """The regression guard for the reader. Italian Serie B 2003/04 is the file
    that falsified the original ragged-row rule; this walks all of them."""
    if not RAW.is_dir():
        pytest.skip(f"no raw files at {RAW}; run scripts/fetch_data.py")
    files = sorted(RAW.rglob("*.csv"))
    if not files:
        pytest.skip("no raw files downloaded")
    for path in files:
        assert read_provider_csv(path), f"{path} parsed to zero rows"


def test_the_raw_manifest_covers_every_cached_file() -> None:
    """A file present but unrecorded is a gap in the provenance chain: the
    canonical table could have been built from bytes nothing checksummed."""
    if not RAW_MANIFEST.is_file():
        pytest.skip("no raw manifest; run scripts/fetch_data.py")
    manifest = read_manifest(RAW_MANIFEST)
    recorded = {entry["path"] for entry in manifest["files"]}  # type: ignore[index,union-attr]
    root = RAW_MANIFEST.parent
    on_disk = {path.relative_to(root).as_posix() for path in root.rglob("*.csv")}
    assert on_disk - recorded == set(), "cached files missing from the manifest"


def test_the_raw_manifest_still_verifies() -> None:
    """Re-checksums every cached file. A mismatch means either local corruption
    or a provider revision — both worth knowing before training on it."""
    if not RAW_MANIFEST.is_file():
        pytest.skip("no raw manifest; run scripts/fetch_data.py")
    result = verify_manifest(read_manifest(RAW_MANIFEST), RAW_MANIFEST.parent)
    assert result.ok, f"raw files changed since the manifest was written: {result.summary()}"


def test_the_fetch_cache_holds_validators_for_what_it_fetched() -> None:
    """Without validators every re-run is a full download, so an entry that
    recorded a fetch but no ETag or Last-Modified is a silent regression."""
    cache_path = SETTINGS.paths.raw_dir / CACHE_FILENAME
    if not cache_path.is_file():
        pytest.skip("no fetch cache; run scripts/fetch_data.py")

    cache = FetchCache.load(cache_path)
    fetched = [
        cache.get(url)
        for url in cache._entries  # noqa: SLF001 - inspecting cache contents is the point
        if not cache._entries[url].missing  # noqa: SLF001
    ]
    assert fetched, "cache recorded no successful fetches"
    without = [entry.url for entry in fetched if entry and not (entry.etag or entry.last_modified)]
    assert not without, f"fetched without recording a validator: {without[:5]}"
