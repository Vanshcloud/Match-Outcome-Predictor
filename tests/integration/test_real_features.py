"""Features against the real table.

Skips without `python scripts/build_features.py`. What these add over the unit
suite is scale and mess: synthetic leagues have every club playing every week
for a whole season, and the real table has promotions, thirty-year gaps,
competitions with no shot data at all, and a club that once went twenty-six
years between appearances.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.registry import FEATURE_COLUMNS, FEATURES
from src.ingestion.manifest import read_manifest, verify_manifest
from src.pipelines.features import FEATURES_FILENAME
from src.pipelines.ingest import MATCHES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.storage.duckdb_store import MATCHES_VIEW, DuckDBStore
from src.utils.config import load_settings
from src.validation.features import feature_checks
from src.validation.report import Outcome, run_checks

pytestmark = pytest.mark.integration

SETTINGS = load_settings()
MATCHES = SETTINGS.paths.processed_dir / MATCHES_FILENAME
FEATURES_PATH = SETTINGS.paths.features_dir / FEATURES_FILENAME
RATINGS = SETTINGS.paths.features_dir / RATINGS_FILENAME


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES.is_file():
        pytest.skip(f"no ingested data at {MATCHES}; run scripts/fetch_data.py")
    return pd.read_parquet(MATCHES)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    if not FEATURES_PATH.is_file():
        pytest.skip(f"no features at {FEATURES_PATH}; run scripts/build_features.py")
    return pd.read_parquet(FEATURES_PATH)


def test_the_real_features_pass_every_blocking_check(
    features: pd.DataFrame, matches: pd.DataFrame
) -> None:
    report = run_checks(features, feature_checks(matches["match_id"]))
    assert report.ok, "\n".join(f"{r.name}: {r.message}" for r in report.blocking)


def test_no_check_is_silently_skipped(features: pd.DataFrame, matches: pd.DataFrame) -> None:
    report = run_checks(features, feature_checks(matches["match_id"]))
    assert report.of(Outcome.SKIPPED) == ()


def test_the_manifest_records_a_verified_build() -> None:
    """The claim is only worth anything if the run that made the file made it."""
    if not FEATURES_PATH.is_file():
        pytest.skip("no features; run scripts/build_features.py")
    manifest = read_manifest(FEATURES_PATH.with_suffix(".manifest.json"))
    assert manifest["kind"] == "features"
    assert manifest["causal"] is True
    assert manifest["features"] == list(FEATURE_COLUMNS)
    assert verify_manifest(manifest, FEATURES_PATH.parent).ok


def test_home_venue_form_beats_overall_form(features: pd.DataFrame) -> None:
    """The whole reason venue form is a separate column.

    A home side's record *at home* is better than its record everywhere, by
    roughly the size of home advantage. If these two ever converged, the venue
    window would have stopped restricting anything.
    """
    overall = float(features["home_form_points_5"].astype("float64").mean())
    at_home = float(features["home_venue_points_5"].astype("float64").mean())
    assert at_home > overall + 0.15, f"venue form {at_home:.3f} against overall {overall:.3f}"


def test_away_venue_form_is_worse_than_overall(features: pd.DataFrame) -> None:
    """The mirror image, and the check that catches home and away swapped in
    the reshape — an error the previous test alone would not see."""
    overall = float(features["away_form_points_5"].astype("float64").mean())
    away = float(features["away_venue_points_5"].astype("float64").mean())
    assert away < overall - 0.05, f"away venue form {away:.3f} against overall {overall:.3f}"


def test_form_is_available_for_almost_every_match(features: pd.DataFrame) -> None:
    """Only a team's first appearance has none, and there are 1,323 teams
    against three hundred thousand matches."""
    assert float(features["home_form_points_5"].notna().mean()) > 0.99


def test_shot_form_follows_the_feeds_that_carry_shots(features: pd.DataFrame) -> None:
    """Coverage is a timeline, not a constant: the provider added shot data to
    most European divisions around 2019/20, and the secondary feed has none at
    all. A shot feature that was suspiciously well covered would mean it had
    invented values where the source had none."""
    coverage = float(features["home_shots_for_5"].notna().mean())
    assert 0.3 < coverage < 0.6, f"shot form coverage {coverage:.3f}"


def test_every_registered_feature_was_actually_built(features: pd.DataFrame) -> None:
    """A column registered but never filled is a promise to a consumer that
    nothing keeps."""
    for feature in FEATURES:
        assert features[feature.name].notna().any(), feature.name


def test_matches_features_and_ratings_all_join() -> None:
    """Three tables, one key. If any pair stops lining up, every model silently
    trains on fewer rows than it thinks."""
    if not FEATURES_PATH.is_file():
        pytest.skip("no features; run scripts/build_features.py")
    if not RATINGS.is_file():
        pytest.skip("no ratings; run scripts/build_ratings.py")
    with DuckDBStore.open_matches(MATCHES, ratings=RATINGS) as store:
        store.attach_parquet("features", FEATURES_PATH)
        joined = store.query(f"""
            SELECT count(*) AS n FROM {MATCHES_VIEW}
            JOIN ratings USING (match_id)
            JOIN features USING (match_id)
            """)
        assert int(joined.iloc[0]["n"]) == store.count()


def test_a_point_in_time_read_carries_its_features() -> None:
    """The read every backtest makes: matches up to a date, with the numbers
    that were knowable before each of them."""
    if not FEATURES_PATH.is_file():
        pytest.skip("no features; run scripts/build_features.py")
    with DuckDBStore.open_matches(MATCHES) as store:
        store.attach_parquet("features", FEATURES_PATH)
        result = store.query(f"""
            SELECT count(*) AS n, max(m.date) AS latest
            FROM {MATCHES_VIEW} AS m JOIN features AS f USING (match_id)
            WHERE m.date <= DATE '2015-06-30' AND f.home_form_points_5 IS NOT NULL
            """)
    assert int(result.iloc[0]["n"]) > 100_000
    assert pd.Timestamp(result.iloc[0]["latest"]) <= pd.Timestamp("2015-06-30")
