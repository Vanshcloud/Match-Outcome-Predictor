"""Ratings against the real table.

Skips without `python scripts/build_ratings.py`. What these add over the unit
suite is scale: synthetic leagues are tidy, and the properties worth checking
here — that Dixon-Coles beats the class prior, that Elo's error lands where it
was measured, that the two tables still join — only mean something over three
decades and thirty-nine competitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.manifest import read_manifest, verify_manifest
from src.pipelines.ingest import MATCHES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.ratings.elo import EloRatings, mean_squared_error
from src.storage.duckdb_store import MATCHES_VIEW, RATINGS_VIEW, DuckDBStore
from src.utils.config import load_settings
from src.validation.ratings import ratings_checks
from src.validation.report import Outcome, run_checks

pytestmark = pytest.mark.integration

SETTINGS = load_settings()
MATCHES = SETTINGS.paths.processed_dir / MATCHES_FILENAME
RATINGS = SETTINGS.paths.features_dir / RATINGS_FILENAME

# Measured over the full 305,499-match ingest with the shipped constants. The
# band is wide enough to survive the provider adding a season and narrow enough
# that a broken update cannot slip through.
ELO_MSE_RANGE = (0.160, 0.165)


@pytest.fixture(scope="module")
def store() -> DuckDBStore:
    if not MATCHES.is_file():
        pytest.skip(f"no ingested data at {MATCHES}; run scripts/fetch_data.py")
    if not RATINGS.is_file():
        pytest.skip(f"no ratings at {RATINGS}; run scripts/build_ratings.py")
    with DuckDBStore.open_matches(MATCHES, ratings=RATINGS) as opened:
        yield opened


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES.is_file():
        pytest.skip(f"no ingested data at {MATCHES}; run scripts/fetch_data.py")
    return pd.read_parquet(MATCHES)


@pytest.fixture(scope="module")
def ratings() -> pd.DataFrame:
    if not RATINGS.is_file():
        pytest.skip(f"no ratings at {RATINGS}; run scripts/build_ratings.py")
    return pd.read_parquet(RATINGS)


# ---- the ratings table ------------------------------------------------------


def test_the_real_ratings_pass_every_blocking_check(
    ratings: pd.DataFrame, matches: pd.DataFrame
) -> None:
    report = run_checks(ratings, ratings_checks(matches["match_id"]))
    assert report.ok, "\n".join(f"{r.name}: {r.message}" for r in report.blocking)


def test_no_check_is_silently_skipped(ratings: pd.DataFrame, matches: pd.DataFrame) -> None:
    report = run_checks(ratings, ratings_checks(matches["match_id"]))
    assert report.of(Outcome.SKIPPED) == ()


def test_the_ratings_manifest_still_verifies() -> None:
    if not RATINGS.is_file():
        pytest.skip("no ratings; run scripts/build_ratings.py")
    manifest_path = RATINGS.with_suffix(".manifest.json")
    manifest = read_manifest(manifest_path)
    assert manifest["kind"] == "ratings"
    assert verify_manifest(manifest, RATINGS.parent).ok


def test_the_build_recorded_that_causality_was_verified() -> None:
    """The claim is only worth anything if the run that made the file made it."""
    if not RATINGS.is_file():
        pytest.skip("no ratings; run scripts/build_ratings.py")
    manifest = read_manifest(RATINGS.with_suffix(".manifest.json"))
    assert manifest["causal"] is True
    assert manifest["causality_verified_on"]


# ---- the two tables together ------------------------------------------------


def test_matches_and_ratings_join_one_to_one(store: DuckDBStore) -> None:
    """Attached side by side rather than joined in, so the join has to work."""
    joined = store.query(
        f"SELECT count(*) AS n FROM {MATCHES_VIEW} JOIN {RATINGS_VIEW} USING (match_id)"
    )
    assert int(joined.iloc[0]["n"]) == store.count()


def test_a_point_in_time_read_can_carry_its_ratings(store: DuckDBStore) -> None:
    """The read every backtest makes: matches up to a date, with the numbers
    that were knowable before each of them."""
    result = store.query(f"""
        SELECT count(*) AS n, max(m.date) AS latest
        FROM {MATCHES_VIEW} AS m JOIN {RATINGS_VIEW} AS r USING (match_id)
        WHERE m.date <= DATE '2015-06-30' AND r.elo_home IS NOT NULL
        """)
    assert int(result.iloc[0]["n"]) > 100_000
    assert pd.Timestamp(result.iloc[0]["latest"]) <= pd.Timestamp("2015-06-30")


# ---- quality ----------------------------------------------------------------


def test_elo_lands_where_it_was_measured(matches: pd.DataFrame) -> None:
    """Every prediction in the walk is made from prior matches only, so this is
    an out-of-sample figure over the whole table rather than a fit."""
    low, high = ELO_MSE_RANGE
    error = mean_squared_error(matches, EloRatings())
    assert low <= error <= high, f"elo mse {error:.5f} outside {ELO_MSE_RANGE}"


def test_dixon_coles_beats_predicting_the_base_rates(
    ratings: pd.DataFrame, matches: pd.DataFrame
) -> None:
    """The lowest bar a probabilistic forecast has to clear: knowing that 45%
    of matches are home wins is free, and a model that cannot beat it has
    learned nothing about the teams."""
    joined = matches[["match_id", "result"]].merge(ratings, on="match_id")
    priced = joined.dropna(subset=["dc_prob_home", "dc_prob_draw", "dc_prob_away"])
    assert len(priced) > 100_000, "too few priced matches to judge"

    index = {"H": 0, "D": 1, "A": 2}
    probabilities = (
        priced[["dc_prob_home", "dc_prob_draw", "dc_prob_away"]].astype("float64").to_numpy()
    )
    actual = np.array([index[value] for value in priced["result"]])

    model = float(
        -np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-15, None)).mean()
    )
    shares = np.array([(actual == value).mean() for value in range(3)])
    prior = float(-np.log(shares[actual]).mean())
    assert model < prior, f"dixon-coles {model:.4f} did not beat the base rates {prior:.4f}"


def test_elo_and_dixon_coles_agree_about_who_is_favoured(
    ratings: pd.DataFrame,
) -> None:
    """Two independent estimates of the same thing. They need not agree on a
    number, but a systematic disagreement about *direction* would mean one of
    them has home and away the wrong way round."""
    priced = ratings.dropna(subset=["elo_expected_home", "dc_prob_home", "dc_prob_away"])
    elo_favours_home = priced["elo_expected_home"].astype("float64") > 0.5
    poisson_favours_home = priced["dc_prob_home"].astype("float64") > priced["dc_prob_away"].astype(
        "float64"
    )
    agreement = float((elo_favours_home == poisson_favours_home).mean())
    assert agreement > 0.8, f"the two ratings disagree on direction {1 - agreement:.1%} of the time"
