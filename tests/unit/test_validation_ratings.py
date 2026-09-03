"""Checks on the ratings table.

These ask a different question from the match checks: not "is this football"
but "is this arithmetically possible". None of them can fail because of the
provider — they can only fail because of us, which is why they run on every
build rather than living in a unit test.
"""

from __future__ import annotations

import pandas as pd

from src.ratings.base import RATINGS_SCHEMA
from src.validation.ratings import ratings_checks
from src.validation.report import Outcome, Severity, run_checks


def ratings(rows: int = 4, **overrides: object) -> pd.DataFrame:
    """A valid ratings table, before whatever the test breaks."""
    base: dict[str, object] = {
        "match_id": [f"m{index}" for index in range(rows)],
        "elo_home": [1500.0] * rows,
        "elo_away": [1490.0] * rows,
        "elo_expected_home": [0.55] * rows,
        "elo_home_played": [10] * rows,
        "elo_away_played": [12] * rows,
        "dc_home_lambda": [1.5] * rows,
        "dc_away_lambda": [1.1] * rows,
        "dc_prob_home": [0.5] * rows,
        "dc_prob_draw": [0.25] * rows,
        "dc_prob_away": [0.25] * rows,
    }
    return pd.DataFrame({**base, **overrides}).astype(RATINGS_SCHEMA)


def failures(frame: pd.DataFrame, **kwargs: object) -> set[str]:
    report = run_checks(frame, ratings_checks(**kwargs))  # type: ignore[arg-type]
    return {result.name for result in report.of(Outcome.FAILED)}


def test_a_valid_table_passes_everything() -> None:
    assert failures(ratings()) == set()


def test_a_null_elo_is_caught() -> None:
    """Elo has a prior for a team it has never seen, so unlike Dixon-Coles it
    is never unable to answer. A null means the walk skipped a row."""
    frame = ratings()
    frame.loc[0, "elo_home"] = pd.NA
    assert failures(frame) == {"elo is always available"}


def test_an_expected_score_outside_the_unit_interval_is_caught() -> None:
    frame = ratings()
    frame.loc[0, "elo_expected_home"] = 1.0
    assert failures(frame) == {"the expected score is a score"}


def test_probabilities_that_do_not_sum_to_one_are_caught() -> None:
    frame = ratings()
    frame.loc[0, "dc_prob_draw"] = 0.4
    assert failures(frame) == {"dixon-coles probabilities sum to one"}


def test_a_probability_outside_zero_and_one_is_caught() -> None:
    frame = ratings(dc_prob_home=[1.0] * 4, dc_prob_draw=[0.0] * 4, dc_prob_away=[0.0] * 4)
    assert "dixon-coles probabilities are probabilities" in failures(frame)


def test_a_non_positive_rate_is_caught() -> None:
    """A Poisson rate of zero would mean the exponential producing it
    underflowed, which the log-likelihood would not have reported."""
    frame = ratings()
    frame.loc[0, "dc_home_lambda"] = 0.0
    assert failures(frame) == {"expected goals are positive"}


def test_a_half_filled_dixon_coles_row_is_caught() -> None:
    """Either the fit could price a match or it could not. Some columns but not
    others means the emit path has a branch that forgets one."""
    frame = ratings()
    frame.loc[0, "dc_prob_draw"] = pd.NA
    assert "a dixon-coles row is all-or-nothing" in failures(frame)


def test_an_entirely_unpriced_table_is_a_warning() -> None:
    """A column that is null everywhere satisfies every arithmetic rule above,
    so a refit that quietly stopped producing fits needs its own check."""
    frame = ratings()
    for column in (
        "dc_home_lambda",
        "dc_away_lambda",
        "dc_prob_home",
        "dc_prob_draw",
        "dc_prob_away",
    ):
        frame[column] = pd.NA
    report = run_checks(frame, ratings_checks())
    coverage = next(r for r in report.results if r.name == "dixon-coles prices most matches")
    assert coverage.outcome is Outcome.FAILED
    assert coverage.severity is Severity.WARNING
    assert report.ok


def test_coverage_is_not_checked_when_dixon_coles_was_not_run() -> None:
    """`--model elo` leaves those columns null on purpose, and a warning there
    is noise rather than a finding — which is how a report earns being read."""
    frame = ratings()
    for column in (
        "dc_home_lambda",
        "dc_away_lambda",
        "dc_prob_home",
        "dc_prob_draw",
        "dc_prob_away",
    ):
        frame[column] = pd.NA
    assert failures(frame, expect_dixon_coles=False) == set()


def test_an_empty_table_passes() -> None:
    assert failures(ratings(rows=0)) == set()


def test_ratings_must_cover_the_matches_exactly() -> None:
    """A rating for a match that does not exist, or a match with no rating,
    means the two tables drifted apart and every join silently drops rows."""
    frame = ratings()
    assert failures(frame, match_ids=["m0", "m1", "m2", "m3"]) == set()
    assert failures(frame, match_ids=["m0", "m1"]) == {
        "every rated match exists, and every match is rated"
    }
    assert failures(frame, match_ids=[f"m{i}" for i in range(6)]) == {
        "every rated match exists, and every match is rated"
    }


def test_duplicate_ratings_are_caught() -> None:
    frame = pd.concat([ratings(), ratings().head(1)], ignore_index=True).astype(RATINGS_SCHEMA)
    assert "match ids are unique" in failures(frame, match_ids=[f"m{i}" for i in range(4)])
