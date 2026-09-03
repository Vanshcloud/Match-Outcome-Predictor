"""Checks on the ratings table.

Separate from :mod:`src.validation.matches` because they answer a different
question. That suite asks whether the *data* is football; this one asks whether
the numbers derived from it are arithmetically possible — three probabilities
that sum to one, an expected score inside the unit interval, a Poisson rate
above zero. None of these can be wrong because of the provider. They can only
be wrong because of us, which is exactly why they are worth checking on every
run rather than trusting a unit test to have covered the case.

The coverage checks are the other half. A rating column that is null everywhere
still satisfies every arithmetic rule, and a refit that silently stopped
producing fits would pass all of them.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS, KEY_COLUMN
from src.validation.report import Check, Severity, check

_PROBABILITIES = ("dc_prob_home", "dc_prob_draw", "dc_prob_away")

MIN_DIXON_COLES_COVERAGE = 0.5
"""Share of matches Dixon-Coles is expected to price.

It cannot price a competition's opening seasons, nor a promoted club's first
matches of a window, and both are real. Measured over the full ingest: 0.921.
The floor sits well below that because a partial run — one competition, or a
short date range — legitimately has a much colder start, and a check that
fires on a legitimate `--competition ENG_1` is a check people learn to
ignore."""


@check("elo is always available")
def _elo_is_complete(ratings: pd.DataFrame) -> str | None:
    """Elo has a prior for a team it has never seen, so unlike Dixon-Coles it
    is never unable to answer. A null here means the walk skipped a row."""
    nulls = {
        column: int(ratings[column].isna().sum())
        for column in ELO_COLUMNS
        if ratings[column].isna().any()
    }
    return f"nulls in elo columns: {nulls}" if nulls else None


@check("the expected score is a score")
def _expected_score_is_bounded(ratings: pd.DataFrame) -> str | None:
    known = ratings["elo_expected_home"].dropna()
    if known.empty:
        return None
    if known.min() <= 0.0 or known.max() >= 1.0:
        return f"elo_expected_home ranges {known.min():.4f}..{known.max():.4f}"
    return None


@check("dixon-coles probabilities sum to one")
def _probabilities_are_a_distribution(ratings: pd.DataFrame) -> str | None:
    priced = ratings.dropna(subset=list(_PROBABILITIES))
    if priced.empty:
        return None
    total = priced[list(_PROBABILITIES)].astype("float64").sum(axis=1)
    worst = float((total - 1.0).abs().max())
    # 1e-9 rather than exact: the grid is renormalised in floating point, so
    # the sum is one to within rounding and nothing tighter is achievable.
    return None if worst < 1e-9 else f"probabilities miss one by up to {worst:.2e}"


@check("dixon-coles probabilities are probabilities")
def _probabilities_are_bounded(ratings: pd.DataFrame) -> str | None:
    priced = ratings.dropna(subset=list(_PROBABILITIES))
    if priced.empty:
        return None
    outside = {
        column: int(((priced[column] <= 0.0) | (priced[column] >= 1.0)).sum())
        for column in _PROBABILITIES
        if ((priced[column] <= 0.0) | (priced[column] >= 1.0)).any()
    }
    return f"probabilities outside (0, 1): {outside}" if outside else None


@check("expected goals are positive")
def _rates_are_positive(ratings: pd.DataFrame) -> str | None:
    """A Poisson rate of zero or below is not a rate. It would mean the
    exponential that produces it underflowed, which the log-likelihood would
    not have reported."""
    for column in ("dc_home_lambda", "dc_away_lambda"):
        known = ratings[column].dropna()
        if not known.empty and known.min() <= 0.0:
            return f"{column} minimum is {known.min()}"
    return None


@check("a dixon-coles row is all-or-nothing")
def _dixon_coles_is_complete_per_row(ratings: pd.DataFrame) -> str | None:
    """Either the fit could price a match or it could not. A row with rates but
    no probabilities, or the reverse, means the emit path has a branch that
    fills some columns and forgets others."""
    present = ratings[list(DIXON_COLES_COLUMNS)].notna()
    partial = int((present.any(axis=1) & ~present.all(axis=1)).sum())
    return f"{partial:,} rows have some dixon-coles columns but not all" if partial else None


@check("dixon-coles prices most matches", severity=Severity.WARNING)
def _dixon_coles_coverage_holds(ratings: pd.DataFrame) -> str | None:
    """A column that is null everywhere satisfies every arithmetic rule above,
    so a refit that quietly stopped producing fits needs its own check."""
    if ratings.empty:
        return None
    coverage = float(ratings["dc_prob_home"].notna().mean())
    if coverage >= MIN_DIXON_COLES_COVERAGE:
        return None
    return f"dixon-coles priced {coverage:.1%} of matches, below {MIN_DIXON_COLES_COVERAGE:.0%}"


_ARITHMETIC_CHECKS: tuple[Check, ...] = (
    _elo_is_complete,
    _expected_score_is_bounded,
    _probabilities_are_a_distribution,
    _probabilities_are_bounded,
    _rates_are_positive,
    _dixon_coles_is_complete_per_row,
)


def ratings_checks(
    match_ids: Iterable[str] | None = None,
    *,
    expect_dixon_coles: bool = True,
) -> tuple[Check, ...]:
    """The suite.

    Args:
        match_ids: The canonical table's keys. When given, the ratings must
            cover them exactly — a rating for a match that does not exist, or a
            match with no rating, means the two tables have drifted apart and
            every join downstream silently drops rows.
        expect_dixon_coles: Whether a Dixon-Coles model was part of the build.
            A run of ``--model elo`` leaves those columns null on purpose, and a
            coverage warning there is noise rather than a finding — which is
            how a report earns the right to be read.
    """
    suite = _ARITHMETIC_CHECKS
    if expect_dixon_coles:
        suite = (*suite, _dixon_coles_coverage_holds)
    if match_ids is None:
        return suite

    expected = set(match_ids)

    @check("every rated match exists, and every match is rated")
    def _covers_exactly(ratings: pd.DataFrame) -> str | None:
        actual = set(ratings[KEY_COLUMN])
        missing = len(expected - actual)
        extra = len(actual - expected)
        if not missing and not extra:
            return None
        return f"{missing:,} matches unrated, {extra:,} ratings without a match"

    @check("match ids are unique")
    def _ids_are_unique(ratings: pd.DataFrame) -> str | None:
        duplicated = int(ratings[KEY_COLUMN].duplicated().sum())
        return f"{duplicated:,} duplicate match ids" if duplicated else None

    return (_covers_exactly, _ids_are_unique, *suite)
