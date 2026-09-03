"""Checks on the feature table.

The arithmetic half is the same idea as :mod:`src.validation.ratings`: points
per game cannot exceed three, a count cannot be negative, rest cannot be
measured in fractions of a day. None of these can be wrong because of the
provider.

The half that is specific to features is the last one, and it is a **leakage
check that works on the table rather than on the code**. Every window feature
has a companion count saying how much history it had, and a value where the
count is zero is a value that came from somewhere it should not have. The
temporal probes are the real guarantee — they recompute the builder over
truncated inputs — but they run on a sample competition, and this runs over
every row that was actually written.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.feature_engineering.registry import FEATURE_COLUMNS, FEATURE_SCHEMA, KEY_COLUMN
from src.validation.report import Check, Severity, check

MAX_POINTS_PER_GAME = 3.0
MAX_PLAUSIBLE_GOAL_RATE = 10.0
"""Goals per game over a five-match window. The record is nowhere near this;
anything above means a window summed instead of averaging."""

MAX_MATCHES_IN_A_FORTNIGHT = 14
"""One a day would be absurd, and it is still the bound: a rule that fires on
a genuinely congested schedule would fire on Christmas."""

MIN_FORM_COVERAGE = 0.9
"""Share of matches with a five-match form figure. Every team's first match has
none, and so does the first match of every club the provider ever introduces —
measured over the full ingest: 0.99."""

# Feature, and the count that says how much history it had. A value where the
# count is zero is a value that was invented.
_HISTORY_GATED: tuple[tuple[str, str], ...] = (
    ("home_form_points_5", "home_matches_played"),
    ("away_form_points_5", "away_matches_played"),
    ("home_goals_for_5", "home_matches_played"),
    ("away_goals_for_5", "away_matches_played"),
    ("home_goals_against_5", "home_matches_played"),
    ("away_goals_against_5", "away_matches_played"),
    ("home_venue_points_5", "home_matches_played"),
    ("away_venue_points_5", "away_matches_played"),
    ("home_rest_days", "home_matches_played"),
    ("away_rest_days", "away_matches_played"),
    ("h2h_home_points", "h2h_matches"),
)

_COUNTS: tuple[str, ...] = (
    "home_matches_played",
    "away_matches_played",
    "home_matches_14d",
    "away_matches_14d",
    "h2h_matches",
)

_POINT_RATES: tuple[str, ...] = (
    "home_form_points_5",
    "away_form_points_5",
    "home_venue_points_5",
    "away_venue_points_5",
    "h2h_home_points",
)

_GOAL_RATES: tuple[str, ...] = (
    "home_goals_for_5",
    "away_goals_for_5",
    "home_goals_against_5",
    "away_goals_against_5",
)


@check("the table is exactly the registered features")
def _schema_matches_the_registry(features: pd.DataFrame) -> str | None:
    """The registry is the single source of truth for this table's shape. A
    column built but not registered is a column nothing documents; one
    registered but not built is a promise to a consumer that nothing keeps.

    Checked against the whole registry regardless of which builders ran: a
    partial build widens the same schema and leaves the rest null, so the
    columns are always all there.
    """
    actual = {str(column): str(dtype) for column, dtype in features.dtypes.items()}
    missing = sorted(set(FEATURE_SCHEMA) - set(actual))
    extra = sorted(set(actual) - set(FEATURE_SCHEMA))
    if missing or extra:
        return f"missing={missing} unregistered={extra}"
    wrong = {
        column: f"{actual[column]} != {expected}"
        for column, expected in FEATURE_SCHEMA.items()
        if actual[column] != expected
    }
    return f"wrong dtypes: {wrong}" if wrong else None


def _present(features: pd.DataFrame, built: frozenset[str], columns: Iterable[str]) -> list[str]:
    """The columns among ``columns`` that this build actually produced.

    A run of ``--builder head_to_head`` leaves every form column null on
    purpose, and a check that called those nulls a defect would fire on a
    legitimate command — which is how a report earns the right to be ignored.
    """
    return [column for column in columns if column in built and column in features.columns]


def _arithmetic_checks(built: frozenset[str]) -> tuple[Check, ...]:
    """The checks that only apply to columns this build claims to have filled."""

    @check("no feature has a value before there is history")
    def _values_require_history(features: pd.DataFrame) -> str | None:
        """The leakage check that works on the table.

        Every window feature has a companion count. A form figure where the
        count is zero did not come from the team's previous matches, because
        there were none — it came from somewhere else, and somewhere else is
        the future.
        """
        offending: dict[str, int] = {}
        for value, counter in _HISTORY_GATED:
            if not _present(features, built, (value,)) or counter not in features.columns:
                continue
            invented = int((features[value].notna() & (features[counter] == 0)).sum())
            if invented:
                offending[value] = invented
        return f"values with no history behind them: {offending}" if offending else None

    @check("counts are never null and never negative")
    def _counts_are_sane(features: pd.DataFrame) -> str | None:
        """A count of prior matches is always knowable — it is zero at worst —
        so a null means the walk skipped a row rather than that the answer was
        unknown."""
        problems: dict[str, str] = {}
        for column in _present(features, built, _COUNTS):
            if features[column].isna().any():
                problems[column] = f"{int(features[column].isna().sum())} null"
            elif len(features) and int(features[column].min()) < 0:
                problems[column] = f"minimum {features[column].min()}"
        return f"bad counts: {problems}" if problems else None

    @check("points per game never exceed three")
    def _point_rates_are_bounded(features: pd.DataFrame) -> str | None:
        for column in _present(features, built, _POINT_RATES):
            known = features[column].dropna()
            if not known.empty and (
                float(known.min()) < 0.0 or float(known.max()) > MAX_POINTS_PER_GAME
            ):
                return f"{column} ranges {known.min():.2f}..{known.max():.2f}"
        return None

    @check("goal rates are plausible")
    def _goal_rates_are_bounded(features: pd.DataFrame) -> str | None:
        """Above this a window summed instead of averaging — an error that
        leaves every value positive and finite, so nothing else would notice."""
        for column in _present(features, built, _GOAL_RATES):
            known = features[column].dropna()
            if not known.empty and (
                float(known.min()) < 0.0 or float(known.max()) > MAX_PLAUSIBLE_GOAL_RATE
            ):
                return f"{column} ranges {known.min():.2f}..{known.max():.2f}"
        return None

    @check("rest is measured in whole days, and at least one")
    def _rest_is_plausible(features: pd.DataFrame) -> str | None:
        """Zero would mean a team informed its own fixture: the window stops at
        the last strictly *earlier* date, so two matches on one day both report
        the gap to the day before, and neither reports zero."""
        for column in _present(features, built, ("home_rest_days", "away_rest_days")):
            known = features[column].dropna()
            if not known.empty and int(known.min()) < 1:
                return f"{column} minimum is {known.min()}"
        return None

    @check("fortnight congestion is possible")
    def _congestion_is_bounded(features: pd.DataFrame) -> str | None:
        for column in _present(features, built, ("home_matches_14d", "away_matches_14d")):
            known = features[column].dropna()
            if not known.empty and int(known.max()) > MAX_MATCHES_IN_A_FORTNIGHT:
                return f"{column} maximum is {known.max()}"
        return None

    @check("most matches have a form figure", severity=Severity.WARNING)
    def _form_coverage_holds(features: pd.DataFrame) -> str | None:
        """A column that is null everywhere satisfies every rule above."""
        columns = _present(features, built, ("home_form_points_5",))
        if features.empty or not columns:
            return None
        coverage = float(features[columns[0]].notna().mean())
        if coverage >= MIN_FORM_COVERAGE:
            return None
        return f"form coverage {coverage:.1%}, below {MIN_FORM_COVERAGE:.0%}"

    return (
        _schema_matches_the_registry,
        _values_require_history,
        _counts_are_sane,
        _point_rates_are_bounded,
        _goal_rates_are_bounded,
        _rest_is_plausible,
        _congestion_is_bounded,
        _form_coverage_holds,
    )


def feature_checks(
    match_ids: Iterable[str] | None = None,
    *,
    built: Iterable[str] | None = None,
) -> tuple[Check, ...]:
    """The suite.

    Args:
        match_ids: The canonical table's keys. When given, the features must
            cover them exactly — a feature row for a match that does not exist,
            or a match with no features, means the two tables have drifted apart
            and every join downstream silently drops rows.
        built: The features this build claims to have produced. Defaults to the
            whole registry. A partial run leaves the rest null on purpose, and
            a suite that called those nulls defects would fire on a legitimate
            command.
    """
    produced = frozenset(built) if built is not None else frozenset(FEATURE_COLUMNS)
    suite = _arithmetic_checks(produced)
    if match_ids is None:
        return suite

    expected = set(match_ids)

    @check("every featured match exists, and every match is featured")
    def _covers_exactly(features: pd.DataFrame) -> str | None:
        actual = set(features[KEY_COLUMN])
        missing = len(expected - actual)
        extra = len(actual - expected)
        if not missing and not extra:
            return None
        return f"{missing:,} matches without features, {extra:,} features without a match"

    @check("match ids are unique")
    def _ids_are_unique(features: pd.DataFrame) -> str | None:
        duplicated = int(features[KEY_COLUMN].duplicated().sum())
        return f"{duplicated:,} duplicate match ids" if duplicated else None

    return (_covers_exactly, _ids_are_unique, *suite)
