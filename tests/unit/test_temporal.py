"""The temporal-integrity probes.

A leakage test that cannot catch a leak is worse than none, so every probe here
is checked against a *deliberately leaky* computation as well as a causal one.
The important result is that the two probes catch different things: a
derivation reading its own row is completely invisible to prefix invariance,
and that is why both exist rather than the simpler one alone.
"""

from __future__ import annotations

import pandas as pd

from src.validation.temporal import (
    _default_cutoffs,
    outcome_independence,
    prefix_invariance,
)
from tests.factories import league_frame

MATCHES = league_frame(seasons=["2020-21", "2021-22"])


def _lagged(matches: pd.DataFrame) -> pd.DataFrame:
    """Causal: each row carries the previous match's home goals."""
    return pd.DataFrame(
        {
            "match_id": matches["match_id"].to_numpy(),
            "value": matches["home_goals"].astype("float64").shift(1).fillna(0.0).to_numpy(),
        }
    )


def _global_mean(matches: pd.DataFrame) -> pd.DataFrame:
    """Leaky: a statistic over the whole table, including the future."""
    return pd.DataFrame(
        {
            "match_id": matches["match_id"].to_numpy(),
            "value": float(matches["home_goals"].mean()),
        }
    )


def _own_result(matches: pd.DataFrame) -> pd.DataFrame:
    """Leaky: each row carries its own result."""
    return pd.DataFrame(
        {
            "match_id": matches["match_id"].to_numpy(),
            "value": matches["home_goals"].astype("float64").to_numpy(),
        }
    )


# ---- the probes catch what they are for -------------------------------------


def test_prefix_invariance_passes_a_causal_derivation() -> None:
    result = prefix_invariance(_lagged, MATCHES, name="lagged")
    assert result.ok
    assert result.checks == 4


def test_prefix_invariance_catches_a_statistic_over_the_whole_table() -> None:
    """A mean over 760 rows is a different number when there are only 38, which
    is what makes truncation a test rather than a formality."""
    result = prefix_invariance(_global_mean, MATCHES, name="global mean")
    assert not result.ok
    assert "['value'] differ" in result.violations[0]


def test_outcome_independence_passes_a_causal_derivation() -> None:
    assert outcome_independence(_lagged, MATCHES, name="lagged").ok


def test_outcome_independence_catches_a_row_reading_its_own_result() -> None:
    result = outcome_independence(_own_result, MATCHES, name="own result")
    assert not result.ok
    assert "at or before the rewritten one" in result.violations[0]


def test_the_two_probes_are_blind_to_different_leaks() -> None:
    """The reason there are two.

    A derivation that reads its own row is *perfectly* prefix-invariant — every
    surviving row keeps its value under truncation, because each value only
    ever depended on its own row. Only rewriting the result finds it.
    """
    assert prefix_invariance(_own_result, MATCHES, name="own result").ok
    assert not outcome_independence(_own_result, MATCHES, name="own result").ok


def test_a_whole_table_statistic_is_caught_by_both() -> None:
    assert not prefix_invariance(_global_mean, MATCHES, name="global").ok
    assert not outcome_independence(_global_mean, MATCHES, name="global").ok


# ---- mechanics --------------------------------------------------------------


def test_cutoffs_can_be_given_explicitly() -> None:
    result = prefix_invariance(_lagged, MATCHES, name="lagged", cutoffs=[10, 20])
    assert result.checks == 2


def test_positions_can_be_given_explicitly() -> None:
    result = outcome_independence(_own_result, MATCHES, name="own", positions=[5])
    assert result.checks == 1
    assert "position=5" in result.violations[0]


def test_the_default_cutoffs_include_an_early_one() -> None:
    """Warm-up logic lives in the first rows, and an off-by-one there is
    invisible at a cutoff of half the table."""
    cutoffs = _default_cutoffs(1_000)
    assert min(cutoffs) < 100
    assert max(cutoffs) < 1_000


def test_a_tiny_table_still_produces_a_cutoff() -> None:
    assert _default_cutoffs(3) == (3,)
    assert prefix_invariance(_lagged, MATCHES.head(3), name="lagged").checks == 1


def test_outcome_independence_needs_two_rows_to_say_anything() -> None:
    result = outcome_independence(_lagged, MATCHES.head(1), name="lagged")
    assert result.checks == 0
    assert result.ok


def test_the_rewrite_flips_whichever_way_the_match_did_not_go() -> None:
    """A 5-0 written over a match that was already 5-0 would prove nothing."""
    seen: list[tuple[int, int]] = []

    def record(matches: pd.DataFrame) -> pd.DataFrame:
        seen.append((int(matches.iloc[1]["home_goals"]), int(matches.iloc[1]["away_goals"])))
        return _lagged(matches)

    outcome_independence(record, MATCHES, name="record", positions=[1])
    original, rewritten = seen[0], seen[1]
    assert rewritten in {(0, 5), (5, 0)}
    assert (original[0] > original[1]) != (rewritten[0] > rewritten[1])


def test_the_summary_names_the_first_violation() -> None:
    passing = prefix_invariance(_lagged, MATCHES, name="lagged")
    assert passing.summary() == "lagged: prefix invariance holds over 4 check(s)"
    failing = prefix_invariance(_global_mean, MATCHES, name="global")
    assert failing.summary().startswith("global: prefix invariance FAILED")


def test_null_compares_equal_to_null() -> None:
    """Otherwise a derivation that legitimately emits nulls during warm-up
    would fail every probe for having done so consistently."""

    def nullable(matches: pd.DataFrame) -> pd.DataFrame:
        values = pd.array([None] * len(matches), dtype="Float64")
        return pd.DataFrame({"match_id": matches["match_id"].to_numpy(), "value": values})

    assert prefix_invariance(nullable, MATCHES, name="nullable").ok
    assert outcome_independence(nullable, MATCHES, name="nullable").ok
