"""Temporal integrity: proving a derived column could not see its own future.

Every leak this project can suffer has the same shape — a value attached to
match *n* that was computed with knowledge of match *n* or later. Reading the
code for it does not scale: a rolling mean with an off-by-one window, a
normalisation over the whole table, a rating updated before it is read, all
look correct and all leak. So the property is *tested* instead, and the two
tests here need to know nothing about how the value was produced.

**Prefix invariance.** Truncate the input after match *n* and recompute. Every
surviving row must be byte-identical. Anything that consulted a later match, or
any statistic over the whole table, moves — a mean over all 303,517 rows is a
different number when there are only 50,000 of them.

**Outcome independence.** Rewrite one match's scoreline and recompute. Rows up
to and including that match must not move; later rows should. This is the more
specific of the two and catches what prefix invariance cannot: a computation
that reads match *n*'s own result while emitting match *n*'s features. Both are
needed, because each is blind to what the other finds.

The pair is generic over ``Callable[[DataFrame], DataFrame]`` rather than tied
to a rating model: it is written for Milestone 4's ratings and is the same test
Milestone 6 runs over feature builders.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

Compute = Callable[[pd.DataFrame], pd.DataFrame]
"""Anything that turns a slice of the canonical table into one row per match."""

DEFAULT_KEY = "match_id"


@dataclass(frozen=True, slots=True)
class TemporalResult:
    """What a temporal-integrity probe found."""

    name: str
    probe: str
    checks: int
    """How many truncations or rewrites were tried."""

    violations: tuple[str, ...]
    """One entry per failing check, naming the cutoff and the columns that moved."""

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        if self.ok:
            return f"{self.name}: {self.probe} holds over {self.checks} check(s)"
        return (
            f"{self.name}: {self.probe} FAILED "
            f"{len(self.violations)}/{self.checks} — {self.violations[0]}"
        )


def _moved_columns(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    """Columns whose values differ, treating null as equal to null.

    Exact comparison, not approximate. Both sides ran the same arithmetic in
    the same order over the same rows, so any difference at all is a real
    dependency on the rows that were removed — a tolerance here would hide
    exactly the small leak that is hardest to find by reading.
    """
    return [column for column in before.columns if not before[column].equals(after[column])]


def _default_cutoffs(total: int) -> tuple[int, ...]:
    """A spread of truncation points, always including a very early one.

    Early cutoffs are where warm-up logic lives and where an off-by-one is most
    likely; late ones are where an accumulated statistic would show up.
    """
    if total < 4:
        return (total,)
    return tuple(sorted({max(2, total // 20), total // 4, total // 2, (3 * total) // 4}))


def prefix_invariance(
    compute: Compute,
    matches: pd.DataFrame,
    *,
    name: str,
    cutoffs: Sequence[int] | None = None,
    key: str = DEFAULT_KEY,
) -> TemporalResult:
    """Check that truncating the input leaves the surviving rows unchanged.

    Args:
        compute: The derivation under test.
        matches: The canonical table, sorted by date.
        name: What is being tested, for the report.
        cutoffs: Row counts to truncate at. Defaults to a spread.
        key: The column joining the output back to the input.
    """
    full = compute(matches).set_index(key)
    points = tuple(cutoffs) if cutoffs is not None else _default_cutoffs(len(matches))

    violations: list[str] = []
    for cutoff in points:
        prefix = matches.iloc[:cutoff]
        truncated = compute(prefix).set_index(key)
        expected = full.loc[truncated.index]
        moved = _moved_columns(expected, truncated)
        if moved:
            violations.append(f"cutoff={cutoff}: {moved} differ from the full-history run")

    return TemporalResult(
        name=name, probe="prefix invariance", checks=len(points), violations=tuple(violations)
    )


def outcome_independence(
    compute: Compute,
    matches: pd.DataFrame,
    *,
    name: str,
    positions: Sequence[int] | None = None,
    key: str = DEFAULT_KEY,
) -> TemporalResult:
    """Check that rewriting one match's score leaves that match's own row alone.

    The rewrite is deliberately extreme — a 5-0 in whichever direction the
    match did *not* go — so a derivation that reads its own result cannot
    coincidentally produce the same number.

    Args:
        compute: The derivation under test.
        matches: The canonical table, sorted by date.
        name: What is being tested, for the report.
        positions: Row positions to rewrite. Defaults to a spread.
        key: The column joining the output back to the input.
    """
    if len(matches) < 2:
        return TemporalResult(name=name, probe="outcome independence", checks=0, violations=())

    baseline = compute(matches).set_index(key)
    points = (
        tuple(positions)
        if positions is not None
        else tuple(sorted({1, len(matches) // 3, (2 * len(matches)) // 3}))
    )

    violations: list[str] = []
    for position in points:
        # Read through a column comparison and write through `.loc`, rather
        # than reaching for individual cells: `iat` is typed as a union of
        # every scalar pandas can hold, so cell-level arithmetic does not
        # type-check even when both cells are plainly integers.
        was_home_win = bool((matches["home_goals"] > matches["away_goals"]).iloc[position])
        rewritten = matches.reset_index(drop=True)
        rewritten.loc[position, "home_goals"] = 0 if was_home_win else 5
        rewritten.loc[position, "away_goals"] = 5 if was_home_win else 0
        rewritten.loc[position, "result"] = "A" if was_home_win else "H"

        after = compute(rewritten).set_index(key)
        upto = baseline.index[: position + 1]
        moved = _moved_columns(baseline.loc[upto], after.loc[upto])
        if moved:
            violations.append(
                f"position={position}: {moved} changed for matches at or before the rewritten one"
            )

    return TemporalResult(
        name=name,
        probe="outcome independence",
        checks=len(points),
        violations=tuple(violations),
    )
