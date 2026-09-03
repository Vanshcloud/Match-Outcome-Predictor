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

**Split boundary.** The two above test a derivation. A split is the other place
the same leak lives: no training row may be dated at or after any evaluation
row, and no match may appear in both halves.

**Observed reads.** Rewrite one input column and see which outputs move. That
is the measured version of a ``reads`` declaration, and the reason to measure
it is that the declaration is the one part of the feature registry that can be
wrong silently — a typo there reclassifies a leaking feature as safe.

All four are generic over ``Callable[[DataFrame], DataFrame]`` rather than tied
to a rating model: the first two were written for Milestone 4's ratings and are
the same tests Milestone 6 runs over every registered producer.
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


def split_boundary(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    name: str,
    date_column: str = "date",
    key: str = DEFAULT_KEY,
) -> TemporalResult:
    """Check that a split's training half ends before its evaluation half starts.

    The two probes above test a *derivation*. This one tests a *split*, which
    is the other place the same leak lives and the place it is hardest to see:
    a fold assembled with a random shuffle, or with a boundary computed off a
    sorted-by-something-else frame, produces a model that has already seen the
    matches it is scored on. Nothing in the output looks wrong — the metrics
    just come out better than they should.

    Ties fail. Two matches on the same date are the same round, and a model
    trained on the 3pm kick-offs is not entitled to predict the 5.30 one; the
    same reasoning that makes every window here cut strictly on the date.

    Args:
        train: The training rows.
        evaluate: The rows the model will be scored on.
        name: What is being tested, for the report.
        date_column: The column carrying kick-off dates.
        key: The column identifying a match, checked for membership in both.
    """
    violations: list[str] = []

    if not train.empty and not evaluate.empty:
        last_train = train[date_column].max()
        first_evaluation = evaluate[date_column].min()
        if last_train >= first_evaluation:
            violations.append(
                f"training runs to {last_train:%Y-%m-%d} but evaluation starts "
                f"{first_evaluation:%Y-%m-%d}"
            )

    shared = set(train[key]) & set(evaluate[key])
    if shared:
        violations.append(f"{len(shared)} match(es) are in both halves, e.g. {sorted(shared)[0]}")

    return TemporalResult(name=name, probe="split boundary", checks=2, violations=tuple(violations))


def _perturbations(matches: pd.DataFrame, column: str) -> tuple[pd.DataFrame, ...]:
    """Two rewrites of one column, each of which changes something different.

    *Flattened* gives every row the same value, which destroys the ordering and
    the gaps between values. *Altered* moves every value while keeping them
    distinct, which is the one that finds a column that was already constant in
    this sample — a flatten of a constant column changes nothing, and a
    dependence probe that reported "unread" there would be reporting on the
    fixture rather than on the code.

    Dates are altered by a cumulative spread rather than a constant offset:
    shifting every kick-off by a day leaves every gap identical, so a feature
    reading rest days would not move. The spread keeps the frame sorted, which
    every producer here requires.
    """
    values = matches[column]
    present = values.dropna()
    if present.empty:
        return ()

    flattened = values.copy()
    flattened[:] = present.iloc[0]

    if pd.api.types.is_datetime64_any_dtype(values):
        altered = values + pd.to_timedelta(range(len(values)), unit="D")
    elif pd.api.types.is_numeric_dtype(values):
        altered = values + 1
    else:
        altered = values + "-x"

    return tuple(matches.assign(**{column: rewritten}) for rewritten in (flattened, altered))


def observed_reads(
    compute: Compute,
    matches: pd.DataFrame,
    columns: Sequence[str],
    *,
    key: str = DEFAULT_KEY,
) -> dict[str, frozenset[str]]:
    """Which input columns each output column actually depends on.

    Rewrite one input column, recompute, and see what moved. What moved read
    it. This is the measured counterpart to a hand-written ``reads``
    declaration, which is worth having because the declaration is the one part
    of the feature registry that can be wrong without anything noticing: a typo
    there reclassifies a leaking feature as safe.

    The answer is a **lower bound**. A column this sample never varies in, or
    one a producer reads only in a branch this sample never takes, will not
    show up. So the property to assert is that a declaration *covers* what was
    observed, never that the two are equal.

    Args:
        compute: The derivation under test.
        matches: The canonical table, sorted by date.
        columns: Input columns to try. ``key`` is skipped if present: it joins
            the output back to the input rather than feeding it.
        key: The column joining the output back to the input.

    Returns:
        One entry per output column, naming the inputs that moved it.
    """
    baseline = compute(matches).set_index(key)
    reads: dict[str, set[str]] = {column: set() for column in baseline.columns}

    for column in columns:
        if column == key:
            continue
        for perturbed in _perturbations(matches, column):
            after = compute(perturbed).set_index(key).reindex(baseline.index)
            for moved in _moved_columns(baseline, after):
                reads[moved].add(column)

    return {column: frozenset(inputs) for column, inputs in reads.items()}
