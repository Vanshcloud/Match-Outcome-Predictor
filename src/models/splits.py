"""Walk-forward folds, cut on the date and gated by the boundary probe.

A random split of football matches is not a split, it is a leak with a
confidence interval. Every fold here trains on matches strictly *earlier* than
the ones it is scored on, which is the only arrangement that answers the
question the project is actually asking: given what was known on Friday, what
happens on Saturday.

**The cut is a date, never a row.** A boundary at row *n* puts two matches
played on the same afternoon on opposite sides of it, ordered only by whatever
tiebreak the sort used — so the model trains on the 3pm results and is scored
on the 5.30 kick-off. Boundaries here are timestamps and a day is indivisible,
the same rule the feature windows follow.

**Every fold is probed, not trusted.** :func:`~src.validation.temporal.split_boundary`
runs on each one before it is yielded, and a violation raises, so a ``<``
changed to ``<=`` below fails loudly instead of leaking quietly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger
from src.validation.temporal import split_boundary

logger = get_logger(__name__)

DEFAULT_FOLDS = 5
DEFAULT_HORIZON_DAYS = 365
"""One year of matches per fold.

A season, in the only sense that generalises: the 39 competitions here start in
August, in January and in April, so a fold boundary drawn on "the season" would
be drawn in a different place for each of them.
"""

MIN_TRAIN_MATCHES = 380
"""A fold trains on at least this many matches or it is skipped.

One English season is 380 fixtures. Below that, a class prior is a sample
rather than a prior, and a fold that reports a number computed from forty
matches is a fold that makes the pooled figure worse for no gain.
"""


class SplitError(RuntimeError):
    """A split cannot be produced, or is not one."""


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward step: everything before ``start``, scored on ``[start, end)``."""

    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    train: pd.DataFrame
    evaluate: pd.DataFrame

    def summary(self) -> str:
        # The last day training could possibly reach, named rather than
        # implied: "train to 2021-09-03" and "evaluate from 2021-09-03" read
        # like an overlap even when the code has none.
        last_train = self.start - pd.Timedelta(1, "D")
        return (
            f"fold {self.index}: train {len(self.train):,} to {last_train:%Y-%m-%d}, "
            f"evaluate {len(self.evaluate):,} in "
            f"[{self.start:%Y-%m-%d}, {self.end:%Y-%m-%d})"
        )


def boundaries(
    matches: pd.DataFrame,
    *,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[pd.Timestamp, ...]:
    """The ``folds + 1`` timestamps that cut the tail of the history into folds.

    Anchored at the end rather than the start: the interesting question is how
    the model does on recent football, and a backtest whose folds begin in 1993
    spends most of its evidence on a sport that was played differently.
    """
    if folds < 1:
        raise SplitError("a walk-forward backtest needs at least one fold")
    if horizon_days < 1:
        raise SplitError("a fold must span at least one day")
    if matches.empty:
        raise SplitError("no matches to split")

    # Exclusive upper bound, so the final fold contains the last match rather
    # than stopping just short of it.
    end = pd.Timestamp(matches["date"].max()) + pd.Timedelta(1, "D")
    span = pd.Timedelta(horizon_days, "D")
    return tuple(end - span * (folds - index) for index in range(folds + 1))


def walk_forward(
    matches: pd.DataFrame,
    *,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    min_train: int = MIN_TRAIN_MATCHES,
) -> Iterator[Fold]:
    """Yield expanding-window folds, oldest first.

    Expanding rather than sliding: there is no reason to throw away 1993 when
    scoring 2024, and the ratings that feed the models already handle recency
    with a decay. A sliding window would be a second, blunter opinion about the
    same thing.

    Args:
        matches: The canonical table, sorted by date.
        folds: How many steps. The tail of the history is cut into this many.
        horizon_days: How much time each step is scored over.
        min_train: Skip a fold with less training history than this.

    Yields:
        One :class:`Fold` per step that has both halves. A step with no matches
        to score, or too few to train on, is logged and skipped rather than
        yielded empty — a fold of nothing is not evidence.

    Raises:
        SplitError: If no fold survives, or if a fold fails the boundary probe.
    """
    if not matches["date"].is_monotonic_increasing:
        raise SplitError(
            "matches must be sorted by date before splitting; an unsorted input "
            "produces folds that look chronological and are not"
        )

    cuts = boundaries(matches, folds=folds, horizon_days=horizon_days)
    dates = matches["date"]
    produced = 0

    for index, (start, end) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
        train = matches[dates < start]
        evaluate = matches[(dates >= start) & (dates < end)]

        if evaluate.empty or len(train) < min_train:
            logger.info(
                "fold %d skipped: %d training and %d evaluation matches",
                index,
                len(train),
                len(evaluate),
            )
            continue

        fold = Fold(index=index, start=start, end=end, train=train, evaluate=evaluate)
        probe = split_boundary(train, evaluate, name=f"fold {index}")
        if not probe.ok:
            raise SplitError(probe.summary())

        produced += 1
        logger.info("%s", fold.summary())
        yield fold

    if produced == 0:
        raise SplitError(
            f"no fold had {min_train} training and one evaluation match; "
            f"the table spans {len(matches):,} matches"
        )
