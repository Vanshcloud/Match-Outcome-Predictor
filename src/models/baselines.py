"""The numbers a model has to beat before it is worth having.

Four forecasters, in ascending order of how hard they are to beat:

**Home always.** The one everybody's first model accidentally learns. It is
right about 45% of the time, which is why accuracy is the wrong target, and it
scores infinite log loss, which is why a proper scoring rule is the right one.
It is here to make both of those visible rather than argued.

**Class prior.** The base rates of home, draw and away, counted on the training
fold and applied to every match in the evaluation fold. This is the honest
floor: a model that cannot beat it has learned nothing about *which* match it
is looking at. Counted per fold rather than taken from a constant, because a
prior read off the whole table is a prior that has seen the future.

**Dixon-Coles.** The goal-rate rating, already three-class and already causal.
The model zoo's real opponent — it is a fitted model, and beating it is the
first evidence that features add something ratings do not.

**Bookmaker.** The closing line with the overround removed. The strongest
public forecast there is, available for a subset of matches, and the reason the
report is built around subsets: comparing a model scored on 12,000 matches with
a line quoted on 8,800 is arithmetic this project exists not to do.

A forecaster returns one row per evaluation match, and **NaN for a match it
cannot price**. Nothing here invents a number to fill a gap; the scoring layer
takes coverage seriously instead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.evaluation.market import MARKET_COLUMNS as ODDS_COLUMNS
from src.evaluation.market import implied_probabilities
from src.evaluation.metrics import CLASSES
from src.ingestion.base import TARGET_COLUMN

DIXON_COLES_PROBABILITIES: tuple[str, str, str] = ("dc_prob_home", "dc_prob_draw", "dc_prob_away")
"""The rating's three-class output, in the same order."""


@runtime_checkable
class Forecaster(Protocol):
    """Turns a training fold and an evaluation fold into probabilities.

    A Protocol, matching every other seam here. The four baselines share a
    shape and no implementation: one counts, one reads a column, one inverts a
    price, and one is a constant.
    """

    @property
    def name(self) -> str:
        """Short identifier, used in the report."""
        ...

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        """Return an ``(n, 3)`` array for ``evaluate``, NaN where it cannot price.

        Args:
            train: Matches strictly earlier than every row of ``evaluate``.
                The split layer guarantees that; a forecaster is entitled to
                use all of it and nothing else.
            evaluate: The matches to price.
        """
        ...


def _empty(rows: int) -> np.ndarray:
    return np.full((rows, len(CLASSES)), np.nan)


class HomeAlways:
    """Certainty that the home side wins. The forecast nobody should make."""

    name = "home_always"

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        del train
        forecast = _empty(len(evaluate))
        forecast[:] = (1.0, 0.0, 0.0)
        return forecast


class ClassPrior:
    """The training fold's own base rates, applied to every match in the next one.

    Counted per fold. A prior taken from the whole table would be a small leak
    of exactly the kind the split layer exists to prevent — and a measurable
    one: the home-win rate has fallen over the thirty years in this dataset.
    """

    name = "class_prior"

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        results = train[TARGET_COLUMN].dropna()
        if results.empty:
            return _empty(len(evaluate))
        counts = np.array([float((results == label).sum()) for label in CLASSES])
        forecast = _empty(len(evaluate))
        forecast[:] = counts / counts.sum()
        return forecast


class DixonColes:
    """The Dixon-Coles rating, read from the ratings table.

    The rating was fitted over the whole history, which sounds like a leak and
    is not: every row of it depends only on matches strictly earlier than its
    own, and that is the property `src/validation/temporal.py` proves on every
    build. A rating recomputed per fold would produce the same numbers at ten
    times the cost.
    """

    name = "dixon_coles"

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        del train
        missing = [column for column in DIXON_COLES_PROBABILITIES if column not in evaluate]
        if missing:
            return _empty(len(evaluate))
        priced = evaluate[list(DIXON_COLES_PROBABILITIES)].to_numpy(dtype=float)
        # A partially filled row would be scored as a valid forecast that
        # happens to sum to less than one, which is worse than not scoring it.
        return np.where(np.isnan(priced).any(axis=1, keepdims=True), np.nan, priced)


class Bookmaker:
    """The closing line, with the overround removed.

    The arithmetic moved to :func:`~src.evaluation.market.implied_probabilities`
    and is not repeated here. Three callers now need "what did
    the market say" — this benchmark, the disagreement table, and the
    dashboard's match page — and three implementations of removing a margin is
    three chances to publish a number that is not the one this benchmark was
    scored against.
    """

    name = "bookmaker"

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        del train
        missing = [column for column in ODDS_COLUMNS if column not in evaluate]
        if missing:
            return _empty(len(evaluate))
        return implied_probabilities(evaluate[list(ODDS_COLUMNS)].to_numpy(dtype=float))


def default_forecasters(evaluate: pd.DataFrame | None = None) -> tuple[Forecaster, ...]:
    """Every baseline, or the ones a given table can actually support.

    Called with a frame, it drops the forecasters whose input columns are
    absent. A baseline reporting 0% coverage because the ratings were never
    built looks exactly like a baseline that failed, and the two deserve
    different words.
    """
    everything: tuple[Forecaster, ...] = (HomeAlways(), ClassPrior(), DixonColes(), Bookmaker())
    if evaluate is None:
        return everything

    available = set(evaluate.columns)
    required: dict[str, tuple[str, ...]] = {
        DixonColes.name: DIXON_COLES_PROBABILITIES,
        Bookmaker.name: ODDS_COLUMNS,
    }
    return tuple(
        forecaster
        for forecaster in everything
        if set(required.get(forecaster.name, ())) <= available
    )
