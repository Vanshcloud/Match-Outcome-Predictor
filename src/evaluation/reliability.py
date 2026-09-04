"""Whether a stated probability happens as often as it says it will.

Log loss answers "how good is this forecast" with one number, and that number
cannot be taken apart by eye into the two different ways a forecast is bad. A
model can be sharp and overconfident, or timid and perfectly honest, and land
on the same loss. Reliability separates them: bin every probability the model
states, and put the mean of each bin beside how often the thing actually
happened. A forecast that says 30% and is right 30% of the time is reliable at
that level whether or not it is any use.

This is a diagnostic, not a score. Nothing here ranks two models — the pooled
table in :mod:`src.pipelines.backtest` does that — and a model can be perfectly
reliable and worthless (the class prior is), which is why the two are reported
beside each other rather than one instead of the other.

**Every probability, not just the confident one.** A three-class forecast makes
three statements about each match, and this bins all ``3n`` of them against the
one-hot outcome. The usual alternative bins only the class the model likes
best, which measures a classifier's confidence and says nothing at all about
the draw column — where a football model is most often wrong, and where the
overconfidence a calibration layer exists to remove actually lives.

**Equal-width bins, and empty ones are dropped.** Ten bins of 0.1 rather than
deciles of the forecast: for a before-and-after table to be readable the edges
have to mean the same thing in both halves, and quantile edges move with the
forecast being measured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.evaluation.metrics import MetricError, _as_probabilities, one_hot

if TYPE_CHECKING:
    from src.evaluation.metrics import Outcomes

DEFAULT_BINS = 10

RELIABILITY_COLUMNS: tuple[str, ...] = (
    "bin",
    "lower",
    "upper",
    "n",
    "predicted",
    "observed",
    "gap",
)
"""The bin's index and edges, how many statements fell in it, their mean, how
often they happened, and the difference — negative where the forecast promised
more than it delivered."""


def reliability(
    probabilities: np.ndarray, outcomes: Outcomes, *, bins: int = DEFAULT_BINS
) -> pd.DataFrame:
    """One row per non-empty bin of stated probability.

    Args:
        probabilities: An ``(n, 3)`` array in
            :data:`~src.evaluation.metrics.CLASSES` order.
        outcomes: What happened, as canonical ``result`` strings.
        bins: How many equal-width bins to cut ``[0, 1]`` into.

    Raises:
        MetricError: On an unusable forecast, or on fewer than one bin.
    """
    if bins < 1:
        raise MetricError(f"reliability needs at least one bin, got {bins}")
    forecast = _as_probabilities(probabilities)
    actual = one_hot(outcomes)
    if len(forecast) != len(actual):
        raise MetricError(f"{len(forecast)} forecasts against {len(actual)} outcomes")

    edges = np.linspace(0.0, 1.0, bins + 1)
    # `edges[1:-1]` rather than every edge: digitize over the interior
    # boundaries returns 0..bins-1 directly, and puts a stated 1.0 in the last
    # bin instead of one past the end.
    index = np.digitize(forecast.ravel(), edges[1:-1])

    grouped = (
        pd.DataFrame({"bin": index, "predicted": forecast.ravel(), "observed": actual.ravel()})
        .groupby("bin")
        .agg(n=("observed", "size"), predicted=("predicted", "mean"), observed=("observed", "mean"))
        .reset_index()
    )
    grouped["lower"] = edges[grouped["bin"]]
    grouped["upper"] = edges[grouped["bin"] + 1]
    grouped["gap"] = grouped["observed"] - grouped["predicted"]
    return grouped.reindex(columns=list(RELIABILITY_COLUMNS))


def expected_calibration_error(table: pd.DataFrame) -> float:
    """The reliability table as one number: mean absolute gap, weighted by bin.

    Weighted rather than plain, because the bins are wildly uneven — most of a
    three-class football forecast sits between 0.15 and 0.55 — and an unweighted
    mean would let a bin holding forty statements count as much as one holding
    forty thousand.

    Raises:
        MetricError: On an empty table. Nothing was forecast, so there is no
            calibration to report, and zero would read as perfect.
    """
    if table.empty:
        raise MetricError("no forecasts to measure reliability over")
    return float(np.average(table["gap"].abs().to_numpy(), weights=table["n"].to_numpy()))
