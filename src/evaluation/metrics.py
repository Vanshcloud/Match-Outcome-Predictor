"""Log loss, RPS and accuracy over three ordered outcomes.

Two proper scoring rules and one improper one, in that order of importance.

**Log loss** is the likelihood of what happened, and it is the metric the whole
project is measured against because it is what the bookmaker's closing line is
usually quoted in. It punishes confidence without mercy: a forecast that gives
the true outcome zero probability scores infinity, and nothing here clips that
away. A baseline that always predicts a home win *is* infinitely bad by this
measure, and reporting it as some large finite number would be a kindness the
metric does not extend.

**RPS** — the ranked probability score — is the one that knows H, D and A are
*ordered*. Being wrong by predicting a draw when the away side won is a smaller
error than predicting a home win, and log loss cannot see the difference. It is
also finite for a degenerate forecast, which is what makes it the metric that
can still rank a baseline log loss has sent to infinity.

**Accuracy** is reported because people ask for it, and it is listed last
because in this problem it is close to meaningless: predicting the home side
every time scores around 45%.

Every function takes probabilities as an ``(n, 3)`` array in :data:`CLASSES`
order and outcomes as the canonical ``result`` strings. Both are checked rather
than assumed — a transposed array or a mis-ordered class vector produces a
number that looks plausible and is wrong, which is the failure mode a metric
module can least afford.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.ingestion.base import Result

if TYPE_CHECKING:
    # What actually happened, as canonical `result` strings. A Series or a
    # plain sequence: callers hold one of each — the pipeline slices a column,
    # a test writes a list — and converting once at the boundary is cheaper
    # than converting at every call site. Type-checking only, because a
    # parameterised Series is not a runtime expression.
    Outcomes = Sequence[str] | pd.Series[Any]

CLASSES: tuple[str, str, str] = (Result.HOME, Result.DRAW, Result.AWAY)
"""Home, draw, away — in that order, everywhere.

The order is not a convention here, it is arithmetic: RPS accumulates over the
classes and only means anything if they are ordered by the quantity they
describe. Home win, draw and away win are consecutive on goal difference.
"""

SUM_TOLERANCE = 1e-6
"""How far a row of probabilities may sum from one.

Loose enough for the rounding in a normalised set of odds, tight enough that a
forecast which forgot to normalise is caught rather than quietly scored.
"""


class MetricError(ValueError):
    """A forecast cannot be scored as given."""


@dataclass(frozen=True, slots=True)
class Scores:
    """One forecaster's performance over one set of matches."""

    n: int
    log_loss: float
    rps: float
    accuracy: float

    def summary(self) -> str:
        return (
            f"{self.n:,} matches: log loss {self.log_loss:.4f}, "
            f"RPS {self.rps:.4f}, accuracy {self.accuracy:.1%}"
        )


def _as_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Check an array is a usable set of forecasts and return it as float."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(CLASSES):
        raise MetricError(f"probabilities must be (n, {len(CLASSES)}), got {values.shape}")
    if np.isnan(values).any():
        raise MetricError("probabilities contain nulls; drop unpriced rows before scoring")
    if (values < 0).any() or (values > 1).any():
        raise MetricError("probabilities outside [0, 1]")
    sums = values.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=SUM_TOLERANCE):
        worst = float(np.abs(sums - 1.0).max())
        raise MetricError(f"probabilities do not sum to one; worst row is off by {worst:.2e}")
    return values


def one_hot(outcomes: Outcomes) -> np.ndarray:
    """Outcomes as an ``(n, 3)`` indicator array in :data:`CLASSES` order."""
    labels = pd.Series(list(outcomes), dtype="string")
    unknown = sorted(set(labels.dropna().unique()) - set(CLASSES))
    if unknown:
        raise MetricError(f"not outcomes: {unknown}")
    if labels.isna().any():
        raise MetricError("outcomes contain nulls; a match with no result cannot be scored")
    return np.column_stack([(labels == label).to_numpy() for label in CLASSES]).astype(float)


def terms(probabilities: np.ndarray, outcomes: Outcomes) -> pd.DataFrame:
    """Per-match contributions, one row per match.

    The metrics are the column means, which is the property that lets a caller
    group these by competition, by fold, or by both, and get the same numbers a
    direct computation over each subset would give. Scoring once and grouping
    afterwards is also the difference between one pass and one per scope.
    """
    forecast = _as_probabilities(probabilities)
    actual = one_hot(outcomes)
    if len(forecast) != len(actual):
        raise MetricError(f"{len(forecast)} forecasts against {len(actual)} outcomes")

    # log(0) is a legitimate answer here, not a warning: a forecaster that ruled
    # out what happened has infinite loss and the report says so.
    with np.errstate(divide="ignore"):
        taken = np.log(np.where(actual > 0, forecast, 1.0)).sum(axis=1)

    # RPS over ordered classes: the squared gap between the two cumulative
    # distributions, averaged over the K-1 thresholds that carry information.
    gaps = np.cumsum(forecast, axis=1)[:, :-1] - np.cumsum(actual, axis=1)[:, :-1]
    ranked = np.square(gaps).sum(axis=1) / (len(CLASSES) - 1)

    return pd.DataFrame(
        {
            "log_loss": -taken,
            "rps": ranked,
            "hit": actual[np.arange(len(actual)), forecast.argmax(axis=1)],
        }
    )


def score(probabilities: np.ndarray, outcomes: Outcomes) -> Scores:
    """The three metrics over one set of matches."""
    contributions = terms(probabilities, outcomes)
    return Scores(
        n=len(contributions),
        log_loss=float(contributions["log_loss"].mean()),
        rps=float(contributions["rps"].mean()),
        accuracy=float(contributions["hit"].mean()),
    )
