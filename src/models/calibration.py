"""Temperature scaling, fitted on a holdout carved out of the training half.

A model can rank matches well and still state the wrong numbers. Gradient
boosting on a log-loss objective comes out slightly overconfident here, and so
does an average of several such models — the temperatures fitted over the
reported folds run from 0.99 to 1.12, all but one of them above one. The defect
is a forecast whose stated probability is not the rate at which the thing
happens, and one scalar removes most of it.

**It buys reliability, not loss.** Measured over 59,001 matches, the scalar
halves the gap between what the model says and what happens — 0.0037 to 0.0020
for the best single family — and moves log loss by 0.00008, in the wrong
direction. That is the answer to the question the milestone asked rather than a
disappointment: these models were already close to proper, and a layer that
made the score better *and* the probabilities honest would have meant the
score was the thing that was wrong.

**One parameter, applied to the probabilities.** ``p ** (1/T)``, renormalised.
Above one it flattens a forecast towards the uniform, below one it sharpens it,
and at exactly one it does nothing — which is the answer for a model that was
already honest, and is a result worth being able to report rather than a case
to special-case away.

Vector and matrix scaling — a weight per class, or a full 3x3 — were not built.
They are the same idea with three and twelve parameters, and there is no
evidence here that the miscalibration differs by class; a shape that big fitted
on one year of holdout would mostly fit the holdout.

**The holdout comes out of the training half, by date.** The last year of what
the fold was given to train on, with the model refitted on everything before
it. That costs a second fit per fold and is the only arrangement that keeps the
guarantee the split layer exists to provide: the temperature is chosen without
the evaluation half being touched, so a calibrated model reads exactly the same
matches an uncalibrated one does.

A thin holdout is not fitted on. Below :data:`MIN_HOLDOUT_MATCHES` the
temperature is left at one and a warning is logged — a scalar fitted on eighty
matches would move the forecast by more than the miscalibration does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.evaluation.metrics import terms
from src.ingestion.base import TARGET_COLUMN
from src.models.baselines import Forecaster
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.evaluation.metrics import Outcomes

logger = get_logger(__name__)

NEUTRAL = 1.0
"""The temperature that changes nothing."""

BOUNDS: tuple[float, float] = (0.2, 5.0)
"""The search interval.

Wide enough to contain any correction a fitted model needs — a temperature of 2
already halves the log-odds — and bounded because the objective is flat far
from one, where an unbounded search wanders for no gain.
"""

HOLDOUT_DAYS = 365
"""How much of the training half is kept back to fit the temperature on.

A year, matching the fold horizon: the calibration is measured over the same
span of football it is applied to, and a shorter window would fit the
temperature on one part of a season.
"""

MIN_HOLDOUT_MATCHES = 380
"""The floor for both halves of the inner split. One English season, the same
number :mod:`src.models.splits` uses, for the same reason."""


def apply(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """``p ** (1/T)``, renormalised. Rows of nulls stay null.

    An unpriced row comes back unpriced rather than as a forecast invented by
    the arithmetic — the scoring layer takes coverage seriously and this is not
    the place to quietly fill a gap.
    """
    powered = np.power(np.asarray(probabilities, dtype=float), 1.0 / temperature)
    scaled: np.ndarray = powered / powered.sum(axis=1, keepdims=True)
    return scaled


def temperature(
    probabilities: np.ndarray, outcomes: Outcomes, *, minimum: int = MIN_HOLDOUT_MATCHES
) -> float:
    """The temperature that minimises log loss on what it is given.

    Unpriced rows are dropped rather than counted: a forecaster that could not
    price half its holdout should have its temperature fitted on the half it
    could, and scoring a null as anything is the one thing this project does
    not do.

    Returns:
        The fitted temperature, or :data:`NEUTRAL` when too little is left to
        fit one on.
    """
    stated = np.asarray(probabilities, dtype=float)
    labels = pd.Series(list(outcomes), dtype="string")
    usable = ~np.isnan(stated).any(axis=1) & labels.notna().to_numpy()
    if int(usable.sum()) < minimum:
        logger.warning(
            "temperature left at %.1f: %d usable holdout match(es), %d needed",
            NEUTRAL,
            int(usable.sum()),
            minimum,
        )
        return NEUTRAL

    priced, happened = stated[usable], labels[usable]

    def loss(value: float) -> float:
        return float(terms(apply(priced, value), happened)["log_loss"].mean())

    found = minimize_scalar(loss, bounds=BOUNDS, method="bounded")
    return float(found.x)


@dataclass(frozen=True, slots=True)
class Calibrated:
    """Any forecaster, with its probabilities rescaled by one fitted scalar.

    A :class:`~src.models.baselines.Forecaster` wrapping a forecaster, so it
    drops into the same backtest on the same folds and can be put in the same
    table as the thing it wraps. It works on a baseline as readily as on a
    model — the bookmaker's line is a forecast like any other — which is what
    makes "was this already calibrated?" a question the report can answer.
    """

    inner: Forecaster
    holdout_days: int = HOLDOUT_DAYS
    min_holdout: int = MIN_HOLDOUT_MATCHES

    @property
    def name(self) -> str:
        return f"{self.inner.name}-calibrated"

    def split(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The training half, cut into what the model is fitted on and what the
        temperature is fitted on. On the date, never on a row count — the same
        rule every other boundary in this project follows."""
        cut = pd.Timestamp(train["date"].max()) - pd.Timedelta(self.holdout_days, "D")
        return train[train["date"] < cut], train[train["date"] >= cut]

    def temperature(self, train: pd.DataFrame) -> float:
        """Refit the inner model on the earlier part, score the later part, fit
        the scalar on that. Never touches the evaluation half."""
        earlier, holdout = self.split(train)
        if len(earlier) < self.min_holdout or len(holdout) < self.min_holdout:
            logger.warning(
                "%s not calibrated: %d/%d matches either side of the holdout cut, %d needed",
                self.inner.name,
                len(earlier),
                len(holdout),
                self.min_holdout,
            )
            return NEUTRAL
        return temperature(
            self.inner.forecast(earlier, holdout),
            holdout[TARGET_COLUMN],
            minimum=self.min_holdout,
        )

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        found = self.temperature(train)
        logger.info("%s: temperature %.4f", self.name, found)
        return apply(self.inner.forecast(train, evaluate), found)
