"""Temperature scaling: the scalar, the wrapper, and the leak it must not be.

The models are stubs with a known defect — a forecaster that always says 90%
about something that happens 45% of the time is overconfident by construction —
so a test can assert which way the temperature moved rather than hoping a real
fit lands somewhere. What is being tested is the layer: that it corrects the
right direction, that it fits on the training half only, and that it does
nothing when there is nothing to fit on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import score
from src.models.baselines import ClassPrior, Forecaster
from src.models.calibration import (
    NEUTRAL,
    Calibrated,
    apply,
    temperature,
)
from src.models.splits import walk_forward
from tests.factories import modelled_frame, season_labels

LEAGUE = modelled_frame(seasons=season_labels(2012, 10), teams=12)
FOLD = next(iter(walk_forward(LEAGUE, folds=2)))

# The synthetic league fields 132 matches a year, so its holdout is a fraction
# of the 380 the shipped default asks for. Lowered here rather than the league
# grown tenfold: what is under test is the arithmetic, not the threshold.
HOLDOUT = 100


class Fixed:
    """A forecaster that says the same thing about every match."""

    def __init__(self, name: str, stated: tuple[float, float, float]) -> None:
        self.name = name
        self.stated = stated

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        del train
        return np.tile(np.array(self.stated), (len(evaluate), 1))


OVERCONFIDENT = Fixed("overconfident", (0.90, 0.05, 0.05))
TIMID = Fixed("timid", (0.34, 0.33, 0.33))


# ---- the scalar --------------------------------------------------------------


def test_a_temperature_of_one_changes_nothing() -> None:
    stated = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    assert apply(stated, NEUTRAL) == pytest.approx(stated)


def test_above_one_flattens_and_below_one_sharpens() -> None:
    stated = np.array([[0.7, 0.2, 0.1]])
    assert apply(stated, 2.0)[0, 0] < 0.7
    assert apply(stated, 0.5)[0, 0] > 0.7


def test_a_scaled_forecast_still_sums_to_one() -> None:
    stated = np.array([[0.7, 0.2, 0.1], [0.2, 0.4, 0.4]])
    assert apply(stated, 1.7).sum(axis=1) == pytest.approx(1.0)


def test_a_row_nobody_could_price_stays_unpriced() -> None:
    """The scoring layer takes coverage seriously; this is not the place to
    quietly invent a forecast for a match with no odds."""
    stated = np.array([[0.7, 0.2, 0.1], [np.nan, np.nan, np.nan]])
    assert np.isnan(apply(stated, 1.5)[1]).all()


def test_an_overconfident_forecast_is_cooled() -> None:
    outcomes = ["H"] * 45 + ["D"] * 27 + ["A"] * 28
    stated = np.tile(np.array(OVERCONFIDENT.stated), (len(outcomes), 1))
    assert temperature(stated, outcomes, minimum=10) > 1.0


def test_a_timid_forecast_is_sharpened() -> None:
    outcomes = ["H"] * 90 + ["D"] * 5 + ["A"] * 5
    stated = np.tile(np.array(TIMID.stated), (len(outcomes), 1))
    assert temperature(stated, outcomes, minimum=10) < 1.0


def test_a_forecast_that_was_already_honest_is_left_alone() -> None:
    """The result worth being able to report. A model whose probabilities are
    already proper gains nothing here, and the layer has to be able to say so
    rather than move it anyway."""
    outcomes = ["H"] * 50 + ["D"] * 25 + ["A"] * 25
    stated = np.tile(np.array([0.5, 0.25, 0.25]), (len(outcomes), 1))
    assert temperature(stated, outcomes, minimum=10) == pytest.approx(NEUTRAL, abs=0.02)


def test_a_temperature_lowers_the_loss_it_was_fitted_on() -> None:
    outcomes = ["H"] * 45 + ["D"] * 27 + ["A"] * 28
    stated = np.tile(np.array(OVERCONFIDENT.stated), (len(outcomes), 1))
    found = temperature(stated, outcomes, minimum=10)
    assert score(apply(stated, found), outcomes).log_loss < score(stated, outcomes).log_loss


def test_unpriced_rows_are_dropped_rather_than_counted() -> None:
    """A forecaster that could price half its holdout gets a temperature fitted
    on the half it could."""
    outcomes = ["H"] * 45 + ["D"] * 27 + ["A"] * 28
    stated = np.tile(np.array(OVERCONFIDENT.stated), (len(outcomes), 1))
    holed = np.concatenate([stated, np.full((200, 3), np.nan)])
    assert temperature(holed, [*outcomes, *(["H"] * 200)], minimum=10) == pytest.approx(
        temperature(stated, outcomes, minimum=10)
    )


def test_too_little_to_fit_on_leaves_the_temperature_at_one() -> None:
    """A scalar fitted on eighty matches moves a forecast further than the
    miscalibration does."""
    outcomes = ["H"] * 10
    stated = np.tile(np.array(OVERCONFIDENT.stated), (10, 1))
    assert temperature(stated, outcomes, minimum=380) == NEUTRAL


# ---- the wrapper -------------------------------------------------------------


def test_a_calibrated_model_is_a_forecaster() -> None:
    """So it drops into the same backtest, on the same folds, in the same table
    as the model it wraps."""
    assert isinstance(Calibrated(OVERCONFIDENT), Forecaster)


def test_it_is_named_for_what_it_wraps() -> None:
    assert Calibrated(OVERCONFIDENT).name == "overconfident-calibrated"


def test_the_holdout_is_cut_on_the_date_not_on_a_row_count() -> None:
    """The rule every other boundary in this project follows. A row-count cut
    puts two matches played on the same afternoon on opposite sides of it."""
    earlier, holdout = Calibrated(OVERCONFIDENT, holdout_days=365).split(FOLD.train)
    assert earlier["date"].max() < holdout["date"].min()
    assert len(earlier) + len(holdout) == len(FOLD.train)


def test_the_temperature_is_fitted_without_touching_the_matches_it_prices() -> None:
    """The property the whole arrangement exists for. Rewriting every result in
    the evaluation half must not move the forecast by a single bit — if it
    does, the calibration read the answers."""
    calibrated = Calibrated(ClassPrior(), min_holdout=HOLDOUT)
    rewritten = FOLD.evaluate.assign(result="A", home_goals=0, away_goals=5)
    assert np.array_equal(
        calibrated.forecast(FOLD.train, FOLD.evaluate),
        calibrated.forecast(FOLD.train, rewritten),
    )


def test_a_training_half_too_thin_to_split_is_not_calibrated() -> None:
    """A fold early enough to have no holdout still has to produce a forecast.
    Refusing would fail a run for a reason that has nothing to do with the
    model."""
    calibrated = Calibrated(OVERCONFIDENT, min_holdout=len(FOLD.train))
    assert calibrated.temperature(FOLD.train) == NEUTRAL
    assert calibrated.forecast(FOLD.train, FOLD.evaluate) == pytest.approx(
        OVERCONFIDENT.forecast(FOLD.train, FOLD.evaluate)
    )


def test_calibrating_an_overconfident_model_lowers_its_loss_on_the_next_fold() -> None:
    """End to end, on matches the temperature never saw: the scalar is fitted
    on the last year of the training half and applied to the year after it."""
    outcomes = FOLD.evaluate["result"]
    calibrated = Calibrated(OVERCONFIDENT, min_holdout=HOLDOUT)
    assert calibrated.temperature(FOLD.train) > 1.0
    before = score(OVERCONFIDENT.forecast(FOLD.train, FOLD.evaluate), outcomes)
    after = score(calibrated.forecast(FOLD.train, FOLD.evaluate), outcomes)
    assert after.log_loss < before.log_loss
