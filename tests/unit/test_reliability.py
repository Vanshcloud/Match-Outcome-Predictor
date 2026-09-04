"""Reliability, and the four ways a calibration table lies about a forecast.

Every case here is a forecast whose honesty is known by construction — a
probability repeated over outcomes counted out by hand — so a table that
disagrees is the table's fault and not a question about the model that produced
it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import MetricError
from src.evaluation.reliability import (
    DEFAULT_BINS,
    expected_calibration_error,
    reliability,
)


def _repeated(probabilities: tuple[float, float, float], outcomes: str) -> np.ndarray:
    """One forecast, repeated once per outcome in ``outcomes``."""
    return np.tile(np.array(probabilities), (len(outcomes), 1))


# 50 home, 25 draw, 25 away — exactly what a (0.5, 0.25, 0.25) forecast claims.
HONEST_OUTCOMES = "H" * 50 + "D" * 25 + "A" * 25
HONEST = (0.5, 0.25, 0.25)


def test_a_forecast_that_happens_at_the_rate_it_states_has_no_gap() -> None:
    table = reliability(_repeated(HONEST, HONEST_OUTCOMES), list(HONEST_OUTCOMES))
    assert table["gap"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert expected_calibration_error(table) == pytest.approx(0.0, abs=1e-12)


def test_an_overconfident_forecast_shows_a_negative_gap() -> None:
    """The defect the calibration layer exists to remove: the model promised
    95% and delivered 60%, and the sign says which way it was wrong."""
    outcomes = "H" * 60 + "A" * 40
    table = reliability(_repeated((0.95, 0.025, 0.025), outcomes), list(outcomes))
    top = table[table["lower"] == 0.9].iloc[0]
    assert top["predicted"] == pytest.approx(0.95)
    assert top["observed"] == pytest.approx(0.6)
    assert top["gap"] < 0


def test_every_stated_probability_is_binned_not_only_the_confident_one() -> None:
    """Three statements per match, all of them measured. Binning only the
    model's favourite class would say nothing about the draw column, which is
    where a football forecast is usually wrong."""
    table = reliability(_repeated(HONEST, HONEST_OUTCOMES), list(HONEST_OUTCOMES))
    assert table["n"].sum() == 3 * len(HONEST_OUTCOMES)


def test_an_empty_bin_is_left_out_rather_than_reported_as_perfect() -> None:
    table = reliability(_repeated(HONEST, HONEST_OUTCOMES), list(HONEST_OUTCOMES))
    # Two distinct probabilities, so two bins of the ten hold anything.
    assert len(table) == 2
    assert sorted(table["bin"]) == [2, 5]


def test_a_stated_certainty_lands_in_the_last_bin_and_not_past_the_end() -> None:
    """The off-by-one a digitize over every edge would produce, silently, on
    the one forecast whose calibration matters most."""
    outcomes = "HHHA"
    table = reliability(_repeated((1.0, 0.0, 0.0), outcomes), list(outcomes))
    assert table["bin"].max() == DEFAULT_BINS - 1
    assert table["upper"].max() == pytest.approx(1.0)
    assert table["n"].sum() == 3 * len(outcomes)


def test_the_error_is_weighted_by_how_many_statements_are_behind_each_bin() -> None:
    """A bin holding forty statements must not count as much as one holding
    forty thousand; the bins in a three-class forecast are wildly uneven."""
    # 999 honest statements at 0.5, one wild one at 0.95 that never happens.
    outcomes = "H" * 500 + "A" * 500
    probabilities = _repeated((0.5, 0.25, 0.25), outcomes)
    probabilities[-1] = (0.95, 0.025, 0.025)
    table = reliability(probabilities, list(outcomes))
    assert expected_calibration_error(table) < 0.01


def test_one_bin_is_allowed_and_collapses_the_whole_range() -> None:
    table = reliability(_repeated(HONEST, HONEST_OUTCOMES), list(HONEST_OUTCOMES), bins=1)
    assert len(table) == 1
    assert table.loc[0, "lower"] == 0.0
    assert table.loc[0, "upper"] == 1.0


def test_fewer_than_one_bin_is_refused() -> None:
    with pytest.raises(MetricError, match="at least one bin"):
        reliability(_repeated(HONEST, HONEST_OUTCOMES), list(HONEST_OUTCOMES), bins=0)


def test_a_forecast_that_does_not_line_up_with_its_outcomes_is_refused() -> None:
    with pytest.raises(MetricError, match="against"):
        reliability(_repeated(HONEST, "HH"), ["H"])


def test_an_unusable_forecast_is_refused_by_the_same_check_the_metrics_use() -> None:
    with pytest.raises(MetricError, match="sum to one"):
        reliability(np.array([[0.5, 0.5, 0.5]]), ["H"])


def test_the_error_of_nothing_is_refused_rather_than_reported_as_perfect() -> None:
    """An empty table means nothing was forecast. Zero would read as a model
    that never misses."""
    empty = reliability(np.zeros((0, 3)), [])
    with pytest.raises(MetricError, match="no forecasts"):
        expected_calibration_error(empty)
