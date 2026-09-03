"""The four baselines, and the one property all of them must have.

Each is a few lines, and each of those lines is the kind that is wrong in a way
nothing downstream can see: a prior counted on the wrong half, an overround
removed twice, a column read in the wrong order. So the assertions are about
arithmetic that can be done by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import CLASSES
from src.models.baselines import (
    Bookmaker,
    ClassPrior,
    DixonColes,
    Forecaster,
    HomeAlways,
    default_forecasters,
)
from src.models.splits import walk_forward
from tests.factories import canonical_frame, league_frame, season_labels

LEAGUE = league_frame(seasons=season_labels(2012, 14), teams=20)
FOLD = next(iter(walk_forward(LEAGUE)))
FORECASTERS: tuple[Forecaster, ...] = (HomeAlways(), ClassPrior(), Bookmaker())


def _rated(frame: pd.DataFrame) -> pd.DataFrame:
    """The evaluation half with a Dixon-Coles column block attached."""
    return frame.assign(dc_prob_home=0.5, dc_prob_draw=0.3, dc_prob_away=0.2)


# ---- the property they share ------------------------------------------------


@pytest.mark.parametrize("forecaster", FORECASTERS, ids=lambda one: one.name)
def test_a_baseline_does_not_read_the_results_it_is_pricing(forecaster: Forecaster) -> None:
    """A forecaster is handed the matches it is scored on, so it *could* cheat.

    The split layer guarantees the training half is earlier; nothing but this
    guarantees the evaluation half's own results were not consulted.
    """
    rewritten = FOLD.evaluate.assign(result="A", home_goals=0, away_goals=5)
    assert np.array_equal(
        forecaster.forecast(FOLD.train, FOLD.evaluate),
        forecaster.forecast(FOLD.train, rewritten),
        equal_nan=True,
    )


@pytest.mark.parametrize("forecaster", FORECASTERS, ids=lambda one: one.name)
def test_a_baseline_returns_one_row_per_match(forecaster: Forecaster) -> None:
    forecast = forecaster.forecast(FOLD.train, FOLD.evaluate)
    assert forecast.shape == (len(FOLD.evaluate), len(CLASSES))


# ---- home always ------------------------------------------------------------


def test_home_always_is_certain_of_a_home_win() -> None:
    forecast = HomeAlways().forecast(FOLD.train, FOLD.evaluate)
    assert (forecast == np.array([1.0, 0.0, 0.0])).all()


# ---- class prior ------------------------------------------------------------


def test_the_prior_is_the_training_half_s_own_base_rates() -> None:
    """Counted on the training fold, not on the table.

    A prior read off the whole table is a small leak and a measurable one: the
    home-win rate has fallen over the thirty years of real data here. The
    synthetic league holds it constant on purpose, which is why this asserts
    the *source* of the number rather than that it differs from the pooled one.
    """
    forecast = ClassPrior().forecast(FOLD.train, FOLD.evaluate)
    expected = [float((FOLD.train["result"] == label).mean()) for label in CLASSES]
    assert forecast[0] == pytest.approx(expected)


def test_the_prior_sums_to_one() -> None:
    assert ClassPrior().forecast(FOLD.train, FOLD.evaluate).sum(axis=1) == pytest.approx(1.0)


def test_a_prior_with_no_history_prices_nothing() -> None:
    """Null, not a uniform guess. "No evidence" and "evenly matched" differ."""
    forecast = ClassPrior().forecast(canonical_frame([]), FOLD.evaluate)
    assert np.isnan(forecast).all()


# ---- Dixon-Coles ------------------------------------------------------------


def test_dixon_coles_reads_the_ratings_columns() -> None:
    forecast = DixonColes().forecast(FOLD.train, _rated(FOLD.evaluate))
    assert forecast[0] == pytest.approx([0.5, 0.3, 0.2])


def test_dixon_coles_prices_nothing_without_a_ratings_table() -> None:
    assert np.isnan(DixonColes().forecast(FOLD.train, FOLD.evaluate)).all()


def test_a_half_filled_rating_row_is_not_priced() -> None:
    """It would otherwise score as a valid forecast summing to less than one."""
    partial = _rated(FOLD.evaluate).copy()
    partial.loc[partial.index[0], "dc_prob_draw"] = np.nan
    forecast = DixonColes().forecast(FOLD.train, partial)
    assert np.isnan(forecast[0]).all()
    assert not np.isnan(forecast[1]).any()


# ---- bookmaker --------------------------------------------------------------


def test_the_overround_is_removed() -> None:
    """The factory's prices imply 1.080; the forecast must imply exactly one."""
    forecast = Bookmaker().forecast(FOLD.train, FOLD.evaluate)
    assert forecast.sum(axis=1) == pytest.approx(1.0)


def test_the_implied_probabilities_are_in_class_order() -> None:
    """A transposed read here scores plausibly and is wrong every time."""
    odds = FOLD.evaluate[["odds_home", "odds_draw", "odds_away"]].iloc[0].to_numpy(dtype=float)
    implied = (1 / odds) / (1 / odds).sum()
    assert Bookmaker().forecast(FOLD.train, FOLD.evaluate)[0] == pytest.approx(implied)


def test_a_match_with_no_price_is_not_priced() -> None:
    unpriced = FOLD.evaluate.copy()
    unpriced.loc[unpriced.index[0], "odds_draw"] = pd.NA
    forecast = Bookmaker().forecast(FOLD.train, unpriced)
    assert np.isnan(forecast[0]).all()
    assert not np.isnan(forecast[1]).any()


def test_an_impossible_price_is_not_priced() -> None:
    """A decimal price of one is certainty, which no bookmaker offers."""
    broken = FOLD.evaluate.copy()
    broken.loc[broken.index[0], "odds_home"] = 1.0
    assert np.isnan(Bookmaker().forecast(FOLD.train, broken)[0]).all()


def test_a_table_without_odds_prices_nothing() -> None:
    without = FOLD.evaluate.drop(columns=["odds_home", "odds_draw", "odds_away"])
    assert np.isnan(Bookmaker().forecast(FOLD.train, without)).all()


# ---- the default set --------------------------------------------------------


def test_every_baseline_is_offered_when_nothing_is_known_about_the_table() -> None:
    assert [one.name for one in default_forecasters()] == [
        "home_always",
        "class_prior",
        "dixon_coles",
        "bookmaker",
    ]


def test_a_baseline_whose_columns_are_absent_is_dropped() -> None:
    """0% coverage and "the ratings were never built" deserve different words."""
    names = [one.name for one in default_forecasters(FOLD.evaluate)]
    assert names == ["home_always", "class_prior", "bookmaker"]


def test_a_rated_table_offers_dixon_coles() -> None:
    assert "dixon_coles" in [one.name for one in default_forecasters(_rated(FOLD.evaluate))]
