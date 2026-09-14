"""The blend, the diagnostic pass, and the rule that decides who is in it.

The selection rule is the part worth testing hardest. An ensemble whose members
are wrong about the same matches is an averaging artefact wearing a model's
name, and the only thing standing between this project and one of those is
:func:`select_members` refusing a candidate it should refuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import Bookmaker, ClassPrior, Forecaster
from src.models.ensemble import (
    BEST,
    FORECAST_COLUMNS,
    MAX_ERROR_CORRELATION,
    MEMBERS,
    Ensemble,
    by_log_loss,
    ensemble,
    error_correlations,
    fold_forecasts,
    select_members,
)
from src.models.splits import walk_forward
from src.models.zoo import FAMILIES, ModelError
from tests.factories import modelled_frame, season_labels

LEAGUE = modelled_frame(seasons=season_labels(2012, 10), teams=12)
FOLD = next(iter(walk_forward(LEAGUE, folds=2)))


class Fixed:
    """A forecaster that says the same thing about every match."""

    def __init__(self, name: str, stated: tuple[float, float, float]) -> None:
        self.name = name
        self.stated = stated

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        del train
        return np.tile(np.array(self.stated), (len(evaluate), 1))


HOME = Fixed("home", (0.6, 0.2, 0.2))
AWAY = Fixed("away", (0.2, 0.2, 0.6))


# ---- the blend ---------------------------------------------------------------


def test_an_ensemble_is_a_forecaster() -> None:
    """So it is scored by the unchanged backtest, on the same folds, beside its
    own members."""
    assert isinstance(Ensemble("blend", (HOME, AWAY)), Forecaster)


def test_the_blend_is_the_mean_of_its_members() -> None:
    forecast = Ensemble("blend", (HOME, AWAY)).forecast(FOLD.train, FOLD.evaluate)
    assert forecast[0] == pytest.approx((0.4, 0.2, 0.4))
    assert forecast.sum(axis=1) == pytest.approx(1.0)


def test_one_member_is_that_member() -> None:
    assert Ensemble("blend", (HOME,)).forecast(FOLD.train, FOLD.evaluate) == pytest.approx(
        HOME.forecast(FOLD.train, FOLD.evaluate)
    )


def test_a_match_one_member_could_not_price_is_not_priced_by_the_blend() -> None:
    """Rather than priced by whichever members happened to have an opinion,
    which would be a forecaster whose membership varied by row."""
    unpriced = FOLD.evaluate.drop(columns=["odds_home", "odds_draw", "odds_away"])
    forecast = Ensemble("blend", (HOME, Bookmaker())).forecast(FOLD.train, unpriced)
    assert np.isnan(forecast).all()


def test_the_blend_is_built_from_the_zoo_by_name() -> None:
    built = ensemble(["logistic_regression", "random_forest"])
    assert [member.name for member in built.members] == [
        "logistic_regression",
        "random_forest",
    ]


def test_a_member_the_zoo_does_not_hold_is_refused() -> None:
    with pytest.raises(ModelError, match="not a model"):
        ensemble(["transformer"])


def test_the_shipped_constants_name_families_that_exist() -> None:
    """A typo in a baked constant is a run that fails a minute in, on a machine
    that has already loaded 250,000 matches."""
    assert BEST in FAMILIES
    assert set(MEMBERS) <= set(FAMILIES)
    assert BEST in MEMBERS


# ---- the diagnostic pass -----------------------------------------------------


def test_every_forecaster_is_recorded_on_every_fold() -> None:
    """What makes two forecasters' rows line up: the same fold, the same
    position within it."""
    forecasts = fold_forecasts(LEAGUE, [HOME, AWAY], folds=2)
    assert set(forecasts["forecaster"]) == {"home", "away"}
    counts = forecasts.groupby(["fold", "forecaster"]).size().unstack()
    assert (counts["home"] == counts["away"]).all()


def test_the_pass_returns_the_matches_the_backtest_only_returns_means_of() -> None:
    forecasts = fold_forecasts(LEAGUE, [HOME], folds=2)
    assert list(forecasts.columns) == [
        "fold",
        "match",
        "match_id",
        "competition_id",
        *FORECAST_COLUMNS,
        "forecaster",
        "result",
    ]
    assert forecasts["result"].isin(["H", "D", "A"]).all()
    # Carried for the "where is it reliable" report, which is a grouping
    # the scored table keeps only as means.
    assert set(forecasts["competition_id"]) == set(LEAGUE["competition_id"])


def test_every_forecast_names_the_fixture_it_is_about() -> None:
    """``match`` lines two forecasters up within a fold;
    ``match_id`` is what joins these rows to the closing odds, and
    re-deriving it from the split somewhere else would be a plausible-looking
    wrong answer waiting for a split parameter to change."""
    forecasts = fold_forecasts(LEAGUE, [HOME], folds=2)
    assert set(forecasts["match_id"]) <= set(LEAGUE["match_id"])
    # One row per fixture per fold: the id identifies the row, not just labels it.
    assert not forecasts.duplicated(["fold", "forecaster", "match_id"]).any()


# ---- who is alike, and who that admits ---------------------------------------


def test_a_forecaster_is_perfectly_correlated_with_itself() -> None:
    matrix = error_correlations(fold_forecasts(LEAGUE, [HOME, AWAY], folds=2))
    assert matrix.loc["home", "home"] == pytest.approx(1.0)
    assert matrix.loc["home", "away"] == pytest.approx(matrix.loc["away", "home"])


def test_two_forecasters_wrong_about_the_same_matches_correlate() -> None:
    """Two constants that differ only in confidence are wrong about exactly the
    same matches, in proportion. That is the shape the rule has to catch."""
    matrix = error_correlations(
        fold_forecasts(LEAGUE, [HOME, Fixed("home_ish", (0.61, 0.2, 0.19))], folds=2)
    )
    assert matrix.loc["home", "home_ish"] > MAX_ERROR_CORRELATION


def test_the_correlation_is_taken_over_the_matches_both_could_price() -> None:
    """The reason the backtest reports a common subset, one level down: a
    correlation over two different sets of matches compares two questions."""
    matrix = error_correlations(fold_forecasts(LEAGUE, [HOME, Bookmaker()], folds=2))
    assert set(matrix.index) == {"home", "bookmaker"}
    assert matrix.notna().all().all()


def test_the_ranking_is_best_first() -> None:
    """Worst last, which is the end :func:`select_members` consumes: a family
    that scores badly is still admitted if it is wrong about different matches,
    but it is considered after the ones that do not need that excuse."""
    hopeless = Fixed("hopeless", (0.02, 0.02, 0.96))
    ranked = by_log_loss(fold_forecasts(LEAGUE, [HOME, ClassPrior(), hopeless], folds=2))
    assert set(ranked) == {"home", "class_prior", "hopeless"}
    assert ranked[-1] == "hopeless"


def test_a_near_substitute_is_not_admitted() -> None:
    matrix = pd.DataFrame(
        [[1.0, 0.999, 0.5], [0.999, 1.0, 0.5], [0.5, 0.5, 1.0]],
        index=["first", "twin", "different"],
        columns=["first", "twin", "different"],
    )
    assert select_members(matrix, ["first", "twin", "different"]) == ("first", "different")


def test_the_order_decides_which_of_two_substitutes_survives() -> None:
    """And nothing else. Passing the ranking sorted by log loss means the best
    of a correlated group is the one kept."""
    matrix = pd.DataFrame(
        [[1.0, 0.999], [0.999, 1.0]], index=["first", "twin"], columns=["first", "twin"]
    )
    assert select_members(matrix, ["twin", "first"]) == ("twin",)


def test_a_candidate_is_judged_against_every_member_already_taken() -> None:
    """Not only against the last one. A third model correlated with the first
    but not the second would otherwise slip in."""
    matrix = pd.DataFrame(
        [[1.0, 0.5, 0.999], [0.5, 1.0, 0.5], [0.999, 0.5, 1.0]],
        index=["first", "second", "third"],
        columns=["first", "second", "third"],
    )
    assert select_members(matrix, ["first", "second", "third"]) == ("first", "second")
