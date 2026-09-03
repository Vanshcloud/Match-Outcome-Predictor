"""The six families, and the four ways a model wrapper goes silently wrong.

Every family is exercised with a deliberately tiny configuration. What is being
tested is the wrapper — class order, a fresh fit per fold, null handling, the
column selection — not whether gradient boosting works, and a suite that spent
four minutes fitting four hundred trees to prove `predict_proba` returns three
columns is a suite people stop running.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import CLASSES, score
from src.models.baselines import ClassPrior, Forecaster
from src.models.dataset import DESIGN_COLUMNS, without
from src.models.splits import walk_forward
from src.models.zoo import FAMILIES, SEED, ModelError, TrainedForecaster, build, default_zoo
from tests.factories import modelled_frame, season_labels

# The tiny MLP below is given forty iterations and does not converge in them.
# That is the point — this file tests a wrapper, not a neural network — and the
# warning is scoped off here rather than in the pytest config, where it would
# also hide a real one from a full-size run.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

LEAGUE = modelled_frame(seasons=season_labels(2015, 6), teams=12)
FOLD = next(iter(walk_forward(LEAGUE, folds=2)))

# One tiny configuration per family, so the whole zoo fits in seconds.
#
# The boosted three also need their learning rate raising. The shipped value is
# around 0.02, which is right for four hundred rounds and leaves a ten-round
# model sitting on the class prior — so a test asserting the model beats the
# prior would be asserting something about the budget rather than the wrapper.
QUICK: dict[str, dict[str, object]] = {
    "logistic_regression": {"max_iter": 60},
    "random_forest": {"n_estimators": 10, "max_depth": 4},
    "xgboost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.3},
    "lightgbm": {"n_estimators": 10, "num_leaves": 7, "learning_rate": 0.3},
    "catboost": {"iterations": 10, "depth": 3, "learning_rate": 0.3},
    "mlp": {"hidden_layer_sizes": (8,), "max_iter": 40},
}


def _quick(name: str, columns: tuple[str, ...] = DESIGN_COLUMNS) -> TrainedForecaster:
    return build(name, columns, **QUICK[name])


NAMES = sorted(FAMILIES)


# ---- the contract every family keeps ----------------------------------------


def test_the_zoo_holds_the_six_families_the_scope_named() -> None:
    assert set(FAMILIES) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
    }


@pytest.mark.parametrize("name", NAMES)
def test_a_model_is_a_forecaster(name: str) -> None:
    """So it drops into the walk-forward backtest without that pipeline
    learning that an estimator exists."""
    assert isinstance(_quick(name), Forecaster)


@pytest.mark.parametrize("name", NAMES)
def test_a_model_returns_one_probability_row_per_match(name: str) -> None:
    forecast = _quick(name).forecast(FOLD.train, FOLD.evaluate)
    assert forecast.shape == (len(FOLD.evaluate), len(CLASSES))
    assert forecast.sum(axis=1) == pytest.approx(1.0)
    assert not np.isnan(forecast).any()


@pytest.mark.parametrize("name", NAMES)
def test_a_model_beats_counting_base_rates(name: str) -> None:
    """The floor. A wrapper that transposed its columns or mis-encoded its
    target still returns a valid-looking forecast — and scores worse than
    counting."""
    outcomes = FOLD.evaluate["result"]
    model = score(_quick(name).forecast(FOLD.train, FOLD.evaluate), outcomes)
    prior = score(ClassPrior().forecast(FOLD.train, FOLD.evaluate), outcomes)
    assert model.log_loss < prior.log_loss


@pytest.mark.parametrize("name", NAMES)
def test_a_model_does_not_read_the_results_it_is_pricing(name: str) -> None:
    """The property every baseline has to have, applied to a fitted model."""
    rewritten = FOLD.evaluate.assign(result="A", home_goals=0, away_goals=5)
    model = _quick(name)
    # Approximate, because a parallel fit reorders float addition — see SEED.
    # Four orders of magnitude below anything a real dependency would move.
    assert np.allclose(
        model.forecast(FOLD.train, FOLD.evaluate),
        model.forecast(FOLD.train, rewritten),
        atol=1e-9,
    )


# ---- the ways a wrapper goes silently wrong ---------------------------------


def test_class_order_is_home_draw_away_and_not_alphabetical() -> None:
    """The single most likely silent bug here.

    Fitted on a training half that is overwhelmingly home wins, the model must
    put its mass in column 0. If the labels reached scikit-learn as strings it
    would sort them A, D, H, and the mass would land in column 2 — a forecast
    that still sums to one and is exactly wrong.
    """
    lopsided = FOLD.train.copy()
    # All but a handful. A training half with one class in it is refused by
    # the solver, which would fail this test for the wrong reason.
    lopsided.loc[lopsided.index[:-5], "result"] = "H"
    forecast = _quick("logistic_regression").forecast(lopsided, FOLD.evaluate)
    assert forecast[:, CLASSES.index("H")].mean() > 0.9


def test_each_fold_gets_a_fresh_estimator() -> None:
    """`build` is a callable, not an instance. A reused estimator would carry
    fold 0's fit into fold 1 — a leak with no symptom, because the scores would
    simply come out better than they should."""
    model = _quick("logistic_regression")
    assert model.build() is not model.build()


def test_a_training_half_missing_a_class_still_produces_three_columns() -> None:
    """Impossible on real football, reachable on a slice. The alternative to
    scattering the estimator's columns into a full array is a silent
    misalignment."""
    two_classes = FOLD.train[FOLD.train["result"] != "D"]
    forecast = _quick("logistic_regression").forecast(two_classes, FOLD.evaluate)
    assert forecast.shape[1] == len(CLASSES)
    assert (forecast[:, CLASSES.index("D")] == 0).all()


def test_a_model_reads_only_the_columns_it_was_given() -> None:
    """What the ablation depends on. A model that quietly read all thirty
    would report every block as worthless."""
    blinded = _quick("logistic_regression", without("elo", "dixon_coles"))
    ratings_removed = FOLD.evaluate.assign(elo_expected_home=0.5, dc_prob_home=0.5)
    assert np.array_equal(
        blinded.forecast(FOLD.train, FOLD.evaluate),
        blinded.forecast(FOLD.train, ratings_removed),
    )


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
def test_a_boosted_family_reads_a_null_directly(name: str) -> None:
    """No imputer in their pipelines, on purpose: "this club has no five-match
    form yet" is information, and a median standing in for it is not."""
    warming = FOLD.evaluate.copy()
    warming["home_form_points_5"] = pd.NA
    forecast = _quick(name).forecast(FOLD.train, warming)
    assert not np.isnan(forecast).any()


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "mlp"])
def test_a_family_that_cannot_read_a_null_imputes_one(name: str) -> None:
    warming = FOLD.evaluate.copy()
    warming["home_form_points_5"] = pd.NA
    assert not np.isnan(_quick(name).forecast(FOLD.train, warming)).any()


def test_a_run_is_reproducible() -> None:
    """One seed, everywhere. A zoo whose ordering changes between runs cannot
    be ablated: the difference a block makes would have to beat the difference
    the seed makes before anyone could see it."""
    linear = _quick("logistic_regression")
    assert np.array_equal(
        linear.forecast(FOLD.train, FOLD.evaluate),
        linear.forecast(FOLD.train, FOLD.evaluate),
    )
    # The forest fits on every core, and a parallel sum reorders float
    # addition, so its guarantee is 1e-9 rather than bit for bit.
    forest = _quick("random_forest")
    assert np.allclose(
        forest.forecast(FOLD.train, FOLD.evaluate),
        forest.forecast(FOLD.train, FOLD.evaluate),
        atol=1e-9,
    )
    assert SEED == 20260904


# ---- building one -----------------------------------------------------------


def test_a_model_carries_the_settings_it_was_built_with() -> None:
    """So a score in a report can be traced to what produced it."""
    assert build("xgboost", max_depth=3).parameters["max_depth"] == 3


def test_an_override_leaves_the_other_settings_alone() -> None:
    built = build("xgboost", max_depth=3)
    assert built.parameters["n_estimators"] == build("xgboost").parameters["n_estimators"]


def test_asking_for_a_model_the_zoo_does_not_hold_is_refused() -> None:
    with pytest.raises(ModelError, match="not a model"):
        build("transformer")


def test_the_default_zoo_is_every_family_in_reporting_order() -> None:
    assert [model.name for model in default_zoo()] == list(FAMILIES)
