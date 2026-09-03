"""The search, and the one line that keeps its results honest.

Most of this file is about `tuning_slice`. Everything else here is plumbing —
Optuna works — but a search that quietly ran over the matches the report scores
would produce settings that look excellent and generalise to nothing, and no
downstream number would show it.
"""

from __future__ import annotations

import optuna
import pytest

from src.evaluation.metrics import terms
from src.models.splits import DEFAULT_FOLDS, boundaries, walk_forward
from src.models.tuning import (
    MLP_SHAPES,
    SPACES,
    TuningResult,
    as_constants,
    tune,
    tune_all,
    tuning_slice,
    unsearched,
    walk_forward_log_loss,
)
from src.models.zoo import FAMILIES, ModelError, build
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

LEAGUE = modelled_frame(seasons=season_labels(2010, 12), teams=14)


# ---- the honesty property ---------------------------------------------------


def test_the_tuning_slice_ends_before_the_first_reported_fold() -> None:
    """The line that makes this milestone's numbers mean something."""
    first = next(iter(walk_forward(LEAGUE)))
    assert tuning_slice(LEAGUE)["date"].max() < first.evaluate["date"].min()


def test_no_reported_match_is_in_the_tuning_slice() -> None:
    tuned = set(tuning_slice(LEAGUE)["match_id"])
    for fold in walk_forward(LEAGUE):
        assert not tuned & set(fold.evaluate["match_id"])


def test_the_slice_is_everything_before_the_first_boundary() -> None:
    cut = boundaries(LEAGUE, folds=DEFAULT_FOLDS)[0]
    assert len(tuning_slice(LEAGUE)) == int((LEAGUE["date"] < cut).sum())


def test_a_wider_report_leaves_a_narrower_slice() -> None:
    """More reported folds eat further back into the history, and the search
    has to give that ground up rather than reuse it."""
    assert len(tuning_slice(LEAGUE, folds=8)) < len(tuning_slice(LEAGUE, folds=3))


# ---- the objective ----------------------------------------------------------


def test_the_objective_pools_by_match_not_by_fold() -> None:
    """Log loss is a mean, so one mean over the union is the right answer; an
    average of fold averages would let a short fold count as much as a long
    one."""
    slice_ = tuning_slice(LEAGUE)
    forecaster = build("logistic_regression")
    contributions = [
        terms(forecaster.forecast(fold.train, fold.evaluate), fold.evaluate["result"])["log_loss"]
        for fold in walk_forward(slice_, folds=2)
    ]
    pooled = sum(part.sum() for part in contributions) / sum(len(part) for part in contributions)
    assert walk_forward_log_loss(slice_, "logistic_regression", {}, folds=2) == pytest.approx(
        pooled
    )


def test_the_objective_reads_the_settings_it_is_given() -> None:
    slice_ = tuning_slice(LEAGUE)
    weak = walk_forward_log_loss(slice_, "logistic_regression", {"C": 1e-6}, folds=2)
    strong = walk_forward_log_loss(slice_, "logistic_regression", {"C": 1.0}, folds=2)
    assert weak != strong


# ---- the search -------------------------------------------------------------


def test_a_search_reports_both_numbers() -> None:
    """A search that cannot beat the constant it is replacing has found
    nothing, and saying so needs the shipped value beside the best one."""
    result = tune(tuning_slice(LEAGUE), "logistic_regression", trials=2, folds=2)
    assert result.trials == 2
    assert result.improvement == pytest.approx(result.baseline_value - result.best_value)
    assert "logistic_regression" in result.summary()


def test_the_best_settings_can_be_built_with() -> None:
    result = tune(tuning_slice(LEAGUE), "logistic_regression", trials=2, folds=2)
    assert build("logistic_regression", **result.best_parameters).parameters["C"] > 0


def test_a_searched_label_is_turned_back_into_what_the_estimator_needs() -> None:
    """Optuna's categorical space holds scalars, so the MLP suggests "64-32"
    and the estimator needs the tuple it names. Reading `study.best_params`
    back would produce settings the zoo cannot be built with."""
    result = tune(tuning_slice(LEAGUE), "mlp", trials=1, folds=2)
    assert result.best_parameters["hidden_layer_sizes"] in MLP_SHAPES.values()
    assert build("mlp", **result.best_parameters)


def test_every_model_in_the_zoo_has_a_search_space() -> None:
    """Or the tuner skips one silently, and its constants never improve."""
    assert unsearched() == ()
    assert set(SPACES) == set(FAMILIES)


def test_searching_a_model_the_zoo_does_not_hold_is_refused() -> None:
    with pytest.raises(ModelError, match="no search space"):
        tune(LEAGUE, "transformer", trials=1)


# ---- handing the result over ------------------------------------------------


def test_the_constants_are_printed_for_a_person_to_paste() -> None:
    """Printed rather than written. Several of these searches find an
    improvement too small to be worth its runtime, and only someone reading
    both numbers can say so."""
    rendered = as_constants(
        TuningResult(
            model="xgboost",
            trials=20,
            best_value=1.0213,
            best_parameters={"max_depth": 5, "learning_rate": 0.019267},
            baseline_value=1.0224,
        )
    )
    assert "XGBOOST: dict[str, Any] = {" in rendered
    assert '"max_depth": 5,' in rendered
    assert '"learning_rate": 0.01927,' in rendered
    assert "+0.0011" in rendered


# ---- the search spaces ------------------------------------------------------


@pytest.mark.parametrize("model", sorted(SPACES))
def test_every_search_space_suggests_settings_the_estimator_accepts(model: str) -> None:
    """Drawn from a real sampler and built with, but never fitted.

    This is where a typo in a hyperparameter name is caught. Left to the search
    itself, the same typo surfaces twenty minutes into a tuning run as an
    exception from inside a library, after the folds have been split.
    """
    trial = optuna.create_study().ask()
    suggested = SPACES[model](trial)
    assert suggested
    assert build(model, **suggested).parameters.keys() >= suggested.keys()


def test_searching_every_model_ranks_them_by_what_the_search_bought() -> None:
    """Worst improvement last, so the models whose constants are worth changing
    are the ones at the top of the printout."""
    slice_ = tuning_slice(LEAGUE)
    found = tune_all(slice_, ["logistic_regression", "mlp"], trials=1, folds=2)
    assert [result.model for result in found] == sorted(
        ["logistic_regression", "mlp"],
        key=lambda name: -next(r.improvement for r in found if r.model == name),
    )
    assert len(found) == 2
