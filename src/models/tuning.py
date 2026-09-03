"""Hyperparameters chosen by search, on matches the report never sees.

The obvious way to tune is to optimise the number the report prints. It is also
the way to publish a number that will not survive contact with next season:
choosing settings by looking at the folds they are then scored on is the same
mistake as fitting a rating on the matches it prices, one level up. This
project has already refused that once — per-competition Elo constants were
built, measured and removed for it — so the search here runs on a **tuning
slice**: every match strictly earlier than the first reported fold.

Within that slice the arrangement is the same walk-forward one, because a
hyperparameter that only works when the training set is the whole history is a
hyperparameter that will not work in the report either.

The winners are then written into `src/models/zoo.py` as documented constants,
the way Dixon-Coles' decay and window are. A tuned parameter living in a JSON
file beside the data is a parameter that is regenerated on one machine and
stale everywhere else.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from src.evaluation.metrics import terms
from src.models.splits import DEFAULT_FOLDS, DEFAULT_HORIZON_DAYS, boundaries, walk_forward
from src.models.zoo import FAMILIES, ModelError, build
from src.utils.logging import get_logger

logger = get_logger(__name__)

TUNING_FOLDS = 3
"""Walk-forward steps inside the tuning slice.

Three rather than five: the slice is what is left after the reported folds are
carved off the end, and a search is a few dozen fits per fold. Two would let a
single unusual season pick the settings; five costs two-thirds more for a
ranking that does not change.
"""

DEFAULT_TRIALS = 20


@dataclass(frozen=True, slots=True)
class TuningResult:
    """What one search found."""

    model: str
    trials: int
    best_value: float
    best_parameters: dict[str, Any]
    baseline_value: float
    """The shipped settings, scored on the same folds. A search that cannot
    beat the constant it is replacing has found nothing, and saying so needs
    both numbers."""

    @property
    def improvement(self) -> float:
        return self.baseline_value - self.best_value

    def summary(self) -> str:
        return (
            f"{self.model}: {self.best_value:.4f} over {self.trials} trials "
            f"({self.improvement:+.4f} against the shipped {self.baseline_value:.4f})"
        )


def tuning_slice(
    matches: pd.DataFrame,
    *,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Every match strictly before the first fold the report scores.

    The one line that makes this milestone's numbers honest. Everything the
    search touches is on this side of it.
    """
    first = boundaries(matches, folds=folds, horizon_days=horizon_days)[0]
    return matches[matches["date"] < first]


def walk_forward_log_loss(
    matches: pd.DataFrame,
    model: str,
    parameters: dict[str, Any],
    *,
    folds: int = TUNING_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> float:
    """Pooled log loss for one set of settings, over the tuning folds.

    Pooled by match rather than by fold: log loss is a mean, so concatenating
    the per-match terms and taking one mean is what a single pass over the
    union would give. Averaging fold averages would let a short fold count as
    much as a long one.
    """
    forecaster = build(model, **parameters)
    contributions = [
        terms(forecaster.forecast(fold.train, fold.evaluate), fold.evaluate["result"])["log_loss"]
        for fold in walk_forward(matches, folds=folds, horizon_days=horizon_days)
    ]
    return float(pd.concat(contributions, ignore_index=True).mean())


# -- Search spaces -------------------------------------------------------------
#
# Deliberately narrow. A space wide enough to contain an absurd model spends
# most of its trials proving the model is absurd, and a trial here is three
# fits over most of the history.
#
# The tree spaces were narrowed once, after measurement rather than by taste.
# LightGBM's first version reached 255 leaves and 800 estimators, and a single
# trial in that corner took longer than the entire XGBoost search — for a model
# nobody would ship on thirty tabular columns. The bounds below are the region
# a person would actually consider, which is where the trials are worth
# spending. Every reported search used the space as it stands here.

Space = Callable[[optuna.Trial], dict[str, Any]]


def _logistic_regression(trial: optuna.Trial) -> dict[str, Any]:
    return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


def _random_forest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 400, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
    }


def _xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
    }


def _lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
    }


def _catboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


MLP_SHAPES: dict[str, tuple[int, ...]] = {
    "32": (32,),
    "64": (64,),
    "64-32": (64, 32),
    "128-64": (128, 64),
}
"""Hidden layer stacks, named. Optuna's categorical space holds scalars, and a
label is more readable in a trial log than a tuple would have been anyway."""


def _mlp(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "hidden_layer_sizes": MLP_SHAPES[
            trial.suggest_categorical("hidden_layer_sizes", list(MLP_SHAPES))
        ],
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


SPACES: dict[str, Space] = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
    "mlp": _mlp,
}


def tune(
    matches: pd.DataFrame,
    model: str,
    *,
    trials: int = DEFAULT_TRIALS,
    folds: int = TUNING_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    seed: int = 0,
) -> TuningResult:
    """Search one model's space over the tuning folds.

    Args:
        matches: The **tuning slice**, not the full table. Passing the full
            table would search over the matches the report scores, which is the
            one thing this module exists to avoid — so the caller does the
            slicing and this signature cannot hide it.
        model: A name from the zoo.
        trials: How many settings to try.
        folds: Walk-forward steps inside the slice.
        horizon_days: Days per step.
        seed: The sampler's seed, so a search is reproducible.

    Raises:
        ModelError: On a model the zoo does not hold.
    """
    if model not in SPACES:
        raise ModelError(f"no search space for {model!r}; the zoo holds {sorted(FAMILIES)}")

    def objective(trial: optuna.Trial) -> float:
        parameters = SPACES[model](trial)
        # Stashed on the trial rather than read back from `study.best_params`,
        # which reports what was *suggested*: the MLP suggests the label
        # "64-32" and the estimator needs the tuple it names. Reading the
        # suggestions back would produce settings the zoo cannot be built with.
        trial.set_user_attr("parameters", parameters)
        value = walk_forward_log_loss(
            matches, model, parameters, folds=folds, horizon_days=horizon_days
        )
        logger.info("%s trial %d: %.4f %s", model, trial.number, value, parameters)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)

    baseline = walk_forward_log_loss(matches, model, {}, folds=folds, horizon_days=horizon_days)
    return TuningResult(
        model=model,
        trials=trials,
        best_value=float(study.best_value),
        best_parameters=dict(study.best_trial.user_attrs["parameters"]),
        baseline_value=baseline,
    )


def tune_all(
    matches: pd.DataFrame,
    models: Sequence[str] | None = None,
    **kwargs: Any,
) -> tuple[TuningResult, ...]:
    """Search every model's space, worst improvement last."""
    chosen = models if models is not None else tuple(SPACES)
    found = tuple(tune(matches, model, **kwargs) for model in chosen)
    return tuple(sorted(found, key=lambda result: -result.improvement))


def unsearched() -> tuple[str, ...]:
    """Models in the zoo with no search space. Empty, or the tuner skips one."""
    return tuple(sorted(set(FAMILIES) - set(SPACES)))


def as_constants(result: TuningResult) -> str:
    """The winning settings, formatted for pasting into ``zoo.py``.

    Printed rather than written. A tuned constant that changes without a diff
    is a constant nobody reviewed, and the review is the point: several of
    these searches find an improvement too small to be worth the runtime it
    costs, and only a person reading both numbers can say so.
    """
    lines = [f"# {result.summary()}", f"{result.model.upper()}: dict[str, Any] = {{"]
    for key, value in sorted(result.best_parameters.items()):
        rendered = (
            f"{value:.4g}"
            if isinstance(value, float) and not float(value).is_integer()
            else repr(value)
        )
        lines.append(f'    "{key}": {rendered},')
    lines.append("}")
    return "\n".join(lines)
