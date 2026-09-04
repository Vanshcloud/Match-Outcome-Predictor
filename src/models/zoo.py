"""Six model families, one wrapper, and a fresh fit for every fold.

Each family is a factory returning a :class:`TrainedForecaster` — the same
:class:`~src.models.baselines.Forecaster` shape the baselines already satisfy,
so a model plugs into the walk-forward backtest without that pipeline learning
anything about estimators. Fitting happens inside ``forecast``, on the training
half it is handed, which is what makes the arrangement causal by construction:
a model cannot see a match it was not given, and the split layer decides what
it is given.

**A fresh estimator per fold.** ``build`` is a callable, not an instance. An
instance reused across folds would carry fold 0's fit into fold 1 — a leak with
no symptom, because the scores would simply come out better than they should.

**Class order is fixed by encoding, not by trust.** Targets are indices in
:data:`~src.evaluation.metrics.CLASSES` order, so ``predict_proba`` comes back
in that order too. Handing scikit-learn the labels would sort them
alphabetically — A, D, H — and transpose every forecast into something that
still sums to one and scores plausibly.

**Nulls are handled where the choice is visible.** The gradient-boosted
families read NaN directly and do better for it: "this club has no five-match
form yet" is information. The three that cannot — logistic regression, the
forest, the MLP — get an explicit imputer in their own pipeline rather than a
silently pre-filled matrix.

**The three boosted libraries are imported when their family is built, not when
this module is.** They are separate wheels of a few hundred megabytes each, and
Milestone 11 serves a blend that contains one of them — so importing all three
at module scope made the serving image carry LightGBM and CatBoost to satisfy
an `import` that nothing in a request ever reaches. The failure was not subtle:
the container would not start. scikit-learn stays at the top because three
families and both preprocessing steps need it, so nothing that imports this
module can avoid it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import CLASSES
from src.models.dataset import DESIGN_COLUMNS, design_matrix, targets

SEED = 20260904
"""One seed, everywhere. A model zoo whose ordering changes between runs is a
zoo that cannot be ablated: the difference a feature block makes would have to
be larger than the difference the seed makes before anyone could see it.

It fixes the model, not the last bits of the arithmetic. Every family here fits
on all cores, and a parallel sum reorders floating-point addition, so two runs
of the forest agree to about 1e-9 rather than bit for bit. Single-threading
them would buy exact reproducibility for roughly four times the runtime, and
the smallest difference this project reads off an ablation is 1e-4."""


class Estimator(Protocol):
    """The scikit-learn surface this module actually uses.

    Three members, not the whole API: a Protocol that named everything
    scikit-learn offers would have to be updated whenever scikit-learn was,
    for no gain, and CatBoost would not satisfy it.
    """

    classes_: np.ndarray

    def fit(self, inputs: np.ndarray, outcomes: np.ndarray) -> Any: ...

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TrainedForecaster:
    """A model family, its hyperparameters, and the columns it is shown."""

    name: str
    build: Callable[[], Estimator]
    columns: tuple[str, ...] = DESIGN_COLUMNS
    parameters: Mapping[str, Any] = field(default_factory=dict)
    """What ``build`` was configured with. Carried for the run log, so a score
    in a report can be traced to the settings that produced it."""

    def fit(self, train: pd.DataFrame) -> Estimator:
        """A fresh estimator, fitted on the matches it is handed.

        Public because Milestone 10 needs the fitted thing itself: SHAP reads
        the trees, and permutation importance predicts nineteen times off one
        fit. Both would otherwise refit for every number they report, and both
        would reimplement the two lines below to do it.
        """
        estimator = self.build()
        estimator.fit(design_matrix(train, self.columns), targets(train))
        return estimator

    def predict(self, estimator: Estimator, evaluate: pd.DataFrame) -> np.ndarray:
        """One probability row per match, in :data:`CLASSES` order."""
        predicted = estimator.predict_proba(design_matrix(evaluate, self.columns))

        # Scattered into a full three-column array rather than returned as-is.
        # A training fold missing a class entirely — impossible on real
        # football, reachable on a small slice — produces a narrower matrix,
        # and the alternative to scattering is a silent misalignment.
        forecast = np.zeros((len(evaluate), len(CLASSES)))
        for column, label in enumerate(estimator.classes_):
            forecast[:, int(label)] = predicted[:, column]
        return forecast

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        return self.predict(self.fit(train), evaluate)


# -- The hyperparameters, and where they came from -----------------------------
#
# Every family was searched by `src/models/tuning.py` over folds that end
# *before* the first fold any of these is reported on, so no setting here was
# chosen by looking at the matches it is scored on. Where the search beat the
# value it started from, its winner is below; the log loss it reached, on the
# tuning folds and against what it replaced, is in docs/MODELS.md.
#
# The budgets differ — twenty trials for the two cheapest families, three for
# the most expensive — and are stated rather than implied, because a trial is
# three fits over most of the history and the whole set took about forty
# minutes. `python scripts/train.py --tune <model>` re-derives any of them.
#
# What the six searches bought, in total, is small: four of them moved the
# fourth decimal place. Two did not, and both are worth naming — LightGBM's
# defaults were too coarse, and the MLP's were simply wrong.

LOGISTIC_REGRESSION: dict[str, Any] = {"C": 0.1541, "max_iter": 1000}
RANDOM_FOREST: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 13,
    "min_samples_leaf": 60,
}
XGBOOST: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.01927,
    "subsample": 0.6988,
    "colsample_bytree": 0.6981,
    "min_child_weight": 25,
    "reg_lambda": 0.06868,
}
LIGHTGBM: dict[str, Any] = {
    "n_estimators": 400,
    "num_leaves": 27,
    "learning_rate": 0.02209,
    "min_child_samples": 101,
    "subsample": 0.7825,
    # LightGBM ignores `subsample` unless it is told how often to resample.
    # Left at the default of 0, the parameter above would be silently inert and
    # the tuner would spend trials discovering that it makes no difference.
    "subsample_freq": 1,
    "colsample_bytree": 0.8274,
    "reg_lambda": 0.01139,
}
CATBOOST: dict[str, Any] = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.02209,
    "l2_leaf_reg": 5.946,
}
MLP: dict[str, Any] = {
    # One hidden layer, not two. The search preferred it by 0.0144 of log loss,
    # which is the largest thing tuning bought anywhere in this zoo and a fair
    # measure of how wrong an untested guess at an architecture can be.
    "hidden_layer_sizes": (64,),
    "alpha": 1.205e-05,
    "learning_rate_init": 0.004626,
    "max_iter": 200,
}


def _scaled(estimator: Estimator) -> Pipeline:
    """Impute, scale, then fit. For the families that cannot read a null.

    Median rather than mean: a form column's nulls are warm-up rows, and the
    mean of a skewed warm-up distribution moves with the size of the slice.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("estimate", estimator),
        ]
    )


def logistic_regression(
    columns: Sequence[str] = DESIGN_COLUMNS, **overrides: Any
) -> TrainedForecaster:
    """Multinomial logistic regression. The model that says whether the
    features are linearly separable at all, and the one to beat before any
    ensemble is worth its build time."""
    settings = {**LOGISTIC_REGRESSION, **overrides}
    return TrainedForecaster(
        name="logistic_regression",
        build=lambda: _scaled(LogisticRegression(random_state=SEED, **settings)),
        columns=tuple(columns),
        parameters=settings,
    )


def random_forest(columns: Sequence[str] = DESIGN_COLUMNS, **overrides: Any) -> TrainedForecaster:
    """Bagged trees. Deep enough to interact, shallow enough not to memorise."""
    settings = {**RANDOM_FOREST, **overrides}
    return TrainedForecaster(
        name="random_forest",
        build=lambda: _scaled(RandomForestClassifier(random_state=SEED, n_jobs=-1, **settings)),
        columns=tuple(columns),
        parameters=settings,
    )


def xgboost(columns: Sequence[str] = DESIGN_COLUMNS, **overrides: Any) -> TrainedForecaster:
    from xgboost import XGBClassifier

    settings = {**XGBOOST, **overrides}
    return TrainedForecaster(
        name="xgboost",
        build=lambda: XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            **settings,
        ),
        columns=tuple(columns),
        parameters=settings,
    )


def lightgbm(columns: Sequence[str] = DESIGN_COLUMNS, **overrides: Any) -> TrainedForecaster:
    from lightgbm import LGBMClassifier

    settings = {**LIGHTGBM, **overrides}
    return TrainedForecaster(
        name="lightgbm",
        build=lambda: LGBMClassifier(
            objective="multiclass",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
            **settings,
        ),
        columns=tuple(columns),
        parameters=settings,
    )


def catboost(columns: Sequence[str] = DESIGN_COLUMNS, **overrides: Any) -> TrainedForecaster:
    from catboost import CatBoostClassifier

    settings = {**CATBOOST, **overrides}
    return TrainedForecaster(
        name="catboost",
        build=lambda: CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=SEED,
            verbose=False,
            # Off, or every fit drops a catboost_info/ directory wherever the
            # command happened to be run from.
            allow_writing_files=False,
            **settings,
        ),
        columns=tuple(columns),
        parameters=settings,
    )


def mlp(columns: Sequence[str] = DESIGN_COLUMNS, **overrides: Any) -> TrainedForecaster:
    """One hidden stack. Present because the scope asks whether a neural net
    beats a boosted tree on thirty tabular columns, which is a question with a
    well-known answer and is worth measuring here rather than citing."""
    settings = {**MLP, **overrides}
    return TrainedForecaster(
        name="mlp",
        build=lambda: _scaled(MLPClassifier(random_state=SEED, **settings)),
        columns=tuple(columns),
        parameters=settings,
    )


FAMILIES: dict[str, Callable[..., TrainedForecaster]] = {
    "logistic_regression": logistic_regression,
    "random_forest": random_forest,
    "xgboost": xgboost,
    "lightgbm": lightgbm,
    "catboost": catboost,
    "mlp": mlp,
}


class ModelError(ValueError):
    """A model was asked for that this zoo does not contain."""


def build(
    name: str, columns: Sequence[str] = DESIGN_COLUMNS, **overrides: Any
) -> TrainedForecaster:
    """One model by name, with its documented settings or overrides of them."""
    if name not in FAMILIES:
        raise ModelError(f"not a model: {name!r}; the zoo holds {sorted(FAMILIES)}")
    return FAMILIES[name](columns, **overrides)


def default_zoo(columns: Sequence[str] = DESIGN_COLUMNS) -> tuple[TrainedForecaster, ...]:
    """Every family, in the order they are reported."""
    return tuple(build(name, columns) for name in FAMILIES)
