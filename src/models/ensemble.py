"""An average of the families whose mistakes are not the same mistakes.

An ensemble is worth building when its members are wrong about different
matches. Averaging four models that fail together produces a fifth model that
fails on exactly those matches with slightly less confidence, and reports a
number that says more about the averaging than about the models — which is why
Milestone 8 deliberately did not build one on top of a ranking whose top four
sat within 0.0003 of each other.

So the members are chosen on **the correlation of their per-match errors**, not
on their individual scores. The scores decide the order candidates are
considered in and nothing else; what admits a candidate is that its log loss
per match moves differently from every model already in the blend.

**Measured where the report cannot see it.** The correlations are computed on
:func:`~src.models.tuning.tuning_slice` — every match strictly earlier than the
first reported fold — for the same reason the hyperparameters are. Choosing
members by looking at errors on the folds they are then scored on is selection
on the test set with an extra step, and it is the mistake this project has
already built and removed one thing for.

**A plain mean of probabilities.** Not a fitted weight per member: with the
candidates this close together, weights fitted on any holdout small enough to
be honest are noise with three decimal places. Not a log pool either, which
would sharpen a blend that measurement says is already slightly too confident —
the fitted temperatures over the reported folds run from 0.99 to 1.09, all but
one of them above one.

What it is worth, measured: 1.0156 against the best single family's 1.0159, on
the same 59,001 matches. Real, repeatable, and about a fiftieth of what the
model layer bought over the rating it replaced. docs/MODELS.md says so in those
words, because a milestone whose honest answer is "barely" is worth more
written down than rerun by the next person who assumes otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import terms
from src.ingestion.base import TARGET_COLUMN
from src.models.baselines import Forecaster
from src.models.dataset import DESIGN_COLUMNS
from src.models.splits import DEFAULT_FOLDS, DEFAULT_HORIZON_DAYS, walk_forward
from src.models.zoo import build
from src.utils.logging import get_logger

logger = get_logger(__name__)

FORECAST_COLUMNS: tuple[str, str, str] = ("prob_home", "prob_draw", "prob_away")
"""The three probabilities, in :data:`~src.evaluation.metrics.CLASSES` order."""

MAX_ERROR_CORRELATION = 0.99
"""How alike two members' mistakes may be.

Read off a gap in the measured matrix rather than chosen in advance. Every pair
in this zoo correlates above 0.91 — they read the same thirty columns about the
same matches, so they agree about which fixtures are hard — but the pairs are
not evenly spread: the four tree-based families sit at 0.9934 to 0.9960 against
each other, logistic regression at 0.9857 against the nearest of them, and the
MLP at 0.92 against everything. The threshold sits in the empty band between
0.9857 and 0.9934, so it separates the substitutes from the rest and nothing
else. `python scripts/train.py --correlations` reprints the matrix;
docs/MODELS.md holds it.
"""

BEST = "xgboost"
"""The family the calibration layer is measured over.

Chosen on the **tuning slice**, where it is the best of the six, and not on the
reported folds, where CatBoost leads it by 0.0002. That difference is smaller
than the one Milestone 8 called indistinguishable, and picking the subject of a
measurement by looking at the folds it is then measured on is the mistake this
module's selection rule is arranged to avoid — it would be odd to avoid it for
three members and commit it for one.
"""

MEMBERS: tuple[str, ...] = ("xgboost", "logistic_regression", "mlp")
"""The blend, chosen by :func:`select_members` on the tuning slice.

Not the top three, which would have been three gradient-boosting libraries
whose errors correlate at 0.9915 and up. The rule takes the best of that
cluster and then the two families that disagree with it — logistic regression,
and the MLP, which is the worst single model here and the only one wrong about
different matches. A blend of near-substitutes reports the averaging rather
than the models, which is the reason Milestone 8 refused to build one.
"""


@dataclass(frozen=True, slots=True)
class Ensemble:
    """The mean of several forecasters, as one forecaster.

    A :class:`~src.models.baselines.Forecaster`, so it drops into the unchanged
    backtest beside its own members and the baselines, on the same folds and in
    the same table.

    Nulls propagate: a match one member could not price is a match the ensemble
    does not price either, rather than one priced by whoever happened to have
    an opinion. The alternative is a forecaster whose coverage and whose
    membership both vary by row.
    """

    name: str
    members: tuple[Forecaster, ...]

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        stated = [member.forecast(train, evaluate) for member in self.members]
        return np.mean(stated, axis=0)


def ensemble(
    members: Sequence[str] = MEMBERS,
    columns: Sequence[str] = DESIGN_COLUMNS,
    *,
    name: str = "ensemble",
) -> Ensemble:
    """The blend, built from the zoo by name."""
    return Ensemble(name=name, members=tuple(build(one, columns) for one in members))


def fold_forecasts(
    matches: pd.DataFrame,
    forecasters: Sequence[Forecaster],
    *,
    folds: int = DEFAULT_FOLDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Every forecaster's per-match probabilities over the folds, as a long frame.

    The diagnostic pass. :mod:`src.pipelines.backtest` deliberately persists
    means rather than rows — one score per fold, competition, forecaster and
    subset — which is the right thing for a report and the wrong thing for the
    two questions this milestone asks: whether two models are wrong about the
    same matches, and whether a stated probability happens at the rate it
    states. Both need the matches back.

    Columns: ``fold``, ``match`` (the row's position within its evaluation
    half, which is what makes two forecasters' rows line up),
    ``competition_id``, ``forecaster``, the three probabilities, and the
    outcome. The competition is carried because Milestone 10 asks where a model
    is reliable rather than whether it is, and that is a grouping this frame
    can answer and the scored table cannot — the backtest keeps competition and
    fold, but only as means.
    """
    collected: list[pd.DataFrame] = []
    for fold in walk_forward(matches, folds=folds, horizon_days=horizon_days):
        outcomes = fold.evaluate[TARGET_COLUMN].to_numpy()
        competitions = fold.evaluate["competition_id"].to_numpy()
        for forecaster in forecasters:
            stated = pd.DataFrame(
                forecaster.forecast(fold.train, fold.evaluate), columns=list(FORECAST_COLUMNS)
            )
            stated.insert(0, "fold", fold.index)
            stated.insert(1, "match", np.arange(len(fold.evaluate)))
            stated.insert(2, "competition_id", competitions)
            stated["forecaster"] = forecaster.name
            stated[TARGET_COLUMN] = outcomes
            collected.append(stated)
        logger.info("fold %d: %d forecaster(s) recorded", fold.index, len(forecasters))
    return pd.concat(collected, ignore_index=True)


def _losses(forecasts: pd.DataFrame) -> pd.DataFrame:
    """The priced rows of a forecast frame, with a per-match log loss column."""
    priced = forecasts.dropna(subset=list(FORECAST_COLUMNS))
    scored = terms(priced[list(FORECAST_COLUMNS)].to_numpy(dtype=float), priced[TARGET_COLUMN])
    return priced.assign(log_loss=scored["log_loss"].to_numpy())


def error_correlations(forecasts: pd.DataFrame) -> pd.DataFrame:
    """How alike two forecasters' per-match log losses are, as a square matrix.

    Restricted to the matches every forecaster priced, for the same reason the
    backtest reports a common subset: a correlation computed over two different
    sets of matches is a comparison of two different questions.
    """
    wide = _losses(forecasts).pivot(
        index=["fold", "match"], columns="forecaster", values="log_loss"
    )
    return wide.dropna().corr().rename_axis(None, axis=0).rename_axis(None, axis=1)


def by_log_loss(forecasts: pd.DataFrame) -> tuple[str, ...]:
    """Forecaster names, best first. The order candidates are considered in,
    which is the tie-break between two near-substitutes and not the criterion —
    see :func:`select_members`."""
    pooled = _losses(forecasts).groupby("forecaster")["log_loss"].mean().sort_values()
    return tuple(str(name) for name in pooled.index)


def _correlation(correlations: pd.DataFrame, one: str, other: str) -> float:
    """One cell of the matrix as a float. Through numpy because a `.loc` on two
    labels is typed as every scalar pandas can hold, and a comparison against a
    threshold has to be a comparison of numbers."""
    return float(np.asarray(correlations.loc[one, other], dtype=float))


def select_members(
    correlations: pd.DataFrame,
    order: Sequence[str],
    *,
    maximum: float = MAX_ERROR_CORRELATION,
) -> tuple[str, ...]:
    """Take candidates in ``order``, keeping each that disagrees enough with
    every one already taken.

    ``order`` is the tie-break, not the criterion: it decides which of two
    near-substitutes is kept, and nothing else. Passing it sorted by log loss —
    which is what the caller does — means the best of any correlated group is
    the one that survives, and a family scoring worse on its own is admitted
    whenever it is wrong about different matches.
    """
    chosen: list[str] = []
    for name in order:
        alike = [taken for taken in chosen if _correlation(correlations, name, taken) >= maximum]
        if alike:
            logger.info("%s not admitted: errors correlate with %s", name, ", ".join(alike))
            continue
        chosen.append(name)
    return tuple(chosen)
