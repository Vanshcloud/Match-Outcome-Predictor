"""Break a block, re-score, and see what the forecast lost.

The model-agnostic half of the pair. It needs nothing from the estimator except
the ability to predict, so it covers all six families — including the three
:mod:`src.explainability.shapley` cannot read — and it reports in **log loss**,
which is the unit the ablation reports in and therefore the only one of the two
methods whose number can be put beside the ablation's directly.

**A block is shuffled jointly, not column by column.** The fourteen form
columns are strongly correlated with each other; permuting them independently
builds matches that never happened — a side with five wins in five and a goal
difference of minus nine — and measures the model's behaviour on nonsense
rather than the block's worth. Permuting the block's rows as a unit keeps every
within-block relationship and destroys only the one being measured: the link
between this block and *this match*.

**One fit, many predictions.** The estimator depends on the training half,
which no permutation touches, so refitting per repeat would produce the same
model at nineteen times the cost. That is what
:meth:`~src.models.zoo.TrainedForecaster.fit` exists to make sayable.

**Compared with the ablation, not confused with it.** The ablation retrains
without a block and answers "would a model built without this be worse". This
permutes at prediction time and answers "does *this* model use it". A block
that scores low here and high there is one whose information the model can
recover from its neighbours — which is exactly what the ablation measures
between Elo and Dixon-Coles, and is a fact about the feature set rather than a
disagreement between two tools.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import terms
from src.ingestion.base import TARGET_COLUMN
from src.models.dataset import BLOCKS
from src.models.zoo import SEED, TrainedForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REPEATS = 5
"""How many shuffles per block.

The spread between repeats is the number that says whether a block's delta is
real. Five is enough to see a standard deviation an order of magnitude below
the deltas this project reads off, and cheap: a repeat is one prediction.
"""

IMPORTANCE_COLUMNS: tuple[str, ...] = ("block", "columns", "delta_log_loss", "spread")


def _log_loss(forecast: np.ndarray, outcomes: pd.Series) -> float:
    return float(terms(forecast, outcomes)["log_loss"].mean())


def shuffled(
    frame: pd.DataFrame, columns: Sequence[str], generator: np.random.Generator
) -> pd.DataFrame:
    """A copy of ``frame`` with ``columns`` moved to other matches, together.

    The rows of the block are reordered as a unit, so each match gets some
    other match's whole block — a real combination of values attached to the
    wrong fixture, which is the thing being measured.

    Reindexed as a frame rather than assigned as an array. A block holding both
    `Float64` and `Int32` columns comes out of ``to_numpy`` as one object array,
    and assigning that back makes every column in the block object dtype — which
    still scores, because the design matrix casts, and is a different input to
    the intact one it is being compared against.
    """
    broken = frame.copy()
    moved = frame[list(columns)].iloc[generator.permutation(len(frame))]
    moved.index = frame.index
    broken[list(columns)] = moved
    return broken


def importance(
    model: TrainedForecaster,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    blocks: dict[str, tuple[str, ...]] | None = None,
    repeats: int = DEFAULT_REPEATS,
    seed: int = SEED,
) -> pd.DataFrame:
    """What each block is worth to a fitted model, in log loss.

    Args:
        model: A family from the zoo. Fitted once, here.
        train: The matches it is fitted on.
        evaluate: The matches it is scored on, and whose blocks are shuffled.
        blocks: Block name to columns. Defaults to the registry's own groups.
        repeats: Shuffles per block.
        seed: The generator's seed, so a report is reproducible.

    Returns:
        One row per block: the mean rise in log loss when it is broken, and the
        standard deviation across repeats. Positive is a block the model uses.
    """
    grouped = blocks if blocks is not None else BLOCKS
    generator = np.random.default_rng(seed)
    estimator = model.fit(train)
    outcomes = evaluate[TARGET_COLUMN]
    intact = _log_loss(model.predict(estimator, evaluate), outcomes)
    logger.info("%s scores %.4f intact over %d matches", model.name, intact, len(evaluate))

    rows: list[dict[str, object]] = []
    for name, members in grouped.items():
        present = [column for column in members if column in model.columns]
        if not present:
            continue
        losses = [
            _log_loss(model.predict(estimator, shuffled(evaluate, present, generator)), outcomes)
            for _ in range(repeats)
        ]
        rows.append(
            {
                "block": name,
                "columns": len(present),
                "delta_log_loss": float(np.mean(losses) - intact),
                "spread": float(np.std(losses)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(IMPORTANCE_COLUMNS))
    return pd.DataFrame(rows).sort_values("delta_log_loss", ascending=False).reset_index(drop=True)
