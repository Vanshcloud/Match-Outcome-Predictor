"""SHAP values over the trees, aggregated to the blocks the ablation used.

A per-column number is not what anyone reads. Thirty columns, fourteen of them
rolling windows of the same idea, produce a ranking whose top of the list moves
between runs and whose meaning is "form matters" either way. So everything here
reports **per block** — the same five blocks Milestone 8 withheld one at a time
— because that is the grain at which the answer can be checked against a second
method.

**Mean absolute value, summed over the three classes.** SHAP decomposes one
prediction into contributions that sum to it, and those contributions are
signed and per class: home form pushes P(home win) up and P(away win) down by
construction. Averaging them signed gives approximately zero and says nothing;
the magnitude is what "this column moved the forecast" means.

**Trees only, on purpose.** ``TreeExplainer`` reads the model's own structure:
exact, and seconds for a boosted forest. The other three families here are
scikit-learn pipelines around an imputer and a scaler, which it refuses
outright, and the general-purpose explainer that would handle them costs hours
on thirty columns for a number :mod:`src.explainability.permutation` produces
exactly, for every family, in seconds. Where SHAP cannot go, the second method
already goes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.models.dataset import BLOCKS, design_matrix
from src.models.zoo import TrainedForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)

TREE_FAMILIES: tuple[str, ...] = ("xgboost", "lightgbm", "catboost")
"""The families whose estimator is the tree ensemble itself.

The other three wrap theirs in a pipeline, and ``TreeExplainer`` raises on a
pipeline rather than reaching inside it — correctly, since the columns it would
then explain are imputed and scaled versions of the ones named here.
"""

DEFAULT_SAMPLE = 5_000
"""How many evaluation matches to explain.

Every one of 12,000 is affordable and pointless: the mean absolute contribution
of a block over 5,000 matches and over 12,000 agree to the fourth decimal, and
the smaller number keeps the command interactive.
"""

ATTRIBUTION_COLUMNS: tuple[str, ...] = ("block", "columns", "shap", "share")


class ExplainError(ValueError):
    """A model cannot be explained by the method asked for."""


def _sampled(frame: pd.DataFrame, sample: int) -> pd.DataFrame:
    """At most ``sample`` rows, taken as every ``n``th.

    Deterministic without a seed to remember, and it spans the whole evaluation
    window instead of over-weighting whichever weeks a shuffle happened to
    pick — the forecast for a January fixture is built from different columns
    than one in August, when half the form windows are still warming up.

    A stride can only undershoot: 12,000 rows capped at 5,000 is every third,
    which is 4,000. That is the price of not drawing at random, and it is paid
    in matches nobody would have read individually.
    """
    if sample <= 0 or len(frame) <= sample:
        return frame
    return frame.iloc[:: -(-len(frame) // sample)]


def shap_values(
    model: TrainedForecaster,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    sample: int = DEFAULT_SAMPLE,
) -> np.ndarray:
    """Per-match, per-column, per-class contributions as an ``(n, c, 3)`` array.

    Raises:
        ExplainError: On a family ``TreeExplainer`` does not accept.
    """
    if model.name not in TREE_FAMILIES:
        raise ExplainError(
            f"no tree to read in {model.name!r}; SHAP here covers {list(TREE_FAMILIES)}, "
            f"and src/explainability/permutation.py covers every family"
        )
    import shap

    explained = _sampled(evaluate, sample)
    logger.info("explaining %d of %d matches for %s", len(explained), len(evaluate), model.name)
    explainer = shap.TreeExplainer(model.fit(train))
    return np.asarray(explainer.shap_values(design_matrix(explained, model.columns)))


def by_block(
    values: np.ndarray,
    columns: Sequence[str],
    *,
    blocks: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Collapse per-column contributions into one row per block.

    Args:
        values: The ``(n, c, 3)`` array :func:`shap_values` returns.
        columns: The design columns, in the order the array's second axis uses.
        blocks: Block name to columns. Defaults to the registry's own groups.

    Returns:
        One row per block: how many columns it holds, the summed mean absolute
        contribution, and that as a share of the total — which is the column
        worth reading, because the raw magnitudes are in log-odds units that
        mean nothing beside another model's.
    """
    grouped = blocks if blocks is not None else BLOCKS
    if values.ndim != 3 or values.shape[1] != len(columns):
        raise ExplainError(
            f"expected (matches, {len(columns)} columns, classes), got {values.shape}"
        )

    # Mean over matches, sum over classes: one number per column, in the units
    # the model's own arithmetic uses.
    per_column = pd.Series(
        np.abs(values).mean(axis=0).sum(axis=1), index=list(columns), dtype=float
    )
    rows = [
        {
            "block": name,
            "columns": len(present),
            "shap": float(per_column[present].sum()),
        }
        for name, members in grouped.items()
        if (present := [column for column in members if column in per_column.index])
    ]
    if not rows:
        return pd.DataFrame(columns=list(ATTRIBUTION_COLUMNS))

    table = pd.DataFrame(rows)
    table["share"] = table["shap"] / table["shap"].sum()
    return table.sort_values("shap", ascending=False).reset_index(drop=True)


def attribution(
    model: TrainedForecaster,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    sample: int = DEFAULT_SAMPLE,
) -> pd.DataFrame:
    """What each block contributed to this model's forecasts, as a table."""
    return by_block(shap_values(model, train, evaluate, sample=sample), model.columns)
