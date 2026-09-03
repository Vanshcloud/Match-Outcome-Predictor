"""What a model is allowed to see, and how it is grouped for the ablation.

Thirty columns: the twenty features from Milestone 5 and the ten rating
columns from Milestone 4. Nothing canonical is passed through directly — not
the scoreline, obviously, and **not the odds**, which are pre-match and still
withheld because they are the benchmark this project measures itself against.
A model given the closing line learns to copy the bookmaker.

The column list is *derived* from the two registries rather than written out
again. A feature added in Milestone 5's registry is a column the zoo sees
without anyone editing this file, and a column that stops existing stops being
requested — which is the failure a hand-maintained list produces a milestone
later, as a table of nulls nobody notices.

Blocks are what the ablation switches off. They come from the same place: the
feature registry already groups its features, and the ratings schema already
separates Elo from Dixon-Coles. Naming a block here that no registry defines is
not possible, which is the point.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import CLASSES
from src.feature_engineering.registry import FEATURES
from src.ingestion.base import TARGET_COLUMN
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS

ELO_BLOCK = "elo"
DIXON_COLES_BLOCK = "dixon_coles"


def _feature_blocks() -> dict[str, tuple[str, ...]]:
    """The feature registry's own groups, plus one block per rating model."""
    blocks: dict[str, list[str]] = {}
    for feature in FEATURES:
        blocks.setdefault(feature.group, []).append(feature.name)
    return {
        **{name: tuple(columns) for name, columns in blocks.items()},
        ELO_BLOCK: tuple(ELO_COLUMNS),
        DIXON_COLES_BLOCK: tuple(DIXON_COLES_COLUMNS),
    }


BLOCKS: dict[str, tuple[str, ...]] = _feature_blocks()
"""Feature groups, in registry order, then the two rating models."""

DESIGN_COLUMNS: tuple[str, ...] = tuple(column for columns in BLOCKS.values() for column in columns)
"""Every column a model sees, in block order."""


class DatasetError(ValueError):
    """A design matrix cannot be built from what was given."""


def columns_for(blocks: Sequence[str] | None = None) -> tuple[str, ...]:
    """The columns belonging to ``blocks``, or every column.

    Raises:
        DatasetError: On a block no registry defines. An ablation that silently
            dropped nothing would report that the missing block was worthless.
    """
    if blocks is None:
        return DESIGN_COLUMNS
    unknown = sorted(set(blocks) - set(BLOCKS))
    if unknown:
        raise DatasetError(f"not feature blocks: {unknown}; known are {sorted(BLOCKS)}")
    chosen = set(blocks)
    return tuple(column for name, columns in BLOCKS.items() for column in columns if name in chosen)


def without(*dropped: str) -> tuple[str, ...]:
    """Every block except the named ones. The ablation's one operation."""
    return columns_for([name for name in BLOCKS if name not in set(dropped)])


def design_matrix(frame: pd.DataFrame, columns: Sequence[str] = DESIGN_COLUMNS) -> np.ndarray:
    """The model's inputs as a float array, nulls preserved as NaN.

    Preserved rather than filled: a null here means "no history yet", and the
    models that can read that directly — every gradient-boosted one — do better
    with it than with a median standing in for it. The ones that cannot get an
    imputer in their own pipeline, where the choice is visible.

    Raises:
        DatasetError: If a requested column is absent. A model quietly trained
            on twenty-five columns because the ratings were not joined is a
            model whose score means nothing.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DatasetError(f"the table is missing {len(missing)} model column(s): {missing[:5]}")
    # Through `Float64` rather than straight to a float array. A column that
    # arrived from a join as object dtype — an all-null block is the usual way
    # — makes the direct conversion fail deep inside pandas' block manager,
    # with a message naming neither the column nor this project.
    return frame[list(columns)].astype("Float64").to_numpy(dtype=float, na_value=np.nan)


def targets(frame: pd.DataFrame) -> np.ndarray:
    """Outcomes as class indices in :data:`~src.evaluation.metrics.CLASSES` order.

    Indices rather than the labels themselves, so an estimator's ``classes_``
    comes back as ``[0, 1, 2]`` and its ``predict_proba`` columns are already in
    the order the metrics expect. Handing scikit-learn the strings would sort
    them alphabetically — A, D, H — and silently transpose every forecast.
    """
    order = {label: index for index, label in enumerate(CLASSES)}
    encoded = frame[TARGET_COLUMN].map(order)
    if encoded.isna().any():
        raise DatasetError("a match has no usable result; it cannot be trained on")
    return encoded.to_numpy(dtype=int)
