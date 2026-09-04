"""SHAP values, the block aggregation, and the families it refuses.

The aggregation is tested on arrays rather than on a fitted model: what can go
wrong there is a transposed axis or a block whose columns are summed from the
wrong place, and both are invisible in a table of plausible-looking numbers.
The fitted half is tested once, against the block this synthetic league was
built around.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.shapley import (
    ATTRIBUTION_COLUMNS,
    DEFAULT_SAMPLE,
    TREE_FAMILIES,
    ExplainError,
    _sampled,
    attribution,
    by_block,
    shap_values,
)
from src.models.dataset import BLOCKS, DESIGN_COLUMNS
from src.models.splits import walk_forward
from src.models.zoo import FAMILIES, build
from tests.factories import modelled_frame, season_labels

LEAGUE = modelled_frame(seasons=season_labels(2012, 8), teams=10)
FOLD = next(iter(walk_forward(LEAGUE, folds=2)))

QUICK = {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.3}

TWO_BLOCKS: dict[str, tuple[str, ...]] = {"first": ("a", "b"), "second": ("c",)}


def _values(rows: list[list[list[float]]]) -> np.ndarray:
    """An ``(n, columns, classes)`` array, written out."""
    return np.asarray(rows, dtype=float)


# ---- the aggregation ---------------------------------------------------------


def test_a_block_is_the_sum_of_its_columns() -> None:
    values = _values([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    table = by_block(values, ["a", "b", "c"], blocks=TWO_BLOCKS).set_index("block")
    assert table.loc["first", "shap"] == pytest.approx(3.0)
    assert table.loc["second", "shap"] == pytest.approx(4.0)
    assert table.loc["first", "columns"] == 2


def test_the_magnitude_is_taken_before_anything_is_added_up() -> None:
    """Home form pushes P(home) up and P(away) down by construction. Summed
    signed, a column that dominates the model reports approximately nothing."""
    cancelling = _values([[[3.0, 0.0, -3.0]]])
    assert by_block(cancelling, ["a"], blocks={"only": ("a",)}).loc[0, "shap"] == pytest.approx(6.0)


def test_contributions_are_averaged_over_matches_not_summed() -> None:
    """Otherwise the numbers scale with the sample and two runs at different
    sample sizes cannot be compared."""
    one = _values([[[2.0, 0.0, 0.0]]])
    ten = _values([[[2.0, 0.0, 0.0]]] * 10)
    single = by_block(one, ["a"], blocks={"only": ("a",)}).loc[0, "shap"]
    many = by_block(ten, ["a"], blocks={"only": ("a",)}).loc[0, "shap"]
    assert single == pytest.approx(many)


def test_the_share_is_of_the_whole_and_sums_to_one() -> None:
    """The column worth reading. The raw magnitudes are log-odds units that
    mean nothing beside another model's."""
    values = _values([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    table = by_block(values, ["a", "b", "c"], blocks=TWO_BLOCKS)
    assert table["share"].sum() == pytest.approx(1.0)
    assert table.loc[0, "share"] == pytest.approx(0.5)


def test_the_table_is_sorted_by_what_the_model_leans_on() -> None:
    values = _values([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    table = by_block(values, ["a", "b", "c"], blocks=TWO_BLOCKS)
    assert list(table["block"]) == ["second", "first"]
    assert list(table.columns) == list(ATTRIBUTION_COLUMNS)


def test_a_block_whose_columns_are_absent_is_left_out() -> None:
    """Not reported as worthless: the model was never shown it."""
    values = _values([[[1.0, 0.0, 0.0]]])
    table = by_block(values, ["a"], blocks=TWO_BLOCKS)
    assert list(table["block"]) == ["first"]


def test_no_block_matching_anything_is_an_empty_table() -> None:
    values = _values([[[1.0, 0.0, 0.0]]])
    table = by_block(values, ["a"], blocks={"elsewhere": ("z",)})
    assert table.empty
    assert list(table.columns) == list(ATTRIBUTION_COLUMNS)


def test_an_array_that_does_not_match_the_columns_is_refused() -> None:
    """The transposed axis. A (n, classes, columns) array aggregates happily
    into numbers that are wrong and look fine."""
    with pytest.raises(ExplainError, match="expected"):
        by_block(_values([[[1.0, 0.0, 0.0]]]), ["a", "b"], blocks=TWO_BLOCKS)


# ---- the sample --------------------------------------------------------------


def test_a_small_frame_is_explained_whole() -> None:
    assert len(_sampled(FOLD.evaluate, DEFAULT_SAMPLE)) == len(FOLD.evaluate)


def test_a_large_frame_is_thinned_by_stride_and_spans_the_window() -> None:
    """Every nth row rather than a random draw: deterministic without a seed to
    remember, and it does not over-weight whichever weeks a shuffle picked. A
    stride can only undershoot the cap, never exceed it."""
    thinned = _sampled(FOLD.evaluate, 20)
    assert 0 < len(thinned) <= 20
    assert thinned["date"].min() == FOLD.evaluate["date"].min()
    assert thinned["date"].max() > thinned["date"].median()


def test_asking_for_nothing_explains_everything() -> None:
    """A sample of zero is "no cap", not "no matches" — the latter is a table
    of nulls where a report should be."""
    assert len(_sampled(FOLD.evaluate, 0)) == len(FOLD.evaluate)


# ---- against a fitted model --------------------------------------------------


def test_the_three_boosted_families_are_the_ones_with_a_tree_to_read() -> None:
    assert set(TREE_FAMILIES) < set(FAMILIES)
    assert set(TREE_FAMILIES) == {"xgboost", "lightgbm", "catboost"}


def test_it_finds_the_block_the_league_was_built_around() -> None:
    model = build("lightgbm", **QUICK)
    table = attribution(model, FOLD.train, FOLD.evaluate, sample=200).set_index("block")
    assert table.index[0] == "elo"
    assert table.loc["elo", "share"] > 0.5


def test_every_design_column_is_accounted_for() -> None:
    """A block missing from the aggregation is contribution silently dropped,
    and the share column would still sum to one."""
    model = build("lightgbm", **QUICK)
    values = shap_values(model, FOLD.train, FOLD.evaluate, sample=100)
    assert values.shape[1] == len(DESIGN_COLUMNS)
    table = by_block(values, model.columns)
    assert set(table["block"]) <= set(BLOCKS)
    assert table["columns"].sum() == len(DESIGN_COLUMNS)


def test_a_family_with_no_tree_to_read_is_refused_by_name() -> None:
    """Rather than by a stack trace from inside the explainer, and with the
    method that does cover it named in the message."""
    with pytest.raises(ExplainError, match="permutation"):
        shap_values(build("logistic_regression"), FOLD.train, FOLD.evaluate)


def test_the_same_matches_give_the_same_answer() -> None:
    model = build("lightgbm", **QUICK)
    first = attribution(model, FOLD.train, FOLD.evaluate, sample=100)
    second = attribution(model, FOLD.train, FOLD.evaluate, sample=100)
    pd.testing.assert_frame_equal(first, second)
