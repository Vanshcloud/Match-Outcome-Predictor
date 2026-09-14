"""What a model is allowed to see, and how the ablation carves it up.

The assertions here are mostly about *absence*: which canonical columns never
reach an estimator. A design matrix that quietly gained the odds, or the
scoreline, would produce a model that looked extraordinary and was worthless,
and nothing downstream of this module could tell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import CLASSES
from src.feature_engineering.registry import FEATURE_COLUMNS
from src.ingestion.base import BENCHMARK_COLUMNS, POST_MATCH_COLUMNS
from src.models.dataset import (
    BLOCKS,
    DESIGN_COLUMNS,
    DatasetError,
    columns_for,
    design_matrix,
    targets,
    without,
)
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS
from tests.factories import modelled_frame, season_labels

LEAGUE = modelled_frame(seasons=season_labels(2015, 4), teams=10)


# ---- what is in it ----------------------------------------------------------


def test_the_design_matrix_is_the_features_and_the_ratings() -> None:
    assert set(DESIGN_COLUMNS) == set(FEATURE_COLUMNS) | set(ELO_COLUMNS) | set(DIXON_COLES_COLUMNS)
    assert len(DESIGN_COLUMNS) == 30


def test_no_post_match_column_reaches_a_model() -> None:
    """Not the scoreline, not the shots, not the referee. The features read
    those through a lag; a model reading one directly is the leak the whole
    project is arranged against."""
    assert not set(DESIGN_COLUMNS) & POST_MATCH_COLUMNS


def test_the_bookmakers_price_is_not_a_feature() -> None:
    """Pre-match and still withheld: it is the benchmark, and a model trained
    on it learns to copy the bookmaker."""
    assert not set(DESIGN_COLUMNS) & BENCHMARK_COLUMNS


def test_the_columns_are_derived_from_the_registries() -> None:
    """A feature added to the feature registry is a column the zoo sees
    without anyone editing a second list."""
    assert set(BLOCKS) == {"form", "schedule", "head_to_head", "elo", "dixon_coles"}
    assert sum(len(columns) for columns in BLOCKS.values()) == len(DESIGN_COLUMNS)


def test_no_column_belongs_to_two_blocks() -> None:
    """The ablation subtracts blocks. Overlapping ones would subtract less than
    they claim to and report the difference as the block being worthless."""
    named = [column for columns in BLOCKS.values() for column in columns]
    assert len(named) == len(set(named))


# ---- carving it up ----------------------------------------------------------


def test_a_block_can_be_selected() -> None:
    assert columns_for(["elo"]) == tuple(ELO_COLUMNS)


def test_selecting_nothing_selects_everything() -> None:
    assert columns_for(None) == DESIGN_COLUMNS


def test_withholding_a_block_leaves_the_rest_in_order() -> None:
    remaining = without("dixon_coles")
    assert set(remaining) == set(DESIGN_COLUMNS) - set(DIXON_COLES_COLUMNS)
    assert remaining == tuple(c for c in DESIGN_COLUMNS if c in set(remaining))


def test_withholding_two_blocks_removes_both() -> None:
    assert set(without("elo", "dixon_coles")) == set(FEATURE_COLUMNS)


def test_a_block_no_registry_defines_is_refused() -> None:
    """An ablation that silently dropped nothing would report the missing block
    as worthless, which is the wrong conclusion drawn with confidence."""
    with pytest.raises(DatasetError, match="not feature blocks"):
        columns_for(["momentum"])


# ---- building it ------------------------------------------------------------


def test_the_matrix_is_one_row_per_match_and_one_column_per_feature() -> None:
    assert design_matrix(LEAGUE).shape == (len(LEAGUE), len(DESIGN_COLUMNS))


def test_nulls_survive_as_nan_rather_than_being_filled() -> None:
    """The families that can read a null do better with it; the ones that
    cannot impute in their own pipeline, where the choice is visible."""
    warming = LEAGUE.copy()
    warming.loc[warming.index[0], "home_form_points_5"] = pd.NA
    assert np.isnan(design_matrix(warming)[0, DESIGN_COLUMNS.index("home_form_points_5")])


def test_a_table_missing_the_model_columns_is_refused() -> None:
    """A model quietly trained on twenty-five columns because the ratings were
    never joined is a model whose score means nothing."""
    with pytest.raises(DatasetError, match="missing 5 model column"):
        design_matrix(LEAGUE.drop(columns=list(ELO_COLUMNS)))


def test_only_the_requested_columns_are_read() -> None:
    matrix = design_matrix(LEAGUE, without("elo"))
    assert matrix.shape[1] == len(DESIGN_COLUMNS) - len(ELO_COLUMNS)


# ---- the target -------------------------------------------------------------


def test_outcomes_are_encoded_in_class_order() -> None:
    """Indices, so an estimator's `predict_proba` columns are already in the
    order the metrics expect. Handing scikit-learn the labels would sort them
    alphabetically — A, D, H — and transpose every forecast."""
    frame = LEAGUE.head(3).assign(result=list(CLASSES))
    assert list(targets(frame)) == [0, 1, 2]


def test_a_match_with_no_result_cannot_be_trained_on() -> None:
    unplayed = LEAGUE.head(2).copy()
    unplayed.loc[unplayed.index[0], "result"] = pd.NA
    with pytest.raises(DatasetError, match="no usable result"):
        targets(unplayed)
