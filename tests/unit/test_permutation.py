"""Permutation importance, and the four ways it lies.

The league here is built so that one block *is* the answer: `modelled_frame`
decides most results from `elo_expected_home`, so a method that cannot find the
elo block on this data cannot find anything on real data either. The rest of
the file is about the mechanics — that the shuffle keeps a block together, that
the model is fitted once, and that a block the model was never shown reports
nothing rather than noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.permutation import (
    IMPORTANCE_COLUMNS,
    importance,
    shuffled,
)
from src.models.dataset import BLOCKS, without
from src.models.splits import walk_forward
from src.models.zoo import build
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

LEAGUE = modelled_frame(seasons=season_labels(2012, 10), teams=12)
FOLD = next(iter(walk_forward(LEAGUE, folds=2)))

QUICK = {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.3}


def _model(**overrides: object) -> object:
    return build("lightgbm", **{**QUICK, **overrides})


# ---- the shuffle -------------------------------------------------------------


def _rows(frame: pd.DataFrame, columns: list[str]) -> list[tuple[float, ...]]:
    """The block's rows as sortable tuples, with nulls made comparable."""
    filled = np.nan_to_num(frame[columns].to_numpy(dtype=float), nan=-999.0)
    return sorted(tuple(row) for row in filled.tolist())


def test_a_block_moves_to_another_match_as_a_unit() -> None:
    """Permuting the fourteen form columns independently builds fixtures that
    never happened — five wins in five and a goal difference of minus nine —
    and measures the model on nonsense. The multiset of *rows* surviving is
    what says the block moved together."""
    columns = list(BLOCKS["form"])
    broken = shuffled(FOLD.evaluate, columns, np.random.default_rng(0))
    assert _rows(broken, columns) == _rows(FOLD.evaluate, columns)
    assert not broken[columns].equals(FOLD.evaluate[columns])


def test_the_block_keeps_its_dtypes() -> None:
    """A block of `Float64` and `Int32` columns assigned back as one numpy
    array comes out object dtype throughout — which still scores, because the
    design matrix casts, and is a different input from the intact one it is
    being compared against."""
    columns = list(BLOCKS["form"])
    broken = shuffled(FOLD.evaluate, columns, np.random.default_rng(0))
    assert list(broken[columns].dtypes) == list(FOLD.evaluate[columns].dtypes)


def test_nothing_outside_the_block_is_touched() -> None:
    columns = list(BLOCKS["form"])
    broken = shuffled(FOLD.evaluate, columns, np.random.default_rng(0))
    untouched = [column for column in FOLD.evaluate.columns if column not in columns]
    pd.testing.assert_frame_equal(broken[untouched], FOLD.evaluate[untouched])


def test_the_frame_handed_in_is_not_modified() -> None:
    """A permutation that mutated its input would leave every later block
    measured against a table someone had already broken."""
    before = FOLD.evaluate.copy()
    shuffled(FOLD.evaluate, list(BLOCKS["elo"]), np.random.default_rng(0))
    pd.testing.assert_frame_equal(FOLD.evaluate, before)


# ---- what it measures --------------------------------------------------------


def test_it_finds_the_block_the_league_was_built_around() -> None:
    """The floor. This synthetic league decides most of its results from the
    elo expectation, so anything that ranks another block first is broken."""
    table = importance(_model(), FOLD.train, FOLD.evaluate, repeats=3)
    assert list(table.columns) == list(IMPORTANCE_COLUMNS)
    assert table.iloc[0]["block"] == "elo"
    assert table.iloc[0]["delta_log_loss"] > 0


def test_breaking_a_block_never_helps_by_much() -> None:
    """A negative delta is possible — noise, on a block the model ignores — but
    a large one would mean the shuffle improved the model, which would mean the
    shuffle is not doing what it says."""
    table = importance(_model(), FOLD.train, FOLD.evaluate, repeats=3)
    assert (table["delta_log_loss"] > -0.01).all()


def test_the_spread_across_repeats_is_reported() -> None:
    """Without it a delta of 0.0003 and a delta of 0.0003 ± 0.002 read the
    same, and only one of them is a finding."""
    table = importance(_model(), FOLD.train, FOLD.evaluate, repeats=4).set_index("block")
    assert (table["spread"] >= 0).all()
    assert table.loc["elo", "spread"] < table.loc["elo", "delta_log_loss"]


def test_a_block_the_model_was_not_shown_is_left_out() -> None:
    """Not reported as worthless. A model blinded to the ratings has no opinion
    about them, and a row of zeros would read as one."""
    blinded = build("lightgbm", without("elo", "dixon_coles"), **QUICK)
    table = importance(blinded, FOLD.train, FOLD.evaluate, repeats=2)
    assert set(table["block"]) == set(BLOCKS) - {"elo", "dixon_coles"}


def test_only_the_named_blocks_are_measured() -> None:
    table = importance(
        _model(), FOLD.train, FOLD.evaluate, blocks={"elo": BLOCKS["elo"]}, repeats=2
    )
    assert list(table["block"]) == ["elo"]


def test_no_blocks_at_all_is_an_empty_table_with_the_right_columns() -> None:
    table = importance(_model(), FOLD.train, FOLD.evaluate, blocks={}, repeats=2)
    assert table.empty
    assert list(table.columns) == list(IMPORTANCE_COLUMNS)


def test_the_same_seed_gives_the_same_answer() -> None:
    """A report whose numbers move between runs cannot be read against an
    ablation whose numbers do not."""
    first = importance(_model(), FOLD.train, FOLD.evaluate, repeats=2, seed=7)
    second = importance(_model(), FOLD.train, FOLD.evaluate, repeats=2, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_moves_the_answer_but_not_the_ranking() -> None:
    first = importance(_model(), FOLD.train, FOLD.evaluate, repeats=3, seed=1)
    second = importance(_model(), FOLD.train, FOLD.evaluate, repeats=3, seed=2)
    assert list(first["block"]) == list(second["block"])


def test_the_model_is_fitted_once_however_many_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Six blocks and five repeats is thirty-one predictions and one fit. The
    estimator depends on the training half, which no permutation touches."""
    model = _model()
    fits: list[int] = []
    original = type(model).fit
    monkeypatch.setattr(
        type(model),
        "fit",
        lambda self, train: (fits.append(1), original(self, train))[1],
    )
    importance(model, FOLD.train, FOLD.evaluate, repeats=5)
    assert len(fits) == 1
