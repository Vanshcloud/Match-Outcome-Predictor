"""The zoo against the real table.

Skips without `scripts/fetch_data.py`, `build_ratings.py` and
`build_features.py`. Two kinds of assertion here, and they cost very different
amounts.

The cheap kind fits **one** family over two folds and checks the wiring holds
on three hundred thousand real matches: thirty columns present, a model that
beats the class prior, folds that pass the boundary probe. That is what a
synthetic league cannot prove.

The expensive kind reads the table `make train` wrote, if it is there, and pins
the numbers this milestone reports. Fitting six families over five folds is ten
minutes and does not belong in a test suite; asserting on its output does.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.metrics import score
from src.models.baselines import ClassPrior
from src.models.dataset import BLOCKS, DESIGN_COLUMNS, design_matrix
from src.models.splits import walk_forward
from src.models.tuning import tuning_slice
from src.models.zoo import build
from src.pipelines.backtest import (
    BACKTEST_FILENAME,
    COMMON,
    per_competition_table,
    pooled_table,
)
from src.pipelines.features import FEATURES_FILENAME
from src.pipelines.ingest import MATCHES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.pipelines.train import ABLATION_SUBDIR, ZOO_SUBDIR, ablation_table
from src.utils.config import load_settings
from src.validation.temporal import split_boundary

pytestmark = pytest.mark.integration

SETTINGS = load_settings()
MATCHES = SETTINGS.paths.processed_dir / MATCHES_FILENAME
RATINGS = SETTINGS.paths.features_dir / RATINGS_FILENAME
FEATURES = SETTINGS.paths.features_dir / FEATURES_FILENAME
TRAINED = SETTINGS.paths.reports_dir / ZOO_SUBDIR / BACKTEST_FILENAME
ABLATED = SETTINGS.paths.reports_dir / ABLATION_SUBDIR / BACKTEST_FILENAME
ABLATED_MODEL = "lightgbm"
"""Which family `make ablation` uses. The top four are within 0.0003 of each
other — smaller than anything the ablation measures — so the fastest of them
is the one that gets run six times."""

# Measured over five yearly folds. Wide enough to survive the provider adding a
# season, narrow enough that a regression or an ordering flip cannot pass.
BEST_LOG_LOSS = (1.00, 1.04)
BOOKMAKER_LOG_LOSS = (0.98, 1.02)
DESIGN_COVERAGE = 0.90


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    for label, path in (("matches", MATCHES), ("ratings", RATINGS), ("features", FEATURES)):
        if not path.is_file():
            pytest.skip(f"no {label} at {path}")
    return (
        pd.read_parquet(MATCHES)
        .merge(pd.read_parquet(RATINGS), on="match_id", how="left")
        .merge(pd.read_parquet(FEATURES), on="match_id", how="left")
    )


@pytest.fixture(scope="module")
def trained() -> pd.DataFrame:
    if not TRAINED.is_file():
        pytest.skip(f"no trained scores at {TRAINED}; run scripts/train.py")
    return pd.read_parquet(TRAINED)


# ---- the wiring, on real data -----------------------------------------------


def test_the_three_tables_join_into_a_full_design_matrix(matches: pd.DataFrame) -> None:
    assert design_matrix(matches).shape == (len(matches), len(DESIGN_COLUMNS))


def test_most_of_the_real_table_carries_the_model_columns(matches: pd.DataFrame) -> None:
    """Nulls are warm-up rows — a club's first matches, a competition's first
    seasons — and the boosted families read them as such. A design matrix that
    was mostly null would mean the join, not the warm-up."""
    filled = pd.DataFrame(design_matrix(matches)).notna().mean().mean()
    assert filled > DESIGN_COVERAGE


def test_a_model_beats_the_class_prior_on_real_matches(matches: pd.DataFrame) -> None:
    """One family, two folds. The floor, on data a synthetic league cannot
    stand in for: thirty-nine competitions, thirty-three years, real warm-up."""
    fold = next(iter(walk_forward(matches, folds=2)))
    outcomes = fold.evaluate["result"]
    model = score(build("lightgbm").forecast(fold.train, fold.evaluate), outcomes)
    prior = score(ClassPrior().forecast(fold.train, fold.evaluate), outcomes)
    assert model.log_loss < prior.log_loss


def test_the_tuning_slice_excludes_every_reported_match(matches: pd.DataFrame) -> None:
    """The line that makes the shipped constants mean something, on the real
    calendar rather than a tidy one."""
    tuned = set(tuning_slice(matches)["match_id"])
    for fold in walk_forward(matches):
        assert not tuned & set(fold.evaluate["match_id"])


def test_every_fold_the_zoo_trains_on_passes_the_boundary_probe(matches: pd.DataFrame) -> None:
    for fold in walk_forward(matches):
        assert split_boundary(fold.train, fold.evaluate, name=f"fold {fold.index}").ok


# ---- the reported numbers ---------------------------------------------------


def test_every_family_was_scored(trained: pd.DataFrame) -> None:
    assert {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
    } <= set(trained["forecaster"])


def test_the_models_are_scored_beside_the_baselines(trained: pd.DataFrame) -> None:
    assert {"class_prior", "dixon_coles", "bookmaker"} <= set(trained["forecaster"])


def test_every_family_beats_the_rating_it_was_built_on(trained: pd.DataFrame) -> None:
    """The question Milestone 8 exists to answer. A zoo that could not beat
    Dixon-Coles would mean the features add nothing to a strength model."""
    pooled = pooled_table(trained, COMMON).set_index("forecaster")
    rating = pooled.loc["dixon_coles", "log_loss"]
    for family in ("logistic_regression", "xgboost", "lightgbm", "catboost"):
        assert pooled.loc[family, "log_loss"] < rating, family


def test_the_bookmaker_still_wins(trained: pd.DataFrame) -> None:
    """The ceiling. A model beating the closing line on public data would be
    evidence of a leak, not of skill."""
    pooled = pooled_table(trained, COMMON).set_index("forecaster")
    assert pooled["log_loss"].idxmin() == "bookmaker"


def test_the_best_model_lands_where_it_was_measured(trained: pd.DataFrame) -> None:
    pooled = pooled_table(trained, COMMON).set_index("forecaster")
    best = pooled.drop(index=["bookmaker"])["log_loss"].min()
    assert BEST_LOG_LOSS[0] <= best <= BEST_LOG_LOSS[1]
    assert BOOKMAKER_LOG_LOSS[0] <= pooled.loc["bookmaker", "log_loss"] <= BOOKMAKER_LOG_LOSS[1]


# ---- the ablation -----------------------------------------------------------


@pytest.fixture(scope="module")
def ablated() -> pd.DataFrame:
    if not ABLATED.is_file():
        pytest.skip(f"no ablation at {ABLATED}; run scripts/train.py --ablate lightgbm")
    return pd.read_parquet(ABLATED)


def test_every_block_was_withheld_in_turn(ablated: pd.DataFrame) -> None:
    table = ablation_table(ablated, ABLATED_MODEL)
    assert set(table["block"]) == set(BLOCKS)


def test_every_variant_is_scored_over_the_same_matches(ablated: pd.DataFrame) -> None:
    """One backtest, not six. Six would each compute their own common subset,
    and those differences are the same size as the ones being measured."""
    assert pooled_table(ablated, COMMON)["n"].nunique() == 1


def test_form_is_the_block_the_model_least_wants_to_lose(ablated: pd.DataFrame) -> None:
    """Bigger than either rating block — and the ratings are a fitted model
    each, where form is a window over a sorted array."""
    table = ablation_table(ablated, ABLATED_MODEL).set_index("block")
    assert table["delta_log_loss"].idxmax() == "form"
    assert table.loc["form", "delta_log_loss"] > table.loc["elo", "delta_log_loss"]


def test_the_schedule_and_head_to_head_blocks_are_worth_almost_nothing(
    ablated: pd.DataFrame,
) -> None:
    """Milestone 5 shipped rest days saying they carried no marginal signal and
    that this ablation would settle it. It has: 0.0003 of log loss."""
    table = ablation_table(ablated, ABLATED_MODEL).set_index("block")
    assert 0 <= table.loc["schedule", "delta_log_loss"] < 0.001
    assert 0 <= table.loc["head_to_head", "delta_log_loss"] < 0.001


def test_no_block_is_actively_harmful(ablated: pd.DataFrame) -> None:
    """A negative delta would mean the model does better without the block —
    a result, not a bug, and one worth knowing before Milestone 9 calibrates."""
    assert (ablation_table(ablated, ABLATED_MODEL)["delta_log_loss"] >= -0.001).all()


# ---- the competition Dixon-Coles could not price -----------------------------


def test_the_zoo_fixes_the_competition_the_rating_failed_on(trained: pd.DataFrame) -> None:
    """Milestone 7 found Dixon-Coles losing to the class prior on the Argentine
    cup — a narrow field fitted per competition on 610 matches. The models have
    somewhere else to look, and none of them needed telling."""
    table = per_competition_table(trained).set_index("competition_id")
    assert table.loc["ARG_CUP", "dixon_coles"] > table.loc["ARG_CUP", "class_prior"]
    assert table.loc["ARG_CUP", "lightgbm"] < table.loc["ARG_CUP", "class_prior"]


def test_the_best_model_beats_the_rating_in_every_competition(trained: pd.DataFrame) -> None:
    table = per_competition_table(trained)
    assert (table["lightgbm"] < table["dixon_coles"]).all()


def test_the_bookmaker_still_wins_in_every_competition(trained: pd.DataFrame) -> None:
    table = per_competition_table(trained)
    assert (table["bookmaker"] < table["lightgbm"]).all()
