"""Training, the ablation, the blend — all of them the *unchanged* backtest.

A trained model is a forecaster that fits inside its own `forecast`, so
Milestone 7's evaluation pipeline runs the zoo without knowing an estimator
exists. What is tested here is that the wiring holds that shape: same folds,
same two subsets, same file, and an ablation whose variants are all scored on
identical matches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.reliability import DEFAULT_BINS
from src.models.baselines import Bookmaker
from src.models.dataset import BLOCKS, DESIGN_COLUMNS, without
from src.models.zoo import ModelError
from src.pipelines.backtest import BACKTEST_FILENAME, COMMON, POOLED, pooled_table
from src.pipelines.train import (
    ABLATION_SUBDIR,
    ALL_BLOCKS,
    ENSEMBLE_SUBDIR,
    ZOO_SUBDIR,
    TrainingReport,
    ablation_forecasters,
    ablation_table,
    ensemble_forecasters,
    reliability_tables,
    run_ablation,
    run_ensemble,
    run_training,
    zoo_forecasters,
)
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

LEAGUE = modelled_frame(seasons=season_labels(2012, 10), teams=12)
QUICK = ("logistic_regression",)


@pytest.fixture
def trained(tmp_path: Path) -> TrainingReport:
    return run_training(LEAGUE, tmp_path, models=QUICK, folds=2)


# ---- the wiring -------------------------------------------------------------


def test_a_model_is_scored_by_the_unchanged_backtest(trained: TrainingReport) -> None:
    assert trained.backtest is not None
    assert trained.backtest.output == (
        trained.backtest.output.parent / BACKTEST_FILENAME  # type: ignore[union-attr]
    )
    assert trained.backtest.output.parent.name == ZOO_SUBDIR


def test_the_baselines_are_scored_in_the_same_run(trained: TrainingReport) -> None:
    """A model's log loss means nothing without the class prior beside it,
    computed on the same matches."""
    assert trained.backtest is not None and trained.backtest.scores is not None
    scored = set(trained.backtest.scores["forecaster"])
    assert {"class_prior", "dixon_coles", "logistic_regression"} <= scored


def test_the_baselines_can_be_left_out(tmp_path: Path) -> None:
    report = run_training(LEAGUE, tmp_path, models=QUICK, baselines=False, folds=2)
    assert report.backtest is not None and report.backtest.scores is not None
    assert set(report.backtest.scores["forecaster"]) == {"logistic_regression"}


def test_a_model_beats_the_prior_on_the_common_subset(trained: TrainingReport) -> None:
    assert trained.backtest is not None and trained.backtest.scores is not None
    pooled = pooled_table(trained.backtest.scores, COMMON).set_index("forecaster")
    assert pooled.loc["logistic_regression", "log_loss"] < pooled.loc["class_prior", "log_loss"]


def test_the_requested_models_are_the_ones_run(trained: TrainingReport) -> None:
    assert trained.models == QUICK


def test_the_whole_zoo_is_the_default() -> None:
    assert len(zoo_forecasters()) == 6


def test_a_model_the_zoo_does_not_hold_is_refused() -> None:
    with pytest.raises(ModelError, match="not a model"):
        zoo_forecasters(DESIGN_COLUMNS, ["transformer"])


def test_a_report_with_nothing_in_it_summarises_without_failing() -> None:
    assert TrainingReport().summary() == "nothing trained"


# ---- tracking ---------------------------------------------------------------


def test_a_run_is_tracked_when_a_model_directory_is_given(tmp_path: Path) -> None:
    report = run_training(
        LEAGUE, tmp_path / "reports", models=QUICK, folds=2, model_dir=tmp_path / "models"
    )
    assert set(report.tracked) == {"logistic_regression"}


def test_no_model_directory_means_no_tracking(trained: TrainingReport) -> None:
    assert trained.tracked == {}


def test_a_model_missing_from_the_common_subset_is_not_tracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reachable when a baseline in the same run could price nothing, which
    empties the subset the tracked metrics are read from."""
    monkeypatch.setattr(
        "src.pipelines.train.pooled_table", lambda *_, **__: pd.DataFrame({"forecaster": []})
    )
    report = run_training(
        LEAGUE, tmp_path / "reports", models=QUICK, folds=2, model_dir=tmp_path / "models"
    )
    assert report.tracked == {}


# ---- the ablation -----------------------------------------------------------


def test_the_ablation_scores_a_control_and_one_variant_per_block() -> None:
    names = [one.name for one in ablation_forecasters("lightgbm")]
    assert names[0] == f"lightgbm-{ALL_BLOCKS}"
    assert names[1:] == [f"lightgbm-no-{block}" for block in BLOCKS]


def test_a_variant_is_the_same_model_with_one_block_withheld() -> None:
    variants = {one.name: one for one in ablation_forecasters("lightgbm")}
    full = variants[f"lightgbm-{ALL_BLOCKS}"]
    reduced = variants["lightgbm-no-elo"]
    assert full.columns == DESIGN_COLUMNS
    assert reduced.columns == without("elo")
    assert reduced.parameters == full.parameters


def test_only_the_named_blocks_are_ablated() -> None:
    names = [one.name for one in ablation_forecasters("lightgbm", ["elo"])]
    assert names == ["lightgbm-none", "lightgbm-no-elo"]


def test_every_variant_is_scored_in_one_run(tmp_path: Path) -> None:
    """One backtest, not six. Six would each compute their own common subset,
    and the differences between those subsets are the same size as the
    differences the ablation is trying to measure."""
    report = run_ablation(
        LEAGUE, tmp_path, model="logistic_regression", blocks=["elo", "form"], folds=2
    )
    assert report.output is not None and report.output.parent.name == ABLATION_SUBDIR
    assert report.scores is not None
    pooled = pooled_table(report.scores, COMMON)
    assert len(pooled) == 3
    assert pooled["n"].nunique() == 1


def test_the_ablation_table_is_a_delta_against_the_full_model(tmp_path: Path) -> None:
    report = run_ablation(LEAGUE, tmp_path, model="logistic_regression", blocks=["elo"], folds=2)
    assert report.scores is not None
    table = ablation_table(report.scores, "logistic_regression")
    pooled = pooled_table(report.scores, COMMON).set_index("forecaster")
    expected = (
        pooled.loc["logistic_regression-no-elo", "log_loss"]
        - pooled.loc["logistic_regression-none", "log_loss"]
    )
    assert list(table["block"]) == ["elo"]
    assert table["delta_log_loss"].iloc[0] == pytest.approx(expected)


def test_the_ablation_finds_the_block_that_carries_the_signal(tmp_path: Path) -> None:
    """Its own sanity check, in both directions.

    The synthetic league's outcome is decided by the Elo expectation and by
    nothing else, so withholding that block must cost and withholding the
    constant Dixon-Coles block must not. An ablation that reported the same
    number for both would be measuring its own noise.
    """
    report = run_ablation(
        LEAGUE, tmp_path, model="logistic_regression", blocks=["elo", "dixon_coles"], folds=2
    )
    assert report.scores is not None
    deltas = ablation_table(report.scores, "logistic_regression").set_index("block")
    assert deltas.loc["elo", "delta_log_loss"] > 0.05
    assert abs(deltas.loc["dixon_coles", "delta_log_loss"]) < 0.001


def test_an_ablation_table_without_its_control_is_empty() -> None:
    """A run that scored no variants has nothing to compare against, and an
    empty table says so better than a division by a missing row."""
    empty = pd.DataFrame(
        {"forecaster": ["class_prior"], "n": [10], "log_loss": [1.0], "rps": [0.2]}
    ).assign(competition_id=POOLED, subset=COMMON, accuracy=0.4, fold=0)
    assert ablation_table(empty, "lightgbm").empty


def test_the_blocks_ablated_are_the_ones_the_registries_define() -> None:
    assert set(BLOCKS) == {"form", "schedule", "head_to_head", "elo", "dixon_coles"}


# ---- the blend and the calibration layer ------------------------------------


def test_the_four_rows_the_milestone_compares_are_built_together() -> None:
    """The model, that model calibrated, the blend, and the blend calibrated.
    One list, so one backtest scores them on identical matches and a difference
    between two rows is the layer rather than the subset."""
    names = [one.name for one in ensemble_forecasters("logistic_regression", QUICK)]
    assert names == [
        "logistic_regression",
        "logistic_regression-calibrated",
        "ensemble",
        "ensemble-calibrated",
    ]


def test_the_blend_is_scored_by_the_unchanged_backtest(tmp_path: Path) -> None:
    report = run_ensemble(
        LEAGUE,
        tmp_path,
        model="logistic_regression",
        members=["logistic_regression", "random_forest"],
        folds=2,
    )
    assert report.output == tmp_path / ENSEMBLE_SUBDIR / BACKTEST_FILENAME
    assert report.scores is not None
    scored = set(report.scores["forecaster"])
    assert {"ensemble", "ensemble-calibrated", "class_prior"} <= scored


def test_the_blend_can_be_scored_without_the_baselines(tmp_path: Path) -> None:
    report = run_ensemble(
        LEAGUE, tmp_path, model="logistic_regression", members=QUICK, baselines=False, folds=2
    )
    assert report.scores is not None
    assert "class_prior" not in set(report.scores["forecaster"])


def test_reliability_is_reported_per_forecaster_over_the_same_folds() -> None:
    tables = reliability_tables(LEAGUE, ensemble_forecasters("logistic_regression", QUICK), folds=2)
    assert set(tables) == {
        "logistic_regression",
        "logistic_regression-calibrated",
        "ensemble",
        "ensemble-calibrated",
    }
    for table in tables.values():
        assert 0 < len(table) <= DEFAULT_BINS


def test_a_forecaster_is_measured_on_the_probabilities_it_actually_issued() -> None:
    """Unpriced rows are dropped per forecaster rather than reduced to a common
    subset: the bookmaker's line is not less honest for being absent on the
    matches it never quoted."""
    half_quoted = LEAGUE.copy()
    half_quoted.loc[half_quoted.index[::2], "odds_home"] = np.nan
    tables = reliability_tables(half_quoted, [Bookmaker()], folds=2)
    both = reliability_tables(LEAGUE, [Bookmaker()], folds=2)
    assert 0 < tables["bookmaker"]["n"].sum() < both["bookmaker"]["n"].sum()
