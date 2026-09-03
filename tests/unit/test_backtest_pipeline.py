"""The backtest: what gets scored, over which matches, and what is averaged.

The arithmetic that matters here is not the metrics — they have their own file
— but the *bookkeeping*: which subset a row belongs to, and whether collapsing
folds gives the number a single pass would have given. Both are places where a
plausible-looking table can be quietly wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import score
from src.models.baselines import ClassPrior, Forecaster, HomeAlways
from src.models.splits import SplitError, walk_forward
from src.pipelines.backtest import (
    BACKTEST_FILENAME,
    COMMON,
    POOLED,
    PRICED,
    SCORE_COLUMNS,
    BacktestReport,
    per_competition_table,
    pooled_table,
    run_backtest,
    score_fold,
)
from tests.factories import league_frame, season_labels

SEASONS = season_labels(2012, 14)


def _two_countries() -> pd.DataFrame:
    england = league_frame(seasons=SEASONS, teams=20)
    spain = league_frame(
        competition_id="ESP_1",
        country="Spain",
        name="La Liga",
        seasons=SEASONS,
        teams=20,
        team_prefix="Club",
    )
    return (
        pd.concat([england, spain], ignore_index=True)
        .sort_values(["date", "competition_id", "match_id"], kind="stable")
        .reset_index(drop=True)
    )


LEAGUE = _two_countries()
FOLD = next(iter(walk_forward(LEAGUE)))


class Unpriceable:
    """A forecaster that can never price anything. The coverage case."""

    name = "unpriceable"

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        del train
        return np.full((len(evaluate), 3), np.nan)


class Half:
    """Prices the first half of a fold and nothing else."""

    name = "half"

    def forecast(self, train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        del train
        forecast = np.full((len(evaluate), 3), np.nan)
        forecast[: len(evaluate) // 2] = (0.4, 0.3, 0.3)
        return forecast


@pytest.fixture
def report(tmp_path: Path) -> BacktestReport:
    return run_backtest(LEAGUE, tmp_path, folds=3)


# ---- what a fold produces ---------------------------------------------------


def test_a_fold_is_scored_per_competition_and_pooled() -> None:
    scored = score_fold(FOLD, (ClassPrior(),))
    assert set(scored["competition_id"]) == {"ENG_1", "ESP_1", POOLED}
    assert set(scored["subset"]) == {PRICED, COMMON}


def test_the_pooled_row_covers_every_competition_at_once() -> None:
    scored = score_fold(FOLD, (ClassPrior(),))
    pooled = scored[(scored["competition_id"] == POOLED) & (scored["subset"] == PRICED)]
    per_competition = scored[(scored["competition_id"] != POOLED) & (scored["subset"] == PRICED)]
    assert int(pooled["n"].iloc[0]) == int(per_competition["n"].sum())


def test_a_scored_fold_matches_a_direct_computation() -> None:
    """The grouping is not allowed to change the answer."""
    scored = score_fold(FOLD, (ClassPrior(),))
    pooled = scored[(scored["competition_id"] == POOLED) & (scored["subset"] == PRICED)]
    direct = score(ClassPrior().forecast(FOLD.train, FOLD.evaluate), FOLD.evaluate["result"])
    assert float(pooled["log_loss"].iloc[0]) == pytest.approx(direct.log_loss)
    assert float(pooled["rps"].iloc[0]) == pytest.approx(direct.rps)


def test_scoring_no_forecasters_produces_an_empty_table_of_the_right_shape() -> None:
    empty = score_fold(FOLD, ())
    assert empty.empty
    assert list(empty.columns) == list(SCORE_COLUMNS)


def test_a_forecaster_that_prices_nothing_produces_no_rows() -> None:
    scored = score_fold(FOLD, (Unpriceable(),))
    assert scored.empty


# ---- the two subsets --------------------------------------------------------


def test_the_common_subset_is_what_every_forecaster_could_price() -> None:
    forecasters: tuple[Forecaster, ...] = (HomeAlways(), Half())
    scored = score_fold(FOLD, forecasters)
    common = scored[(scored["competition_id"] == POOLED) & (scored["subset"] == COMMON)]
    assert set(common["n"]) == {len(FOLD.evaluate) // 2}


def test_a_forecaster_is_scored_over_more_matches_than_the_common_subset() -> None:
    """The reason there are two tables: the bookmaker prices fewer matches than
    the rating does, and putting the two numbers side by side compares two
    different questions."""
    scored = score_fold(FOLD, (HomeAlways(), Half()))
    rows = scored[(scored["competition_id"] == POOLED) & (scored["forecaster"] == "home_always")]
    priced = int(rows[rows["subset"] == PRICED]["n"].iloc[0])
    common = int(rows[rows["subset"] == COMMON]["n"].iloc[0])
    assert priced > common


def test_a_forecaster_that_prices_nothing_empties_the_common_subset() -> None:
    """A real answer — usually that the ratings were built for other matches."""
    scored = score_fold(FOLD, (HomeAlways(), Unpriceable()))
    assert scored[scored["subset"] == COMMON].empty
    assert pooled_table(scored, COMMON).empty


# ---- collapsing folds -------------------------------------------------------


def test_pooling_folds_weights_by_the_matches_behind_each(report: BacktestReport) -> None:
    """An average of averages would over-weight a short fold. This is the check."""
    assert report.scores is not None
    rows = report.scores[
        (report.scores["competition_id"] == POOLED)
        & (report.scores["subset"] == COMMON)
        & (report.scores["forecaster"] == "class_prior")
    ]
    weights = rows["n"].to_numpy(dtype=float)
    expected = float(np.average(rows["log_loss"].to_numpy(dtype=float), weights=weights))
    assert pooled_table(report.scores).set_index("forecaster").loc["class_prior", "log_loss"] == (
        pytest.approx(expected)
    )


def test_the_pooled_table_is_ordered_by_log_loss(report: BacktestReport) -> None:
    assert report.scores is not None
    losses = pooled_table(report.scores)["log_loss"].to_list()
    assert losses == sorted(losses)


def test_the_per_competition_table_has_one_row_per_competition(
    report: BacktestReport,
) -> None:
    assert report.scores is not None
    table = per_competition_table(report.scores)
    assert set(table["competition_id"]) == {"ENG_1", "ESP_1"}
    assert "class_prior" in table.columns


def test_the_per_competition_table_is_ordered_by_size(report: BacktestReport) -> None:
    assert report.scores is not None
    counts = per_competition_table(report.scores)["n"].to_list()
    assert counts == sorted(counts, reverse=True)


def test_an_empty_score_table_produces_empty_summaries() -> None:
    empty = pd.DataFrame(columns=list(SCORE_COLUMNS)).astype(SCORE_COLUMNS)
    assert pooled_table(empty).empty
    assert per_competition_table(empty).empty


# ---- the run ----------------------------------------------------------------


def test_a_run_scores_every_fold(report: BacktestReport) -> None:
    assert report.folds == 3
    assert report.matches == sum(len(fold.evaluate) for fold in walk_forward(LEAGUE, folds=3))


def test_coverage_is_the_share_of_evaluation_matches_priced(tmp_path: Path) -> None:
    run = run_backtest(LEAGUE, tmp_path, forecasters=(HomeAlways(), Half()), folds=3)
    assert run.coverage["home_always"] == pytest.approx(1.0)
    assert run.coverage["half"] == pytest.approx(0.5, abs=0.01)


def test_the_table_is_written_with_a_manifest_beside_it(report: BacktestReport) -> None:
    assert report.output is not None
    assert report.output.name == BACKTEST_FILENAME
    manifest = json.loads(report.output.with_suffix(".manifest.json").read_text())
    assert manifest["kind"] == "backtest"
    assert manifest["folds"] == 3


def test_a_table_too_short_to_split_fails_rather_than_reporting_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(SplitError):
        run_backtest(league_frame(seasons=["2020-21"], teams=6), tmp_path)


def test_a_summary_names_every_forecaster(report: BacktestReport) -> None:
    summary = report.summary()
    assert "3 fold(s)" in summary
    assert "class_prior" in summary


def test_a_report_with_nothing_in_it_summarises_without_failing() -> None:
    """The default-constructed case, which the CLI can reach on an empty run."""
    assert BacktestReport().pooled() == {}
    assert "0 matches" in BacktestReport().summary()
