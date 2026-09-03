"""The training entry point. Only the contract a caller depends on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.models.dataset import DESIGN_COLUMNS
from src.pipelines.backtest import BACKTEST_FILENAME
from src.pipelines.features import FEATURES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.pipelines.train import ABLATION_SUBDIR, ZOO_SUBDIR
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS
from src.utils.paths import PROJECT_ROOT
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "train_cli", PROJECT_ROOT / "scripts" / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()
LEAGUE = modelled_frame(seasons=season_labels(2012, 10), teams=12)
RATING_COLUMNS = [*ELO_COLUMNS, *DIXON_COLES_COLUMNS]
FEATURE_COLUMNS = [column for column in DESIGN_COLUMNS if column not in RATING_COLUMNS]


@pytest.fixture
def tables(tmp_path: Path) -> dict[str, Path]:
    """The three tables the trainer needs, written where it looks for them."""
    matches = tmp_path / "matches.parquet"
    ratings = tmp_path / RATINGS_FILENAME
    features = tmp_path / FEATURES_FILENAME
    LEAGUE.drop(columns=list(DESIGN_COLUMNS)).to_parquet(matches, index=False)
    LEAGUE[["match_id", *RATING_COLUMNS]].to_parquet(ratings, index=False)
    LEAGUE[["match_id", *FEATURE_COLUMNS]].to_parquet(features, index=False)
    return {"matches": matches, "ratings": ratings, "features": features}


def _argv(tables: dict[str, Path], output: Path, *extra: str) -> list[str]:
    return [
        "--matches",
        str(tables["matches"]),
        "--ratings",
        str(tables["ratings"]),
        "--features",
        str(tables["features"]),
        "--output",
        str(output),
        "--folds",
        "2",
        "--no-tracking",
        *extra,
    ]


def test_a_clean_run_exits_zero(tables: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "reports"
    code = CLI.main(_argv(tables, out, "--model", "logistic_regression"))
    assert code == 0
    assert (out / ZOO_SUBDIR / BACKTEST_FILENAME).is_file()


def test_the_three_tables_are_joined_before_training(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The features and ratings live in separate files; a model needs all
    thirty columns, and a run that silently trained on five is worthless."""
    code = CLI.main(_argv(tables, tmp_path / "reports", "--model", "logistic_regression"))
    assert code == 0
    assert "logistic_regression" in capsys.readouterr().out


def test_markdown_renders_a_table(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    CLI.main(_argv(tables, tmp_path / "reports", "--model", "logistic_regression", "--markdown"))
    printed = capsys.readouterr().out
    assert "| forecaster | n | log_loss | rps | accuracy |" in printed
    assert "| class_prior |" in printed


def test_the_models_can_be_scored_without_the_baselines(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    CLI.main(
        _argv(tables, tmp_path / "reports", "--model", "logistic_regression", "--no-baselines")
    )
    assert "class_prior" not in capsys.readouterr().out


def test_an_ablation_reports_a_delta_per_block(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "reports"
    code = CLI.main(_argv(tables, out, "--ablate", "logistic_regression"))
    assert code == 0
    assert (out / ABLATION_SUBDIR / BACKTEST_FILENAME).is_file()
    printed = capsys.readouterr().out
    assert "delta_log_loss" in printed
    assert "elo" in printed


def test_a_search_runs_on_matches_earlier_than_every_reported_fold(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = CLI.main(
        _argv(tables, tmp_path / "reports", "--tune", "logistic_regression", "--trials", "2")
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "earlier than the first reported fold" in printed
    assert "LOGISTIC_REGRESSION: dict[str, Any] = {" in printed


def test_a_search_that_found_nothing_says_so(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Several of these searches find nothing. The command has to say it rather
    than print settings that look like an improvement."""
    CLI.main(_argv(tables, tmp_path / "reports", "--tune", "logistic_regression", "--trials", "1"))
    printed = capsys.readouterr().out
    assert ("did not beat the shipped settings" in printed) or ("+0.0" in printed)


def test_a_missing_table_exits_non_zero(tables: dict[str, Path], tmp_path: Path) -> None:
    """The state of a clean checkout. A model needs all three tables."""
    assert CLI.main(_argv({**tables, "features": tmp_path / "absent.parquet"}, tmp_path)) == 1


def test_an_unknown_competition_exits_non_zero(tables: dict[str, Path], tmp_path: Path) -> None:
    assert CLI.main(_argv(tables, tmp_path, "--competition", "NOWHERE_1")) == 1


def test_a_table_too_short_to_split_exits_non_zero(tmp_path: Path) -> None:
    short = modelled_frame(seasons=["2020-21"], teams=6)
    paths = {
        "matches": tmp_path / "m.parquet",
        "ratings": tmp_path / RATINGS_FILENAME,
        "features": tmp_path / FEATURES_FILENAME,
    }
    short.drop(columns=list(DESIGN_COLUMNS)).to_parquet(paths["matches"], index=False)
    short[["match_id", *RATING_COLUMNS]].to_parquet(paths["ratings"], index=False)
    short[["match_id", *FEATURE_COLUMNS]].to_parquet(paths["features"], index=False)
    assert CLI.main(_argv(paths, tmp_path / "out", "--model", "logistic_regression")) == 1


def test_an_empty_table_renders_as_nothing_to_report() -> None:
    assert "nothing to report" in CLI.render(pd.DataFrame(), markdown=False)


def test_tracking_is_on_unless_it_is_turned_off(
    tables: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run log goes under the configured model directory, never the
    working directory."""
    seen: list[Path] = []
    monkeypatch.setattr(
        "src.pipelines.train.log_run",
        lambda *_, model_dir, **__: seen.append(model_dir) or "run-id",
    )
    argv = [
        arg
        for arg in _argv(tables, tmp_path / "reports", "--model", "logistic_regression")
        if arg != "--no-tracking"
    ]
    assert CLI.main(argv) == 0
    assert seen and seen[0].name == "models"
