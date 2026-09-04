"""The attribution entry point. Only the contract a caller depends on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.models.dataset import DESIGN_COLUMNS
from src.pipelines.backtest import BACKTEST_FILENAME
from src.pipelines.train import ABLATION_SUBDIR, run_ablation
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS
from src.utils.paths import PROJECT_ROOT
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "explain_cli", PROJECT_ROOT / "scripts" / "explain.py"
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
    matches = tmp_path / "matches.parquet"
    ratings = tmp_path / "ratings.parquet"
    features = tmp_path / "features.parquet"
    LEAGUE.drop(columns=list(DESIGN_COLUMNS)).to_parquet(matches, index=False)
    LEAGUE[["match_id", *RATING_COLUMNS]].to_parquet(ratings, index=False)
    LEAGUE[["match_id", *FEATURE_COLUMNS]].to_parquet(features, index=False)
    return {"matches": matches, "ratings": ratings, "features": features}


@pytest.fixture(scope="module")
def ablated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real ablation of one family, so the third column has something to be."""
    out = tmp_path_factory.mktemp("reports")
    run_ablation(LEAGUE, out, model="logistic_regression", folds=2)
    return out / ABLATION_SUBDIR / BACKTEST_FILENAME


def _argv(tables: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--matches",
        str(tables["matches"]),
        "--ratings",
        str(tables["ratings"]),
        "--features",
        str(tables["features"]),
        "--folds",
        "2",
        "--repeats",
        "2",
        "--sample",
        "100",
        *extra,
    ]


def test_a_clean_run_reports_both_methods(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = CLI.main(_argv(tables, "--model", "lightgbm", "--ablation", str(tmp_path / "none")))
    assert code == 0
    printed = capsys.readouterr().out
    assert "What breaking each block costs" in printed
    assert "What each block contributes" in printed
    assert "elo" in printed


def test_a_family_with_no_tree_says_so_rather_than_failing(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Permutation covers every family; SHAP covers three. A command that
    exited non-zero on the other three would be reporting a scope decision as
    an error."""
    code = CLI.main(_argv(tables, "--model", "mlp", "--ablation", str(tmp_path / "none")))
    assert code == 0
    printed = capsys.readouterr().out
    assert "no SHAP for mlp" in printed
    assert "What breaking each block costs" in printed


def test_several_families_are_reported_in_the_order_asked_for(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    CLI.main(
        _argv(
            tables,
            "--model",
            "lightgbm",
            "--model",
            "logistic_regression",
            "--ablation",
            str(tmp_path / "none"),
        )
    )
    printed = capsys.readouterr().out
    assert printed.index("## lightgbm") < printed.index("## logistic_regression")


def test_the_ablation_is_joined_when_it_is_the_same_family(
    tables: dict[str, Path], ablated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three numbers per row that are about the same model, or two that are."""
    CLI.main(_argv(tables, "--model", "logistic_regression", "--ablation", str(ablated)))
    printed = capsys.readouterr().out
    assert "comparing against the persisted ablation of logistic_regression" in printed
    assert "ablation" in printed


def test_the_ablation_is_not_joined_onto_a_different_family(
    tables: dict[str, Path], ablated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A LightGBM ablation beside an XGBoost attribution is three numbers in a
    row that are not about the same model."""
    CLI.main(_argv(tables, "--model", "xgboost", "--ablation", str(ablated)))
    assert "ablation" not in capsys.readouterr().out.split("## xgboost")[1]


def test_a_scores_file_that_is_not_an_ablation_is_ignored(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It has no control row, so there is nothing to take a delta against."""
    not_an_ablation = tmp_path / "scores.parquet"
    pd.DataFrame({"forecaster": ["lightgbm"], "log_loss": [1.0]}).to_parquet(
        not_an_ablation, index=False
    )
    code = CLI.main(_argv(tables, "--model", "lightgbm", "--ablation", str(not_an_ablation)))
    assert code == 0
    assert "ablation" not in capsys.readouterr().out


def test_markdown_renders_a_table(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    CLI.main(
        _argv(tables, "--model", "lightgbm", "--markdown", "--ablation", str(tmp_path / "none"))
    )
    assert "| block | columns | delta_log_loss | spread |" in capsys.readouterr().out


def test_an_empty_table_renders_as_nothing_to_report() -> None:
    assert "nothing to report" in CLI.render(pd.DataFrame(), markdown=False)


def test_a_missing_table_exits_non_zero(tables: dict[str, Path], tmp_path: Path) -> None:
    assert CLI.main(_argv({**tables, "features": tmp_path / "absent.parquet"})) == 1


def test_a_table_too_short_to_split_exits_non_zero(tmp_path: Path) -> None:
    short = modelled_frame(seasons=["2020-21"], teams=6)
    paths = {
        "matches": tmp_path / "m.parquet",
        "ratings": tmp_path / "r.parquet",
        "features": tmp_path / "f.parquet",
    }
    short.drop(columns=list(DESIGN_COLUMNS)).to_parquet(paths["matches"], index=False)
    short[["match_id", *RATING_COLUMNS]].to_parquet(paths["ratings"], index=False)
    short[["match_id", *FEATURE_COLUMNS]].to_parquet(paths["features"], index=False)
    assert CLI.main(_argv(paths, "--model", "lightgbm", "--ablation", str(tmp_path / "none"))) == 1
