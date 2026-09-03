"""The backtest entry point. Only the contract a caller depends on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.pipelines.backtest import BACKTEST_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.utils.paths import PROJECT_ROOT
from tests.factories import league_frame, season_labels

SEASONS = season_labels(2012, 14)


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "backtest_cli", PROJECT_ROOT / "scripts" / "backtest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()
LEAGUE = league_frame(seasons=SEASONS, teams=20)


@pytest.fixture
def matches(tmp_path: Path) -> Path:
    path = tmp_path / "matches.parquet"
    LEAGUE.to_parquet(path, index=False)
    return path


@pytest.fixture
def ratings(tmp_path: Path) -> Path:
    """A ratings table for exactly these matches, with a plausible forecast."""
    path = tmp_path / RATINGS_FILENAME
    pd.DataFrame(
        {
            "match_id": LEAGUE["match_id"],
            "dc_prob_home": 0.45,
            "dc_prob_draw": 0.27,
            "dc_prob_away": 0.28,
        }
    ).to_parquet(path, index=False)
    return path


def test_a_clean_run_exits_zero(matches: Path, tmp_path: Path) -> None:
    code = CLI.main(["--matches", str(matches), "--output", str(tmp_path / "out"), "--folds", "3"])
    assert code == 0
    assert (tmp_path / "out" / BACKTEST_FILENAME).is_file()


def test_the_ratings_bring_in_the_dixon_coles_baseline(
    matches: Path, ratings: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = CLI.main(
        [
            "--matches",
            str(matches),
            "--ratings",
            str(ratings),
            "--output",
            str(tmp_path / "out"),
            "--folds",
            "3",
        ]
    )
    assert code == 0
    assert "dixon_coles" in capsys.readouterr().out


def test_without_ratings_the_run_drops_dixon_coles(
    matches: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dropped rather than reported as a forecaster that priced nothing."""
    code = CLI.main(
        [
            "--matches",
            str(matches),
            "--ratings",
            str(tmp_path / "absent.parquet"),
            "--output",
            str(tmp_path / "out"),
            "--folds",
            "3",
        ]
    )
    assert code == 0
    assert "dixon_coles" not in capsys.readouterr().out


def test_markdown_renders_a_table_without_a_formatting_dependency(
    matches: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = CLI.main(
        [
            "--matches",
            str(matches),
            "--output",
            str(tmp_path / "out"),
            "--folds",
            "3",
            "--markdown",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "| forecaster | coverage | n | log_loss | rps | accuracy |" in printed
    assert "|---|---|---|---|---|---|" in printed
    assert "| class_prior |" in printed


def test_a_missing_table_exits_non_zero(tmp_path: Path) -> None:
    """The state of a clean checkout. It must say what to run, not traceback."""
    assert CLI.main(["--matches", str(tmp_path / "absent.parquet")]) == 1


def test_an_unknown_competition_exits_non_zero(matches: Path) -> None:
    """A filter that silently scores nothing looks exactly like a broken run."""
    assert CLI.main(["--matches", str(matches), "--competition", "NOWHERE_1"]) == 1


def test_a_table_too_short_to_split_exits_non_zero(tmp_path: Path) -> None:
    path = tmp_path / "short.parquet"
    league_frame(seasons=["2020-21"], teams=6).to_parquet(path, index=False)
    assert CLI.main(["--matches", str(path), "--output", str(tmp_path / "out")]) == 1


def test_an_empty_common_subset_is_reported_rather_than_crashing(
    matches: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ratings built for a different set of matches: a real state, not a bug."""
    unrelated = tmp_path / "other.parquet"
    pd.DataFrame(
        {
            "match_id": ["nobody:2020-21:00001"],
            "dc_prob_home": [0.4],
            "dc_prob_draw": [0.3],
            "dc_prob_away": [0.3],
        }
    ).to_parquet(unrelated, index=False)
    code = CLI.main(
        [
            "--matches",
            str(matches),
            "--ratings",
            str(unrelated),
            "--output",
            str(tmp_path / "out"),
            "--folds",
            "3",
        ]
    )
    assert code == 0
    assert "nothing priced by every forecaster" in capsys.readouterr().out
