"""The ratings entry point.

Only the contract a caller depends on: the exit code, and that `--fit-until`
is a report rather than a pipeline step. A build command that prints FAILED and
exits zero is worse than no command, because a pipeline believes it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.pipelines.ratings import RATINGS_FILENAME
from src.utils.paths import PROJECT_ROOT
from tests.factories import league_frame


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_ratings_cli", PROJECT_ROOT / "scripts" / "build_ratings.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


@pytest.fixture
def matches(tmp_path: Path) -> Path:
    path = tmp_path / "matches.parquet"
    league_frame(seasons=["2019-20", "2020-21", "2021-22"], teams=8).to_parquet(path, index=False)
    return path


def test_a_clean_build_exits_zero(matches: Path, tmp_path: Path) -> None:
    code = CLI.main(
        ["--matches", str(matches), "--output", str(tmp_path / "out"), "--model", "elo"]
    )
    assert code == 0
    assert (tmp_path / "out" / RATINGS_FILENAME).is_file()


def test_a_missing_table_exits_non_zero(tmp_path: Path) -> None:
    """The state of a clean checkout. It must say what to run, not traceback."""
    assert CLI.main(["--matches", str(tmp_path / "absent.parquet")]) == 1


def test_a_competition_filter_selects_a_subset(matches: Path, tmp_path: Path) -> None:
    code = CLI.main(
        [
            "--matches",
            str(matches),
            "--output",
            str(tmp_path / "out"),
            "--model",
            "elo",
            "--competition",
            "ENG_1",
        ]
    )
    assert code == 0
    written = pd.read_parquet(tmp_path / "out" / RATINGS_FILENAME)
    assert len(written) == len(pd.read_parquet(matches))


def test_an_unknown_competition_exits_non_zero(matches: Path) -> None:
    """A filter that silently rates nothing looks exactly like a broken model."""
    assert CLI.main(["--matches", str(matches), "--competition", "NOWHERE_1"]) == 1


def test_fit_until_reports_and_does_not_write(
    matches: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A maintenance tool, not a pipeline step: fitting on rows the ratings
    later predict is a leak, so it prints constants for a human to adopt."""
    output = tmp_path / "out"
    assert (
        CLI.main(["--matches", str(matches), "--output", str(output), "--fit-until", "2021-01-01"])
        == 0
    )
    printed = capsys.readouterr().out
    assert "home_advantage" in printed
    assert "src/ratings/elo.py" in printed
    assert not output.exists()


def test_fit_until_before_any_data_exits_non_zero(matches: Path) -> None:
    assert CLI.main(["--matches", str(matches), "--fit-until", "1900-01-01"]) == 1


def test_selecting_models_by_name() -> None:
    assert [model.name for model in CLI.select_models([])] == ["dixon_coles", "elo"]
    assert [model.name for model in CLI.select_models(["elo"])] == ["elo"]
