"""The feature entry point. Only the contract a caller depends on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.pipelines.features import FEATURES_FILENAME
from src.utils.paths import PROJECT_ROOT
from tests.factories import league_frame


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_features_cli", PROJECT_ROOT / "scripts" / "build_features.py"
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
    code = CLI.main(["--matches", str(matches), "--output", str(tmp_path / "out")])
    assert code == 0
    assert (tmp_path / "out" / FEATURES_FILENAME).is_file()


def test_a_missing_table_exits_non_zero(tmp_path: Path) -> None:
    """The state of a clean checkout. It must say what to run, not traceback."""
    assert CLI.main(["--matches", str(tmp_path / "absent.parquet")]) == 1


def test_an_unknown_competition_exits_non_zero(matches: Path) -> None:
    """A filter that silently builds nothing looks exactly like a broken builder."""
    assert CLI.main(["--matches", str(matches), "--competition", "NOWHERE_1"]) == 1


def test_a_single_builder_can_be_selected(matches: Path, tmp_path: Path) -> None:
    code = CLI.main(
        [
            "--matches",
            str(matches),
            "--output",
            str(tmp_path / "out"),
            "--builder",
            "head_to_head",
            "--no-verify",
        ]
    )
    assert code == 0
    written = pd.read_parquet(tmp_path / "out" / FEATURES_FILENAME)
    assert written["h2h_matches"].notna().all()
    assert written["home_form_points_5"].isna().all()


def test_the_registry_can_be_listed(capsys: pytest.CaptureFixture[str]) -> None:
    """The one command that answers "what does this project know about a
    fixture", and it says which side of kick-off each answer comes from."""
    assert CLI.main(["--list"]) == 0
    printed = capsys.readouterr().out
    assert "home_form_points_5" in printed
    assert "post-match" in printed
    assert "pre-match" in printed


def test_selecting_builders_by_name() -> None:
    assert [b.name for b in CLI.select_builders([])] == ["head_to_head", "team_history"]
    assert [b.name for b in CLI.select_builders(["team_history"])] == ["team_history"]
