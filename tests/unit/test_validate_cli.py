"""The validation entry point.

Only the contract a caller depends on: the exit code. A validation command that
prints "FAILED" and exits zero is worse than no command at all, because a
pipeline believes it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT
from tests.factories import league_frame


def _load_cli() -> ModuleType:
    """Import the script by path.

    ``scripts/`` is not a package — the files are entry points, not importable
    modules — so this is the only way to exercise them without turning them
    into one.
    """
    spec = importlib.util.spec_from_file_location(
        "validate_data_cli", PROJECT_ROOT / "scripts" / "validate_data.py"
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
    league_frame().to_parquet(path, index=False)
    return path


def test_a_valid_table_exits_zero(matches: Path, tmp_path: Path) -> None:
    code = CLI.main(["--matches", str(matches), "--card", str(tmp_path / "card.md")])
    assert code == 0
    assert (tmp_path / "card.md").is_file()


def test_a_missing_table_exits_non_zero(tmp_path: Path) -> None:
    """The state of a clean checkout. It must say what to run, not traceback."""
    assert CLI.main(["--matches", str(tmp_path / "absent.parquet")]) == 1


def test_a_blocking_failure_exits_non_zero(tmp_path: Path) -> None:
    path = tmp_path / "matches.parquet"
    frame = league_frame()
    frame.loc[0, "result"] = "H" if frame.loc[0, "result"] != "H" else "A"
    frame.to_parquet(path, index=False)
    assert CLI.main(["--matches", str(path), "--no-card"]) == 1


def test_a_warning_alone_does_not_fail_the_run(tmp_path: Path) -> None:
    """One Argentinian match filed under the wrong season is not a reason to
    refuse three hundred thousand rows."""
    path = tmp_path / "matches.parquet"
    frame = league_frame()
    frame.loc[0, "date"] = pd.Timestamp("2019-01-01")
    frame = frame.sort_values(["date", "competition_id", "match_id"], kind="stable")
    frame.reset_index(drop=True).to_parquet(path, index=False)
    assert CLI.main(["--matches", str(path), "--no-card"]) == 0
    assert CLI.main(["--matches", str(path), "--no-card", "--strict"]) == 1


def test_the_report_can_be_written_as_json(matches: Path, tmp_path: Path) -> None:
    destination = tmp_path / "reports" / "validation.json"
    CLI.main(["--matches", str(matches), "--no-card", "--json", str(destination)])
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["rows"] == 5320


def test_no_card_leaves_the_committed_one_untouched(matches: Path, tmp_path: Path) -> None:
    """The default destination is a file in the repository. A test that forgot
    --no-card would rewrite it from synthetic data."""
    sentinel = tmp_path / "card.md"
    CLI.main(["--matches", str(matches), "--no-card", "--card", str(sentinel)])
    assert not sentinel.exists()
