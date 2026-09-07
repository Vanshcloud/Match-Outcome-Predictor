"""The drift entry point. Only the contract a caller depends on.

No database and no service. The prediction log is injected — the seam
:func:`~src.storage.predictions.open_prediction_log` exists for — so this drives
the whole command, including the exits, without a server.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.ingestion.base import Result
from src.models.ensemble import SHIPPED
from src.utils.paths import PROJECT_ROOT
from tests.unit.test_archive import archive, folds, played, served


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "archive_cli", PROJECT_ROOT / "scripts" / "archive.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()
DSN = "postgresql://predictor:predictor@localhost:5432/predictions"


class StubLog:
    """A prediction log that never opened a connection."""

    def __init__(self, rows: pd.DataFrame) -> None:
        self._rows = rows
        self.closed = False
        self.limit: int | None = None

    def ensure_schema(self) -> None:
        return None

    def recent(self, limit: int = 20) -> pd.DataFrame:
        self.limit = limit
        return self._rows

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data directory holding the two tables the command joins."""
    processed, reports = tmp_path / "processed", tmp_path / "reports" / "ensemble"
    processed.mkdir(parents=True)
    reports.mkdir(parents=True)
    matches = played(("a", Result.HOME), ("b", Result.AWAY))
    matches.assign(competition_id="ENG_1", date=pd.Timestamp("2026-09-05")).to_parquet(
        processed / "matches.parquet", index=False
    )
    folds().to_parquet(reports / "forecasts.parquet", index=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PREDICTION_LOG_DSN", DSN)
    return tmp_path


def _log(monkeypatch: pytest.MonkeyPatch, rows: pd.DataFrame) -> StubLog:
    """Stand the command's log up without a server."""
    stub = StubLog(rows)
    monkeypatch.setattr(CLI, "open_prediction_log", lambda *_, **__: stub)
    return stub


def test_a_log_that_has_never_been_written_to_is_reported_not_refused(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A service nobody has called yet is the ordinary state of a fresh
    deployment. The command still writes the file and still prints how much
    archive a verdict would need."""
    _log(monkeypatch, pd.DataFrame())
    assert CLI.main([]) == 0
    printed = capsys.readouterr().out
    assert "0 row(s) read" in printed
    assert "How much archive a verdict would need" in printed
    assert (workspace / "reports" / "ensemble" / "archive.parquet").exists()


def test_the_report_is_written_beside_the_other_report_tables(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _log(monkeypatch, archive(served("a"), served("b")))
    assert CLI.main([]) == 0
    written = pd.read_parquet(workspace / "reports" / "ensemble" / "archive.parquet")
    assert list(written["model"]) == [SHIPPED]
    assert int(written["n"].iloc[0]) == 2
    assert (workspace / "reports" / "ensemble" / "archive.manifest.json").exists()


@pytest.mark.usefixtures("workspace")
def test_the_whole_log_is_asked_for_rather_than_the_default_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``recent`` defaults to twenty rows, which is right for a page of a
    dashboard and would silently score a fifth of a real archive."""
    stub = _log(monkeypatch, archive(served("a")))
    assert CLI.main([]) == 0
    assert stub.limit == CLI.DEFAULT_LIMIT
    assert stub.closed


def test_a_dry_run_prints_the_report_and_writes_nothing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _log(monkeypatch, archive(served("a")))
    assert CLI.main(["--dry-run"]) == 0
    assert "nothing written" in capsys.readouterr().out
    assert not (workspace / "reports" / "ensemble" / "archive.parquet").exists()


@pytest.mark.usefixtures("workspace")
def test_without_a_dsn_the_command_says_which_variable_to_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The log is optional everywhere else in this project, so its absence has
    to be a sentence rather than a traceback."""
    monkeypatch.delenv("PREDICTION_LOG_DSN")
    assert CLI.main([]) == 1


def test_without_a_match_table_there_is_nothing_to_score_against(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "processed" / "matches.parquet").unlink()
    _log(monkeypatch, archive(served("a")))
    assert CLI.main([]) == 1


def test_without_the_fold_forecasts_there_is_no_baseline_to_compare_to(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A served log loss on its own is a number with nothing to be read
    against, and the spread that says whether it means anything comes from the
    same table."""
    (workspace / "reports" / "ensemble" / "forecasts.parquet").unlink()
    _log(monkeypatch, archive(served("a")))
    assert CLI.main([]) == 1


def test_an_empty_frame_renders_as_a_sentence_rather_than_a_blank() -> None:
    assert "nothing to report" in CLI.render(pd.DataFrame())


def test_the_command_is_wired_into_the_makefile() -> None:
    """Its own target rather than part of `make card`: its input is the
    prediction log, and `make reproduce` cannot rebuild it."""
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "scripts/archive.py" in makefile
    assert "\narchive:" in makefile


def test_the_horizons_the_command_prints_are_the_ones_the_page_shows() -> None:
    """Two lists of the same five numbers is two lists that eventually differ,
    and the difference would be a page and a command disagreeing about how much
    archive is still needed."""
    from dashboard.views import performance

    assert CLI.WORTH_SEEING == performance.WORTH_SEEING
