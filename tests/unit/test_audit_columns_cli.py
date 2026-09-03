"""The audit entry point. Only the contract a caller depends on.

The audit is expensive by construction — it recomputes every producer twice per
canonical column — so these run against a small synthetic league with the
column list cut down to what the assertions are actually about. What is being
tested here is the CLI's exit codes and its sampling, not the tracing; that has
its own file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT
from src.validation import leakage
from tests.factories import league_frame

SEASONS = ("2019-20", "2020-21", "2021-22")


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_columns_cli", PROJECT_ROOT / "scripts" / "audit_columns.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


@pytest.fixture
def matches(tmp_path: Path) -> Path:
    england = league_frame(seasons=SEASONS, teams=8)
    spain = league_frame(
        competition_id="ESP_1",
        country="Spain",
        name="La Liga",
        seasons=SEASONS,
        teams=8,
        team_prefix="Club",
    )
    path = tmp_path / "matches.parquet"
    pd.concat([england, spain], ignore_index=True).sort_values(
        ["date", "competition_id", "match_id"], kind="stable"
    ).to_parquet(path, index=False)
    return path


@pytest.fixture
def cheap_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace two columns rather than thirty-one, so the CLI tests are seconds.

    The audit's cost is linear in the columns it perturbs and these tests are
    about exit codes, not about the trace. `test_leakage_suite.py` runs the
    full one.
    """
    monkeypatch.setattr(
        CLI,
        "audit",
        lambda frame, found=None, **_: leakage.audit(
            frame, found, columns=("home_goals", "home_team_id")
        ),
    )


@pytest.mark.usefixtures("cheap_trace")
def test_a_clean_audit_exits_zero(matches: Path) -> None:
    assert CLI.main(["--matches", str(matches), "--limit", "60"]) == 0


@pytest.mark.usefixtures("cheap_trace")
def test_the_markdown_table_names_every_traced_column(
    matches: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert CLI.main(["--matches", str(matches), "--limit", "60", "--markdown"]) == 0
    printed = capsys.readouterr().out
    assert "| `elo_home` | elo |" in printed
    assert "| `h2h_matches` | head_to_head |" in printed


@pytest.mark.usefixtures("cheap_trace")
def test_named_competitions_are_both_sampled(
    matches: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = CLI.main(
        [
            "--matches",
            str(matches),
            "--competition",
            "ENG_1",
            "--competition",
            "ESP_1",
            "--limit",
            "40",
        ]
    )
    assert code == 0
    assert "80 matches from ENG_1, ESP_1" in capsys.readouterr().out


def test_a_missing_table_exits_non_zero(tmp_path: Path) -> None:
    """The state of a clean checkout. It must say what to run, not traceback."""
    assert CLI.main(["--matches", str(tmp_path / "absent.parquet")]) == 1


def test_an_empty_table_exits_non_zero(tmp_path: Path) -> None:
    """There is no sample to audit, which is a different failure from a clean one."""
    path = tmp_path / "empty.parquet"
    league_frame(seasons=["2020-21"]).head(0).to_parquet(path, index=False)
    assert CLI.main(["--matches", str(path)]) == 1


@pytest.mark.usefixtures("cheap_trace")
def test_a_failing_probe_exits_non_zero(matches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    planted = leakage.TemporalResult(
        name="planted", probe="prefix invariance", checks=1, violations=("cutoff=2",)
    )
    monkeypatch.setattr(CLI, "run_suite", lambda *_, **__: (planted,))
    assert CLI.main(["--matches", str(matches), "--limit", "60"]) == 1


def test_a_column_reading_the_benchmark_exits_non_zero(
    matches: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure the audit exists to catch, planted."""

    def cheat(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {leakage.KEY_COLUMN: frame[leakage.KEY_COLUMN], "tip": frame["odds_home"].to_numpy()}
        )

    planted = leakage.Producer(
        name="cheat", kind="feature", compute=cheat, outputs=("tip",), declared_reads=None
    )
    monkeypatch.setattr(CLI, "producers", lambda: (planted,))
    monkeypatch.setattr(
        CLI,
        "audit",
        lambda frame, found=None, **_: leakage.audit(frame, found, columns=("odds_home",)),
    )
    assert CLI.main(["--matches", str(matches), "--limit", "60"]) == 1
    assert "BENCHMARK LEAK: tip" in capsys.readouterr().out


def test_a_feature_reading_more_than_it_declares_exits_non_zero(
    matches: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def understating(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                leakage.KEY_COLUMN: frame[leakage.KEY_COLUMN],
                "h2h_matches": frame["home_goals"].to_numpy(),
            }
        )

    planted = leakage.Producer(
        name="understating",
        kind="feature",
        compute=understating,
        outputs=("h2h_matches",),
        declared_reads=frozenset({"date"}),
    )
    monkeypatch.setattr(CLI, "producers", lambda: (planted,))
    monkeypatch.setattr(
        CLI,
        "audit",
        lambda frame, found=None, **_: leakage.audit(frame, found, columns=("home_goals",)),
    )
    assert CLI.main(["--matches", str(matches), "--limit", "60"]) == 1
    assert "UNDECLARED: h2h_matches" in capsys.readouterr().out
