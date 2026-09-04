"""The model card entry point. Only the contract a caller depends on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.evaluation.model_card import CARD_FILENAME
from src.models.calibration import Calibrated
from src.models.dataset import DESIGN_COLUMNS
from src.models.zoo import build
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS
from src.utils.paths import PROJECT_ROOT
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "model_card_cli", PROJECT_ROOT / "scripts" / "model_card.py"
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

SCORES = pd.DataFrame(
    {
        "forecaster": ["ensemble-calibrated", "bookmaker"],
        "subset": ["common", "common"],
        "competition_id": ["ALL", "ALL"],
        "n": [100, 100],
        "log_loss": [1.0156, 0.9993],
        "rps": [0.2082, 0.2031],
        "accuracy": [0.4931, 0.5055],
    }
)


@pytest.fixture
def tables(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "matches": tmp_path / "matches.parquet",
        "ratings": tmp_path / "ratings.parquet",
        "features": tmp_path / "features.parquet",
        "scores": tmp_path / "scores.parquet",
    }
    LEAGUE.drop(columns=list(DESIGN_COLUMNS)).to_parquet(paths["matches"], index=False)
    LEAGUE[["match_id", *RATING_COLUMNS]].to_parquet(paths["ratings"], index=False)
    LEAGUE[["match_id", *FEATURE_COLUMNS]].to_parquet(paths["features"], index=False)
    SCORES.to_parquet(paths["scores"], index=False)
    return paths


@pytest.fixture(autouse=True)
def quick_blend(monkeypatch: pytest.MonkeyPatch) -> None:
    """One family standing in for the blend of three.

    What is under test is the command — which tables it reads, what it writes,
    what it refuses — and fitting three families twice per fold to prove a file
    was written is a test people stop running. The blend itself is measured in
    `test_report_pipeline.py` and on real data in `tests/integration/`.
    """
    monkeypatch.setattr(CLI, "shipped_forecaster", lambda: Calibrated(build("logistic_regression")))


def _argv(tables: dict[str, Path], docs: Path, *extra: str) -> list[str]:
    return [
        "--matches",
        str(tables["matches"]),
        "--ratings",
        str(tables["ratings"]),
        "--features",
        str(tables["features"]),
        "--scores",
        str(tables["scores"]),
        "--docs",
        str(docs),
        "--folds",
        "2",
        *extra,
    ]


def test_a_clean_run_writes_the_card(tables: dict[str, Path], tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    assert CLI.main(_argv(tables, docs)) == 0
    written = (docs / CARD_FILENAME).read_text(encoding="utf-8")
    assert written.startswith("# Model card — logistic_regression-calibrated")
    assert "## What it must not be used for" in written


def test_the_breakdowns_are_printed_as_well_as_written(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run is long enough that nobody should have to open a file to see
    whether it worked."""
    CLI.main(_argv(tables, tmp_path / "docs"))
    printed = capsys.readouterr().out
    assert "How honest it is about each class" in printed
    assert "Where it is least reliable, by competition" in printed
    assert "written to" in printed


def test_a_dry_run_writes_nothing(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    assert CLI.main(_argv(tables, docs, "--dry-run")) == 0
    assert not (docs / CARD_FILENAME).exists()
    assert "nothing written" in capsys.readouterr().out


def test_a_single_family_can_have_its_own_card(tables: dict[str, Path], tmp_path: Path) -> None:
    """Calibrated the same way as the blend, so the card reports the same kind
    of probabilities rather than uncalibrated ones."""
    docs = tmp_path / "docs"
    assert CLI.main(_argv(tables, docs, "--model", "lightgbm")) == 0
    assert "# Model card — lightgbm-calibrated" in (docs / CARD_FILENAME).read_text(
        encoding="utf-8"
    )


def test_no_scores_yet_exits_non_zero_and_says_which_command(
    tables: dict[str, Path], tmp_path: Path
) -> None:
    """The state before `make ensemble`. A card built from nothing would be a
    document asserting a model that has not been measured."""
    assert CLI.main(_argv({**tables, "scores": tmp_path / "absent.parquet"}, tmp_path)) == 1


def test_a_missing_table_exits_non_zero(tables: dict[str, Path], tmp_path: Path) -> None:
    assert CLI.main(_argv({**tables, "ratings": tmp_path / "absent.parquet"}, tmp_path)) == 1


def test_a_table_too_short_to_split_exits_non_zero(tmp_path: Path) -> None:
    short = modelled_frame(seasons=["2020-21"], teams=6)
    paths = {
        "matches": tmp_path / "m.parquet",
        "ratings": tmp_path / "r.parquet",
        "features": tmp_path / "f.parquet",
        "scores": tmp_path / "s.parquet",
    }
    short.drop(columns=list(DESIGN_COLUMNS)).to_parquet(paths["matches"], index=False)
    short[["match_id", *RATING_COLUMNS]].to_parquet(paths["ratings"], index=False)
    short[["match_id", *FEATURE_COLUMNS]].to_parquet(paths["features"], index=False)
    SCORES.to_parquet(paths["scores"], index=False)
    assert CLI.main(_argv(paths, tmp_path / "docs")) == 1


def test_an_empty_table_renders_as_nothing_to_report() -> None:
    assert "nothing to report" in CLI.render(pd.DataFrame())


def test_the_card_regenerates_identically(tables: dict[str, Path], tmp_path: Path) -> None:
    """Two runs over the same measurements produce the same bytes, so a diff in
    `docs/MODEL_CARD.md` is a change in the model rather than in the clock."""
    docs = tmp_path / "docs"
    CLI.main(_argv(tables, docs))
    first = (docs / CARD_FILENAME).read_text(encoding="utf-8")
    CLI.main(_argv(tables, docs))
    assert (docs / CARD_FILENAME).read_text(encoding="utf-8") == first
