"""The model-building entry point. Only the contract a caller depends on."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.models.dataset import DESIGN_COLUMNS
from src.pipelines.serving import MANIFEST_FILENAME, MODEL_FILENAME, load_servable
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS
from src.utils.paths import PROJECT_ROOT
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_model_cli", PROJECT_ROOT / "scripts" / "build_model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()
LEAGUE = modelled_frame(seasons=season_labels(2012, 8), teams=12)
RATING_COLUMNS = [*ELO_COLUMNS, *DIXON_COLES_COLUMNS]
FEATURE_COLUMNS = [column for column in DESIGN_COLUMNS if column not in RATING_COLUMNS]


@pytest.fixture
def tables(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "matches": tmp_path / "matches.parquet",
        "ratings": tmp_path / "ratings.parquet",
        "features": tmp_path / "features.parquet",
    }
    LEAGUE.drop(columns=list(DESIGN_COLUMNS)).to_parquet(paths["matches"], index=False)
    LEAGUE[["match_id", *RATING_COLUMNS]].to_parquet(paths["ratings"], index=False)
    LEAGUE[["match_id", *FEATURE_COLUMNS]].to_parquet(paths["features"], index=False)
    return paths


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
        # One family standing in for the blend of three. What is under test is
        # the command — which tables it reads, what it writes, what it refuses
        # — and fitting an MLP to prove a file appeared is a test people stop
        # running.
        "--member",
        "logistic_regression",
        *extra,
    ]


def test_a_clean_run_writes_an_artefact_and_its_manifest(
    tables: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "models"
    assert CLI.main(_argv(tables, output)) == 0

    assert (output / MODEL_FILENAME).is_file()
    manifest = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["kind"] == "servable-model"
    assert manifest["members"] == ["logistic_regression"]
    assert manifest["trained_matches"] == len(LEAGUE)


def test_what_it_wrote_loads_back(tables: dict[str, Path], tmp_path: Path) -> None:
    """The command and the service have to agree about the file, which is only
    provable by writing one with the first and reading it with the second."""
    output = tmp_path / "models"
    CLI.main(_argv(tables, output))
    assert load_servable(output).model.trained_matches == len(LEAGUE)


def test_the_run_reports_the_window_it_fitted_on(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fit takes minutes on the real table. Nobody should have to open a JSON
    file to find out what it did.

    Read off stderr rather than through ``caplog``: the command configures
    logging itself, which replaces the root handlers — pytest's capture handler
    among them — so the fixture would see nothing and the assertion would pass
    only by being empty."""
    CLI.main(_argv(tables, tmp_path / "models"))
    logged = capsys.readouterr().err
    assert str(LEAGUE["date"].max().date()) in logged
    assert "logistic_regression" in logged


def test_a_dry_run_writes_nothing(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "models"
    assert CLI.main(_argv(tables, output, "--dry-run")) == 0
    assert not (output / MODEL_FILENAME).exists()
    assert "not written" in capsys.readouterr().err


def test_the_history_can_be_restricted_to_named_competitions(
    tables: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "models"
    assert CLI.main(_argv(tables, output, "--competition", "ENG_1")) == 0
    assert load_servable(output).model.trained_matches == len(LEAGUE)


def test_no_tables_exits_non_zero_rather_than_writing_half_a_model(tmp_path: Path) -> None:
    """The clean-checkout state. `load_modelling_frame` has already logged which
    table is missing, so this exits rather than repeating it less
    specifically."""
    absent = {name: tmp_path / f"{name}.parquet" for name in ("matches", "ratings", "features")}
    output = tmp_path / "models"
    assert CLI.main(_argv(absent, output)) == 1
    assert not (output / MODEL_FILENAME).exists()


def test_a_competition_filter_that_selects_nothing_exits_non_zero(
    tables: dict[str, Path], tmp_path: Path
) -> None:
    assert CLI.main(_argv(tables, tmp_path / "models", "--competition", "NOT_A_LEAGUE")) == 1


def test_the_written_path_is_printed(
    tables: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    CLI.main(_argv(tables, tmp_path / "models"))
    assert MODEL_FILENAME in capsys.readouterr().out


def test_the_default_members_are_the_blend_the_card_describes() -> None:
    """The command's default has to be the model the documentation is about,
    not whichever family is convenient."""
    from src.models.ensemble import MEMBERS

    assert CLI.parse_args([]).member is None
    assert MEMBERS == ("xgboost", "logistic_regression", "mlp")


def test_dates_survive_the_round_trip_through_parquet(tables: dict[str, Path]) -> None:
    """The training window is read off the table the command loads, not off the
    frame in memory, so a dtype lost in Parquet would move it silently."""
    matches = pd.read_parquet(tables["matches"])
    assert matches["date"].max() == LEAGUE["date"].max()
