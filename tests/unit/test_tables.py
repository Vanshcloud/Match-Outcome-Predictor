"""The three tables, joined once for every command that needs them.

What is tested here is the seam, not the store: that a missing table is a
reported absence rather than a stack trace, that the join keeps matches the
ratings have never heard of, and that an override is honoured — because the
version of this that lived in one script was on its way into two more.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models.dataset import DESIGN_COLUMNS
from src.pipelines.tables import (
    TablePaths,
    load_modelling_frame,
    read_scores,
    resolve_tables,
)
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS
from src.utils.config import load_settings
from tests.factories import modelled_frame, season_labels

LEAGUE = modelled_frame(seasons=season_labels(2015, 4), teams=8)
RATING_COLUMNS = [*ELO_COLUMNS, *DIXON_COLES_COLUMNS]
FEATURE_COLUMNS = [column for column in DESIGN_COLUMNS if column not in RATING_COLUMNS]


@pytest.fixture
def tables(tmp_path: Path) -> TablePaths:
    paths = TablePaths(
        matches=tmp_path / "matches.parquet",
        ratings=tmp_path / "ratings.parquet",
        features=tmp_path / "features.parquet",
    )
    LEAGUE.drop(columns=list(DESIGN_COLUMNS)).to_parquet(paths.matches, index=False)
    LEAGUE[["match_id", *RATING_COLUMNS]].to_parquet(paths.ratings, index=False)
    LEAGUE[["match_id", *FEATURE_COLUMNS]].to_parquet(paths.features, index=False)
    return paths


def test_the_join_produces_every_model_column(tables: TablePaths) -> None:
    frame = load_modelling_frame(tables)
    assert frame is not None
    assert set(DESIGN_COLUMNS) <= set(frame.columns)
    assert len(frame) == len(LEAGUE)


def test_a_match_the_ratings_never_heard_of_keeps_its_row(tables: TablePaths) -> None:
    """An inner join would drop every club's first fixtures and improve every
    score. The models read the missing columns as the null they are."""
    thin = pd.read_parquet(tables.ratings).head(10)
    thin.to_parquet(tables.ratings, index=False)
    frame = load_modelling_frame(tables)
    assert frame is not None
    assert len(frame) == len(LEAGUE)
    assert frame["elo_home"].isna().sum() == len(LEAGUE) - 10


def test_one_competition_can_be_asked_for(tables: TablePaths) -> None:
    assert load_modelling_frame(tables, competitions=["ENG_1"]) is not None
    assert load_modelling_frame(tables, competitions=["NOWHERE_1"]) is None


def test_a_missing_table_is_reported_rather_than_raised(tables: TablePaths, tmp_path: Path) -> None:
    """The state of a clean checkout. A command that stack-traced there would
    be failing at the expected state."""
    absent = TablePaths(
        matches=tables.matches, ratings=tables.ratings, features=tmp_path / "gone.parquet"
    )
    assert absent.missing() == ("features",)
    assert load_modelling_frame(absent) is None


def test_every_absent_table_is_named_at_once(tmp_path: Path) -> None:
    """Three separate runs to discover three missing files is three runs."""
    nothing = TablePaths(
        matches=tmp_path / "a.parquet",
        ratings=tmp_path / "b.parquet",
        features=tmp_path / "c.parquet",
    )
    assert nothing.missing() == ("match table", "ratings", "features")


def test_the_defaults_come_from_the_configured_directories() -> None:
    paths = load_settings().paths
    resolved = resolve_tables(paths)
    assert resolved.matches.parent == paths.processed_dir
    assert resolved.ratings.parent == paths.features_dir
    assert resolved.features.parent == paths.features_dir


def test_an_override_replaces_only_what_it_names(tmp_path: Path) -> None:
    paths = load_settings().paths
    resolved = resolve_tables(paths, matches=tmp_path / "elsewhere.parquet")
    assert resolved.matches == tmp_path / "elsewhere.parquet"
    assert resolved.ratings == resolve_tables(paths).ratings


def test_scores_are_read_back_through_the_store(tmp_path: Path) -> None:
    """Not with a direct read. `src/storage` is the only package allowed to
    know what format these are in."""
    written = pd.DataFrame({"forecaster": ["a", "b"], "log_loss": [1.0, 2.0]})
    path = tmp_path / "scores.parquet"
    written.to_parquet(path, index=False)
    read = read_scores(path)
    assert read is not None
    assert list(read["forecaster"]) == ["a", "b"]


def test_scores_that_are_not_there_yet_are_not_an_error(tmp_path: Path) -> None:
    assert read_scores(tmp_path / "never-run.parquet") is None
