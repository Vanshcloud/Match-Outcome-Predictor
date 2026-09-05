"""Persisting the model, reading it back, and finding the fixture to price.

The three things that stand between a fitted object and a served probability:
a file that round-trips, a loader that refuses anything it cannot vouch for,
and an index that resolves a match the way a caller would name one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.manifest import checksum
from src.models.artifact import fit_servable
from src.models.dataset import DESIGN_COLUMNS
from src.pipelines.serving import (
    LIBRARY_DISTRIBUTIONS,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    FixtureIndex,
    LoadedModel,
    ServingError,
    _library_versions,
    build_index,
    build_servable,
    describe_fixture,
    load_index,
    load_servable,
    predict_fixtures,
    save_servable,
)
from src.pipelines.tables import TablePaths, resolve_tables
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS
from src.utils.config import PathsConfig
from tests.factories import modelled_frame, season_labels

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

LEAGUE = modelled_frame(seasons=season_labels(2012, 8), teams=12)
MODEL = fit_servable(LEAGUE, members=("logistic_regression",))

RATING_COLUMNS = [*ELO_COLUMNS, *DIXON_COLES_COLUMNS]
FEATURE_COLUMNS = [column for column in DESIGN_COLUMNS if column not in RATING_COLUMNS]


@pytest.fixture
def written(tmp_path: Path) -> Path:
    """A model directory holding an artefact and its manifest."""
    save_servable(MODEL, tmp_path)
    return tmp_path


def _table_paths(root: Path, frame: pd.DataFrame = LEAGUE) -> TablePaths:
    """The three Parquet files a modelling frame is assembled from."""
    root.mkdir(parents=True, exist_ok=True)
    paths = TablePaths(
        matches=root / "matches.parquet",
        ratings=root / "ratings.parquet",
        features=root / "features.parquet",
    )
    frame.drop(columns=list(DESIGN_COLUMNS)).to_parquet(paths.matches, index=False)
    frame[["match_id", *RATING_COLUMNS]].to_parquet(paths.ratings, index=False)
    frame[["match_id", *FEATURE_COLUMNS]].to_parquet(paths.features, index=False)
    return paths


# ---- the round trip ----------------------------------------------------------


def test_a_saved_model_reads_back_as_the_same_model(written: Path) -> None:
    loaded = load_servable(written)
    assert isinstance(loaded, LoadedModel)
    assert loaded.model.name == MODEL.name
    assert loaded.model.member_names == MODEL.member_names
    assert loaded.model.temperature == pytest.approx(MODEL.temperature)
    assert loaded.model.trained_through == MODEL.trained_through


def test_the_reloaded_model_states_the_same_probabilities(written: Path) -> None:
    """The point of the artefact. A pickle that round-trips structurally and
    predicts differently is worse than one that fails to load."""
    before = MODEL.predict(LEAGUE.head(20))
    after = load_servable(written).model.predict(LEAGUE.head(20))
    assert (before == after).all()


def test_the_manifest_records_what_it_takes_to_reproduce_the_fit(written: Path) -> None:
    manifest = json.loads((written / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["kind"] == "servable-model"
    assert manifest["model"] == MODEL.name
    assert manifest["members"] == list(MODEL.member_names)
    assert manifest["trained_matches"] == len(LEAGUE)
    assert manifest["trained_through"] == str(LEAGUE["date"].max().date())
    assert set(manifest["libraries"]) == {"scikit-learn", "xgboost", "numpy"}
    assert manifest["files"][0]["path"] == MODEL_FILENAME


# ---- what the loader refuses -------------------------------------------------


def test_no_artefact_names_the_command_that_makes_one(tmp_path: Path) -> None:
    """The clean-checkout state. A stack trace here would be the first thing a
    new reader saw."""
    with pytest.raises(ServingError, match="make model"):
        load_servable(tmp_path)


def test_an_artefact_with_no_manifest_is_refused(written: Path) -> None:
    (written / MANIFEST_FILENAME).unlink()
    with pytest.raises(ServingError, match="cannot be identified"):
        load_servable(written)


def test_an_unreadable_manifest_is_refused(written: Path) -> None:
    (written / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ServingError, match="unreadable manifest"):
        load_servable(written)


def test_a_manifest_with_no_checksum_for_the_artefact_is_refused(written: Path) -> None:
    manifest = json.loads((written / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["files"] = []
    (written / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ServingError, match="records no checksum"):
        load_servable(written)


def test_a_truncated_artefact_is_caught_before_it_is_unpickled(written: Path) -> None:
    """An interrupted write or a half-copied file. Unpickling one raises at
    best and produces something plausible at worst, so the bytes are checked
    first — one hash, once per process."""
    path = written / MODEL_FILENAME
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
    with pytest.raises(ServingError, match="does not match its manifest checksum"):
        load_servable(written)


def test_a_file_that_is_not_a_pickle_at_all_is_reported_as_such(written: Path) -> None:
    path = written / MODEL_FILENAME
    path.write_bytes(b"this is not a pickle")
    _restamp(written, path)
    with pytest.raises(ServingError, match="could not unpickle"):
        load_servable(written)


def test_a_pickle_of_the_wrong_type_is_refused(written: Path) -> None:
    """A directory holding somebody else's joblib file. It unpickles fine and
    has no `predict`, which would otherwise be discovered on the first
    request."""
    import joblib

    path = written / MODEL_FILENAME
    joblib.dump({"not": "a model"}, path)
    _restamp(written, path)
    with pytest.raises(ServingError, match="holds dict, not a servable model"):
        load_servable(written)


def _restamp(model_dir: Path, path: Path) -> None:
    """Update the manifest's checksum, so a test about content is not a test
    about the checksum it already has its own case for."""
    manifest = json.loads((model_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = checksum(path)
    (model_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")


# ---- library provenance ------------------------------------------------------


def test_a_library_that_has_moved_since_the_fit_is_reported(written: Path) -> None:
    manifest = json.loads((written / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["libraries"]["numpy"] = "0.0.1"
    (written / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    mismatches = load_servable(written).library_mismatches
    assert len(mismatches) == 1
    assert mismatches[0].startswith("numpy: fitted with 0.0.1, running ")


def test_matching_libraries_report_nothing(written: Path) -> None:
    assert load_servable(written).library_mismatches == ()


def test_a_manifest_with_no_library_block_reports_nothing(written: Path) -> None:
    """An artefact from before the manifest recorded them. Silence is the right
    answer: there is nothing to compare, which is not the same as a mismatch."""
    manifest = json.loads((written / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    del manifest["libraries"]
    (written / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    assert load_servable(written).library_mismatches == ()


# ---- the fixture index -------------------------------------------------------


INDEX = build_index(LEAGUE)
SOME = LEAGUE.iloc[200]


def test_a_fixture_resolves_by_its_id() -> None:
    row = INDEX.resolve(match_id=str(SOME["match_id"]))
    assert row is not None
    assert row.iloc[0]["match_id"] == SOME["match_id"]


def test_a_fixture_resolves_by_competition_clubs_and_date() -> None:
    row = INDEX.resolve(
        competition_id=str(SOME["competition_id"]),
        home_team=str(SOME["home_team"]),
        away_team=str(SOME["away_team"]),
        date=str(SOME["date"].date()),
    )
    assert row is not None
    assert row.iloc[0]["match_id"] == SOME["match_id"]


def test_club_names_are_matched_past_case_and_spacing() -> None:
    """A provider's spacing is not something a caller should have to
    reproduce."""
    row = INDEX.resolve(
        competition_id=str(SOME["competition_id"]),
        home_team=f"  {str(SOME['home_team']).upper()}  ",
        away_team=str(SOME["away_team"]).lower(),
        date=str(SOME["date"].date()),
    )
    assert row is not None
    assert row.iloc[0]["match_id"] == SOME["match_id"]


def test_an_unknown_fixture_resolves_to_nothing() -> None:
    assert INDEX.resolve(match_id="no such match") is None
    assert (
        INDEX.resolve(
            competition_id="ENG_1",
            home_team="A Club That Does Not Exist",
            away_team=str(SOME["away_team"]),
            date=str(SOME["date"].date()),
        )
        is None
    )


def test_half_a_natural_key_resolves_to_nothing_rather_than_guessing() -> None:
    assert INDEX.resolve(competition_id="ENG_1", home_team=str(SOME["home_team"])) is None
    assert INDEX.resolve() is None


def test_two_fixtures_sharing_a_natural_key_keep_the_first_and_say_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The validation suite already fails a table with a duplicated fixture.
    What must not happen here is serving whichever row happened to be last."""
    doubled = pd.concat([LEAGUE.head(3), LEAGUE.head(3)], ignore_index=True)
    doubled.loc[3:, "match_id"] = ["dup-0", "dup-1", "dup-2"]
    with caplog.at_level("WARNING"):
        index = build_index(doubled)
    assert "share a natural key" in caplog.text

    first = doubled.iloc[0]
    row = index.resolve(
        competition_id=str(first["competition_id"]),
        home_team=str(first["home_team"]),
        away_team=str(first["away_team"]),
        date=str(first["date"].date()),
    )
    assert row is not None
    assert row.iloc[0]["match_id"] == first["match_id"]


def test_len_reports_what_was_indexed() -> None:
    assert len(INDEX) == len(LEAGUE)


# ---- searching ---------------------------------------------------------------


def test_search_returns_the_most_recent_first_and_respects_the_limit() -> None:
    found = INDEX.search(limit=5)
    assert len(found) == 5
    assert found["date"].is_monotonic_decreasing


def test_search_filters_by_competition_team_and_window() -> None:
    club = str(SOME["home_team"])
    by_team = INDEX.search(team=club, limit=200)
    assert len(by_team) > 0
    assert ((by_team["home_team"] == club) | (by_team["away_team"] == club)).all()

    assert (INDEX.search(competition_id="ENG_1", limit=5)["competition_id"] == "ENG_1").all()
    assert INDEX.search(competition_id="NOT_A_LEAGUE", limit=5).empty

    windowed = INDEX.search(since="2015-01-01", until="2015-12-31", limit=200)
    assert windowed["date"].min() >= pd.Timestamp("2015-01-01")
    assert windowed["date"].max() <= pd.Timestamp("2015-12-31")


# ---- describing and pricing --------------------------------------------------


def test_a_described_fixture_carries_no_result_and_no_odds() -> None:
    described = describe_fixture(SOME)
    assert described["match_id"] == SOME["match_id"]
    assert described["date"] == str(SOME["date"].date())
    assert "result" not in described
    assert not any(key.startswith("odds_") for key in described)


def test_a_null_descriptive_column_comes_back_as_null_not_as_the_string_nan() -> None:
    described = describe_fixture(SOME.copy().replace({SOME["country"]: None}))
    assert described["country"] is None


def test_predicting_returns_one_record_per_row_with_its_provenance() -> None:
    rows = LEAGUE.head(3)
    predictions = predict_fixtures(MODEL, rows, model_version="9.9.9")

    assert len(predictions) == 3
    first = predictions[0]
    assert first.model == MODEL.name
    assert first.model_version == "9.9.9"
    assert first.in_sample is True
    assert sum(first.probabilities.values()) == pytest.approx(1.0)
    assert set(first.probabilities) == {"home", "draw", "away"}


def test_a_prediction_record_is_the_row_the_log_stores() -> None:
    record = predict_fixtures(MODEL, LEAGUE.head(1))[0].as_record()
    assert record["match_id"] == LEAGUE.iloc[0]["match_id"]
    assert record["model"] == MODEL.name
    assert record["prob_home"] + record["prob_draw"] + record["prob_away"] == pytest.approx(1.0)
    assert record["in_sample"] is True


def test_pricing_rows_that_cannot_be_priced_is_a_serving_error() -> None:
    """A design column missing from the table means the feature build and the
    artefact disagree about what a match is. That is a deployment fault, and it
    surfaces under this package's own error rather than the model layer's."""
    with pytest.raises(ServingError, match="model column"):
        predict_fixtures(MODEL, LEAGUE.head(2).drop(columns=["elo_home"]))


# ---- building from the tables ------------------------------------------------


def test_a_model_can_be_built_straight_from_the_three_tables(tmp_path: Path) -> None:
    built = build_servable(_table_paths(tmp_path / "tables"), members=("logistic_regression",))
    assert built is not None
    assert built.trained_matches == len(LEAGUE)


def test_building_without_the_tables_returns_nothing_rather_than_raising(
    tmp_path: Path,
) -> None:
    """A clean checkout. `load_modelling_frame` has already logged which table
    is missing, and a stack trace at the expected state helps nobody."""
    absent = resolve_tables(PathsConfig(data_dir=tmp_path, model_dir=tmp_path))
    assert build_servable(absent) is None
    assert load_index(absent) is None


def test_an_index_can_be_loaded_from_the_three_tables(tmp_path: Path) -> None:
    index = load_index(_table_paths(tmp_path / "tables"))
    assert isinstance(index, FixtureIndex)
    assert len(index) == len(LEAGUE)


# ---- which distribution provides a library -----------------------------------


def test_a_library_is_found_under_either_of_its_distribution_names() -> None:
    """`xgboost-cpu` is the same module under a different distribution name.

    The serving image installs it to leave the CUDA runtime out, and a lookup
    by import name finds no metadata for it at all — which raised out of
    `/version`, the one endpoint whose job is to say what is running.
    """
    assert LIBRARY_DISTRIBUTIONS["xgboost"] == ("xgboost", "xgboost-cpu")
    assert set(_library_versions()) == set(LIBRARY_DISTRIBUTIONS)


def test_a_library_nothing_provides_is_reported_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance degrades to "unknown"; it does not become a 500.

    The drift check compares strings, so an unestablished version still reads
    as a mismatch against the manifest — which is the warning wanted.
    """
    monkeypatch.setitem(LIBRARY_DISTRIBUTIONS, "numpy", ("no-such-distribution",))
    assert _library_versions()["numpy"] == "unknown"
