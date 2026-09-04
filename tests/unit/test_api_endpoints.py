"""The endpoints with a model and a table actually loaded.

One artefact, fitted once for the module and served through the real
application — including the lifespan, which is what makes this a test of the
service rather than of a hand-assembled object that resembles it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.service import PredictionService
from src.models.artifact import fit_servable
from src.models.dataset import DESIGN_COLUMNS
from src.pipelines.features import FEATURES_FILENAME
from src.pipelines.ingest import MATCHES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.pipelines.serving import build_index, save_servable
from src.ratings.base import DIXON_COLES_COLUMNS, ELO_COLUMNS
from src.storage.predictions import PostgresPredictionLog
from src.utils.config import ApiConfig, LoggingConfig, PathsConfig, Settings
from tests.factories import modelled_frame, season_labels
from tests.unit.test_predictions_log import FakeConnection

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

LEAGUE = modelled_frame(seasons=season_labels(2012, 8), teams=12)
MODEL = fit_servable(LEAGUE, members=("logistic_regression",))
INDEX = build_index(LEAGUE)

SOME = LEAGUE.iloc[100]
RATING_COLUMNS = [*ELO_COLUMNS, *DIXON_COLES_COLUMNS]
FEATURE_COLUMNS = [column for column in DESIGN_COLUMNS if column not in RATING_COLUMNS]


def by_id() -> dict[str, Any]:
    return {"match_id": str(SOME["match_id"])}


def by_key() -> dict[str, Any]:
    return {
        "competition_id": str(SOME["competition_id"]),
        "home_team": str(SOME["home_team"]),
        "away_team": str(SOME["away_team"]),
        "date": str(SOME["date"].date()),
    }


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """A real tree: the artefact under ``models/``, the three tables under
    ``data/``. Built through the same functions the deployment uses, so this
    covers `build_service`'s successful path rather than a stand-in for it."""
    root = tmp_path_factory.mktemp("deployment")
    paths = PathsConfig(data_dir=root / "data", model_dir=root / "models")

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.features_dir.mkdir(parents=True, exist_ok=True)
    LEAGUE.drop(columns=list(DESIGN_COLUMNS)).to_parquet(
        paths.processed_dir / MATCHES_FILENAME, index=False
    )
    LEAGUE[["match_id", *RATING_COLUMNS]].to_parquet(
        paths.features_dir / RATINGS_FILENAME, index=False
    )
    LEAGUE[["match_id", *FEATURE_COLUMNS]].to_parquet(
        paths.features_dir / FEATURES_FILENAME, index=False
    )
    save_servable(MODEL, paths.model_dir)

    return Settings(
        paths=paths,
        api=ApiConfig(max_batch=3),
        logging=LoggingConfig(level="WARNING"),
    )


@pytest.fixture(scope="module")
def client(loaded: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(loaded)) as entered:
        yield entered


# ---- readiness and provenance ------------------------------------------------


def test_a_loaded_service_is_healthy(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    components = {part["name"]: part for part in body["components"]}
    assert components["model"]["ready"] is True
    assert components["model"]["detail"] == MODEL.name
    assert components["fixtures"]["detail"] == f"{len(LEAGUE):,} fixtures indexed"


def test_version_reports_what_the_model_was_fitted_on(client: TestClient) -> None:
    body = client.get("/version").json()
    assert body["model"] == MODEL.name
    assert body["members"] == list(MODEL.member_names)
    assert body["design_columns"] == len(DESIGN_COLUMNS)
    assert body["temperature"] == pytest.approx(MODEL.temperature)
    assert body["trained_matches"] == len(LEAGUE)
    assert body["trained_through"] == str(LEAGUE["date"].max().date())
    assert body["fixtures_indexed"] == len(LEAGUE)
    assert body["library_mismatches"] == []


def test_the_limitations_name_the_loaded_model(client: TestClient) -> None:
    assert client.get("/model-card/limitations").json()["model"] == MODEL.name


# ---- finding a fixture -------------------------------------------------------


def test_fixtures_are_listed_newest_first(client: TestClient) -> None:
    body = client.get("/fixtures", params={"limit": 5}).json()
    assert body["count"] == 5
    dates = [row["date"] for row in body["fixtures"]]
    assert dates == sorted(dates, reverse=True)


def test_fixtures_can_be_filtered_to_one_club(client: TestClient) -> None:
    club = str(SOME["home_team"])
    body = client.get("/fixtures", params={"team": club, "limit": 10}).json()
    assert body["count"] > 0
    assert all(club in (row["home_team"], row["away_team"]) for row in body["fixtures"])


def test_a_filter_that_matches_nothing_is_an_empty_list_not_a_404(client: TestClient) -> None:
    """No fixtures is data, not an error."""
    body = client.get("/fixtures", params={"competition_id": "NOT_A_LEAGUE"}).json()
    assert body == {"fixtures": [], "count": 0}


# ---- one prediction ----------------------------------------------------------


def test_a_fixture_named_by_id_is_priced(client: TestClient) -> None:
    body = client.post("/predict", json=by_id()).json()

    assert body["fixture"]["match_id"] == SOME["match_id"]
    assert body["fixture"]["home_team"] == SOME["home_team"]
    assert "result" not in body["fixture"]
    assert body["model"] == MODEL.name
    assert body["in_sample"] is True
    assert body["limitations_url"] == "/model-card/limitations"

    stated = body["probabilities"]
    assert set(stated) == {"home", "draw", "away"}
    assert sum(stated.values()) == pytest.approx(1.0)


def test_naming_the_same_fixture_either_way_gives_the_same_answer(client: TestClient) -> None:
    """Two spellings of one match, so they must not be two forecasts."""
    assert (
        client.post("/predict", json=by_id()).json()["probabilities"]
        == client.post("/predict", json=by_key()).json()["probabilities"]
    )


def test_a_prediction_matches_the_model_called_directly(client: TestClient) -> None:
    """Nothing between the estimator and the response reshapes the numbers."""
    direct = MODEL.predict(INDEX.resolve(match_id=str(SOME["match_id"])))
    served = client.post("/predict", json=by_id()).json()["probabilities"]
    assert served["home"] == pytest.approx(direct[0][0])
    assert served["draw"] == pytest.approx(direct[0][1])
    assert served["away"] == pytest.approx(direct[0][2])


def test_a_fixture_the_table_does_not_hold_is_404(client: TestClient) -> None:
    response = client.post("/predict", json={"match_id": "ENG_1:1066-67:00001"})
    assert response.status_code == 404
    assert "no fixture" in response.json()["detail"]


# ---- batches -----------------------------------------------------------------


def test_a_batch_prices_every_fixture_it_can_resolve(client: TestClient) -> None:
    ids = [{"match_id": str(row)} for row in LEAGUE["match_id"].head(3)]
    body = client.post("/predict/batch", json={"fixtures": ids}).json()

    assert len(body["predictions"]) == 3
    assert body["unresolved"] == []
    assert [one["fixture"]["match_id"] for one in body["predictions"]] == [
        one["match_id"] for one in ids
    ]


def test_one_bad_fixture_does_not_cost_the_others(client: TestClient) -> None:
    """Forty-nine forecasts and a named gap beats an error about all fifty."""
    body = client.post(
        "/predict/batch",
        json={"fixtures": [by_id(), {"match_id": "not a match"}]},
    ).json()

    assert len(body["predictions"]) == 1
    assert len(body["unresolved"]) == 1
    assert body["unresolved"][0]["match_id"] == "not a match"


def test_a_batch_where_nothing_resolves_is_404(client: TestClient) -> None:
    response = client.post(
        "/predict/batch", json={"fixtures": [{"match_id": "no"}, {"match_id": "nope"}]}
    )
    assert response.status_code == 404


def test_a_batch_over_the_configured_limit_is_refused_with_the_limit(
    client: TestClient,
) -> None:
    """Unbounded is the cheapest denial of service there is, and the limit is
    configuration rather than a literal, so the message has to state it."""
    response = client.post("/predict/batch", json={"fixtures": [by_id()] * 4})
    assert response.status_code == 422
    assert "the limit is 3" in response.json()["detail"]


def test_an_empty_batch_is_rejected_by_the_schema(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"fixtures": []}).status_code == 422


# ---- the prediction log ------------------------------------------------------


def _service_with_log(connection: FakeConnection) -> PredictionService:
    return PredictionService(
        model=None, index=INDEX, log=PostgresPredictionLog(lambda: connection), max_batch=10
    )


def test_a_served_prediction_is_written_to_the_log() -> None:
    """The whole reason the log exists: the served model's calibration has to
    be measurable against outcomes later, rather than assumed to match the
    backtest."""
    from src.pipelines.serving import predict_fixtures

    connection = FakeConnection()
    service = _service_with_log(connection)
    predictions = predict_fixtures(MODEL, INDEX.resolve(match_id=str(SOME["match_id"])))

    assert service.record(predictions) == 1
    query, params = connection.statements[-1]
    assert query.startswith("INSERT INTO predictions")
    assert params[0][1] == SOME["match_id"]


def test_a_log_that_refuses_does_not_fail_the_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A prediction answered and not logged is a gap in an audit trail. A
    prediction not answered because the audit trail was down is an outage."""
    from src.pipelines.serving import predict_fixtures

    service = _service_with_log(FakeConnection(fail_on="INSERT"))
    predictions = predict_fixtures(MODEL, INDEX.resolve(match_id=str(SOME["match_id"])))

    with caplog.at_level("ERROR"):
        assert service.record(predictions) == 0
    assert "prediction log rejected" in caplog.text


def test_recording_nothing_is_not_a_write() -> None:
    connection = FakeConnection()
    assert _service_with_log(connection).record([]) == 0
    assert connection.statements == []


def test_the_batch_response_says_how_many_were_logged(client: TestClient) -> None:
    """Zero with a log configured is how a caller can tell it is not working;
    zero with no log configured is the documented default."""
    body = client.post("/predict/batch", json={"fixtures": [by_id()]}).json()
    assert body["recorded"] == 0
