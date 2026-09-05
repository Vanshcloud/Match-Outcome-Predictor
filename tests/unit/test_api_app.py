"""The application: what it does with nothing loaded, and how failures come out.

Nothing here needs a fitted model. That is the point — the state this module
covers is a clean checkout, which is what a reader meets first and the state
most likely to be met by accident in a deployment.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import REQUEST_ID_HEADER, build_service, create_app
from api.routes import get_service
from api.service import PredictionService, ServingError
from src import __version__
from src.utils.config import ApiConfig, LoggingConfig, PathsConfig, Settings


def settings_for(root: Path, **api: object) -> Settings:
    """Configuration pointing at an empty tree, so nothing loads."""
    return Settings(
        paths=PathsConfig(data_dir=root / "data", model_dir=root / "models"),
        api=ApiConfig(**api),  # type: ignore[arg-type]
        logging=LoggingConfig(level="WARNING"),
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A client with the lifespan actually run.

    As a context manager, deliberately. Starlette only fires startup and
    shutdown when the test client is entered, and a bare ``TestClient(app)``
    answers requests against an application whose state was never populated —
    which fails with an ``AttributeError`` about ``state.service`` rather than
    with anything that names the cause.
    """
    with TestClient(create_app(settings_for(tmp_path))) as entered:
        yield entered


# ---- starting with nothing ---------------------------------------------------


def test_the_service_starts_without_a_model_or_a_table(tmp_path: Path) -> None:
    """Refusing to start would make the first thing a new reader tries fail
    with a stack trace rather than a sentence."""
    service = build_service(settings_for(tmp_path))
    assert not service.ready
    assert service.model is None
    assert service.index is None
    assert "make model" in str(service.model_error)
    assert "missing" in str(service.index_error)


def test_health_is_200_and_degraded_rather_than_503(client: TestClient) -> None:
    """ "The container is wedged" and "the model has not been built" need
    different responses from a person, so they get different signals."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "degraded"
    assert body["version"] == __version__
    components = {part["name"]: part for part in body["components"]}
    assert components["model"]["ready"] is False
    assert components["fixtures"]["ready"] is False
    assert components["prediction_log"]["detail"] == "disabled"


def test_version_answers_without_a_model_and_says_what_it_knows(client: TestClient) -> None:
    body = client.get("/version").json()
    assert body["version"] == __version__
    assert body["model"] is None
    assert body["members"] == []
    assert body["design_columns"] == 30
    assert body["fixtures_indexed"] == 0
    assert body["prediction_log"] == "disabled"


def test_predicting_with_nothing_loaded_is_503_naming_the_fix(client: TestClient) -> None:
    response = client.post("/predict", json={"match_id": "anything"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "model" in detail
    assert "make reproduce" in detail


def test_listing_fixtures_with_no_table_is_503(client: TestClient) -> None:
    assert client.get("/fixtures").status_code == 503


def test_the_limitations_are_served_with_no_model_loaded(client: TestClient) -> None:
    """They describe the model this project ships, not the process's own state,
    so a degraded service still has to be able to say what it must not be used
    for."""
    body = client.get("/model-card/limitations").json()
    assert body["model"] is None
    assert len(body["limitations"]) == 4
    assert body["limitations"][0].startswith("**Betting.**")
    assert "MODEL_CARD.md" in body["model_card_url"]


def test_the_limitations_are_the_card_s_own_text() -> None:
    """Served from the constant the card renders from, so the endpoint and the
    document cannot drift apart."""
    from src.evaluation.model_card import LIMITATIONS

    served = PredictionService().limitations().limitations
    assert len(served) == sum(1 for line in LIMITATIONS if line.startswith("- "))
    assert "in-play" in served[-1]


# ---- request validation ------------------------------------------------------


def test_a_fixture_named_neither_way_is_422(client: TestClient) -> None:
    assert client.post("/predict", json={}).status_code == 422


def test_a_fixture_named_both_ways_is_422(client: TestClient) -> None:
    """Two names for one match is a request nobody can answer, and it should
    fail saying so rather than 404 as though the match were missing."""
    response = client.post(
        "/predict",
        json={
            "match_id": "ENG_1:2024-25:00001",
            "competition_id": "ENG_1",
            "home_team": "A",
            "away_team": "B",
            "date": "2025-01-01",
        },
    )
    assert response.status_code == 422
    assert "exactly one way" in response.text


def test_an_unknown_field_is_rejected_rather_than_ignored(client: TestClient) -> None:
    """A caller who sends `home` where the field is `home_team` should be told,
    not quietly given a prediction about whatever the default resolved to."""
    response = client.post("/predict", json={"match_id": "x", "home": "Arsenal"})
    assert response.status_code == 422
    assert "home" in response.text


def test_a_malformed_date_filter_is_rejected(client: TestClient) -> None:
    assert client.get("/fixtures", params={"since": "01-01-2025"}).status_code == 422


def test_an_out_of_range_limit_is_rejected(client: TestClient) -> None:
    assert client.get("/fixtures", params={"limit": 0}).status_code == 422
    assert client.get("/fixtures", params={"limit": 10_000}).status_code == 422


# ---- error mapping -----------------------------------------------------------


class ExplodingService(PredictionService):
    """Ready, and unable to price anything. Stands in for the one failure that
    is a deployment fault: an artefact and a feature table that disagree about
    what a match looks like."""

    @property
    def ready(self) -> bool:
        return True

    def predict(self, lookups: list[dict[str, str | None]]) -> tuple[list, list[int]]:
        raise ServingError(f"the table is missing 3 model column(s) for {len(lookups)} fixture(s)")


def test_a_model_that_cannot_price_the_row_is_a_500_not_a_client_error(
    tmp_path: Path,
) -> None:
    """Reporting it as a 4xx would send someone to fix the request."""
    app = create_app(settings_for(tmp_path))
    # A lambda, not the class: FastAPI inspects a callable dependency's own
    # signature, and `PredictionService` is a dataclass whose fields would be
    # read as query parameters.
    app.dependency_overrides[get_service] = lambda: ExplodingService()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"match_id": "anything"})
    assert response.status_code == 500
    assert "model column" in response.json()["detail"]


# ---- middleware and documentation --------------------------------------------


def test_a_request_id_is_echoed_back(client: TestClient) -> None:
    """So a client can quote the log line it is asking about."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "abc-123"})
    assert response.headers[REQUEST_ID_HEADER] == "abc-123"


def test_a_request_without_an_id_gets_no_header_invented_for_it(client: TestClient) -> None:
    assert REQUEST_ID_HEADER not in client.get("/health").headers


def test_every_request_is_logged_with_its_status_and_duration(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger="api.main"):
        client.get("/health")
    assert "method=GET path=/health status=200" in caplog.text
    assert "duration_ms=" in caplog.text


def test_the_openapi_document_describes_every_endpoint(client: TestClient) -> None:
    document = client.get("/openapi.json").json()
    assert document["info"]["version"] == __version__
    assert set(document["paths"]) == {
        "/health",
        "/version",
        "/model-card/limitations",
        "/fixtures",
        "/predict",
        "/predict/batch",
    }
    assert document["info"]["license"]["name"] == "MIT"


def test_the_interactive_documentation_is_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200


def test_the_description_warns_before_it_explains(client: TestClient) -> None:
    """The first thing the documentation says about a probability from here is
    that the bookmaker beats it. That is the honest order."""
    description = client.get("/openapi.json").json()["info"]["description"]
    assert "/model-card/limitations" in description
    assert description.index("closing line") < description.index("/fixtures")


# ---- configuration -----------------------------------------------------------


def test_the_batch_limit_comes_from_configuration(tmp_path: Path) -> None:
    assert build_service(settings_for(tmp_path, max_batch=7)).max_batch == 7


def test_the_prediction_log_is_off_unless_a_dsn_is_configured(tmp_path: Path) -> None:
    service = build_service(settings_for(tmp_path))
    assert service.log_kind == "disabled"
    assert service.log.enabled is False


def test_an_unreachable_prediction_log_degrades_instead_of_killing_the_process(
    tmp_path: Path,
) -> None:
    """The regression this catches is a crash loop, not a missing row.

    ``open_prediction_log`` creates the table, so a configured database that is
    down raised out of the lifespan and took the whole service with it — while
    every docstring in the serving layer said the log was optional and that a
    log which refuses never fails a request. A node reboot that restarts the API
    and PostgreSQL together is exactly when a forecaster that still answers is
    worth having.
    """
    settings = settings_for(
        tmp_path, prediction_log_dsn="postgresql://u:p@127.0.0.1:1/none?connect_timeout=1"
    )
    service = build_service(settings)

    assert service.log_error is not None
    assert service.log_kind == "unavailable"
    assert service.log.enabled is False

    log = next(part for part in service.components() if part.name == "prediction_log")
    assert log.ready is False
    assert log.detail == service.log_error
    # Recording is a no-op rather than an error: the request still gets served.
    assert service.record([]) == 0
