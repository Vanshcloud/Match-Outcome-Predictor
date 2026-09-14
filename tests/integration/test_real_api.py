"""The service against the real tables and the real artefact.

Skips without them. What a synthetic league cannot prove is here: that three
hundred thousand fixtures index in a time a service can start in, that a real
artefact unpickles into something that prices a real match, and that the
probabilities the API returns are the ones the model states — on the data the
numbers in docs/MODEL_CARD.md were measured on.

The prediction log is a separate module: this one runs with it off, which is
the configuration a reader gets by default.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.models.dataset import DESIGN_COLUMNS
from src.pipelines.features import FEATURES_FILENAME
from src.pipelines.ingest import MATCHES_FILENAME
from src.pipelines.ratings import RATINGS_FILENAME
from src.pipelines.serving import MODEL_FILENAME
from src.utils.config import load_settings

pytestmark = pytest.mark.integration

SETTINGS = load_settings()
MATCHES = SETTINGS.paths.processed_dir / MATCHES_FILENAME
RATINGS = SETTINGS.paths.features_dir / RATINGS_FILENAME
FEATURES = SETTINGS.paths.features_dir / FEATURES_FILENAME
ARTEFACT = SETTINGS.paths.model_dir / MODEL_FILENAME

STARTUP_BUDGET_SECONDS = 120.0
"""What loading the artefact and indexing the whole table may take.

Generous, because it is a bound rather than a benchmark: what it rules out is
the shape of mistake that makes startup scale badly — a per-row `DataFrame`
scan while building the index, say — rather than a slow morning.
"""


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    for label, path in (
        ("matches", MATCHES),
        ("ratings", RATINGS),
        ("features", FEATURES),
        ("artefact", ARTEFACT),
    ):
        if not path.is_file():
            pytest.skip(f"no {label} at {path}; run `make reproduce` then `make model`")
    started = time.perf_counter()
    with TestClient(create_app(SETTINGS)) as entered:
        assert time.perf_counter() - started < STARTUP_BUDGET_SECONDS
        yield entered


def test_the_real_service_comes_up_ready(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert all(part["ready"] for part in body["components"])


def test_it_indexed_the_whole_table(client: TestClient) -> None:
    body = client.get("/version").json()
    assert body["fixtures_indexed"] > 200_000
    assert body["design_columns"] == len(DESIGN_COLUMNS)
    assert body["members"] == ["xgboost", "logistic_regression", "mlp"]
    assert body["library_mismatches"] == [], "the artefact was fitted by other library versions"


def test_a_real_match_can_be_found_and_then_priced(client: TestClient) -> None:
    """The two endpoints in the order a caller uses them: discovery is how a
    match gets named at all.

    ``until`` is what makes this about a *played* match. When the index held
    only results every row in it was one, so the first row of ``/fixtures`` was
    necessarily inside the artefact's training window; now the index also holds
    matches that have not kicked off, and they sort first. Asking for a match
    before the artefact's cut-off is asking the question this test is about.
    """
    found = client.get(
        "/fixtures", params={"competition_id": "ENG_1", "until": "2026-01-01", "limit": 1}
    ).json()
    assert found["count"] == 1
    fixture = found["fixtures"][0]

    priced = client.post("/predict", json={"match_id": fixture["match_id"]}).json()
    assert priced["fixture"]["match_id"] == fixture["match_id"]
    assert priced["model"] == "ensemble-calibrated"
    assert sum(priced["probabilities"].values()) == pytest.approx(1.0)
    # A match inside the training window: the artefact was fitted on it. The
    # flag exists so that stays visible rather than being something a reader
    # has to work out.
    assert priced["in_sample"] is True


def test_an_upcoming_fixture_prices_out_of_sample(client: TestClient) -> None:
    """Pricing a fixture before kick-off, against the real tables.

    Skipped where `make fixtures` has not run, which is a clean checkout and
    every week the provider has nothing to publish. Where it has, the forecast
    is for a match the artefact could not have trained on — and that is what
    the archive needs before it can score anything.
    """
    today = dt.date.today().isoformat()
    found = client.get("/fixtures", params={"since": today, "limit": 1}).json()
    if not found["count"]:
        pytest.skip("no upcoming fixtures indexed; run `make fixtures`")

    priced = client.post("/predict", json={"match_id": found["fixtures"][0]["match_id"]}).json()
    assert sum(priced["probabilities"].values()) == pytest.approx(1.0)
    assert priced["in_sample"] is False


def test_the_same_match_prices_identically_by_id_and_by_name(client: TestClient) -> None:
    fixture = client.get("/fixtures", params={"competition_id": "ESP_1", "limit": 1}).json()[
        "fixtures"
    ][0]
    by_id = client.post("/predict", json={"match_id": fixture["match_id"]}).json()
    by_key = client.post(
        "/predict",
        json={
            "competition_id": fixture["competition_id"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "date": fixture["date"],
        },
    ).json()
    assert by_id["probabilities"] == by_key["probabilities"]


def test_a_real_batch_is_priced_in_one_pass(client: TestClient) -> None:
    found = client.get("/fixtures", params={"limit": 20}).json()["fixtures"]
    body = client.post(
        "/predict/batch",
        json={"fixtures": [{"match_id": one["match_id"]} for one in found]},
    ).json()

    assert len(body["predictions"]) == len(found)
    assert body["unresolved"] == []
    assert all(
        sum(one["probabilities"].values()) == pytest.approx(1.0) for one in body["predictions"]
    )


def test_the_served_probabilities_are_plausible_football(client: TestClient) -> None:
    """Not a score — the backtest owns that. What this rules out is a service
    that transposed the classes somewhere between the estimator and the JSON,
    which would still sum to one and would still look like a forecast.

    Home advantage is the largest, most stable fact in this data: the measured
    home-win rate over 303,517 matches is 0.450 and the draw rate 0.267. A
    pooled mean far from those means the columns moved.
    """
    found = client.get("/fixtures", params={"limit": 200}).json()["fixtures"]
    body = client.post(
        "/predict/batch",
        json={"fixtures": [{"match_id": one["match_id"]} for one in found[:50]]},
    ).json()

    stated = [one["probabilities"] for one in body["predictions"]]
    home = sum(one["home"] for one in stated) / len(stated)
    draw = sum(one["draw"] for one in stated) / len(stated)
    away = sum(one["away"] for one in stated) / len(stated)

    assert 0.35 < home < 0.55
    assert 0.20 < draw < 0.32
    assert home > away
