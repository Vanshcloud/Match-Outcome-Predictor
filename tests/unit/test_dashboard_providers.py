"""Where football comes from, and what happens when it does not come.

Three providers and the registry that picks between them. What is worth
testing here is almost entirely the failure half: this is the layer that meets
a missing table, a service that is not running and a provider name nobody
registered, and every one of those has to become an empty answer with a reason
rather than an exception on a page that is mostly about something else.

No network and no model. The service is a stub, and the match table is a
synthetic league written to a temporary directory.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dashboard import providers
from dashboard.client import ServiceError
from dashboard.domain.match import MatchStatus
from dashboard.providers.api import ApiPredictions, as_prediction, to_fixtures
from dashboard.providers.base import FixtureProvider, PredictionProvider, ResultProvider
from dashboard.providers.historical import HistoricalResults
from dashboard.providers.historical import to_fixtures as results_to_fixtures
from dashboard.providers.null import NullFixtures
from tests.factories import league_frame, season_labels

LEAGUE = league_frame(seasons=season_labels(2024, 2), teams=8)

API_ROW = {
    "match_id": "abc",
    "competition_id": "ENG_1",
    "date": "2026-08-31",
    "home_team": "Aston Villa",
    "away_team": "Arsenal",
}
ANSWER: dict[str, Any] = {
    "probabilities": {"home": 0.178, "draw": 0.241, "away": 0.581},
    "model": "ensemble-calibrated",
    "model_version": "0.12.0",
    "in_sample": True,
}


@pytest.fixture
def table(tmp_path: Path) -> Path:
    path = tmp_path / "matches.parquet"
    LEAGUE.to_parquet(path, index=False)
    return path


class StubClient:
    """The prediction service, without the service."""

    base_url = "http://stub.test"

    def __init__(
        self,
        *,
        health: dict[str, Any] | None = None,
        fixtures: list[dict[str, Any]] | None = None,
        answer: dict[str, Any] | None = None,
        fails: Exception | None = None,
    ) -> None:
        self._health = health or {"status": "ok", "components": []}
        self._fixtures = [API_ROW] if fixtures is None else fixtures
        self._answer = answer or ANSWER
        self._fails = fails
        self.asked: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        if self._fails is not None:
            raise self._fails
        return self._health

    def fixtures(self, **filters: Any) -> list[dict[str, Any]]:
        if self._fails is not None:
            raise self._fails
        self.asked.append(filters)
        return self._fixtures

    def predict(self, _fixture: dict[str, str]) -> dict[str, Any]:
        if self._fails is not None:
            raise self._fails
        return self._answer


def predictions(**kwargs: Any) -> ApiPredictions:
    return ApiPredictions(client=StubClient(**kwargs))  # type: ignore[arg-type]


# ---- the protocols are actually implemented ----------------------------------


def test_every_shipped_provider_satisfies_the_interface_it_claims(table: Path) -> None:
    """The check that stops the protocols from being documentation. Milestone
    13's provider passes exactly this and nothing else has to change."""
    assert isinstance(HistoricalResults(table), ResultProvider)
    assert isinstance(NullFixtures(), FixtureProvider)
    assert isinstance(predictions(), PredictionProvider)


# ---- results, from the canonical table ---------------------------------------


def test_results_come_back_newest_first_and_finished(table: Path) -> None:
    found = HistoricalResults(table).results(limit=5)
    assert len(found) == 5
    assert [one.date for one in found] == sorted((one.date for one in found), reverse=True)
    assert {one.status for one in found} == {MatchStatus.FINISHED}
    assert all(one.has_score for one in found)


def test_a_missing_table_is_unavailable_and_answers_nothing(tmp_path: Path) -> None:
    """The clean-checkout state. Every page reads this as an empty section with
    `make data` named, not as an error."""
    absent = HistoricalResults(tmp_path / "nothing.parquet")
    assert not absent.available
    assert absent.results() == []
    assert absent.latest() is None
    assert absent.frame().empty


def test_the_latest_date_is_read_without_loading_the_whole_table(table: Path) -> None:
    provider = HistoricalResults(table)
    assert provider.latest() == pd.Timestamp(LEAGUE["date"].max()).date()


def test_filters_narrow_the_read(table: Path) -> None:
    provider = HistoricalResults(table)
    cutoff = pd.Timestamp(LEAGUE["date"].max()).date() - dt.timedelta(days=30)
    recent = provider.results(since=cutoff)
    assert recent
    assert all(one.date >= cutoff for one in recent)
    assert provider.results(competitions=["NOWHERE_1"]) == []


def test_a_row_with_no_kickoff_recorded_carries_none_rather_than_nan() -> None:
    """NaN reaches the template as the string 'nan', which is worse than a
    blank because it looks like a kick-off time."""
    frame = LEAGUE.head(3).copy()
    frame["kickoff"] = pd.NA
    assert [one.kickoff for one in results_to_fixtures(frame)] == [None, None, None]


def test_an_empty_frame_is_an_empty_list() -> None:
    assert results_to_fixtures(LEAGUE.head(0)) == []


# ---- the fixture feed that is not connected ----------------------------------


def test_the_null_feed_answers_nothing_and_says_why() -> None:
    """It is a real implementation, not a special case: every empty state on
    the dashboard is rendered from this answer."""
    feed = NullFixtures()
    assert not feed.available
    assert feed.live() == []
    today = dt.date(2026, 9, 5)
    assert feed.scheduled(since=today, until=today + dt.timedelta(days=7)) == []
    assert "results" in feed.reason


def test_the_registry_returns_the_null_feed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(providers.FIXTURE_PROVIDER_ENV, raising=False)
    assert providers.fixtures().name == "none"


def test_the_provider_is_chosen_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole of the Milestone 13 wiring: a class, a registry entry, and
    this variable."""
    monkeypatch.setenv(providers.FIXTURE_PROVIDER_ENV, "none")
    assert providers.fixtures().name == "none"


def test_an_unknown_provider_name_falls_back_rather_than_refusing_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(providers.FIXTURE_PROVIDER_ENV, "sportmonks")
    feed = providers.fixtures()
    assert not feed.available
    assert feed.name == "none"


# ---- the model, over HTTP ----------------------------------------------------


def test_a_healthy_service_is_available_and_prices_a_fixture() -> None:
    provider = predictions()
    assert provider.available
    priced = provider.predict("abc")
    assert priced is not None
    assert priced.outcome == "away"
    assert priced.in_sample
    assert provider.error is None


def test_a_service_that_is_not_running_is_unavailable_with_the_reason_kept() -> None:
    """Not an exception: the prediction section is one of six on the page."""
    provider = predictions(fails=ServiceError("the service could not be reached"))
    assert not provider.available
    assert "could not be reached" in str(provider.error)
    assert provider.predict("abc") is None
    assert provider.priceable() == []


def test_a_degraded_service_names_the_component_that_is_missing() -> None:
    provider = predictions(
        health={
            "status": "degraded",
            "components": [
                {"name": "model", "ready": False, "detail": "no model; run `make model`"},
                {"name": "fixtures", "ready": True, "detail": "indexed"},
            ],
        }
    )
    assert not provider.available
    assert "make model" in str(provider.error)


def test_a_degraded_service_with_no_detail_still_produces_a_sentence() -> None:
    provider = predictions(health={"status": "degraded", "components": []})
    assert not provider.available
    assert "not ready" in str(provider.error)


def test_priceable_fixtures_are_asked_for_once_per_followed_competition() -> None:
    """The endpoint takes one competition, not a list. Merging here is what
    keeps that shape out of every view."""
    client = StubClient()
    provider = ApiPredictions(client=client)  # type: ignore[arg-type]
    provider.priceable(competitions=["ENG_1", "ESP_1"], limit=5)
    assert [one["competition_id"] for one in client.asked] == ["ENG_1", "ESP_1"]


def test_priceable_with_no_filter_asks_once_for_everything() -> None:
    client = StubClient()
    ApiPredictions(client=client).priceable()  # type: ignore[arg-type]
    assert client.asked == [{"competition_id": None, "limit": 20}]


def test_fixtures_from_the_service_carry_no_status_they_were_not_told() -> None:
    found = to_fixtures([API_ROW])
    assert found[0].status is MatchStatus.UNKNOWN
    assert not found[0].has_score


def test_a_date_already_parsed_is_left_alone() -> None:
    """Pydantic serialises a date as text over the wire and as an object in
    process, and both reach this function."""
    row = {**API_ROW, "date": dt.date(2026, 8, 31)}
    assert to_fixtures([row])[0].date == dt.date(2026, 8, 31)


def test_a_prediction_body_becomes_the_domain_type() -> None:
    stated = as_prediction("abc", ANSWER)
    assert stated.match_id == "abc"
    assert stated.model == "ensemble-calibrated"
    assert stated.model_version == "0.12.0"


# ---- the edges of a real table -----------------------------------------------


def test_a_table_that_exists_but_holds_no_rows_has_no_latest_date(tmp_path: Path) -> None:
    """Between `make data` creating the file and the first competition
    finishing its parse. A page that raised here would fail mid-ingest."""
    path = tmp_path / "matches.parquet"
    LEAGUE.head(0).to_parquet(path, index=False)
    provider = HistoricalResults(path)
    assert provider.available
    assert provider.latest() is None
    assert provider.results() == []


def test_a_row_with_no_goals_recorded_carries_none_rather_than_a_number() -> None:
    """`as_int` on a null. The projection can legitimately arrive without the
    scoreline, and ``int(NA)`` raises rather than returning something wrong."""
    frame = LEAGUE.head(2).copy()
    frame["home_goals"] = pd.NA
    found = results_to_fixtures(frame)
    assert [one.home_goals for one in found] == [None, None]
    assert not found[0].has_score
