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
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from dashboard import providers
from dashboard.client import ServiceError
from dashboard.domain.match import EventKind, Fixture, MatchEvent, MatchStatus
from dashboard.providers import football_data_org, webhook
from dashboard.providers.api import ApiPredictions, as_prediction, to_fixtures
from dashboard.providers.base import (
    FixtureProvider,
    Notifier,
    PredictionProvider,
    ResultProvider,
)
from dashboard.providers.football_data_org import FootballDataOrgFixtures
from dashboard.providers.historical import HistoricalResults
from dashboard.providers.historical import to_fixtures as results_to_fixtures
from dashboard.providers.null import NullFixtures, NullNotifier
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
    """The check that stops the protocols from being documentation. A new
    provider passes exactly this and nothing else has to change."""
    assert isinstance(HistoricalResults(table), ResultProvider)
    assert isinstance(NullFixtures(), FixtureProvider)
    assert isinstance(predictions(), PredictionProvider)
    assert NullFixtures().crests("ENG_1") == {}


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
    """The whole of the provider wiring: a class, a registry entry, and
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
    assert client.asked == [{"competition_id": None, "limit": 20, "since": None, "until": None}]


def test_priceable_sends_a_date_window_as_iso_days() -> None:
    client = StubClient()
    day = dt.date(2026, 9, 14)
    ApiPredictions(client=client).priceable(since=day, until=day)  # type: ignore[arg-type]
    assert client.asked[0]["since"] == client.asked[0]["until"] == "2026-09-14"


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
    assert (stated.home_team, stated.date) == ("", None)


def test_the_fixture_the_service_priced_is_carried_onto_the_prediction() -> None:
    """An upcoming fixture is in no table the dashboard reads, so the clubs and
    date in the service's answer are what its page can put in the title."""
    fixture = {
        "home_team": "Mechelen",
        "away_team": "Anderlecht",
        "competition_id": "BEL_1",
        "date": "2026-09-11",
    }
    stated = as_prediction("abc", {**ANSWER, "fixture": fixture})
    assert (stated.home_team, stated.away_team, stated.competition_id) == (
        "Mechelen",
        "Anderlecht",
        "BEL_1",
    )
    assert stated.date == dt.date(2026, 9, 11)


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


# ---- the fixture feed that is connected: football-data.org -------------------
#
# No network: the client is a stub whose `get` records what was
# asked and answers a canned body, which is enough because the only thing this
# provider does with the wire is turn it into `Fixture` objects and turn every
# failure into an empty answer with a reason.


class StubResponse:
    """A ``requests.Response`` as far as this provider reads one."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class StubHttp:
    """An :class:`~src.utils.http.HttpClient` that never leaves the process."""

    def __init__(self, payload: Any = None, fails: Exception | None = None) -> None:
        self._payload = {"matches": []} if payload is None else payload
        self._fails = fails
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> StubResponse:
        return self._record(url, kwargs)

    def post(self, url: str, **kwargs: Any) -> StubResponse:
        """The notifier posts through the same client the feeds read with."""
        return self._record(url, kwargs)

    def close(self) -> None:
        """`HttpClient` owns a connection pool, so standing in for one means
        standing in for closing it."""

    def _record(self, url: str, kwargs: dict[str, Any]) -> StubResponse:
        self.calls.append({"url": url, **kwargs})
        if self._fails is not None:
            raise self._fails
        return StubResponse(self._payload)


def fd_row(**overrides: Any) -> dict[str, Any]:
    """One ``/v4/matches`` row, shaped the way the feed shapes it."""
    row: dict[str, Any] = {
        "id": 497821,
        "utcDate": "2026-09-06T14:00:00Z",
        "status": "TIMED",
        "competition": {"code": "PL", "name": "Premier League"},
        "area": {"name": "England"},
        "homeTeam": {
            "name": "Manchester United FC",
            "shortName": "Man United",
            "crest": "https://crests.example/mu.png",
        },
        "awayTeam": {"name": "Arsenal FC", "shortName": "Arsenal"},
        "score": {"fullTime": {"home": None, "away": None}},
    }
    row.update(overrides)
    return row


def feed(
    payload: Any = None, fails: Exception | None = None, key: str = "k"
) -> tuple[FootballDataOrgFixtures, StubHttp]:
    http = StubHttp(payload, fails)
    return FootballDataOrgFixtures(api_key=key, http=http), http  # type: ignore[arg-type]


@pytest.fixture
def east_of_utc() -> Iterator[None]:
    """A host five and a half hours ahead of the feed's calendar.

    Pinned rather than inherited, because the bug this guards against is
    invisible on a machine running in UTC — which is what CI runs in, and is
    exactly why the live smoke test found it and the suite had not.
    """
    original = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Kolkata"
    time.tzset()
    try:
        yield
    finally:
        if original is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = original
        time.tzset()


@pytest.fixture(autouse=True)
def _no_memo_between_cases() -> Any:
    """The memo is module-level on purpose — the context builds a fresh provider
    on every Streamlit rerun — so it has to be cleared between cases."""
    football_data_org._fetch.cache_clear()
    football_data_org._fetch_crests.cache_clear()
    yield
    football_data_org._fetch.cache_clear()
    football_data_org._fetch_crests.cache_clear()


def test_the_connected_feed_satisfies_the_interface_the_views_are_written_against() -> None:
    """The provider-layer claim, checked rather than asserted in prose."""
    assert isinstance(FootballDataOrgFixtures(api_key="k"), FixtureProvider)
    assert providers.FIXTURE_PROVIDERS["football-data.org"] is FootballDataOrgFixtures


def test_the_feed_is_selected_by_the_same_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(providers.FIXTURE_PROVIDER_ENV, "football-data.org")
    monkeypatch.setenv(football_data_org.API_KEY_ENV, "from-the-environment")
    chosen = providers.fixtures()
    assert isinstance(chosen, FootballDataOrgFixtures)
    assert chosen.api_key == "from-the-environment"


def test_no_key_is_unavailable_and_names_the_variable_to_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary state of a fresh checkout, and it must read as a
    configuration sentence rather than as an outage."""
    monkeypatch.delenv(football_data_org.API_KEY_ENV, raising=False)
    provider = FootballDataOrgFixtures()
    assert not provider.available
    assert football_data_org.API_KEY_ENV in str(provider.error)


def test_a_feed_that_answers_is_available_and_asks_the_feeds_own_calendar() -> None:
    """Availability is the same request `live()` makes, and it is anchored to
    UTC — from yesterday there, which is certain to hold every match in play
    whatever the host's timezone."""
    provider, http = feed({"matches": [fd_row(status="IN_PLAY", minute=63)]})
    assert provider.available
    assert provider.error is None
    asked = http.calls[0]
    assert asked["url"].endswith("/v4/matches")
    assert asked["headers"] == {"X-Auth-Token": "k"}
    utc_today = dt.datetime.now(dt.UTC).date()
    assert asked["params"]["dateFrom"] == (utc_today - dt.timedelta(days=1)).isoformat()
    assert asked["params"]["dateTo"] == (utc_today + dt.timedelta(days=9)).isoformat()


def test_a_cold_home_page_asks_the_feed_once() -> None:
    """Regression: availability, the live strip and the week of fixtures were two
    requests of several seconds each; with the reader's date equal to the UTC
    date they are one."""
    provider, http = feed(_three_leagues("IN_PLAY"))
    today = dt.datetime.now(dt.UTC).date()
    assert provider.available
    provider.live(competitions=["ENG_1"])
    provider.scheduled(since=today, until=today + dt.timedelta(days=7))
    assert len(http.calls) == 1


def test_a_feed_that_refuses_the_key_is_unavailable_with_the_status_kept() -> None:
    """A 403 must not render as "no fixtures scheduled": that is a sentence
    about football, and this is a sentence about a key."""
    response = requests.Response()
    response.status_code = 403
    response.reason = "Forbidden"
    provider, _ = feed(fails=requests.HTTPError(response=response))
    assert not provider.available
    assert "403" in str(provider.error)


def test_a_feed_that_cannot_be_reached_says_so_rather_than_raising() -> None:
    provider, _ = feed(fails=requests.ConnectionError("no route to host"))
    assert provider.scheduled(since=dt.date(2026, 9, 6), until=dt.date(2026, 9, 13)) == []
    assert provider.error == "football-data.org could not be reached (ConnectionError)"


def test_a_body_that_is_not_json_is_an_empty_answer_with_a_reason() -> None:
    provider, _ = feed(ValueError("Expecting value"))
    assert provider.live() == []
    assert "could not be reached" in str(provider.error)


def test_a_body_that_is_not_the_document_the_feed_promises_is_no_fixtures() -> None:
    provider, _ = feed(["not", "a", "document"])
    assert provider.live() == []
    assert provider.error is None


@pytest.mark.usefixtures("east_of_utc")
def test_a_wire_row_becomes_a_fixture_a_card_can_render() -> None:
    provider, _ = feed({"matches": [fd_row(status="IN_PLAY", minute=63)]})
    (one,) = provider.live()
    assert one.match_id == "fdorg-497821"
    assert one.competition_id == "ENG_1"
    assert one.date == dt.date(2026, 9, 6)
    assert one.teams == ("Man United", "Arsenal")
    assert one.status is MatchStatus.LIVE
    assert one.minute == 63
    assert one.kickoff == "19:30 IST"
    assert one.country == "England"
    assert one.home_crest_url == "https://crests.example/mu.png"
    assert one.away_crest_url is None


# ---- the two clocks ----------------------------------------------------------
#
# Found by the live smoke test, not by this file: the feed indexes by UTC and
# the callers ask about the host's today, and on a machine at UTC+05:30 those
# are different days for five and a half hours of every one.


@pytest.mark.usefixtures("east_of_utc")
def test_a_kick_off_the_feed_calls_yesterday_is_tonight_for_the_reader() -> None:
    """21:00 UTC on the 5th is 02:30 on the 6th in Kolkata. A live centre that
    asked the feed for the host's date would go blank exactly during Saturday
    evening in Europe."""
    (one,) = football_data_org.to_fixtures([fd_row(utcDate="2026-09-05T21:00:00Z")])
    assert one.date == dt.date(2026, 9, 6)
    assert one.kickoff == "02:30 IST"


def test_the_window_asked_of_the_feed_is_widened_to_cover_the_offset() -> None:
    """The caller's calendar is the host's; the feed's is UTC. A day at each
    end covers every real offset, and nothing else has to know."""
    provider, http = feed()
    provider.scheduled(since=dt.date(2026, 9, 6), until=dt.date(2026, 9, 12))
    assert http.calls[0]["params"] == {"dateFrom": "2026-09-05", "dateTo": "2026-09-14"}


@pytest.mark.usefixtures("east_of_utc")
def test_the_widening_does_not_leak_days_nobody_asked_for() -> None:
    """Widened on the wire, narrowed in the answer: a match on the extra day
    the request covered is not a fixture the caller asked to see."""
    provider, _ = feed(
        {
            "matches": [
                fd_row(id=1, utcDate="2026-09-05T09:00:00Z"),  # 14:30 local, the 5th
                fd_row(id=2, utcDate="2026-09-06T09:00:00Z"),  # 14:30 local, the 6th
                fd_row(id=3, utcDate="2026-09-07T09:00:00Z"),  # 14:30 local, the 7th
            ]
        }
    )
    found = provider.scheduled(since=dt.date(2026, 9, 6), until=dt.date(2026, 9, 6))
    assert [one.match_id for one in found] == ["fdorg-2"]


@pytest.mark.usefixtures("east_of_utc")
def test_a_stamp_with_no_offset_is_read_as_utc_not_as_the_hosts_clock() -> None:
    """The field is named `utcDate`. Reading it as local time would put a match
    on the wrong evening the day the feed changed its formatting."""
    (one,) = football_data_org.to_fixtures([fd_row(utcDate="2026-09-06T21:00:00")])
    assert one.date == dt.date(2026, 9, 7)
    assert one.kickoff == "02:30 IST"


def test_a_club_the_feed_gives_no_short_name_for_keeps_its_full_one() -> None:
    row = fd_row(homeTeam={"name": "1. FC Union Berlin"}, competition={"code": "BL1", "name": "x"})
    assert football_data_org.to_fixtures([row])[0].home_team == "1. FC Union Berlin"


def test_a_finished_score_is_carried_and_an_unplayed_one_is_none() -> None:
    played = fd_row(status="FINISHED", score={"fullTime": {"home": 2, "away": 1}})
    (one,) = football_data_org.to_fixtures([played])
    assert (one.home_goals, one.away_goals) == (2, 1)
    assert football_data_org.to_fixtures([fd_row()])[0].has_score is False


def test_scheduled_returns_what_is_still_to_come_earliest_first() -> None:
    """Live matches are today's fixtures too; finished and postponed ones are
    not, and a postponed kick-off time on a card is the one thing the null
    provider exists to refuse."""
    provider, _ = feed(
        {
            "matches": [
                fd_row(id=3, utcDate="2026-09-06T19:00:00Z"),
                fd_row(id=2, utcDate="2026-09-06T14:00:00Z", status="IN_PLAY"),
                fd_row(id=1, utcDate="2026-09-05T14:00:00Z", status="FINISHED"),
                fd_row(id=4, utcDate="2026-09-07T14:00:00Z", status="POSTPONED"),
            ]
        }
    )
    found = provider.scheduled(since=dt.date(2026, 9, 5), until=dt.date(2026, 9, 12))
    assert [one.match_id for one in found] == ["fdorg-2", "fdorg-3"]


def test_a_misreported_status_on_an_unscored_match_is_a_scheduled_card() -> None:
    """The feed intermittently puts a kick-off time where Brazil's "TIMED" goes;
    a postponed match, or a scored one, is still not a card."""
    garbled = football_data_org.to_fixtures([fd_row(status="2026-09-06 14:00:00Z")])
    assert garbled[0].status is MatchStatus.SCHEDULED
    scored = fd_row(status="2026-09-06 14:00:00Z", score={"fullTime": {"home": 1, "away": 0}})
    assert football_data_org.to_fixtures([scored])[0].status is MatchStatus.UNKNOWN
    postponed = football_data_org.to_fixtures([fd_row(status="POSTPONED")])
    assert postponed[0].status is MatchStatus.UNKNOWN


def test_live_is_only_what_is_being_played() -> None:
    provider, _ = feed({"matches": [fd_row(), fd_row(id=9, status="PAUSED")]})
    assert [one.match_id for one in provider.live()] == ["fdorg-9"]


def _three_leagues(status: str = "SCHEDULED") -> dict[str, object]:
    return {
        "matches": [
            fd_row(id=1, status=status, competition={"code": "PL", "name": "Premier League"}),
            fd_row(id=2, status=status, competition={"code": "BL1", "name": "Bundesliga"}),
            fd_row(
                id=3, status=status, competition={"code": "CL", "name": "UEFA Champions League"}
            ),
        ]
    }


def test_the_competitions_a_reader_follows_filter_the_answer() -> None:
    provider, _ = feed(_three_leagues())
    found = provider.scheduled(
        since=dt.date(2026, 9, 5), until=dt.date(2026, 9, 12), competitions=["ENG_1", "GER_1"]
    )
    assert sorted(one.competition_id for one in found) == ["ENG_1", "GER_1"]


def test_a_competition_outside_the_plan_is_ignored_in_the_filter() -> None:
    provider, _ = feed(_three_leagues("IN_PLAY"))
    assert [one.match_id for one in provider.live(competitions=["ENG_1", "SCO_2"])] == ["fdorg-1"]


def test_the_champions_league_is_shown_only_when_it_is_followed() -> None:
    provider, _ = feed(_three_leagues("IN_PLAY"))
    assert [one.competition_id for one in provider.live(competitions=["UEFA_CL"])] == ["UEFA_CL"]


def test_availability_and_a_filtered_live_section_share_one_request() -> None:
    """Regression: each feed request takes seconds, and the home page's cold
    load was three of them because a filtered ``live`` missed the cache entry
    ``available`` had just filled for the same window."""
    provider, http = feed(_three_leagues("IN_PLAY"))
    assert provider.available
    provider.live(competitions=["ENG_1", "GER_1"])
    assert len(http.calls) == 1
    assert "competitions" not in http.calls[0]["params"]


def test_the_champions_league_is_a_feed_only_card_with_the_feeds_own_name() -> None:
    row = fd_row(competition={"code": "CL", "name": "UEFA Champions League"})
    (one,) = football_data_org.to_fixtures([row])
    assert (one.competition_id, one.competition) == ("UEFA_CL", "UEFA Champions League")


def test_following_only_competitions_this_feed_lacks_says_so_and_asks_nothing() -> None:
    provider, http = feed()
    assert provider.live(competitions=["SCO_2", "RUS_1"]) == []
    assert http.calls == []
    assert "plan" in str(provider.error)


def test_no_competition_filter_asks_for_everything_the_plan_covers() -> None:
    provider, http = feed()
    provider.live()
    assert "competitions" not in http.calls[0]["params"]


def test_a_row_this_application_cannot_place_is_dropped_rather_than_shown() -> None:
    """A competition with no id here, an unparseable kick-off, an undrawn tie
    with no teams, and something that is not a row at all."""
    rows = [
        fd_row(competition={"code": "WC", "name": "FIFA World Cup"}),
        fd_row(utcDate="whenever"),
        fd_row(homeTeam={"name": None}),
        "not a row",
    ]
    assert football_data_org.to_fixtures(rows) == ()


def test_a_crest_that_is_not_https_is_not_put_in_an_image_tag() -> None:
    """The trust boundary: this URL decides what a reader's browser fetches."""
    row = fd_row(homeTeam={"name": "X", "crest": "javascript:alert(1)"})
    assert football_data_org.to_fixtures([row])[0].home_crest_url is None


def test_a_minute_the_feed_did_not_give_is_none_and_true_is_not_a_number() -> None:
    assert football_data_org.to_fixtures([fd_row()])[0].minute is None
    assert football_data_org.to_fixtures([fd_row(minute=True)])[0].minute is None


def test_the_same_window_asked_twice_is_one_request() -> None:
    """Ten calls a minute, and Streamlit reruns the script on every click. The
    memo is what makes those two facts compatible."""
    provider, http = feed()
    assert provider.available
    provider.live()
    assert len(http.calls) == 1


def test_the_memo_expires_so_a_score_is_not_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, float(football_data_org.CACHE_SECONDS)])
    monkeypatch.setattr(football_data_org.time, "monotonic", lambda: next(clock))
    provider, http = feed()
    provider.live()
    provider.live()
    assert len(http.calls) == 2


def test_without_an_injected_client_one_is_built_and_closed() -> None:
    """The path production takes. `HttpClient` owns the timeout and the retry
    policy, and a client built here is closed here — Streamlit reruns this
    script on every interaction, and a pool leaked per rerun is a pool nobody
    is tracking."""
    with football_data_org._client(None) as made:
        assert made.timeout_seconds == football_data_org.TIMEOUT_SECONDS


# ---- the ten-day cap ---------------------------------------------------------
#
# Also found by the live smoke test: `dateFrom`/`dateTo` more than ten days
# apart is answered `400 Specified period must not exceed 10 days`, so a
# fourteen-day ask came back empty. Splitting is the fix; truncating would have
# been a page missing fixtures with nothing on it saying so.


def test_a_window_wider_than_the_feed_allows_becomes_consecutive_requests() -> None:
    provider, http = feed()
    provider.scheduled(since=dt.date(2026, 9, 6), until=dt.date(2026, 9, 30))
    windows = [(one["params"]["dateFrom"], one["params"]["dateTo"]) for one in http.calls]
    assert windows == [
        ("2026-09-05", "2026-09-15"),
        ("2026-09-15", "2026-09-25"),
        ("2026-09-25", "2026-10-02"),
    ]
    assert all(
        (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
        <= football_data_org.MAX_WINDOW_DAYS
        for start, end in windows
    ), "eleven days apart is refused; ten is answered"
    # No day falls between two chunks: each window ends on the morning the next
    # one begins, which is the same instant.
    assert [end for _, end in windows][:-1] == [start for start, _ in windows][1:]


def test_every_chunk_of_a_long_window_reaches_the_answer() -> None:
    """The half that truncating would have lost. Three chunks, three distinct
    matches, and all three on the page."""
    windows = iter(
        [
            {"matches": [fd_row(id=1, utcDate="2026-09-08T09:00:00Z")]},
            {"matches": [fd_row(id=2, utcDate="2026-09-18T09:00:00Z")]},
            {"matches": [fd_row(id=3, utcDate="2026-09-28T09:00:00Z")]},
        ]
    )

    class PerWindow(StubHttp):
        def get(self, url: str, **kwargs: Any) -> StubResponse:
            self.calls.append({"url": url, **kwargs})
            return StubResponse(next(windows))

    provider = FootballDataOrgFixtures(api_key="k", http=PerWindow())  # type: ignore[arg-type]
    found = provider.scheduled(since=dt.date(2026, 9, 6), until=dt.date(2026, 9, 30))
    assert [one.match_id for one in found] == ["fdorg-1", "fdorg-2", "fdorg-3"]


def test_an_inverted_window_asks_nothing_rather_than_looping_forever() -> None:
    day = dt.date(2026, 9, 6)
    assert football_data_org._chunks(day, day - dt.timedelta(days=2)) == []
    provider, http = feed()
    assert provider.scheduled(since=day, until=day - dt.timedelta(days=4)) == []
    assert http.calls == []


def test_a_match_on_the_seam_between_two_chunks_is_one_card_not_two() -> None:
    """The feed includes both ends of its window, so consecutive chunks share
    an instant — and every Brazilian evening kick-off is at exactly midnight
    UTC, which is that instant."""
    provider, _ = feed({"matches": [fd_row(id=77, utcDate="2026-09-15T00:00:00Z")]})
    found = provider.scheduled(since=dt.date(2026, 9, 6), until=dt.date(2026, 9, 30))
    assert [one.match_id for one in found] == ["fdorg-77"]


def test_an_error_body_that_is_not_json_falls_back_to_the_status_line() -> None:
    """A gateway between here and the feed answers HTML, not the feed's own
    JSON message."""
    response = requests.Response()
    response.status_code = 502
    response.reason = "Bad Gateway"
    response._content = b"<html>upstream is unwell</html>"
    provider, _ = feed(fails=requests.HTTPError(response=response))
    assert not provider.available
    assert provider.error == "football-data.org answered 502: Bad Gateway"


def test_the_feeds_own_message_is_what_a_reader_is_shown() -> None:
    """This feed sends an empty HTTP reason phrase and puts the explanation in
    the body: without reading it, an invalid token and a window that is too
    wide are both `answered 400: `."""
    response = requests.Response()
    response.status_code = 400
    response.reason = ""
    response._content = b'{"message":"Your API token is invalid.","errorCode":400}'
    provider, _ = feed(fails=requests.HTTPError(response=response))
    assert not provider.available
    assert provider.error == "football-data.org answered 400: Your API token is invalid."


# ---- it has to be *this* service ---------------------------------------------


def test_another_project_answering_on_the_port_is_not_a_prediction_service() -> None:
    """Found by running the dashboard: `DASHBOARD_API_URL` defaults to port
    8000, an unrelated service was listening there, and its `/health` answered
    `{"status": "ok"}` — so the sidebar reported a healthy prediction service
    while every `/predict` would have come back 404."""
    provider = predictions(
        health={
            "status": "ok",
            "model_loaded": True,
            "model_name": "lstm_predictive_maintenance",
            "version": "1.0.0",
        }
    )
    assert not provider.available
    assert "not this project's API" in str(provider.error)
    assert "DASHBOARD_API_URL" in str(provider.error)


def test_the_check_is_the_documents_shape_not_a_list_of_component_names() -> None:
    """A service that is ours but has grown a component this build has never
    heard of is still ours. The set of components is a thing later changes
    add to, and a check that enumerated them would fail on the one that does."""
    provider = predictions(health={"status": "ok", "components": [{"name": "odds", "ready": True}]})
    assert provider.available
    assert provider.error is None


def test_a_service_answering_a_component_list_that_is_not_a_list_is_refused() -> None:
    provider = predictions(health={"status": "ok", "components": "all good"})
    assert not provider.available
    assert "not this project's API" in str(provider.error)


# ---- where an event is sent -----------------------------------------------


def event(kind: MatchEvent | EventKind = EventKind.GOAL) -> MatchEvent:
    fixture = Fixture(
        match_id="fdorg-1",
        competition_id="ENG_1",
        date=dt.date(2026, 9, 6),
        home_team="Arsenal",
        away_team="Chelsea",
        status=MatchStatus.LIVE,
        home_goals=2,
        away_goals=1,
        kickoff="20:30 IST",
    )
    return MatchEvent(kind, fixture)  # type: ignore[arg-type]


def test_with_no_transport_configured_events_go_nowhere_and_say_so() -> None:
    """The default. The toast is in the page and needs no transport; this is
    the other half, and it must not pretend an event was delivered."""
    sink = NullNotifier()
    assert not sink.available
    assert not sink.send(event())
    assert "DASHBOARD_WEBHOOK_URL" in sink.reason


def test_a_webhook_with_no_url_names_the_variable_rather_than_posting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(webhook.WEBHOOK_URL_ENV, raising=False)
    sink = webhook.WebhookNotifier()
    assert not sink.available
    assert not sink.send(event())
    assert webhook.WEBHOOK_URL_ENV in str(sink.error)


def test_an_event_is_posted_in_the_shape_slack_discord_and_a_script_all_read() -> None:
    http = StubHttp()
    sink = webhook.WebhookNotifier(url="https://hooks.test/abc", http=http)  # type: ignore[arg-type]
    assert sink.send(event())
    assert sink.error is None
    (call,) = http.calls
    assert call["url"] == "https://hooks.test/abc"
    body = call["json"]
    assert body["text"] == "Goal: Arsenal 2-1 Chelsea"  # Slack
    assert body["content"] == body["text"]  # Discord
    assert body["event"] == "goal"  # anything programmatic
    assert body["match_id"] == "fdorg-1"
    assert body["score"] == "2-1"


def test_a_webhook_that_refuses_is_a_false_and_a_caption_not_an_exception() -> None:
    """The live section it hangs off is about football. A transport that 404s
    must not cost a reader the scores."""
    response = requests.Response()
    response.status_code = 404
    response.reason = "Not Found"
    http = StubHttp(fails=requests.HTTPError(response=response))
    sink = webhook.WebhookNotifier(url="https://hooks.test/gone", http=http)  # type: ignore[arg-type]
    assert not sink.send(event())
    assert "answered 404" in str(sink.error)


def test_a_webhook_that_cannot_be_reached_says_that_instead() -> None:
    http = StubHttp(fails=requests.ConnectionError("no route to host"))
    sink = webhook.WebhookNotifier(url="https://hooks.test/abc", http=http)  # type: ignore[arg-type]
    assert not sink.send(event())
    assert "could not be reached" in str(sink.error)


def test_an_unreachable_webhook_never_puts_its_url_in_the_error_or_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The URL is the credential. ``requests`` embeds it — path and query —
    in a connection error's message, which is exactly what used to be logged."""
    url = "https://user:hunter2@hooks.example.test/services/T0SECRET/B0SECRET/PATHTOKEN?token=QUERYTOKEN"
    leaked = requests.ConnectionError(
        "HTTPSConnectionPool(host='hooks.example.test', port=443): Max retries exceeded "
        "with url: /services/T0SECRET/B0SECRET/PATHTOKEN?token=QUERYTOKEN"
    )
    sink = webhook.WebhookNotifier(url=url, http=StubHttp(fails=leaked))  # type: ignore[arg-type]
    with caplog.at_level("DEBUG"):
        assert not sink.send(event())

    said = f"{sink.error}\n{caplog.text}"
    for secret in ("T0SECRET", "B0SECRET", "PATHTOKEN", "QUERYTOKEN", "hunter2", "/services/"):
        assert secret not in said
    assert "hooks.example.test" in str(sink.error)
    assert "ConnectionError" in str(sink.error)


def test_a_url_too_malformed_to_parse_still_fails_with_a_sentence() -> None:
    """The likeliest reason a post failed; naming the host must not raise too."""
    http = StubHttp(fails=requests.exceptions.InvalidURL("Invalid URL 'http://[::1/SECRET'"))
    sink = webhook.WebhookNotifier(url="http://[::1/SECRET", http=http)  # type: ignore[arg-type]
    assert not sink.send(event())
    assert sink.error == "the webhook at the configured host could not be reached (InvalidURL)"


def test_a_client_this_transport_built_is_closed_and_an_injected_one_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streamlit reruns the script on every interaction; a pool leaked per
    rerun is a pool nobody is tracking. An injected one belongs to its owner."""
    closed: list[str] = []

    class Owned(StubHttp):
        def close(self) -> None:
            closed.append("closed")

    made = Owned()
    monkeypatch.setattr(webhook, "HttpClient", lambda **_kwargs: made)
    assert webhook.WebhookNotifier(url="https://hooks.test/abc").send(event())
    assert closed == ["closed"]

    injected = Owned()
    assert webhook.WebhookNotifier(url="https://hooks.test/abc", http=injected).send(event())  # type: ignore[arg-type]
    assert closed == ["closed"]


def test_the_url_is_read_from_the_environment_like_every_other_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "https://hooks.test/from-env")
    assert webhook.WebhookNotifier().url == "https://hooks.test/from-env"


def test_the_transport_is_chosen_by_environment_and_falls_back_on_a_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(providers.NOTIFIER_ENV, "webhook")
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "https://hooks.test/abc")
    assert providers.notifier().name == "webhook"
    monkeypatch.setenv(providers.NOTIFIER_ENV, "telegram")
    assert providers.notifier().name == "none"
    monkeypatch.delenv(providers.NOTIFIER_ENV)
    assert providers.notifier().name == "none"


def test_every_shipped_transport_satisfies_the_interface() -> None:
    assert isinstance(NullNotifier(), Notifier)
    assert isinstance(webhook.WebhookNotifier(url="https://hooks.test/abc"), Notifier)


# ---- club names -----------------------------------------------------------------


def test_club_words_folds_accents_closes_apostrophes_and_drops_club_suffixes() -> None:
    """The one normaliser the card matching compares both feeds' names with."""
    assert football_data_org.club_words("Grêmio") == "gremio"
    assert football_data_org.club_words("Nott'm Forest") == "nottm forest"
    assert football_data_org.club_words("Brighton & Hove Albion FC") == "brighton hove albion"
    assert football_data_org.club_words("AFC") == ""


# ---- crests ------------------------------------------------------------------


def test_a_competitions_crests_are_keyed_by_both_of_the_feeds_names_for_a_club() -> None:
    provider, http = feed(
        {
            "teams": [
                {
                    "name": "Manchester City FC",
                    "shortName": "Man City",
                    "crest": "https://c/65.png",
                },
                {"name": "No Badge FC", "crest": None},
                "not a team",
            ]
        }
    )
    crests = provider.crests("ENG_1")
    assert crests == {"manchester city": "https://c/65.png", "man city": "https://c/65.png"}
    assert http.calls[0]["url"].endswith("/competitions/PL/teams")
    provider.crests("ENG_1")
    assert len(http.calls) == 1  # a day's memo, not a request per rerun


def test_crests_are_empty_off_the_plan_without_a_key_or_when_the_feed_fails() -> None:
    provider, http = feed()
    assert provider.crests("ARG_1") == {}
    assert http.calls == []
    assert feed(key="")[0].crests("ENG_1") == {}
    failing, _ = feed(fails=requests.ConnectionError("down"))
    assert failing.crests("ENG_1") == {}
    assert failing.error is None  # missing crests are not a feed that is down
    assert feed({"teams": "garbled"})[0].crests("GER_1") == {}
