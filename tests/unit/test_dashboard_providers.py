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
from dashboard.domain.match import MatchStatus
from dashboard.providers import football_data_org
from dashboard.providers.api import ApiPredictions, as_prediction, to_fixtures
from dashboard.providers.base import FixtureProvider, PredictionProvider, ResultProvider
from dashboard.providers.football_data_org import FootballDataOrgFixtures
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


# ---- the fixture feed that is connected: football-data.org -------------------
#
# Milestone 13. No network: the client is a stub whose `get` records what was
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
    yield
    football_data_org._fetch.cache_clear()


def test_the_connected_feed_satisfies_the_interface_the_views_are_written_against() -> None:
    """The Milestone 13 claim, checked rather than asserted in prose."""
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
    UTC — yesterday and today there, which is the smallest window certain to
    hold every match in play whatever the host's timezone."""
    provider, http = feed({"matches": [fd_row(status="IN_PLAY", minute=63)]})
    assert provider.available
    assert provider.error is None
    asked = http.calls[0]
    assert asked["url"].endswith("/v4/matches")
    assert asked["headers"] == {"X-Auth-Token": "k"}
    utc_today = dt.datetime.now(dt.UTC).date()
    assert asked["params"]["dateFrom"] == (utc_today - dt.timedelta(days=1)).isoformat()
    assert asked["params"]["dateTo"] == (utc_today + dt.timedelta(days=1)).isoformat()


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
    assert "could not be reached" in str(provider.error)


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


def test_live_is_only_what_is_being_played() -> None:
    provider, _ = feed({"matches": [fd_row(), fd_row(id=9, status="PAUSED")]})
    assert [one.match_id for one in provider.live()] == ["fdorg-9"]


def test_the_competitions_a_reader_follows_are_sent_as_the_feeds_own_codes() -> None:
    provider, http = feed()
    provider.scheduled(
        since=dt.date(2026, 9, 6), until=dt.date(2026, 9, 13), competitions=["ENG_1", "GER_1"]
    )
    assert http.calls[0]["params"]["competitions"] == "PL,BL1"


def test_a_competition_outside_the_plan_is_dropped_from_the_filter_not_sent() -> None:
    """There is no code here to send for it — this project's ids are its own.
    (The live API ignores a paid competition left in the filter rather than
    refusing the request, so the drop costs nothing; the alternative would be
    an unfiltered request for competitions the reader did not ask about.)"""
    provider, http = feed()
    provider.live(competitions=["ENG_1", "SCO_2"])
    assert http.calls[0]["params"]["competitions"] == "PL"


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
        fd_row(competition={"code": "CL", "name": "UEFA Champions League"}),
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
