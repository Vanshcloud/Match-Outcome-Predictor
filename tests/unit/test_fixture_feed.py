"""The live feed as a fixture source: rows the results provider would have written.

No network. Each test hands :mod:`src.ingestion.fixture_feed` feed-shaped
matches and checks the one thing that matters downstream — that a row comes back
in the provider's own date, time and club spelling, or does not come back.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from src.ingestion import fixture_feed
from src.ingestion.fixtures import to_frame
from src.ingestion.registry import load_registry


def feed_match(**overrides: Any) -> dict[str, Any]:
    match: dict[str, Any] = {
        "status": "TIMED",
        "utcDate": "2026-09-15T19:00:00Z",
        "competition": {"code": "PD"},
        "homeTeam": {"shortName": "Rayo Vallecano", "name": "Rayo Vallecano de Madrid"},
        "awayTeam": {"shortName": "Espanyol", "name": "RCD Espanyol de Barcelona"},
    }
    match.update(overrides)
    return match


SPAIN = {"ESP_1": ["Ath Madrid", "Barcelona", "Espanol", "Real Madrid", "Vallecano"]}


def test_a_feed_match_becomes_a_published_row_in_uk_time_and_table_spelling() -> None:
    (row,) = fixture_feed.to_rows([feed_match()], load_registry(), SPAIN)
    assert row == {
        "Div": "SP1",
        "Date": "15/09/2026",
        "Time": "20:00",
        "HomeTeam": "Vallecano",
        "AwayTeam": "Espanol",
    }


def test_a_late_evening_kick_off_keeps_the_uk_date_the_played_match_will_carry() -> None:
    """00:30 in India on Tuesday is Monday night in London, and the match id
    is built from London's date."""
    (row,) = fixture_feed.to_rows(
        [feed_match(utcDate="2026-09-14T23:30:00Z")], load_registry(), SPAIN
    )
    assert (row["Date"], row["Time"]) == ("15/09/2026", "00:30")
    (row,) = fixture_feed.to_rows(
        [feed_match(utcDate="2026-09-14T19:00:00Z")], load_registry(), SPAIN
    )
    assert (row["Date"], row["Time"]) == ("14/09/2026", "20:00")


def test_the_rows_build_the_same_match_id_as_the_published_file_would() -> None:
    published = {**fixture_feed.to_rows([feed_match()], load_registry(), SPAIN)[0]}
    ours = to_frame([published], load_registry())
    theirs = to_frame(
        [
            {
                "Div": "SP1",
                "Date": "15/09/2026",
                "Time": "20:00",
                "HomeTeam": "Vallecano",
                "AwayTeam": "Espanol",
            }
        ],
        load_registry(),
    )
    assert ours["match_id"].tolist() == theirs["match_id"].tolist()


def test_a_brazilian_match_becomes_a_row_filed_under_the_country_code() -> None:
    match = feed_match(
        competition={"code": "BSA"},
        utcDate="2026-09-15T23:00:00Z",
        homeTeam={"shortName": "Bahia", "name": "EC Bahia"},
        awayTeam={"shortName": "Mineiro", "name": "CA Mineiro"},
    )
    (row,) = fixture_feed.to_rows([match], load_registry(), {"BRA_1": ["Atletico-MG", "Bahia"]})
    assert row == {
        "Div": "BRA",
        "Date": "16/09/2026",
        "Time": "00:00",
        "HomeTeam": "Bahia",
        "AwayTeam": "Atletico-MG",
    }


def test_a_misreported_status_on_an_unscored_match_is_still_a_fixture() -> None:
    """The feed intermittently puts a kick-off time where Brazil's "TIMED" goes."""
    garbled = feed_match(status="2026-09-15 19:00:00Z", score={"fullTime": {"home": None}})
    assert len(fixture_feed.to_rows([garbled], load_registry(), SPAIN)) == 1
    scored = feed_match(status="2026-09-15 19:00:00Z", score={"fullTime": {"home": 1}})
    postponed = feed_match(status="POSTPONED")
    assert fixture_feed.to_rows([scored, postponed], load_registry(), SPAIN) == []


def test_clubs_resolve_by_alias_by_their_words_or_by_the_short_name() -> None:
    clubs = ["Ath Bilbao", "Ath Madrid", "Man City", "Man United", "PSV Eindhoven", "Utrecht"]
    resolve = fixture_feed.resolve
    assert resolve({"name": "Club Atlético de Madrid", "shortName": "Atleti"}, clubs) == (
        "Ath Madrid"
    )
    assert resolve({"name": "Manchester United FC", "shortName": "Man United"}, clubs) == (
        "Man United"
    )
    assert resolve({"name": "PSV", "shortName": "PSV"}, clubs) == "PSV Eindhoven"


def test_a_city_in_a_full_name_never_names_another_club() -> None:
    """The measured failure: Espanyol's full name contains Barcelona."""
    clubs = ["Barcelona", "Espanol", "Vallecano"]
    espanyol = {"name": "RCD Espanyol de Barcelona", "shortName": "Espanyol"}
    assert fixture_feed.resolve(espanyol, clubs) == "Espanol"
    assert fixture_feed.resolve({**espanyol, "name": "RCD Espanyol"}, clubs) is None


def test_a_club_that_matches_none_or_two_is_not_guessed() -> None:
    resolve = fixture_feed.resolve
    assert (
        resolve({"name": "Manchester FC", "shortName": "Man"}, ["Man City", "Man United"]) is None
    )
    assert resolve({"name": "Nobody FC", "shortName": "Nobody"}, ["Man City"]) is None
    assert resolve({"name": "", "shortName": ""}, ["Man City"]) is None
    # An alias whose target is not a current club is no answer either.
    assert resolve({"name": "Athletic Club", "shortName": "Athletic"}, ["Man City"]) is None


def test_unresolved_clubs_finished_matches_and_unknown_competitions_are_dropped(
    caplog: Any,
) -> None:
    matches = [
        feed_match(homeTeam={"shortName": "Nobody", "name": "Nobody FC"}),
        feed_match(status="FINISHED"),
        feed_match(competition={"code": "CL"}),
        feed_match(utcDate="whenever"),
        feed_match(utcDate="2026-09-15T19:00:00"),
    ]
    assert fixture_feed.to_rows(matches, load_registry(), SPAIN) == []
    assert "Nobody FC" in caplog.text


def test_recent_clubs_takes_both_sides_across_the_country() -> None:
    today = pd.Timestamp("2026-09-01")
    matches = pd.DataFrame(
        {
            "competition_id": ["ENG_1", "ENG_2", "ENG_1"],
            "date": [today, today, today - pd.Timedelta(900, "D")],
            "home_team": ["Arsenal", "Coventry", "Relegated Long Ago"],
            "away_team": ["Chelsea", "Hull", "Chelsea"],
        }
    )
    clubs = fixture_feed.recent_clubs(matches)
    assert clubs["ENG_1"] == ["Arsenal", "Chelsea", "Coventry", "Hull"]
    assert clubs["ESP_1"] == []
    assert fixture_feed.recent_clubs(matches.head(0)) == {}


class StubResponse:
    def __init__(self, body: object) -> None:
        self._body = body

    def json(self) -> object:
        return self._body


class StubClient:
    def __init__(self, *bodies: object) -> None:
        self.bodies = list(bodies)
        self.asked: list[dict[str, Any]] = []

    def get(self, _url: str, **kwargs: Any) -> StubResponse:
        self.asked.append(kwargs["params"])
        return StubResponse(self.bodies.pop(0))


def test_fetch_asks_in_windows_the_feed_accepts_and_keeps_only_matches() -> None:
    client = StubClient({"matches": [feed_match(), "not a match"]}, ["not a payload"])
    found = fixture_feed.fetch(client, "key", since=dt.date(2026, 9, 14), days=12)  # type: ignore[arg-type]
    assert len(found) == 1
    assert [(one["dateFrom"], one["dateTo"]) for one in client.asked] == [
        ("2026-09-14", "2026-09-23"),
        ("2026-09-24", "2026-09-26"),
    ]
    assert client.asked[0]["competitions"].split(",")[0] == "PL"
