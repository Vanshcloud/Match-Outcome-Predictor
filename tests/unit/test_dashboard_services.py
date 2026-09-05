"""Orchestration and caching: what a page is handed, and why it might be empty.

Two modules. :mod:`dashboard.services.history` derives the three small tables a
fixture page shows — form, head-to-head, a record — and none of them is a model
input, which is the property worth holding still. :mod:`dashboard.services.matchday`
assembles the home page from three providers, and its whole job is keeping
"nothing happened" apart from "nothing can answer".

The providers here are stubs. Nothing reads a service and nothing loads a
model; the match table is a synthetic league on disk.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

from dashboard.domain.match import Fixture, MatchStatus
from dashboard.providers.null import NullFixtures
from dashboard.services import history, matchday
from tests.factories import league_frame, season_labels

LEAGUE = league_frame(seasons=season_labels(2024, 2), teams=8)
CLUB = "Team 00"
RIVAL = "Team 01"


@pytest.fixture
def table(tmp_path: Path) -> str:
    """A match table on disk, with Streamlit's global cache cleared around it.

    The cache outlives one ``tmp_path``, so a second test reading a second
    table would otherwise be handed the first one's rows.
    """
    path = tmp_path / "matches.parquet"
    LEAGUE.to_parquet(path, index=False)
    st.cache_data.clear()
    return str(path)


def played(**overrides: object) -> Fixture:
    fields: dict[str, object] = {
        "match_id": "abc",
        "competition_id": "ENG_1",
        "date": dt.date(2026, 8, 31),
        "home_team": CLUB,
        "away_team": RIVAL,
        "status": MatchStatus.FINISHED,
    }
    fields.update(overrides)
    return Fixture(**fields)  # type: ignore[arg-type]


# ---- history -----------------------------------------------------------------


def test_recent_is_bounded_relative_to_the_table_and_not_to_today(table: str) -> None:
    """The table is rebuilt by `make data`. A reader who has not run it this
    month should see the last round ingested, not an empty page."""
    frame = history.recent(table, days=30)
    latest = history.latest_date(table)
    assert latest is not None
    assert not frame.empty
    assert frame["date"].min() >= pd.Timestamp(latest - dt.timedelta(days=30))


def test_an_absent_table_gives_every_reader_an_empty_frame(tmp_path: Path) -> None:
    st.cache_data.clear()
    missing = str(tmp_path / "nothing.parquet")
    assert history.latest_date(missing) is None
    assert history.recent(missing).empty
    assert history.all_matches(missing).empty
    assert history.for_competition(missing, "ENG_1").empty


def test_a_competition_page_reads_one_season(table: str) -> None:
    frame = history.for_competition(table, "ENG_1")
    assert frame["season"].nunique() == 1
    assert frame["season"].iloc[0] == max(LEAGUE["season"])


def test_a_competition_with_no_rows_is_empty_rather_than_an_error(table: str) -> None:
    assert history.for_competition(table, "NOWHERE_1").empty


def test_a_fixture_is_found_by_its_id_and_missing_ids_are_none(table: str) -> None:
    frame = history.all_matches(table)
    wanted = str(frame["match_id"].iloc[0])
    assert history.by_id(frame, wanted) is not None
    assert history.by_id(frame, "not-a-match") is None
    assert history.by_id(history.for_competition(table, "NOWHERE_1"), wanted) is None


def test_form_reads_left_to_right_with_the_most_recent_match_last(table: str) -> None:
    """The direction a form string is read. Reversing it at the point of
    rendering is how half the callers end up reading it backwards."""
    frame = history.all_matches(table)
    letters = history.form(frame, CLUB)
    assert len(letters) == history.FORM_MATCHES
    assert set(letters) <= {"W", "D", "L"}

    latest = history.for_team(frame, CLUB).iloc[0]
    expected = "W" if latest["result"] == ("H" if latest["home_team"] == CLUB else "A") else None
    if expected is not None:
        assert letters[-1] == "W"


def test_form_over_a_club_with_no_matches_is_empty(table: str) -> None:
    assert history.form(history.all_matches(table), "Nobody FC") == []
    assert history.form(history.for_competition(table, "NOWHERE_1"), CLUB) == []


def test_a_head_to_head_finds_the_meetings_both_ways_round(table: str) -> None:
    frame = history.all_matches(table)
    meetings = history.head_to_head(frame, CLUB, RIVAL)
    assert not meetings.empty
    pairs = {frozenset((row["home_team"], row["away_team"])) for _, row in meetings.iterrows()}
    assert pairs == {frozenset((CLUB, RIVAL))}


def test_a_head_to_head_between_strangers_is_empty(table: str) -> None:
    frame = history.all_matches(table)
    assert history.head_to_head(frame, CLUB, "Nobody FC").empty
    assert history.head_to_head(history.for_competition(table, "NOWHERE_1"), CLUB, RIVAL).empty


def test_a_record_adds_up_to_the_meetings_it_was_given(table: str) -> None:
    meetings = history.head_to_head(history.all_matches(table), CLUB, RIVAL)
    wins, draws, losses = history.record(meetings, CLUB)
    assert wins + draws + losses == len(meetings)


def test_a_row_with_no_result_counts_as_a_draw_rather_than_a_loss() -> None:
    """An unplayed row cannot reach the table — validation rejects it — so this
    is about a projection that dropped the column, and guessing "lost" would be
    the one wrong answer."""
    frame = pd.DataFrame(
        [{"date": "2026-01-01", "home_team": CLUB, "away_team": RIVAL, "result": None}]
    )
    assert history.record(frame, CLUB) == (0, 1, 0)


def test_clubs_are_listed_from_both_sides_of_the_fixture(table: str) -> None:
    names = history.teams_in(history.all_matches(table))
    assert CLUB in names and RIVAL in names
    assert names == sorted(names)
    assert history.teams_in(history.for_competition(table, "NOWHERE_1")) == []


def test_following_nobody_narrows_nothing(table: str) -> None:
    frame = history.all_matches(table)
    assert len(history.follow(frame, [])) == len(frame)
    assert len(history.follow(frame, [CLUB])) < len(frame)
    assert history.follow(history.for_competition(table, "NOWHERE_1"), [CLUB]).empty


# ---- the home page -----------------------------------------------------------


class StubResults:
    """A results provider that answers from a list."""

    name = "stub-results"

    def __init__(self, fixtures: list[Fixture] | None = None, *, available: bool = True) -> None:
        self._fixtures = fixtures if fixtures is not None else [played()]
        self.available = available
        self.asked: dict[str, object] = {}

    def results(self, **filters: object) -> list[Fixture]:
        self.asked = filters
        return list(self._fixtures)

    def latest(self) -> dt.date | None:
        return dt.date(2026, 8, 31)


class StubPredictions:
    """A prediction provider that answers from a list, or refuses."""

    name = "stub-predictions"

    def __init__(self, *, available: bool = True, error: str | None = None) -> None:
        self.available = available
        self.error = error

    def priceable(self, **_filters: object) -> list[Fixture]:
        return [played(status=MatchStatus.UNKNOWN)]

    def predict(self, _match_id: str) -> None:
        return None


class StubFixtures:
    """A connected fixture feed, which is what Milestone 13 will register."""

    name = "stub-feed"
    available = True

    def scheduled(self, **_filters: object) -> list[Fixture]:
        return [played(status=MatchStatus.SCHEDULED)]

    def live(self, **_filters: object) -> list[Fixture]:
        return [played(status=MatchStatus.LIVE, minute=63)]


def test_with_no_fixture_feed_the_two_forward_sections_carry_its_reason() -> None:
    """The distinction the whole type exists for: this is not "no matches
    tonight", it is "nothing here can tell you"."""
    page = matchday.home_page(
        fixtures=NullFixtures(),
        results=StubResults(),
        predictions=StubPredictions(),
    )
    assert page.live.unavailable is not None
    assert "results" in page.live.unavailable
    assert page.upcoming.unavailable == page.live.unavailable
    assert not page.live.has_fixtures


def test_with_a_feed_connected_the_same_sections_fill_up_and_nothing_else_changes() -> None:
    page = matchday.home_page(
        fixtures=StubFixtures(),
        results=StubResults(),
        predictions=StubPredictions(),
    )
    assert page.live.unavailable is None
    assert page.live.has_fixtures
    assert page.live.fixtures[0].minute == 63
    assert page.upcoming.has_fixtures


def test_the_four_sections_come_back_in_reading_order() -> None:
    page = matchday.home_page(
        fixtures=NullFixtures(), results=StubResults(), predictions=StubPredictions()
    )
    assert [one.title for one in page.sections] == [
        "Live now",
        "Today and next",
        "Just finished",
        "The model can price these",
    ]


def test_a_missing_match_table_names_the_command_that_builds_one() -> None:
    section = matchday.finished_section(StubResults(available=False), None, None)
    assert section.unavailable is not None
    assert "make data" in section.unavailable


def test_results_are_narrowed_to_the_clubs_a_reader_follows() -> None:
    provider = StubResults([played(), played(home_team="Other", away_team="Else")])
    kept = matchday.finished_section(provider, None, [CLUB])
    assert len(kept.fixtures) == 1
    assert kept.fixtures[0].home_team == CLUB


def test_the_window_is_measured_from_the_latest_match_the_provider_holds() -> None:
    provider = StubResults()
    matchday.finished_section(provider, None, None, days=14)
    assert provider.asked["since"] == dt.date(2026, 8, 31) - dt.timedelta(days=14)


def test_a_service_that_is_not_answering_is_a_caption_not_a_failure() -> None:
    section = matchday.priceable_section(
        StubPredictions(available=False, error="the service could not be reached"), None
    )
    assert section.unavailable == "the service could not be reached"


def test_a_provider_with_no_stated_reason_still_gets_a_sentence() -> None:
    section = matchday.priceable_section(StubPredictions(available=False), None)
    assert section.unavailable == "No provider is configured for this."


def test_the_upcoming_window_is_taken_from_the_date_it_is_given() -> None:
    """Injectable, so a test states the date rather than depending on the one
    the machine happens to be on."""

    class Recording(StubFixtures):
        asked: dict[str, object] = {}

        def scheduled(self, **filters: object) -> list[Fixture]:
            Recording.asked = filters
            return []

    matchday.upcoming_section(Recording(), None, today=dt.date(2026, 9, 5), days=3)
    assert Recording.asked["since"] == dt.date(2026, 9, 5)
    assert Recording.asked["until"] == dt.date(2026, 9, 8)


def test_filtering_an_empty_frame_by_club_gives_an_empty_frame(table: str) -> None:
    """A page can ask for a club's matches before the table has any."""
    empty = history.for_competition(table, "NOWHERE_1")
    assert history.for_team(empty, CLUB).empty


def test_a_record_counts_a_defeat_as_a_defeat() -> None:
    """Home wins, draws and defeats, from one club's point of view — the third
    of which a round-robin fixture list does not guarantee a test sees."""
    frame = pd.DataFrame(
        [
            {"date": "2026-01-01", "home_team": CLUB, "away_team": RIVAL, "result": "H"},
            {"date": "2026-01-08", "home_team": CLUB, "away_team": RIVAL, "result": "D"},
            {"date": "2026-01-15", "home_team": CLUB, "away_team": RIVAL, "result": "A"},
            {"date": "2026-01-22", "home_team": RIVAL, "away_team": CLUB, "result": "H"},
        ]
    )
    assert history.record(frame, CLUB) == (1, 1, 2)


def test_narrowed_rows_become_the_cards_a_view_renders(table: str) -> None:
    frame = history.all_matches(table)
    assert len(history.as_fixtures(frame, limit=3)) == 3
    assert len(history.as_fixtures(frame)) == len(frame)
