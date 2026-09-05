"""The vocabulary: what a match and a forecast mean, and what a reader follows.

Value types, so these are the cheapest tests in the suite and the ones every
other dashboard test rests on. What is worth asserting here is the handful of
decisions that are easy to get wrong later and silent when they are: the order
of the three outcomes, what an unknown status means, and the difference between
"follow nothing" and "follow no leagues".
"""

from __future__ import annotations

import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.domain import competition
from dashboard.domain.match import Fixture, MatchStatus, Prediction

KICKOFF = dt.date(2026, 8, 31)


def fixture(**overrides: object) -> Fixture:
    fields: dict[str, object] = {
        "match_id": "abc",
        "competition_id": "ENG_1",
        "date": KICKOFF,
        "home_team": "Aston Villa",
        "away_team": "Arsenal",
    }
    fields.update(overrides)
    return Fixture(**fields)  # type: ignore[arg-type]


# ---- a match -----------------------------------------------------------------


def test_a_fixture_with_no_goals_has_no_score() -> None:
    """A card branches on this: no score means no numbers beside the clubs,
    which is different from a goalless draw."""
    assert not fixture().has_score
    assert fixture(home_goals=0, away_goals=0).has_score


def test_a_club_is_recognised_however_the_reader_spells_the_spacing() -> None:
    played = fixture(home_team="  Aston   Villa ")
    assert played.involves("aston villa")
    assert not played.involves("Villa")


def test_unknown_is_a_status_and_not_a_missing_one() -> None:
    """``/fixtures`` says which matches can be priced and nothing about whether
    one has been played. Rendering that as FINISHED would put a blank score on
    a card and invite a reader to read it as 0-0."""
    assert fixture().status is MatchStatus.UNKNOWN
    assert MatchStatus.UNKNOWN != MatchStatus.FINISHED


# ---- a forecast --------------------------------------------------------------


def test_a_prediction_names_the_outcome_it_committed_to() -> None:
    stated = Prediction("abc", {"home": 0.178, "draw": 0.241, "away": 0.581}, "m", "1", False)
    assert stated.outcome == "away"
    assert stated.stated == pytest.approx(0.581)


def test_a_flat_forecast_still_names_one_outcome() -> None:
    """Three equal legs is a real answer from a calibrated model, and a page
    that raised on it would fail on the least surprising input there is."""
    flat = Prediction("abc", {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}, "m", "1", False)
    assert flat.outcome in {"home", "draw", "away"}


# ---- the catalogue -----------------------------------------------------------


def test_the_catalogue_is_the_shipped_registry_and_not_a_second_list() -> None:
    """The property this module exists for: a league is added in
    ``configs/leagues.yaml`` and appears here without a code change."""
    identifiers = {one.id for one in competition.competitions()}
    assert {"ENG_1", "ESP_1", "GER_1", "BRA_1", "USA_1"} <= identifiers


def test_a_competition_is_labelled_the_way_a_reader_would_name_it() -> None:
    assert competition.label("ENG_1") == "England · Premier League"
    assert competition.short_label("ESP_1") == "La Liga"


def test_an_unknown_competition_falls_back_to_its_id() -> None:
    """A report from an older run can name a league the registry has since
    renamed, and a page that raised there would be a page that cannot open an
    archive."""
    assert competition.label("XXX_9") == "XXX_9"
    assert competition.short_label("XXX_9") == "XXX_9"


def test_competitions_are_grouped_by_country_and_ordered_by_tier() -> None:
    grouped = competition.by_country()
    assert [one.id for one in grouped["England"]] == [
        "ENG_1",
        "ENG_2",
        "ENG_3",
        "ENG_4",
        "ENG_5",
    ]


# ---- favourites --------------------------------------------------------------
#
# Session state needs a script run, so each of these is a tiny app. That is the
# real environment: `st.session_state` outside one is not the thing under test.


def run(source: str) -> AppTest:
    app = AppTest.from_string(f"from dashboard.domain import favourites\n{source}")
    return app.run()


def test_nothing_is_followed_to_begin_with() -> None:
    app = run("import streamlit as st; st.text(str(favourites.leagues() + favourites.teams()))")
    assert app.exception == []
    assert app.text[0].value == "[]"


def test_following_a_league_and_a_club_survives_within_the_session() -> None:
    app = run(
        "import streamlit as st\n"
        "favourites.remember_leagues(['ENG_1'])\n"
        "favourites.remember_teams(['Arsenal'])\n"
        "st.text(f'{favourites.leagues()} {favourites.teams()}')"
    )
    assert app.text[0].value == "['ENG_1'] ['Arsenal']"


def test_toggling_is_a_switch_and_reports_which_way_it_went() -> None:
    app = run(
        "import streamlit as st\n"
        "on = favourites.toggle_team('Arsenal')\n"
        "off = favourites.toggle_team('Arsenal')\n"
        "league = favourites.toggle_league('ENG_1')\n"
        "st.text(f'{on} {off} {league} {favourites.teams()} {favourites.leagues()}')"
    )
    assert app.text[0].value == "True False True [] ['ENG_1']"


def test_following_no_league_is_no_filter_rather_than_a_filter_matching_nothing() -> None:
    """The distinction every reader downstream depends on. A reader who has
    followed nothing wants all the football, not none of it."""
    app = run(
        "import streamlit as st\n"
        "empty = favourites.league_filter()\n"
        "chosen = favourites.league_filter(['ENG_1'])\n"
        "st.text(f'{empty} {chosen}')"
    )
    assert app.text[0].value == "None ['ENG_1']"
