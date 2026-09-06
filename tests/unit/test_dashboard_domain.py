"""The vocabulary: what a match and a forecast mean, and what a reader follows.

Value types, so these are the cheapest tests in the suite and the ones every
other dashboard test rests on. What is worth asserting here is the handful of
decisions that are easy to get wrong later and silent when they are: the order
of the three outcomes, what an unknown status means, and the difference between
"follow nothing" and "follow no leagues".
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.domain import competition, identity, store
from dashboard.domain.match import (
    EventKind,
    Fixture,
    MatchEvent,
    MatchStatus,
    Prediction,
)

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


# ---- an event ----------------------------------------------------------------
#
# One wording per kind, and the same wording in a toast and in a webhook post:
# two phrasings of one event is two things to keep in step, and the first time
# they disagree is the first time somebody doubts both.


def test_each_kind_of_event_reads_as_a_person_would_say_it() -> None:
    playing = fixture(status=MatchStatus.LIVE, home_goals=2, away_goals=1)
    assert MatchEvent(EventKind.GOAL, playing).message == "Goal: Aston Villa 2-1 Arsenal"
    assert MatchEvent(EventKind.FULL_TIME, playing).message == "Full time: Aston Villa 2-1 Arsenal"
    assert MatchEvent(EventKind.KICK_OFF, playing).message == "Kick-off: Aston Villa v Arsenal"


def test_a_match_with_no_score_yet_has_no_score_in_its_message() -> None:
    """A feed can report a match in play before it reports a scoreline, and
    "Arsenal  Chelsea" with a hole in it is worse than no number."""
    goalless = MatchEvent(EventKind.KICK_OFF, fixture(status=MatchStatus.LIVE))
    assert goalless.score == ""
    assert goalless.message == "Kick-off: Aston Villa v Arsenal"
    assert MatchEvent(EventKind.FULL_TIME, goalless.fixture).message == (
        "Full time: Aston Villa Arsenal"
    )


def test_an_event_carries_the_fields_a_transport_might_branch_on() -> None:
    payload = MatchEvent(
        EventKind.GOAL, fixture(status=MatchStatus.LIVE, home_goals=1, away_goals=0)
    ).as_payload()
    assert payload["event"] == "goal"
    assert payload["match_id"] == "abc"
    assert payload["score"] == "1-0"
    assert payload["home_team"] == "Aston Villa"


def test_an_unplayed_fixture_reports_no_score_rather_than_a_dash() -> None:
    assert MatchEvent(EventKind.KICK_OFF, fixture()).as_payload()["score"] is None


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
#
# The store underneath them is a temporary file per test — see the autouse
# fixture in tests/conftest.py, which exists because the first run without it
# wrote real profiles into the developer's data directory.


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


def test_a_favourite_outlives_the_session_that_set_it() -> None:
    """Milestone 14, in one assertion. Two separate app runs are two browser
    sessions; before this, the second one started empty."""
    first = run("favourites.remember_leagues(['ENG_1'])")
    assert first.exception == []
    second = run("import streamlit as st; st.text(str(favourites.leagues()))")
    assert second.text[0].value == "['ENG_1']"


# ---- who the reader is -------------------------------------------------------


@pytest.fixture(autouse=True)
def _forget_the_chosen_profile() -> Iterator[None]:
    """Bare-mode session state is process-global, so a profile chosen by one
    test would be the profile of every test after it."""
    yield
    st.session_state.pop(identity.PROFILE_KEY, None)


def as_user(monkeypatch: pytest.MonkeyPatch, **claims: object) -> None:
    """Stand in for ``st.user``, which is whatever the identity provider said.

    A plain dict: the real object is a read-only Mapping, and every access in
    :mod:`dashboard.domain.identity` goes through ``.get`` precisely so that it
    works on both — with no provider configured, ``st.user`` has no attributes
    at all and ``st.user.is_logged_in`` is an ``AttributeError``.
    """
    monkeypatch.setattr(st, "user", claims)


def test_with_no_provider_configured_nobody_is_signed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary state, and the one that must not raise: Streamlit omits the
    key entirely rather than setting it False."""
    as_user(monkeypatch)
    assert not identity.signed_in()
    assert not identity.provider_configured()
    assert not identity.sign_in_offered()
    assert identity.account() is None


def test_a_configured_provider_is_visible_before_anyone_signs_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streamlit adds `is_logged_in = False` only when secrets carry an [auth]
    section, which is how the sidebar knows whether to offer the button."""
    as_user(monkeypatch, is_logged_in=False)
    assert identity.provider_configured()
    assert not identity.signed_in()


def test_the_button_is_only_offered_when_login_would_actually_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`st.login()` raises without `streamlit[auth]`, and this image does not
    install it. A button that always errors is worse than no button."""
    as_user(monkeypatch, is_logged_in=False)
    monkeypatch.setattr(identity.importlib.util, "find_spec", lambda _name: object())
    assert identity.sign_in_offered()
    monkeypatch.setattr(identity.importlib.util, "find_spec", lambda _name: None)
    assert not identity.sign_in_offered()


def test_a_signed_in_reader_is_keyed_on_the_verified_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_user(monkeypatch, is_logged_in=True, email="reader@example.com")
    assert identity.account() == "reader@example.com"
    assert identity.current() == "account:reader@example.com"
    assert identity.label() == "reader@example.com"


def test_a_provider_that_returned_no_email_falls_back_to_a_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Email is a claim, not a guarantee — a provider can be configured to
    withhold it, and a key of `account:None` would be one shared inbox for
    everyone it happened to."""
    as_user(monkeypatch, is_logged_in=True)
    assert identity.account() is None
    assert identity.current() == "profile:Guest"


def test_a_profile_named_after_an_account_cannot_read_its_favourites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the keys are namespaced. Small here; the same shape is serious in an
    application that stored anything worth taking."""
    as_user(monkeypatch, is_logged_in=True, email="reader@example.com")
    signed_in_key = identity.current()
    as_user(monkeypatch)
    identity.use_profile("reader@example.com")
    assert identity.current() != signed_in_key


def test_a_reader_who_picks_nothing_is_a_real_profile_not_a_null_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which is the whole user-visible point: favourites persist without anyone
    opening the picker."""
    as_user(monkeypatch)
    assert identity.profile() == identity.GUEST
    assert identity.current() == "profile:Guest"


def test_a_typed_profile_name_is_tidied_before_it_becomes_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_user(monkeypatch)
    assert identity.use_profile("  Vansh   Tomar ") == "Vansh Tomar"
    assert identity.current() == "profile:Vansh Tomar"
    assert identity.use_profile("   ") == identity.GUEST


def test_a_store_key_reads_back_as_a_person() -> None:
    assert identity.display("profile:Vansh") == "Vansh"
    assert identity.display("account:a@b.c") == "a@b.c"
    assert identity.display("neither") == "neither"


def test_the_picker_lists_profiles_and_never_the_accounts() -> None:
    """An unauthenticated page must not enumerate the email addresses of
    everyone who has ever signed in to it."""
    names = identity.profile_names(["profile:Vansh", "account:a@b.c", "profile:Guest"])
    assert names == ["Guest", "Vansh"]


def test_a_profile_just_named_is_in_its_own_picker_before_it_has_a_row() -> None:
    """A profile is created by being chosen and has nothing saved until it has
    a favourite. Found by driving the real sidebar: without this the box read
    "New profile…" while the page below it had already switched, and the
    reader could not select the name they had just typed."""
    identity.use_profile("Someone")
    assert identity.profile_names([]) == ["Guest", "Someone"]


# ---- the store ---------------------------------------------------------------


def test_the_store_is_where_the_environment_says(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(store.STORE_ENV, "/somewhere/profiles.json")
    assert store.path() == Path("/somewhere/profiles.json")


def test_without_the_variable_the_store_sits_under_the_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(store.STORE_ENV, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert store.path() == tmp_path / "dashboard" / "profiles.json"


def test_a_reader_the_file_has_never_seen_follows_nothing() -> None:
    assert store.saved("profile:Nobody") == {"leagues": [], "teams": []}


def test_saving_and_reading_back_is_one_profile_among_others() -> None:
    store.save("profile:Vansh", leagues=["ENG_1"], teams=["Arsenal"])
    store.save("profile:Someone", leagues=["ESP_1"], teams=[])
    assert store.saved("profile:Vansh") == {"leagues": ["ENG_1"], "teams": ["Arsenal"]}
    assert store.identities() == ["profile:Someone", "profile:Vansh"]


def test_a_file_that_is_not_there_yet_is_the_ordinary_first_run() -> None:
    assert store.read() == {}
    assert store.identities() == []


def test_a_corrupt_file_is_an_empty_dashboard_rather_than_a_stack_trace() -> None:
    """A truncated write from a full disk must not make every page raise."""
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text('{"profile:Vansh": {"leagues": ["EN', encoding="utf-8")
    assert store.read() == {}


def test_a_hand_edited_file_is_read_for_what_it_does_carry() -> None:
    """Editing it by hand is a feature of a JSON store, so the parser meets
    whatever a person typed."""
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text(
        '{"profile:A": {"leagues": ["ENG_1"], "teams": "Arsenal"},'
        ' "profile:B": "not a profile", "profile:C": {"teams": [1, 2]}}',
        encoding="utf-8",
    )
    assert store.read() == {
        "profile:A": {"leagues": ["ENG_1"], "teams": []},
        "profile:C": {"leagues": [], "teams": ["1", "2"]},
    }


def test_a_file_holding_something_that_is_not_a_document_is_no_profiles() -> None:
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text("[1, 2, 3]", encoding="utf-8")
    assert store.read() == {}


def test_a_write_that_cannot_land_is_a_sentence_rather_than_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`data/` is mounted read-only in the compose file, so this is a state a
    real deployment reaches — and losing a preference must not lose the page."""
    monkeypatch.setattr(store, "_write_atomically", _refuse)
    assert not store.save("profile:Vansh", leagues=["ENG_1"], teams=[])
    assert "could not be saved" in str(store.last_error)
    store.last_error = None


def test_a_failed_write_leaves_no_half_written_file_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason the write goes through a temporary file at all: a partial
    JSON document is unreadable, and it would be read on the next page load."""
    store.path().parent.mkdir(parents=True, exist_ok=True)
    store.path().write_text('{"profile:Vansh": {"leagues": ["ENG_1"], "teams": []}}')
    monkeypatch.setattr(store.json, "dump", _refuse)
    assert not store.save("profile:Vansh", leagues=["ESP_1"], teams=[])
    assert store.saved("profile:Vansh") == {"leagues": ["ENG_1"], "teams": []}
    assert list(store.path().parent.glob("*.part")) == []
    store.last_error = None


def _refuse(*_args: object, **_kwargs: object) -> None:
    raise OSError("read-only file system")
