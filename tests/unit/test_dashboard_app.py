"""The pages themselves, run by Streamlit's own harness.

``AppTest`` executes a script exactly as the server would and hands back the
elements it produced, so these are tests of rendered pages rather than of the
functions behind them. What they are for is the states a screenshot would not
catch: a clean checkout with no tables, a service that is not running, no
fixture feed connected, a query parameter naming a match that does not exist,
and the in-sample warning that must appear on every fixture the shipped model
was fitted on.

Each page is driven through a two-line script rather than
``AppTest.from_function``, which copies a function's *source* into a temporary
file and loses the module it was defined in — so a view that referred to an
import would fail with ``NameError`` for reasons that have nothing to do with
the view.

No test here needs a network or a built model. The match table and the reports
are written to a temporary directory, and the providers are stubs.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.context import Context
from dashboard.domain import identity, store
from dashboard.domain.match import Fixture, MatchEvent, MatchStatus, Player, Prediction, Squad
from dashboard.providers import football_data_org
from dashboard.providers.historical import HistoricalOdds, HistoricalResults
from dashboard.providers.null import NullFixtures, NullNotifier, NullSquads
from dashboard.services import watch
from dashboard.views import home as dashboard_home
from dashboard.views import match as match_page
from src.models.ensemble import SHIPPED
from src.utils.config import load_settings
from src.utils.paths import PROJECT_ROOT
from tests.factories import league_frame, season_labels
from tests.unit.test_dashboard_reports import (
    archive_frame,
    forecasts_frame,
    market_frame,
    scores_frame,
)

APP = str(PROJECT_ROOT / "dashboard" / "app.py")

SEASONS = season_labels(2024, 2)

ENGLAND = league_frame(seasons=SEASONS, teams=14)
"""Fourteen clubs, which is two more than the search page will list.

The cap and the "narrow the query" line under it are only reachable with more
clubs than `search.TEAM_RESULTS`, and a fixture list that never reaches it is a
fixture list that cannot test it.
"""

GERMANY = league_frame(
    competition_id="GER_1",
    country="Germany",
    name="Bundesliga",
    seasons=SEASONS,
    teams=6,
    team_prefix="Klub",
)
"""A second competition the *reports* say nothing about.

`forecasts_frame` scores ENG_1 and ESP_1 only, so a fixture from here is the
one case that reaches "no scored forecasts for this competition" — which is
what a reader opening a league the backtest never covered actually sees.
"""

LEAGUE = pd.concat([ENGLAND, GERMANY], ignore_index=True)
CLUB, RIVAL = "Team 00", "Team 01"

GUEST = f"{identity.PROFILE_PREFIX}{identity.GUEST}"
"""The store key a reader who has picked no profile is using."""
TIMEOUT = 120

PAGES = {
    "home": "from dashboard.views import home; home.render()",
    "live": "from dashboard.views import home; home.render_live()",
    "competitions": "from dashboard.views import competitions; competitions.render()",
    "match": "from dashboard.views import match; match.render()",
    "search": "from dashboard.views import search; search.render()",
    "model": "from dashboard.views import performance; performance.render()",
}

ANSWER: dict[str, Any] = {
    "probabilities": {"home": 0.178, "draw": 0.241, "away": 0.581},
    "model": SHIPPED,
    "model_version": "0.12.0",
    "in_sample": True,
}


# ---- stubs -------------------------------------------------------------------


class StubPredictions:
    """The prediction service, without the service."""

    name = "stub"

    def __init__(
        self,
        *,
        available: bool = True,
        error: str | None = None,
        fixtures: list[Fixture] | None = None,
        answer: dict[str, Any] | None = None,
    ) -> None:
        self.available = available
        self.error = error
        self._fixtures = fixtures
        self._answer = answer or ANSWER

    def priceable(self, **_filters: object) -> list[Fixture]:
        if self._fixtures is not None:
            return self._fixtures
        return [
            Fixture(
                match_id="priceable",
                competition_id="ENG_1",
                date=dt.date(2026, 8, 31),
                home_team=CLUB,
                away_team=RIVAL,
            )
        ]

    def predict(self, match_id: str) -> Prediction | None:
        if not self.available:
            return None
        return Prediction(
            match_id=match_id,
            probabilities=self._answer["probabilities"],
            model=str(self._answer["model"]),
            model_version=str(self._answer["model_version"]),
            in_sample=bool(self._answer["in_sample"]),
        )


class Recorder:
    """A transport that keeps what it was told instead of posting it."""

    name = "recorder"
    available = True

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, event: MatchEvent) -> bool:
        self.sent.append(event.message)
        return True


class ConnectedFeed:
    """The fixture feed Milestone 13 registers, stubbed."""

    name = "stub-feed"
    available = True

    def scheduled(self, **_filters: object) -> list[Fixture]:
        return [_fixture(status=MatchStatus.SCHEDULED, kickoff="20:00")]

    def live(self, **_filters: object) -> list[Fixture]:
        return [_fixture(status=MatchStatus.LIVE, minute=63)]


def _fixture(**overrides: object) -> Fixture:
    fields: dict[str, object] = {
        "match_id": "live-1",
        "competition_id": "ENG_1",
        "date": dt.date(2026, 9, 5),
        "home_team": CLUB,
        "away_team": RIVAL,
    }
    fields.update(overrides)
    return Fixture(**fields)  # type: ignore[arg-type]


# ---- driving a page ----------------------------------------------------------


def ratings_frame(frame: pd.DataFrame = LEAGUE) -> pd.DataFrame:
    """Dixon-Coles goal rates for every match but the first.

    The first is null on purpose: the model refits on a rolling window and has
    nothing to say before its first fit, and "no rates for this fixture" is a
    state the page has to render rather than an error.
    """
    return pd.DataFrame(
        {
            "match_id": frame["match_id"].to_numpy(),
            "dc_home_lambda": [None, *([1.42] * (len(frame) - 1))],
            "dc_away_lambda": [None, *([1.03] * (len(frame) - 1))],
        }
    )


def write_data(
    root: Path,
    *,
    matches: bool = True,
    reports: bool = True,
    ratings: bool = True,
    archive: pd.DataFrame | None = None,
) -> None:
    """A data directory holding whichever tables the test wants present.

    ``archive`` is off unless a test asks for it, unlike the other three: no
    drift report is the ordinary state — it needs a prediction log and a
    service that has been called — and every other page test should meet that
    state rather than a synthetic archive.
    """
    if matches:
        (root / "processed").mkdir(parents=True, exist_ok=True)
        LEAGUE.to_parquet(root / "processed" / "matches.parquet", index=False)
    if ratings:
        (root / "features").mkdir(parents=True, exist_ok=True)
        ratings_frame().to_parquet(root / "features" / "ratings.parquet", index=False)
    if reports:
        directory = root / "reports" / "ensemble"
        directory.mkdir(parents=True, exist_ok=True)
        scores_frame().to_parquet(directory / "backtest.parquet", index=False)
        forecasts_frame().to_parquet(directory / "forecasts.parquet", index=False)
        market_frame().to_parquet(directory / "market.parquet", index=False)
    if archive is not None:
        directory = root / "reports" / "ensemble"
        directory.mkdir(parents=True, exist_ok=True)
        archive.to_parquet(directory / "archive.parquet", index=False)


def _stub_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    predictions: object | None = None,
    feed: object | None = None,
    notifier: object | None = None,
    squads: object | None = None,
    matches: bool = True,
    reports: bool = True,
    ratings: bool = True,
    archive: pd.DataFrame | None = None,
) -> None:
    """Point the dashboard at a temporary data directory and stub providers.

    ``DATA_DIR`` rather than a patched settings object: it is the documented
    environment override, so a test drives the page the way a deployment does.
    The providers are replaced at :func:`dashboard.context.resolve`, which is
    the one place in the package that names a concrete one — which is the
    architecture being asserted as much as it is a convenience here.
    """
    write_data(tmp_path, matches=matches, reports=reports, ratings=ratings, archive=archive)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def resolve() -> Context:
        settings = load_settings()
        return Context(
            settings=settings,
            results=HistoricalResults(settings.paths.processed_dir / "matches.parquet"),
            predictions=predictions if predictions is not None else StubPredictions(),  # type: ignore[arg-type]
            fixtures=feed if feed is not None else NullFixtures(),  # type: ignore[arg-type]
            odds=HistoricalOdds(settings.paths.processed_dir / "matches.parquet"),
            squads=squads if squads is not None else NullSquads(),  # type: ignore[arg-type]
            notifier=notifier if notifier is not None else NullNotifier(),  # type: ignore[arg-type]
        )

    monkeypatch.setattr("dashboard.context.resolve", resolve)
    # Streamlit's cache is global and outlives one tmp_path, so it is cleared
    # rather than relied upon to miss.
    st.cache_data.clear()


def run(
    page: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    predictions: object | None = None,
    feed: object | None = None,
    notifier: object | None = None,
    squads: object | None = None,
    matches: bool = True,
    reports: bool = True,
    ratings: bool = True,
    archive: pd.DataFrame | None = None,
    query: dict[str, str] | None = None,
) -> AppTest:
    """Render one page against a temporary data directory and stub providers.

    ``DATA_DIR`` rather than a patched settings object: it is the documented
    environment override, so the test drives the page the way a deployment
    does. The providers are replaced at :func:`dashboard.context.resolve`,
    which is the one place that names a concrete one.
    """
    _stub_context(
        tmp_path,
        monkeypatch,
        predictions=predictions,
        feed=feed,
        notifier=notifier,
        squads=squads,
        matches=matches,
        reports=reports,
        ratings=ratings,
        archive=archive,
    )
    prelude = "".join(
        f"import streamlit as st\nst.query_params['{key}'] = {value!r}\n"
        for key, value in (query or {}).items()
    )
    return AppTest.from_string(prelude + PAGES[page], default_timeout=TIMEOUT).run()


def run_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    predictions: object | None = None,
    feed: object | None = None,
    notifier: object | None = None,
    matches: bool = True,
) -> AppTest:
    """The whole application, chrome and all, with stub providers behind it."""
    _stub_context(
        tmp_path,
        monkeypatch,
        predictions=predictions,
        feed=feed,
        notifier=notifier,
        matches=matches,
        reports=True,
    )
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def followed(**lists: list[str]) -> dict[str, list[str]]:
    """Seed the default profile's favourites, where Milestone 14 keeps them.

    Tests used to write ``st.session_state`` directly, which asserted the
    storage mechanism rather than the behaviour — and that is exactly the thing
    the account store replaced. The store file is a temporary one per test; see
    the autouse fixture in ``tests/conftest.py``.
    """
    saved = {"leagues": [], "teams": [], **lists}
    store.save(GUEST, leagues=saved["leagues"], teams=saved["teams"])
    return saved


def favourites_now() -> dict[str, list[str]]:
    """What the store holds for the profile a test's reader is using."""
    return store.saved(GUEST)


def text_of(app: AppTest) -> str:
    """Everything the page said, as one string to search."""
    parts = [
        *(one.value for one in app.markdown),
        *(one.value for one in app.caption),
        *(one.value for one in app.info),
        *(one.value for one in app.warning),
        *(one.value for one in app.error),
        *(one.value for one in app.title),
        *(one.value for one in app.subheader),
    ]
    return " ".join(str(part) for part in parts)


# ---- every page renders ------------------------------------------------------


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_page_renders_without_an_exception(
    page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(page, tmp_path, monkeypatch).exception == []


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_page_renders_on_a_clean_checkout(
    page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state CI runs in and the one a new reader meets first: no match
    table, no reports, and no service."""
    app = run(
        page,
        tmp_path,
        monkeypatch,
        matches=False,
        reports=False,
        predictions=StubPredictions(available=False, error="the service could not be reached"),
    )
    assert app.exception == []


def test_the_shell_says_which_of_the_two_data_paths_are_answering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dashboard whose sections are quietly empty is one a reader debugs.
    There are exactly two reasons for an empty section, and the sidebar names
    both."""
    app = run_shell(tmp_path, monkeypatch)
    assert app.exception == []
    said = " ".join(str(one.value) for one in app.sidebar.caption)
    assert "✓ Match table" in said
    assert "✓ Prediction service" in said
    assert "No fixture feed" in said


def test_the_shell_names_the_fixture_feed_that_is_connected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Milestone 13. The line used to name the milestone; it now names the
    feed, because a status bar that cites an unshipped milestone after it ships
    is the kind of small lie a reader stops checking the rest of against."""
    app = run_shell(tmp_path, monkeypatch, feed=ConnectedFeed())
    said = " ".join(str(one.value) for one in app.sidebar.caption)
    assert "✓ Fixture feed · stub-feed" in said


def test_the_shell_names_what_is_missing_on_a_clean_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run_shell(
        tmp_path,
        monkeypatch,
        matches=False,
        predictions=StubPredictions(available=False, error="the service could not be reached"),
    )
    said = " ".join(str(one.value) for one in app.sidebar.caption)
    assert "make data" in said
    assert "could not be reached" in said


def test_following_a_league_from_the_sidebar_is_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run_shell(tmp_path, monkeypatch)
    app.sidebar.multiselect[0].select("ENG_1").run()
    assert app.exception == []
    assert favourites_now()["leagues"] == ["ENG_1"]


# ---- home --------------------------------------------------------------------


def test_home_says_why_there_are_no_live_matches_rather_than_showing_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction the whole `Section` type exists for. "No matches
    tonight" and "no fixture feed" are the same empty list and completely
    different sentences."""
    said = text_of(run("home", tmp_path, monkeypatch))
    assert "results" in said
    assert "Live now" in said


def test_home_shows_finished_matches_from_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("home", tmp_path, monkeypatch))
    assert "Just finished" in said
    assert "mop-card" in said


def test_home_names_the_command_that_builds_a_missing_match_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("home", tmp_path, monkeypatch, matches=False))
    assert "make data" in said


def test_a_connected_feed_fills_the_two_forward_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Milestone 13, rehearsed: one class, and the empty states fill up with no
    other change anywhere."""
    said = text_of(run("home", tmp_path, monkeypatch, feed=ConnectedFeed()))
    assert "mop-pill live" in said
    assert "63" in said


def test_home_reports_a_service_that_is_not_answering_without_losing_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(
        "home",
        tmp_path,
        monkeypatch,
        predictions=StubPredictions(available=False, error="the service could not be reached"),
    )
    said = text_of(app)
    assert app.exception == []
    assert "could not be reached" in said
    # The three sections that do not leave the process are unaffected. That is
    # why the one that does is last.
    assert "mop-card" in said


def test_the_live_centre_explains_what_connecting_a_provider_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("live", tmp_path, monkeypatch))
    assert "in-play" in said.lower()


# ---- competitions ------------------------------------------------------------


def test_the_browser_lists_every_registered_competition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("competitions", tmp_path, monkeypatch))
    assert "Premier League" in said
    assert "La Liga" in said


def test_a_competition_page_shows_that_leagues_own_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("competitions", tmp_path, monkeypatch, query={"competition": "ENG_1"})
    said = text_of(app)
    assert app.exception == []
    assert "Standings" in said
    assert "mop-card" in said
    assert "Season 2025-26" in said


def test_a_competition_with_no_ingested_matches_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("competitions", tmp_path, monkeypatch, query={"competition": "BRA_1"}))
    assert "No matches ingested" in said


def test_an_unknown_competition_falls_back_to_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("competitions", tmp_path, monkeypatch, query={"competition": "XXX_9"}))
    assert "Competitions" in said


def test_a_competition_page_with_no_match_table_names_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(
        run("competitions", tmp_path, monkeypatch, matches=False, query={"competition": "ENG_1"})
    )
    assert "make data" in said


# ---- the match page ----------------------------------------------------------


def match_id() -> str:
    """A fixture the reports have scored forecasts for."""
    return str(ENGLAND["match_id"].iloc[-1])


def unscored_match_id() -> str:
    """A fixture in a competition the backtest never covered."""
    return str(GERMANY["match_id"].iloc[-1])


def test_without_a_fixture_the_match_page_offers_a_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("match", tmp_path, monkeypatch)
    assert app.exception == []
    assert "Pick a match" in text_of(app)
    assert app.selectbox


def test_the_picker_reports_a_service_that_cannot_list_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(
        run(
            "match",
            tmp_path,
            monkeypatch,
            predictions=StubPredictions(fixtures=[], error="the service answered 503"),
        )
    )
    assert "503" in said


def test_the_picker_with_no_fixtures_and_no_error_names_the_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("match", tmp_path, monkeypatch, predictions=StubPredictions(fixtures=[])))
    assert "make model" in said


def test_a_fixture_page_shows_the_forecast_the_form_and_the_head_to_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("match", tmp_path, monkeypatch, query={"match": match_id()})
    assert app.exception == []
    # First occurrence of each label, not last: Milestone 17 put a second
    # Home/Draw/Away trio on this page — what the market said — under its own
    # heading, and the forecast is the one rendered first.
    labels: dict[str, str] = {}
    for one in app.metric:
        labels.setdefault(one.label, one.value)
    assert labels["Home"] == "17.8%"
    assert labels["Draw"] == "24.1%"
    assert labels["Away"] == "58.1%"
    assert "Head to head" in text_of(app)


def test_a_fixture_the_model_was_fitted_on_carries_the_in_sample_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`make model` fits on the whole history, so every servable fixture is
    in-sample today — and a reader quoting that as the backtest's number is the
    mistake the boolean exists to prevent."""
    app = run("match", tmp_path, monkeypatch, query={"match": match_id()})
    assert any("training window" in one.value for one in app.warning)


def test_an_out_of_sample_fixture_gets_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(
        "match",
        tmp_path,
        monkeypatch,
        predictions=StubPredictions(answer={**ANSWER, "in_sample": False}),
        query={"match": match_id()},
    )
    assert not any("training window" in one.value for one in app.warning)


def test_a_fixture_page_reports_what_the_stated_probability_is_worth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measurement that turns a probability into a claim a reader can
    check, from the same reliability tables the model card is generated from."""
    app = run("match", tmp_path, monkeypatch, query={"match": match_id()})
    labels = {one.label for one in app.metric}
    assert {"Stated here", "Happened, in that band", "Over"} <= labels


def test_without_the_per_match_forecasts_the_page_names_make_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("match", tmp_path, monkeypatch, reports=False, query={"match": match_id()}))
    assert "make card" in said


def test_a_fixture_the_service_cannot_price_still_shows_its_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(
        "match",
        tmp_path,
        monkeypatch,
        predictions=StubPredictions(available=False, error="the service could not be reached"),
        query={"match": match_id()},
    )
    said = text_of(app)
    assert app.exception == []
    assert "Head to head" in said
    assert "could not be reached" in said


def test_a_fixture_in_neither_the_table_nor_the_service_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(
        "match",
        tmp_path,
        monkeypatch,
        predictions=StubPredictions(available=False, error="not answering"),
        matches=False,
        query={"match": "nothing-like-this"},
    )
    assert any("nothing-like-this" in one.value for one in app.error)


def test_a_fixture_the_service_can_price_but_the_table_lacks_still_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine that has a model and no ingested results. The forecast is the
    whole page, and the page says why the rest is missing."""
    app = run("match", tmp_path, monkeypatch, matches=False, query={"match": "elsewhere"})
    said = text_of(app)
    assert app.exception == []
    assert "make data" in said
    assert "Head to head" not in said


# ---- search ------------------------------------------------------------------


def test_search_shows_suggestions_before_anything_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("search", tmp_path, monkeypatch))
    assert "Start here" in said


def test_search_finds_a_competition_by_country_or_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("search", tmp_path, monkeypatch)
    app.text_input[0].set_value("bundesliga").run()
    assert app.exception == []
    said = text_of(app)
    assert "Bundesliga" in said


def test_search_finds_a_club_and_its_recent_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("search", tmp_path, monkeypatch)
    app.text_input[0].set_value(CLUB).run()
    assert app.exception == []
    said = text_of(app)
    assert "mop-card" in said
    assert app.button


def test_a_query_that_finds_nothing_says_so_in_each_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("search", tmp_path, monkeypatch)
    app.text_input[0].set_value("zzzz").run()
    said = text_of(app)
    assert "No competition by that name" in said
    assert "No club by that name" in said


def test_search_without_a_match_table_says_what_cannot_be_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("search", tmp_path, monkeypatch, matches=False)
    app.text_input[0].set_value(CLUB).run()
    assert "make data" in text_of(app)


def test_following_a_club_from_search_is_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("search", tmp_path, monkeypatch)
    app.text_input[0].set_value(CLUB).run()
    app.button[0].click().run()
    assert app.exception == []
    assert CLUB in favourites_now()["teams"]


def test_a_query_matching_many_clubs_is_capped_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("search", tmp_path, monkeypatch)
    app.text_input[0].set_value("team").run()
    assert app.exception == []


# ---- the model page ----------------------------------------------------------


def test_the_model_page_reports_over_everything_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("model", tmp_path, monkeypatch)
    labels = {one.label: one.value for one in app.metric}
    assert labels["Model"] == SHIPPED
    assert labels["Matches"] == "600"


def test_filtering_to_one_competition_changes_the_matches_it_is_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the panel: the card reports one pooled number, and
    this says what it is an average over."""
    app = run("model", tmp_path, monkeypatch)
    app.multiselect[0].select("ENG_1").run()
    assert app.exception == []
    assert {one.label: one.value for one in app.metric}["Matches"] == "300"


def test_a_filter_that_selects_nothing_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("model", tmp_path, monkeypatch)
    app.multiselect[0].select("ENG_1").run()
    app.multiselect[1].select(1).run()
    assert app.exception == []
    assert any("No matches" in one.value for one in app.info)


def test_the_bin_count_is_the_readers_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("model", tmp_path, monkeypatch)
    app.slider[0].set_value(20).run()
    assert app.exception == []


def test_the_model_page_names_both_commands_when_the_reports_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("model", tmp_path, monkeypatch, reports=False)
    said = text_of(app)
    assert "make ensemble" in said
    assert "make card" in said


# ---- the interactions that change what a page shows ---------------------------


def test_a_club_the_reader_follows_can_be_dropped_from_the_sidebar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Followed clubs lead every page, so unfollowing has to be reachable from
    every page — which is why it is in the chrome and not on Search."""
    followed(teams=[CLUB])
    app = run_shell(tmp_path, monkeypatch)
    assert any(CLUB in one.label for one in app.sidebar.button)
    app.sidebar.button[0].click().run()
    assert favourites_now()["teams"] == []


def test_following_a_league_from_the_competitions_browser_is_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("competitions", tmp_path, monkeypatch)
    app.multiselect[0].select("ESP_1").run()
    assert app.exception == []
    assert favourites_now()["leagues"] == ["ESP_1"]


def test_the_browsers_filter_hides_the_countries_that_do_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("competitions", tmp_path, monkeypatch)
    app.text_input[0].set_value("bundesliga").run()
    said = text_of(app)
    assert app.exception == []
    assert "Bundesliga" in said
    assert "Premier League" not in said


def test_a_competition_page_can_be_followed_and_unfollowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("competitions", tmp_path, monkeypatch, query={"competition": "ENG_1"})
    app.button[0].click().run()
    assert favourites_now()["leagues"] == ["ENG_1"]
    app.button[0].click().run()
    assert favourites_now()["leagues"] == []


def test_the_picker_opens_the_fixture_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The picker writes the query parameter every card links with, so the
    page a reader lands on is the same page either way in."""
    app = run("match", tmp_path, monkeypatch)
    app.button[0].click().run()
    assert app.exception == []
    assert app.query_params["match"] == ["priceable"]


def test_a_fixture_in_a_competition_the_backtest_never_covered_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forecast still stands; what is missing is the measurement of how
    often forecasts like it come true, and the page says which."""
    app = run("match", tmp_path, monkeypatch, query={"match": unscored_match_id()})
    said = text_of(app)
    assert app.exception == []
    assert "No scored forecasts for Bundesliga" in said
    assert "Head to head" in said


def test_a_probability_outside_every_measured_band_is_reported_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model stating 0.95 in a competition where nothing was ever stated
    above 0.7 has no measured band to be checked against, and inventing the
    nearest one would be the wrong answer rather than a missing one."""
    app = run(
        "match",
        tmp_path,
        monkeypatch,
        predictions=StubPredictions(
            answer={**ANSWER, "probabilities": {"home": 0.95, "draw": 0.03, "away": 0.02}}
        ),
        query={"match": match_id()},
    )
    assert "fell in this probability band" in text_of(app)


def test_more_clubs_than_the_page_lists_is_capped_and_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run("search", tmp_path, monkeypatch)
    app.text_input[0].set_value("team").run()
    said = text_of(app)
    assert app.exception == []
    assert "more clubs match" in said


def test_a_section_with_no_cards_says_so_rather_than_rendering_a_blank_strip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader who follows a club that has not played recently gets a
    sentence, not an empty column they will read as a bug."""
    followed(teams=["Nobody FC"])
    app = run("home", tmp_path, monkeypatch)
    assert "Nothing finished recently" in text_of(app)


# ---- the context itself ------------------------------------------------------


def test_the_real_context_resolves_three_providers_and_reaches_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one test that runs :func:`dashboard.context.resolve` rather than
    replacing it. Nothing here opens a connection: the client is a frozen
    dataclass that builds its session per call, and the fixture feed is the
    null one."""
    write_data(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DASHBOARD_FIXTURE_PROVIDER", raising=False)
    from dashboard import context as context_module

    resolved = context_module.resolve()
    assert resolved.has_matches
    assert resolved.matches_path.endswith("matches.parquet")
    assert resolved.reports_dir == resolved.settings.paths.reports_dir
    assert resolved.fixtures.name == "none"
    assert resolved.predictions.client.base_url == resolved.settings.dashboard.api_url


# ---- the entry point, in the environment Streamlit actually gives it ---------


def test_the_entry_point_imports_with_only_its_own_directory_on_the_path() -> None:
    """`streamlit run dashboard/app.py` puts *this file's directory* on
    ``sys.path``, not the project root.

    Every other test in this file is blind to that: pytest puts the rootdir on
    the path before any of them run, and the container sets ``PYTHONPATH=/app``,
    so the page rendered green in CI, in the image and in this suite while
    `make dashboard` died on the first browser session with
    ``ModuleNotFoundError: No module named 'dashboard'``.

    So this runs the script the way the server does — an isolated interpreter
    (`-I`, which drops the working directory from the path) with only
    ``dashboard/`` prepended — and asserts nothing failed to import. Streamlit
    calls outside a session merely warn, so what is being tested is the import
    graph and not the render.
    """
    script = PROJECT_ROOT / "dashboard" / "app.py"
    finished = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
            "runpy.run_path(sys.argv[2], run_name='__main__')",
            str(script.parent),
            str(script),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert "ModuleNotFoundError" not in finished.stderr, finished.stderr[-2000:]
    assert "ImportError" not in finished.stderr, finished.stderr[-2000:]


# ---- who the favourites belong to (Milestone 14) ------------------------------


def test_the_sidebar_offers_the_profiles_the_store_already_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.save(f"{identity.PROFILE_PREFIX}Vansh", leagues=["ESP_1"], teams=[])
    app = run_shell(tmp_path, monkeypatch)
    assert app.exception == []
    offered = list(app.sidebar.selectbox[0].options)
    assert offered[:2] == [identity.GUEST, "Vansh"]
    assert offered[-1] == identity.NEW_PROFILE


def test_switching_profile_switches_which_favourites_are_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The milestone's point, through the chrome a reader actually uses."""
    store.save(f"{identity.PROFILE_PREFIX}Vansh", leagues=["ESP_1"], teams=[])
    followed(leagues=["ENG_1"])
    app = run_shell(tmp_path, monkeypatch)
    assert app.sidebar.multiselect[0].value == ["ENG_1"]
    app.sidebar.selectbox[0].select("Vansh").run()
    assert app.exception == []
    assert app.sidebar.multiselect[0].value == ["ESP_1"]


def test_a_new_profile_starts_empty_and_leaves_the_old_one_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    followed(leagues=["ENG_1"])
    app = run_shell(tmp_path, monkeypatch)
    app.sidebar.selectbox[0].select(identity.NEW_PROFILE).run()
    app.sidebar.text_input[0].set_value("Someone").run()
    assert app.exception == []
    assert app.sidebar.multiselect[0].value == []
    assert favourites_now()["leagues"] == ["ENG_1"]


def test_a_signed_in_reader_is_named_and_offered_the_way_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No picker for an account: the identity provider decided who this is."""
    monkeypatch.setattr(st, "user", {"is_logged_in": True, "email": "reader@example.com"})
    app = run_shell(tmp_path, monkeypatch)
    assert app.exception == []
    said = " ".join(str(one.value) for one in app.sidebar.caption)
    assert "reader@example.com" in said
    assert any("Sign out" in one.label for one in app.sidebar.button)
    assert app.sidebar.selectbox == []


def test_the_sign_in_button_appears_only_where_signing_in_would_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(st, "user", {"is_logged_in": False})
    monkeypatch.setattr(identity.importlib.util, "find_spec", lambda _name: object())
    app = run_shell(tmp_path, monkeypatch)
    assert app.exception == []
    assert any("Sign in" in one.label for one in app.sidebar.button)


def test_favourites_that_could_not_be_saved_say_so_in_the_chrome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`data/` is read-only in the compose file. A preference that silently
    fails to save is the kind of thing a reader discovers a week later."""
    monkeypatch.setattr(store, "last_error", "favourites could not be saved to /x: read-only")
    app = run_shell(tmp_path, monkeypatch)
    said = " ".join(str(one.value) for one in app.sidebar.caption)
    assert "could not be saved" in said


def test_signing_out_is_wired_to_streamlits_own_logout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The button has to call the framework rather than clear a key itself: the
    session cookie is Streamlit's, and a "sign out" that only forgot the name
    would leave the reader signed in."""
    called: list[str] = []
    monkeypatch.setattr(st, "user", {"is_logged_in": True, "email": "reader@example.com"})
    monkeypatch.setattr(st, "logout", lambda: called.append("logout"))
    app = run_shell(tmp_path, monkeypatch)
    app.sidebar.button[0].click().run()
    assert app.exception == []
    assert called == ["logout"]


def test_signing_in_is_wired_to_streamlits_own_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(st, "user", {"is_logged_in": False})
    monkeypatch.setattr(st, "login", lambda: called.append("login"))
    monkeypatch.setattr(identity.importlib.util, "find_spec", lambda _name: object())
    app = run_shell(tmp_path, monkeypatch)
    signin = [one for one in app.sidebar.button if "Sign in" in one.label]
    signin[0].click().run()
    assert app.exception == []
    assert called == ["login"]


# ---- what changed while the reader was looking (Milestone 15) -----------------


class ChangingFeed:
    """A feed whose live answer differs between looks, like a real one."""

    name = "stub-feed"
    available = True
    error = None

    def __init__(self, *answers: list[Fixture]) -> None:
        self._answers = list(answers)

    def live(self, **_filters: object) -> list[Fixture]:
        return self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]

    def scheduled(self, **_filters: object) -> list[Fixture]:
        return []


def _live(**overrides: object) -> Fixture:
    goals: dict[str, object] = {"home_goals": 0, "away_goals": 0}
    goals.update(overrides)
    return _fixture(status=MatchStatus.LIVE, **goals)


def test_the_first_look_at_the_live_strip_toasts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening the page during a match must not announce a kick-off that
    happened an hour ago."""
    app = run("home", tmp_path, monkeypatch, feed=ChangingFeed([_live()]))
    assert app.exception == []
    assert [one.value for one in app.toast] == []
    assert CLUB in text_of(app)


def test_a_goal_while_the_page_is_open_is_a_toast_and_a_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The milestone, end to end through the view: one change, one toast, one
    delivery to whatever transport is configured."""
    recorder = Recorder()
    feed = ChangingFeed([_live()], [_live(home_goals=1)])
    app = run("home", tmp_path, monkeypatch, feed=feed, notifier=recorder)
    app.run()
    assert app.exception == []
    assert [one.value for one in app.toast] == [f"Goal: {CLUB} 1-0 {RIVAL}"]
    assert recorder.sent == [f"Goal: {CLUB} 1-0 {RIVAL}"]


def test_the_live_strip_repaints_on_its_own_clock() -> None:
    """A fragment rather than a whole-page rerun: everything else on this page
    is a file read or an HTTP call, and repainting all of it to move one score
    would be the most expensive way to show the cheapest change."""
    assert dashboard_home.live_now.__wrapped__ is not None  # it is a fragment
    assert watch.REFRESH_SECONDS == football_data_org.CACHE_SECONDS


def test_with_no_transport_configured_a_goal_is_still_a_toast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The toast is in the page and needs nothing configured. The transport is
    the half that is optional."""
    feed = ChangingFeed([_live()], [_live(away_goals=1)])
    app = run("home", tmp_path, monkeypatch, feed=feed)
    app.run()
    assert [one.value for one in app.toast] == [f"Goal: {CLUB} 0-1 {RIVAL}"]
    assert app.exception == []


# ---- Milestone 17: the market, and what a gap from it means -------------------


def test_the_match_page_shows_the_closing_line_beside_the_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Milestone 12 kept the odds off this page because showing both invites
    the comparison to be made without the folds. Milestone 17 makes the
    comparison *with* the folds and shows the answer instead."""
    said = text_of(run("match", tmp_path, monkeypatch, query={"match": match_id()}))
    assert "What the market said" in said
    assert "overround" in said


def test_a_gap_from_the_market_is_reported_as_the_model_s_likely_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding, on the page: a gap is not an edge. A page that presented
    it as one would contradict the measurement three directories away."""
    said = text_of(run("match", tmp_path, monkeypatch, query={"match": match_id()}))
    assert "A gap is not an edge" in said
    assert "more likely to be wrong about this match" in said


def test_without_the_measurement_the_page_says_which_command_makes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`make card` writes the disagreement table. Before it has run, the page
    still shows both forecasts and says how far apart they are — it just
    declines to say what that has been worth."""
    said = text_of(run("match", tmp_path, monkeypatch, reports=False, query={"match": match_id()}))
    assert "make card" in said
    assert "A gap is not an edge" not in said


def test_the_match_page_shows_the_goal_model_s_expectation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Labelled as a goal-rate model's expectation, not as xG. Nothing in this
    project has ever seen a shot map, and the label is the whole point."""
    app = run("match", tmp_path, monkeypatch, query={"match": match_id()})
    said = text_of(app)
    assert "not shot-quality xG" in said
    labels = {one.label: one.value for one in app.metric}
    assert labels["Total"] == "2.45"
    assert labels["Supremacy"] == "+0.39"


def test_a_fixture_the_goal_model_never_rated_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dixon-Coles refits on a rolling window and has nothing to say before its
    first fit. A null there means exactly that."""
    first = str(LEAGUE.sort_values("date").iloc[0]["match_id"])
    said = text_of(run("match", tmp_path, monkeypatch, query={"match": first}))
    assert "before its first fit" in said


def test_with_no_ratings_table_the_page_names_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    said = text_of(run("match", tmp_path, monkeypatch, ratings=False, query={"match": match_id()}))
    assert "make ratings" in said


# ---- Milestone 18: who is registered ------------------------------------------


class StubSquads:
    """A squad source, without a source. Answers for whichever clubs it was
    given and sets the error the real one would for the rest."""

    name = "stub"
    available = True

    def __init__(self, *, known: dict[str, int] | None = None) -> None:
        self._known = known or {}
        self.error: str | None = None
        self.asked: list[tuple[str, str | None]] = []

    def squad(self, team: str, *, competition_id: str | None = None) -> Squad | None:
        self.asked.append((team, competition_id))
        size = self._known.get(team)
        if size is None:
            self.error = f"No club in this feed's squad list is spelled like **{team}**."
            return None
        self.error = None
        return Squad(
            team=team,
            full_name=f"{team} FC",
            players=tuple(
                Player(
                    name=f"Player {index}",
                    position="Goalkeeper" if index == 0 else "Midfield",
                    date_of_birth=dt.date(1996, 1, 1),
                )
                for index in range(size)
            ),
            source="stub",
            competition_id=competition_id,
        )


def teams_of(match: str) -> tuple[str, str]:
    """The two clubs of a fixture, as the match table spells them."""
    row = LEAGUE[LEAGUE["match_id"] == match].iloc[0]
    return str(row["home_team"]), str(row["away_team"])


def test_the_match_page_shows_both_registered_squads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Milestone 18, and the caption is as much of it as the numbers: a squad
    is who is registered, which is an upper bound on who is available."""
    home, away = teams_of(match_id())
    source = StubSquads(known={home: 3, away: 4})
    app = run("match", tmp_path, monkeypatch, squads=source, query={"match": match_id()})
    said = text_of(app)
    assert "Who is registered" in said
    assert "Registered is not available" in said
    assert "Registered squads from stub" in said
    assert [one.value for one in app.metric if one.label == "Registered"] == ["3", "4"]
    assert source.asked == [(home, "ENG_1"), (away, "ENG_1")]
    assert app.exception == []


def test_a_club_the_source_could_not_match_says_so_beside_the_one_it_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason has to be read per lookup: a provider carries the *last*
    failure it had, so a shared read would caption the missing club with why
    the other one failed — or, since the other succeeded, with nothing."""
    home, away = teams_of(match_id())
    app = run(
        "match",
        tmp_path,
        monkeypatch,
        squads=StubSquads(known={home: 3}),
        query={"match": match_id()},
    )
    said = text_of(app)
    assert f"spelled like **{away}**" in said
    assert [one.value for one in app.metric if one.label == "Registered"] == ["3"]
    assert app.exception == []


def test_with_no_squad_source_the_section_says_what_would_configure_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default, and it must read as a boundary rather than as an outage."""
    said = text_of(run("match", tmp_path, monkeypatch, query={"match": match_id()}))
    assert "Who is registered" in said
    assert "DASHBOARD_SQUAD_PROVIDER" in said


def test_injuries_are_now_named_as_having_no_source_rather_than_a_milestone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Milestone 18 shipped the squad and found the rest has nowhere to come
    from — no injury endpoint at any tier, and an empty lineup on this plan."""
    said = text_of(run("match", tmp_path, monkeypatch, query={"match": match_id()}))
    assert "no injury endpoint at any tier" in said
    # The heading no longer promises a milestone that has been spent.
    assert "Milestone 18" not in {milestone for _, milestone, _ in match_page.FUTURE_SECTIONS}


def test_a_fixture_outside_the_match_table_has_no_competition_to_look_up_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A card from the live feed opens a page with no table row. The squad
    lookup needs the competition that row carries, and says so rather than
    asking the feed about a competition it guessed."""
    said = text_of(
        run(
            "match",
            tmp_path,
            monkeypatch,
            squads=StubSquads(known={CLUB: 3}),
            query={"match": "fdorg-497821"},
        )
    )
    assert "no competition to look a squad up in" in said


def test_when_neither_club_is_found_the_reason_is_said_once_not_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A competition the source does not cover fails both lookups with the
    same sentence, and printing it under two empty columns reads as two
    problems."""
    home, _ = teams_of(match_id())
    app = run(
        "match", tmp_path, monkeypatch, squads=StubSquads(known={}), query={"match": match_id()}
    )
    assert text_of(app).count(f"spelled like **{home}**") == 1
    assert [one.value for one in app.metric if one.label == "Registered"] == []


# ---- Milestone 19: what the service actually served ---------------------------


def test_with_no_prediction_log_the_drift_panel_names_the_three_things_it_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary state, and it must not read as a missing command: this
    report needs a log, a service that has been called, and matches that have
    since been played."""
    said = text_of(run("model", tmp_path, monkeypatch))
    assert "What the service actually served" in said
    assert "PREDICTION_LOG_DSN" in said
    assert "make archive" in said


def test_a_young_archive_is_reported_as_not_evidence_rather_than_as_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding this milestone is mostly about. A difference under the noise
    floor at the archive's size is not a small drift — it is no measurement,
    and the page has to say the second thing."""
    app = run("model", tmp_path, monkeypatch, archive=archive_frame())
    said = text_of(app)
    assert "not evidence of drift" in said
    assert app.warning == []
    labels = {one.label: one.value for one in app.metric}
    assert labels["Scored"] == "40"
    assert labels["In sample"] == "8"
    assert labels["No result yet"] == "4"


def test_the_panel_prices_the_archive_that_is_still_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Not yet" is only useful with "how long" beside it."""
    app = run("model", tmp_path, monkeypatch, archive=archive_frame())
    assert "told from noise" in text_of(app)
    horizons = [frame for frame in app.dataframe if "forecasts needed" in list(frame.value.columns)]
    assert horizons and list(horizons[0].value["forecasts needed"]) == [61, 243, 1519, 2286, 6073]


def test_a_difference_larger_than_the_noise_floor_is_raised_rather_than_captioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is a threshold, not a refusal: a served model that has fallen
    over has to be visible."""
    app = run("model", tmp_path, monkeypatch, archive=archive_frame(distinguishable=True))
    assert app.warning != []
    assert "worth explaining" in text_of(app)


def test_an_archive_with_nothing_scorable_still_says_what_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows logged, none scored: the state of a deployment in its first week."""
    empty = archive_frame(scored=0)
    said = text_of(run("model", tmp_path, monkeypatch, archive=empty))
    assert "Nothing to score yet" in said
    assert "fitted on the whole history" in said


def test_with_no_measured_spread_the_page_prices_nothing_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`needed()` refuses an unknown spread, and refusing is right — but a page
    is the wrong place for it to be right, so the table is simply absent."""
    unmeasured = archive_frame(scored=0, spread=float("nan"))
    app = run("model", tmp_path, monkeypatch, archive=unmeasured)
    said = text_of(app)
    assert "Nothing to score yet" in said
    assert "told from noise" not in said
    assert app.exception == []
