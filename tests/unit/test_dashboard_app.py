"""The page itself, run by Streamlit's own test harness.

``AppTest`` executes ``dashboard/app.py`` exactly as the server would and hands
back the elements it produced, so these are tests of the rendered page rather
than of the functions behind it. What they are for is the states a screenshot
would not catch: a clean checkout with no reports, a service that is not
running, a filter that selects nothing, and the in-sample warning that must
appear on every fixture the shipped model was fitted on.

No test here needs a network or a built model. The reports are written to a
temporary directory as Parquet, and the service is a stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.client import ServiceError
from src.models.ensemble import SHIPPED
from src.utils.paths import PROJECT_ROOT
from tests.unit.test_dashboard_data import forecasts_frame, scores_frame

APP = str(PROJECT_ROOT / "dashboard" / "app.py")
TIMEOUT = 60

FIXTURE = {
    "match_id": "abc123",
    "competition_id": "ENG_1",
    "date": "2025-03-01",
    "home_team": "Arsenal",
    "away_team": "Chelsea",
}
ANSWER: dict[str, Any] = {
    "fixture": FIXTURE,
    "probabilities": {"home": 0.48, "draw": 0.26, "away": 0.26},
    "model": SHIPPED,
    "in_sample": True,
}


class StubService:
    """The API, without the API. Every method mirrors ``PredictionClient``."""

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
        self._fixtures = [FIXTURE] if fixtures is None else fixtures
        self._answer = answer or ANSWER
        self._fails = fails

    def health(self) -> dict[str, Any]:
        if self._fails is not None:
            raise self._fails
        return self._health

    def fixtures(self, **_filters: object) -> list[dict[str, Any]]:
        return self._fixtures

    def predict(self, _fixture: dict[str, str]) -> dict[str, Any]:
        return self._answer


def write_reports(root: Path, *, scores: bool = True, forecasts: bool = True) -> Path:
    """A reports tree holding whichever tables the test wants present."""
    directory = root / "reports" / "ensemble"
    directory.mkdir(parents=True, exist_ok=True)
    if scores:
        scores_frame().to_parquet(directory / "backtest.parquet", index=False)
    if forecasts:
        forecasts_frame().to_parquet(directory / "forecasts.parquet", index=False)
    return root


def run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    service: object | None = None,
    scores: bool = True,
    forecasts: bool = True,
) -> AppTest:
    """Run the page against a temporary data directory and a stubbed service.

    ``DATA_DIR`` rather than a patched settings object: it is the documented
    environment override, so the test drives the app the way a deployment does.
    """
    write_reports(tmp_path, scores=scores, forecasts=forecasts)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Patched on `dashboard.client`, not on `dashboard.app`. AppTest executes
    # the script's source in a fresh namespace rather than importing the
    # module, so `from dashboard.client import PredictionClient` runs again on
    # every `run()` and resolves against the source module — patching the app
    # module patches a name the running script never reads.
    monkeypatch.setattr(
        "dashboard.client.PredictionClient",
        lambda **_kwargs: service if service is not None else StubService(),
    )
    # Every test gets its own tmp_path, but Streamlit's cache is global and
    # outlives one, so it is cleared rather than relied upon to miss.
    st.cache_data.clear()
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


# ---- the page renders at all -------------------------------------------------


def test_the_page_renders_four_tabs_and_no_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(tmp_path, monkeypatch)
    assert app.exception == []
    assert [tab.label for tab in app.get("tab")][:4] == [
        "Scoreboard",
        "Reliability",
        "By competition",
        "Price a fixture",
    ]


def test_a_clean_checkout_names_both_commands_rather_than_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state CI runs in and the one a new reader meets first."""
    app = run(tmp_path, monkeypatch, scores=False, forecasts=False)
    assert app.exception == []
    warned = " ".join(warning.value for warning in app.warning)
    assert "make ensemble" in warned and "make card" in warned
    told = " ".join(info.value for info in app.info)
    assert "make ensemble" in told and "make card" in told


def test_scores_without_forecasts_still_renders_the_scoreboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(tmp_path, monkeypatch, forecasts=False)
    assert app.exception == []
    assert any("make card" in info.value for info in app.info)


# ---- the reliability panel, which is what the milestone is for ---------------


def test_the_reliability_panel_reports_over_everything_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(tmp_path, monkeypatch)
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Model"] == SHIPPED
    assert labels["Matches"] == "600"


def test_filtering_to_one_competition_changes_the_matches_it_is_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the panel: the card reports one pooled number, and
    this says what it is an average over."""
    app = run(tmp_path, monkeypatch)
    app.multiselect[0].select("ENG_1").run()
    assert app.exception == []
    assert {metric.label: metric.value for metric in app.metric}["Matches"] == "300"


def test_a_filter_that_selects_nothing_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(tmp_path, monkeypatch)
    app.multiselect[0].select("ENG_1").run()
    app.multiselect[1].select(1).run()
    assert app.exception == []
    assert any("No matches" in info.value for info in app.info)


def test_the_bin_count_is_the_readers_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(tmp_path, monkeypatch)
    app.slider[0].set_value(20).run()
    assert app.exception == []


# ---- the panel that leaves the process ---------------------------------------


def test_a_service_that_is_not_running_warns_and_leaves_the_page_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The predict panel is last and is not a precondition for the others: a
    reader with no service running still gets every table."""
    app = run(
        tmp_path, monkeypatch, service=StubService(fails=ServiceError("could not be reached"))
    )
    assert app.exception == []
    assert any("could not be reached" in warning.value for warning in app.warning)
    assert any("make api" in warning.value for warning in app.warning)
    assert {metric.label for metric in app.metric} >= {"Matches", "Calibration error"}


def test_a_degraded_service_names_the_component_that_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    degraded = StubService(
        health={
            "status": "degraded",
            "components": [
                {"name": "model", "ready": False, "detail": "no model; run `make model`"},
                {"name": "fixtures", "ready": True, "detail": "indexed"},
            ],
        }
    )
    app = run(tmp_path, monkeypatch, service=degraded)
    assert app.exception == []
    assert any("make model" in warning.value for warning in app.warning)


def test_no_fixtures_matching_is_reported_rather_than_rendered_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run(tmp_path, monkeypatch, service=StubService(fixtures=[]))
    assert app.exception == []
    assert any("No fixtures matched" in info.value for info in app.info)


def test_pricing_a_fixture_shows_three_probabilities_and_the_in_sample_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`make model` fits on the whole history, so every servable fixture is
    in-sample today — and a reader quoting that number as the backtest's is the
    mistake the boolean exists to prevent."""
    app = run(tmp_path, monkeypatch)
    app.button[0].click().run()
    assert app.exception == []
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Home"] == "48.0%" and labels["Draw"] == "26.0%" and labels["Away"] == "26.0%"
    assert any("training window" in warning.value for warning in app.warning)


def test_an_out_of_sample_fixture_gets_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = {**ANSWER, "in_sample": False}
    app = run(tmp_path, monkeypatch, service=StubService(answer=answer))
    app.button[0].click().run()
    assert app.exception == []
    assert not any("training window" in warning.value for warning in app.warning)


def test_a_service_that_refuses_the_fixture_list_is_an_error_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusing(StubService):
        def fixtures(self, **_filters: object) -> list[dict[str, Any]]:
            raise ServiceError("the service answered 503: Bad")

    app = run(tmp_path, monkeypatch, service=Refusing())
    assert app.exception == []
    assert any("503" in error.value for error in app.error)


def test_a_prediction_that_fails_is_an_error_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failing(StubService):
        def predict(self, _fixture: dict[str, str]) -> dict[str, Any]:
            raise ServiceError("the service answered 404: no fixture matches this request")

    app = run(tmp_path, monkeypatch, service=Failing())
    app.button[0].click().run()
    assert app.exception == []
    assert any("404" in error.value for error in app.error)
