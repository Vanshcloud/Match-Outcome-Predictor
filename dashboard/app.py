"""The page: four panels over measurements this project already made.

``streamlit run dashboard/app.py``

Each panel is a function taking what it needs and returning nothing, so the
layout is readable as a list and a panel can be tested by handing it a frame.
Streamlit reruns this whole script on every interaction, which is why the two
report tables are behind :func:`st.cache_data` — a reader dragging a filter
should not re-read a two-megabyte Parquet each time.

**What the page will not do.** It computes no metric, fits nothing, and holds
no model. The tables come from :mod:`src.pipelines`, the probabilities come
from the service over HTTP, and where a number here appears in
``docs/MODEL_CARD.md`` it is the same number from the same function.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import data
from dashboard.charts import competition_bars, reliability_diagram, score_bars
from dashboard.client import PredictionClient, ServiceError
from src.models.ensemble import SHIPPED
from src.utils.config import Settings, load_settings

TITLE = "Match Outcome Predictor"
CAPTION = (
    "Calibrated home / draw / away probabilities for 39 competitions. "
    "Every table here is read from a report the pipelines wrote; "
    "every probability is answered by the service."
)


@st.cache_data(show_spinner="reading the reports…")
def _reports(reports_dir: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Both report tables, cached on the directory they came from.

    Keyed by a plain string because Streamlit hashes a function's arguments to
    decide whether the cache is still valid, and a ``Path`` or a pydantic model
    is not something it can hash cheaply. Returns the two frames rather than
    the :class:`~dashboard.data.Reports` that holds them, for the same reason:
    what goes into the cache should be what pandas already knows how to store.
    """
    loaded = data.load_reports(Path(reports_dir))
    return loaded.scores, loaded.forecasts


def load(settings: Settings) -> data.Reports:
    """The reports, through the cache."""
    reports_dir = settings.paths.reports_dir
    scores, forecasts = _reports(str(reports_dir))
    return data.Reports(
        scores=scores, forecasts=forecasts, reports_dir=reports_dir / data.ENSEMBLE_SUBDIR
    )


# ---- panels ------------------------------------------------------------------


def scoreboard_panel(reports: data.Reports) -> None:
    """Every forecaster, over the matches all of them could price."""
    st.subheader("How the forecasters compare")
    if not reports.has_scores or reports.scores is None:
        _missing("the backtest scores", "make ensemble")
        return

    table = data.leaderboard(reports.scores)
    st.caption(
        f"{int(table['n'].iloc[0]):,} matches every forecaster could price. "
        "Log loss and RPS are the metrics models are selected on; accuracy is "
        "reported and is not one of them."
    )
    st.plotly_chart(score_bars(table), width="stretch")
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "`home_always` scores infinite log loss and it is not clipped — a "
        "forecast that ruled out what happened was infinitely wrong. RPS still "
        "ranks it, because RPS is bounded and knows H, D and A are ordered."
    )


def reliability_panel(reports: data.Reports) -> None:
    """The panel the milestone exists for: reliability, filtered by the reader.

    The model card reports one pooled calibration error. The interesting
    question is what it is an average over, and that is a filter rather than a
    second document.
    """
    st.subheader("Is a stated probability the rate at which it happens?")
    if not reports.has_forecasts or reports.forecasts is None:
        _missing("the per-match forecasts", "make card")
        return

    forecasts = reports.forecasts
    competitions = sorted(forecasts["competition_id"].dropna().unique())
    folds = sorted(int(fold) for fold in forecasts["fold"].dropna().unique())

    left, right = st.columns(2)
    chosen_competitions = left.multiselect("Competitions", competitions, default=[])
    chosen_folds = right.multiselect("Folds", folds, default=[])
    bins = st.slider("Bins", min_value=5, max_value=20, value=10)

    rows = data.filtered_forecasts(
        forecasts, competitions=chosen_competitions or None, folds=chosen_folds or None
    )
    if rows.empty:
        st.info("No matches match that filter.")
        return

    table = data.reliability_table(rows, bins=bins)
    error = data.calibration_error(table)
    scope = "all 39 competitions" if not chosen_competitions else ", ".join(chosen_competitions)

    metrics = st.columns(3)
    metrics[0].metric("Matches", f"{len(rows):,}")
    metrics[1].metric("Calibration error", f"{error:.4f}")
    metrics[2].metric("Model", SHIPPED)

    st.plotly_chart(reliability_diagram(table, title=f"Reliability — {scope}"), width="stretch")
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Marker area is the matches in a bin. The bins are wildly uneven — most "
        "of a three-class football forecast sits between 0.15 and 0.55 — which "
        "is why the calibration error above weights them the same way."
    )


def competitions_panel(reports: data.Reports) -> None:
    """Where the model is least honest, and where it scores worst.

    Two different questions, and the milestone puts them side by side because
    they have different answers: the competition with the worst log loss is not
    the one with the worst calibration.
    """
    st.subheader("By competition")
    if not reports.has_forecasts or reports.forecasts is None:
        _missing("the per-match forecasts", "make card")
        return

    rows = data.filtered_forecasts(reports.forecasts)
    worst = data.worst_competitions(rows)
    st.plotly_chart(competition_bars(worst), width="stretch")
    st.dataframe(worst, width="stretch", hide_index=True)

    if reports.has_scores and reports.scores is not None:
        st.markdown("**Log loss per competition, every forecaster**")
        st.dataframe(data.by_competition(reports.scores), width="stretch", hide_index=True)


def predict_panel(client: PredictionClient) -> None:
    """The one panel that leaves this process.

    It asks the running service rather than loading the artefact, so there is
    exactly one implementation of "what does the model say" and no second one
    to drift from it. When the service is not running the panel says so and the
    rest of the page is unaffected — which is why this is the last panel and
    not a precondition for the others.
    """
    st.subheader("Price a fixture")
    st.caption(
        f"Answered by the service at `{client.base_url}`, not by this process. "
        "The provider publishes results rather than a fixture list, so the "
        "matches that can be priced are the ones in the feature table."
    )

    try:
        health = client.health()
    except ServiceError as error:
        st.warning(f"{error}\n\nStart it with `make api`, or set `DASHBOARD_API_URL`.")
        return

    if health.get("status") != "ok":
        st.warning(
            "The service is up but not ready: "
            + ", ".join(
                f"{part['name']} ({part.get('detail')})"
                for part in health.get("components", [])
                if not part.get("ready")
            )
        )
        return

    left, right = st.columns([2, 1])
    competition = left.text_input("Competition id", value="", placeholder="ENG_1")
    limit = right.number_input("How many", min_value=1, max_value=50, value=10)

    try:
        fixtures = client.fixtures(competition_id=competition or None, limit=int(limit))
    except ServiceError as error:
        st.error(str(error))
        return

    if not fixtures:
        st.info("No fixtures matched.")
        return

    labels = {
        f"{f['date']}  {f['home_team']} v {f['away_team']}  ({f['competition_id']})": f["match_id"]
        for f in fixtures
    }
    chosen = st.selectbox("Fixture", list(labels))
    if not st.button("Price it", type="primary"):
        return

    try:
        answer = client.predict({"match_id": labels[chosen]})
    except ServiceError as error:
        st.error(str(error))
        return

    probabilities = answer["probabilities"]
    columns = st.columns(3)
    for column, (outcome, key) in zip(
        columns, (("Home", "home"), ("Draw", "draw"), ("Away", "away")), strict=True
    ):
        column.metric(outcome, f"{probabilities[key]:.1%}")

    if answer.get("in_sample"):
        st.warning(
            "**This fixture was inside the served model's training window.** "
            "The probability above is in-sample and is not what the "
            "walk-forward folds measured — see the model card."
        )


# ---- page --------------------------------------------------------------------


def _missing(what: str, command: str) -> None:
    """The empty state, which is what a clean checkout produces."""
    st.info(f"No {what} yet. Run `{command}` to produce them.")


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title=TITLE, page_icon="⚽", layout="wide")
    settings = load_settings()
    client = PredictionClient(
        base_url=settings.dashboard.api_url,
        timeout_seconds=settings.dashboard.request_timeout_seconds,
    )

    st.title(TITLE)
    st.caption(CAPTION)

    reports = load(settings)
    absent = reports.missing()
    if absent:
        st.warning("Missing: " + "; ".join(absent))

    scoreboard, calibration, competitions, predict = st.tabs(
        ["Scoreboard", "Reliability", "By competition", "Price a fixture"]
    )
    with scoreboard:
        scoreboard_panel(reports)
    with calibration:
        reliability_panel(reports)
    with competitions:
        competitions_panel(reports)
    with predict:
        predict_panel(client)


main()
