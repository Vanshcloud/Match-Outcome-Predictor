"""How good the model is, measured — the three panels Milestone 12 shipped.

Unchanged in substance from the tabs this page had before it became an
application: the scoreboard, the reliability diagram and the per-competition
breakdown, moved onto a page of their own so the home page can be about
football and this one about the forecaster.

**It computes nothing.** Every table here is produced by
:mod:`src.pipelines.report` or :mod:`src.pipelines.backtest`, and where a
number here appears in ``docs/MODEL_CARD.md`` it is the same number from the
same function. A dashboard that recomputed a calibration error would be a
second measurement of the model with no test holding it to the first.
"""

from __future__ import annotations

import streamlit as st

from dashboard import context
from dashboard.charts import competition_bars, reliability_diagram, score_bars
from dashboard.domain import competition as catalogue
from dashboard.services import reports as report_service
from src.models.ensemble import SHIPPED


def render() -> None:
    """The model page."""
    ctx = context.resolve()
    st.title("Model")
    st.caption(
        "What the shipped forecaster scores, how honest its probabilities are, "
        "and where it is least honest. Every table is read from a report the "
        "pipelines wrote."
    )

    reports = report_service.reports(ctx.reports_dir)
    absent = reports.missing()
    if absent:
        st.warning("Missing: " + "; ".join(absent))

    scoreboard, calibration, competitions = st.tabs(["Scoreboard", "Reliability", "By competition"])
    with scoreboard:
        scoreboard_panel(reports)
    with calibration:
        reliability_panel(reports)
    with competitions:
        competitions_panel(reports)


def _missing(what: str, command: str) -> None:
    """The empty state, which is what a clean checkout produces."""
    st.info(f"No {what} yet. Run `{command}` to produce them.")


def scoreboard_panel(reports: report_service.Reports) -> None:
    """Every forecaster, over the matches all of them could price."""
    st.subheader("How the forecasters compare")
    if not reports.has_scores or reports.scores is None:
        _missing("the backtest scores", "make ensemble")
        return

    table = report_service.leaderboard(reports.scores)
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


def reliability_panel(reports: report_service.Reports) -> None:
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
    chosen_competitions = left.multiselect(
        "Competitions", competitions, default=[], format_func=catalogue.label
    )
    chosen_folds = right.multiselect("Folds", folds, default=[])
    bins = st.slider("Bins", min_value=5, max_value=20, value=10)

    rows = report_service.filtered_forecasts(
        forecasts, competitions=chosen_competitions or None, folds=chosen_folds or None
    )
    if rows.empty:
        st.info("No matches match that filter.")
        return

    table = report_service.reliability_table(rows, bins=bins)
    error = report_service.calibration_error(table)
    scope = (
        f"all {len(competitions)} competitions"
        if not chosen_competitions
        else ", ".join(chosen_competitions)
    )

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


def competitions_panel(reports: report_service.Reports) -> None:
    """Where the model is least honest, and where it scores worst.

    Two different questions, and the page puts them side by side because they
    have different answers: the competition with the worst log loss is not the
    one with the worst calibration.
    """
    st.subheader("By competition")
    if not reports.has_forecasts or reports.forecasts is None:
        _missing("the per-match forecasts", "make card")
        return

    rows = report_service.filtered_forecasts(reports.forecasts)
    worst = report_service.worst_competitions(rows)
    st.plotly_chart(competition_bars(worst), width="stretch")
    st.dataframe(worst, width="stretch", hide_index=True)

    if reports.has_scores and reports.scores is not None:
        st.markdown("**Log loss per competition, every forecaster**")
        st.dataframe(
            report_service.by_competition(reports.scores), width="stretch", hide_index=True
        )
