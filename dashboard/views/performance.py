"""How good the model is, measured — the three panels Milestone 12 shipped.

Unchanged in substance from the tabs this page had before it became an
application: the scoreboard, the reliability diagram and the per-competition
breakdown, moved onto a page of their own so the home page can be about
football and this one about the forecaster.

Milestone 19 adds a fourth panel, and it is the only one on this dashboard
whose subject is not the backtest: what the *service* actually answered, scored
against what happened afterwards.

**It computes nothing.** Every table here is produced by
:mod:`src.pipelines.report`, :mod:`src.pipelines.backtest` or
:mod:`src.evaluation.archive`, and where a
number here appears in ``docs/MODEL_CARD.md`` it is the same number from the
same function. A dashboard that recomputed a calibration error would be a
second measurement of the model with no test holding it to the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import context
from dashboard.charts import competition_bars, reliability_diagram, score_bars
from dashboard.domain import competition as catalogue
from dashboard.services import reports as report_service
from src.evaluation.archive import horizon
from src.models.ensemble import SHIPPED

WORTH_SEEING: tuple[float, ...] = (0.10, 0.05, 0.02, 0.0163, 0.01)
"""The shifts the horizon table is priced for, the same five `make archive`
prints. 0.0163 is here because it is this project's own number — the gap
between the shipped model and the closing line — so a reader can see what it
would take to notice a change that size."""


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

    scoreboard, calibration, competitions, served = st.tabs(
        ["Scoreboard", "Reliability", "By competition", "What we served"]
    )
    with scoreboard:
        scoreboard_panel(reports)
    with calibration:
        reliability_panel(reports)
    with competitions:
        competitions_panel(reports)
    with served:
        archive_panel(reports)


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


def archive_panel(reports: report_service.Reports) -> None:
    """What the service served, scored — and whether that can mean anything yet.

    Milestone 19. Every other panel on this page is about the walk-forward
    backtest, which is a measurement of a model. This one is about the
    deployment: the forecasts that were actually answered, joined to the
    results that arrived afterwards, against what the folds said the same model
    does.

    **The size of the archive is rendered before the drift figure, not after
    it.** A mean log loss over eleven matches drawn from a distribution with a
    per-match spread of 0.4 is noise, and a page that showed it as a headline
    with the caveat underneath would be read headline-first. So the counts come
    first, the smallest detectable shift sits beside the drift, and a
    difference under it is reported as "not distinguishable" rather than as a
    number with a sign.
    """
    st.subheader("What the service actually served")
    absent = reports.missing_archive()
    if absent or reports.archive is None:
        st.info(
            f"No drift report: {absent}. It needs three things a backtest does "
            "not — a prediction log (`PREDICTION_LOG_DSN`), a service that has "
            "been called, and matches that have since been played. Then run "
            "`make archive`. This is the one report `make reproduce` cannot "
            "rebuild: it is a record of things that happened."
        )
        return

    table = reports.archive
    st.dataframe(table, width="stretch", hide_index=True)
    for _, row in table.iterrows():
        _version(row)


def _version(row: pd.Series) -> None:
    """One served model version: what it answered, and what that is worth."""
    st.markdown(f"**{row['model']} {row['model_version']}**")
    counts = st.columns(4)
    counts[0].metric("Logged", f"{int(row['logged']):,}")
    counts[1].metric("In sample", f"{int(row['in_sample']):,}")
    counts[2].metric("No result yet", f"{int(row['unresolved']):,}")
    counts[3].metric("Scored", f"{int(row['n']):,}")

    scored, reach = int(row["n"]), float(row["detectable"])
    if not scored:
        st.caption(
            "Nothing to score yet. In-sample forecasts are excluded because the "
            "shipped model is fitted on the whole history — a match inside it "
            "was trained on, and scoring it against the walk-forward folds "
            "would compare a memorised answer to an out-of-sample one."
        )
        _horizon(float(row["spread"]))
        return

    drift = float(row["drift"])
    figures = st.columns(3)
    figures[0].metric("Served", f"{float(row['log_loss']):.4f}")
    figures[1].metric("Backtest", f"{float(row['baseline']):.4f}")
    figures[2].metric("Difference", f"{drift:+.4f}", delta=f"±{reach:.4f}")
    if bool(row["distinguishable"]):
        st.warning(
            f"The served forecasts score {drift:+.4f} against the folds, over "
            f"{scored:,} matches — larger than the {reach:.4f} that could be "
            "noise at this sample size. That is a difference worth explaining."
        )
    else:
        st.caption(
            f"{drift:+.4f} against the folds over {scored:,} scored forecasts. "
            f"Anything under {reach:.4f} is indistinguishable from noise at "
            f"this size, so **this is not evidence of drift** — it is an "
            "archive that cannot yet answer the question."
        )
        _horizon(float(row["spread"]))


def _horizon(spread: float) -> None:
    """How many scored forecasts each size of shift would take to see.

    The same function `make archive` prints, so the page and the command
    cannot disagree about how much archive is still needed.
    """
    if not np.isfinite(spread):  # no baseline was measured
        return
    st.caption(
        "How many scored forecasts a difference would need before it could be "
        f"told from noise, at a per-match spread of {spread:.4f}:"
    )
    st.dataframe(
        horizon(spread, WORTH_SEEING).rename(
            columns={"difference": "shift in log loss", "needed": "forecasts needed"}
        ),
        width="stretch",
        hide_index=True,
    )
