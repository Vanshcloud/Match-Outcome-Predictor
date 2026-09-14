"""How good the model is, measured.

A headline — what it predicts, what it scores, what the closing line scores,
how reliable it is — over five panels: the scoreboard, the reliability diagram
and the per-competition breakdown; what the *service* actually answered,
scored against what happened afterwards, the only panel here whose subject is
not the backtest; and the
limitations, so that what the model must not be read as saying sits on the page
that says how good it is rather than only in the card.

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

from dashboard import context, ui
from dashboard.charts import competition_bars, reliability_diagram, score_bars
from dashboard.domain import competition as catalogue
from dashboard.services import reports as report_service
from src.evaluation.archive import horizon
from src.models.ensemble import SHIPPED

LIMITATIONS: tuple[str, ...] = (
    "**It does not beat the closing line** in any competition it was scored on. "
    "It is a forecaster to be measured, not a betting signal.",
    "**Forecasts for played matches are in-sample.** The served model is fitted "
    "on the whole history, so its answer for a played match comes from a model "
    "that saw the result; the match page flags these. The numbers on this page "
    "are walk-forward and out of sample.",
    "**Inputs are what free, match-level data carries:** results, shots where the "
    "competition records them, dates and ratings. No lineups, injuries, transfers "
    "or shot-level xG.",
    "**A club with no history** gets forecasts close to the base rates.",
    "**Reliability varies by competition,** and the rare statements of 80% or more "
    "are somewhat overconfident. The By competition tab shows where.",
    "**Pre-match only.** Nothing is fitted on in-play state, and the live scores "
    "elsewhere on this dashboard never change a forecast.",
)
"""What the model must not be read as saying. The same limits as the model card."""

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
    summary_panel(reports)

    scoreboard, calibration, competitions, served, limits = st.tabs(
        ["Scoreboard", "Reliability", "By competition", "What we served", "Limitations"]
    )
    with scoreboard:
        scoreboard_panel(reports)
    with calibration:
        reliability_panel(reports)
    with competitions:
        competitions_panel(reports)
    with served:
        archive_panel(reports)
    with limits:
        limitations_panel()


def summary_panel(reports: report_service.Reports) -> None:
    """The headline numbers, each read from a report, before any tab.

    What the model predicts, how it was scored and against what, in the first
    screen of the page. Nothing here is computed from anything but the same
    tables the tabs below render.
    """
    if reports.scores is None or reports.forecasts is None:
        return
    if reports.scores.empty or reports.forecasts.empty:
        return
    table = report_service.leaderboard(reports.scores).set_index("forecaster")
    rows = report_service.filtered_forecasts(reports.forecasts)
    error = report_service.calibration_error(report_service.reliability_table(rows))

    columns = st.columns(4)
    columns[0].metric("Out-of-sample forecasts", f"{len(rows):,}")
    columns[1].metric("Model log loss", _score(table, SHIPPED))
    columns[2].metric("Closing-line log loss", _score(table, "bookmaker"))
    columns[3].metric("Pooled calibration error", f"{error:.4f}")
    st.caption(
        "Predicts the full-time result (home, draw or away) before kick-off. "
        f"Scored on {rows['fold'].nunique()} expanding walk-forward folds, each "
        "trained only on matches before its first evaluation day. Log loss is over "
        f"the {int(table['n'].iloc[0]):,} matches every forecaster could price, and "
        "lower is better. The closing line is the benchmark, never a model input."
    )


def _log_loss_text(value: float) -> str:
    """One log loss for the scoreboard, with infinity written out."""
    return f"{value:.4f}" if np.isfinite(value) else "\u221e"


def _score(table: pd.DataFrame, forecaster: str) -> str:
    """One forecaster's pooled log loss, or a dash when it was not scored."""
    if forecaster not in table.index:
        return "—"
    return f"{table.loc[forecaster, 'log_loss']:.4f}"


def limitations_panel() -> None:
    """What the model must not be read as saying."""
    st.subheader("What this model is not")
    st.markdown("\n".join(f"- {one}" for one in LIMITATIONS))
    st.caption("The generated model card, `docs/MODEL_CARD.md`, lists them with the evidence.")


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
    st.plotly_chart(score_bars(table, labels=ui.FORECASTER_LABELS), width="stretch")
    st.dataframe(
        table.assign(
            forecaster=table["forecaster"].map(ui.forecaster_label),
            # A string, because a NumberColumn renders an infinite log loss as
            # an empty cell, and a blank where every other row has a number
            # reads as data this project failed to compute rather than as the
            # one forecaster the metric is unbounded for.
            log_loss=table["log_loss"].map(_log_loss_text),
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "n": st.column_config.NumberColumn("matches", format="localized"),
            "log_loss": st.column_config.TextColumn("log loss"),
            "rps": st.column_config.NumberColumn("RPS", format="%.4f"),
            "accuracy": st.column_config.NumberColumn("accuracy", format="percent"),
        },
    )
    st.caption(
        "Always home scores infinite log loss and it is not clipped — a "
        "forecast that ruled out what happened was infinitely wrong. RPS still "
        "ranks it, because RPS is bounded and knows H, D and A are ordered."
    )


def reliability_panel(reports: report_service.Reports) -> None:
    """The page's central panel: reliability, filtered by the reader.

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
        else ", ".join(catalogue.short_label(one) for one in chosen_competitions)
    )

    metrics = st.columns(3)
    metrics[0].metric("Matches", f"{len(rows):,}")
    metrics[1].metric("Calibration error", f"{error:.4f}")
    metrics[2].metric("Model", ui.forecaster_label(SHIPPED))

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
    worst = _named(report_service.worst_competitions(rows))
    st.plotly_chart(competition_bars(worst), width="stretch")
    st.dataframe(
        worst.rename(
            columns={"competition_id": "competition", "calibration_error": "calibration error"}
        ),
        width="stretch",
        hide_index=True,
    )

    if reports.has_scores and reports.scores is not None:
        st.markdown("**Log loss per competition, every forecaster**")
        st.dataframe(
            _named(report_service.by_competition(reports.scores)).rename(
                columns={"competition_id": "competition", "n": "matches", **ui.FORECASTER_LABELS}
            ),
            width="stretch",
            hide_index=True,
        )


def _named(table: pd.DataFrame) -> pd.DataFrame:
    """The same rows with each competition id replaced by the name a reader knows."""
    return table.assign(competition_id=table["competition_id"].map(catalogue.short_label))


def archive_panel(reports: report_service.Reports) -> None:
    """What the service served, scored — and whether that can mean anything yet.

    Every other panel on this page is about the walk-forward
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

    for _, row in reports.archive.iterrows():
        _version(row)


def _version(row: pd.Series) -> None:
    """One served model version: what it answered, and what that is worth."""
    st.markdown(
        f"**{ui.forecaster_label(str(row['model']))}** · version {row['model_version']} · "
        f"served {row['first_served']:%d %b %Y} to {row['last_served']:%d %b %Y}"
    )
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
