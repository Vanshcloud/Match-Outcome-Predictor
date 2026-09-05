"""One fixture, in as much depth as this project can honestly go.

The page a card links to. It answers four questions and is explicit about the
fifth it cannot:

1. **What does the model say?** Three calibrated probabilities, from the
   service over HTTP. Never computed here — see :mod:`dashboard.client`.
2. **What is that worth?** The rate at which forecasts stated near this one
   actually happened, over this competition, from the same reliability tables
   ``docs/MODEL_CARD.md`` is generated from. A stated probability with no
   measured reliability beside it is the number this project exists to stop
   people quoting.
3. **What happened when these two met before?** Head-to-head, from the match
   table.
4. **How have they been playing?** Form, likewise.

The fifth — expected goals, injuries, availability, bookmaker odds, in-play
statistics — has no source in this repository, and the sections for them say so
and name the milestone rather than being absent.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from dashboard import context, ui
from dashboard.domain import competition as catalogue
from dashboard.domain.match import Prediction
from dashboard.services import history, reports

MATCH_PARAM = "match"
"""The query parameter every card links with. ``match?match=<match_id>``."""

PICKER_FIXTURES = 25

FUTURE_SECTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "Expected goals",
        "Milestone 17",
        "The primary feed carries shots and shots on target; it carries no xG "
        "and this project fits none. A shot-quality model is a modelling "
        "milestone with its own ablation, not a panel.",
    ),
    (
        "Bookmaker odds",
        "Milestone 17",
        "Closing odds are in the match table and are deliberately not served "
        "beside a forecast: they are the benchmark this project measures "
        "itself against, and a page that showed both invites the comparison to "
        "be made without the walk-forward folds that make it meaningful. "
        "<code>docs/EVALUATION.md</code> is where that comparison lives.",
    ),
    (
        "Injuries and availability",
        "Milestone 18",
        "No squad or availability data is ingested. The model has never seen a "
        "team sheet, which <code>docs/MODEL_CARD.md</code> lists among its "
        "limitations — it is not a gap in this page.",
    ),
    (
        "In-play statistics",
        "Milestone 15",
        "Needs the same fixture provider the live centre does. The card and "
        "the <code>Fixture</code> it renders already carry a minute and a "
        "score.",
    ),
)


def render() -> None:
    """The prediction detail page."""
    ctx = context.resolve()
    match_id = str(st.query_params.get(MATCH_PARAM, "") or "")
    if not match_id:
        _picker(ctx)
        return

    row = _lookup(ctx, match_id)
    prediction = _price(ctx, match_id)
    if row is None and prediction is None:
        st.error(f"No match `{match_id}` in the table, and the service could not price it.")
        _picker(ctx)
        return

    _header(row)
    _forecast(ctx, row, prediction)
    if row is not None:
        _history(ctx, row)
    _future()


# ---- finding the fixture -----------------------------------------------------


def _lookup(ctx: context.Context, match_id: str) -> pd.Series | None:
    """The match table's row for this fixture, if there is a table and a row.

    Absent is ordinary rather than exceptional: the service can price a fixture
    on a machine that never ingested the matches, and the page then shows the
    forecast without the history.
    """
    if not ctx.has_matches:
        return None
    return history.by_id(history.all_matches(ctx.matches_path), match_id)


def _price(ctx: context.Context, match_id: str) -> Prediction | None:
    """The service's answer, or ``None`` with the reason on the page.

    A warning rather than an error: a reader who opened this page from a
    results card with no service running still gets the fixture, the form and
    the head-to-head, and that is most of the page.
    """
    prediction = ctx.predictions.predict(match_id)
    if prediction is None:
        st.warning(
            f"{ctx.predictions.error}\n\nStart the service with `make api`, "
            "or set `DASHBOARD_API_URL`."
        )
    return prediction


def _picker(ctx: context.Context) -> None:
    """What the page shows when it was opened without a fixture."""
    st.title("Pick a match")
    st.caption("Every card on this dashboard links here. Or choose one directly.")
    found = ctx.predictions.priceable(limit=PICKER_FIXTURES)
    if not found:
        st.info(
            ctx.predictions.error
            or "The service has no fixtures indexed yet. Run `make model` and `make data`."
        )
        return
    labels = {
        f"{one.date}  {one.home_team} v {one.away_team}  "
        f"({catalogue.short_label(one.competition_id)})": one.match_id
        for one in found
    }
    chosen = st.selectbox("Fixture", list(labels))
    if st.button("Open", type="primary"):
        st.query_params[MATCH_PARAM] = labels[chosen]
        st.rerun()


# ---- the page ----------------------------------------------------------------


def _header(row: pd.Series | None) -> None:
    """Who played, where and when.

    From the match table, which is the only source here that has a scoreline, a
    season and a kick-off time. A fixture the table does not hold still gets
    its forecast — the section below this one — and gets a title saying so
    rather than a blank one.
    """
    if row is None:
        st.title("A fixture the match table does not hold")
        st.caption(
            "The service can price it, so it is in the feature table. Run "
            "`make data` to ingest the results this page reads."
        )
        return

    home, away = str(row["home_team"]), str(row["away_team"])
    competition_id = str(row["competition_id"])
    st.title(f"{home} v {away}")
    when = pd.Timestamp(str(row["date"])).strftime("%A %d %B %Y")
    kickoff = str(row.get("kickoff") or "").strip()
    season = str(row.get("season") or "").strip()
    parts = [
        catalogue.label(competition_id),
        when if kickoff in ui.UNKNOWN_KICKOFF else f"{when}, {kickoff}",
        f"season {season}" if season else "",
    ]
    st.caption(" · ".join(part for part in parts if part))

    if pd.notna(row.get("home_goals")):
        left, right = st.columns(2)
        left.metric(home, int(row["home_goals"]))
        right.metric(away, int(row["away_goals"]))


def _forecast(ctx: context.Context, row: pd.Series | None, prediction: Prediction | None) -> None:
    """The probabilities, what they were produced by, and what they are worth."""
    ui.section("The forecast", "three calibrated probabilities, answered by the service")
    if prediction is None:
        st.caption("No forecast: the service did not answer.")
        return

    probabilities = prediction.probabilities
    st.markdown(ui.probability_bar(probabilities), unsafe_allow_html=True)
    st.markdown(ui.confidence_pill(probabilities), unsafe_allow_html=True)

    columns = st.columns(3)
    for column, (label, key) in zip(
        columns, (("Home", "home"), ("Draw", "draw"), ("Away", "away")), strict=True
    ):
        column.metric(label, f"{probabilities[key]:.1%}")

    st.caption(
        f"{prediction.model or 'the model'} v{prediction.model_version or '?'} · "
        "the confidence band above is the largest of the three probabilities, "
        "not a measurement. What it is worth is below."
    )

    if prediction.in_sample:
        st.warning(
            "**This fixture was inside the served model's training window.** "
            "The probabilities above are in-sample and are not what the "
            "walk-forward folds measured — see `docs/MODEL_CARD.md`."
        )

    _calibration(ctx, row, probabilities)


def _calibration(
    ctx: context.Context, row: pd.Series | None, probabilities: Mapping[str, float]
) -> None:
    """How often forecasts stated near this one actually happened.

    The measurement that turns a probability into a claim a reader can check.
    It comes from :mod:`dashboard.services.reports` — the same functions the
    model card is generated from, over the persisted per-match forecasts — so
    the number here and the number in the card are the same number from the
    same code.
    """
    ui.section("What that probability is worth", "measured on the walk-forward folds")
    loaded = reports.reports(ctx.reports_dir)
    if not loaded.has_forecasts or loaded.forecasts is None:
        st.info("No per-match forecasts yet. Run `make card` to produce them.")
        return

    competition_id = str(row["competition_id"]) if row is not None else None
    rows = reports.filtered_forecasts(
        loaded.forecasts, competitions=[competition_id] if competition_id else None
    )
    scope = catalogue.short_label(competition_id) if competition_id else "all competitions"
    if rows.empty:
        st.caption(f"No scored forecasts for {scope}.")
        return

    table = reports.reliability_table(rows)
    _, _, stated = ui.confidence(probabilities)
    bin_row = _bin_for(table, stated)
    if bin_row is None:
        st.caption(f"No scored forecast in {scope} fell in this probability band.")
        return

    columns = st.columns(3)
    columns[0].metric("Stated here", f"{stated:.1%}")
    columns[1].metric("Happened, in that band", f"{float(bin_row['observed']):.1%}")
    columns[2].metric("Over", f"{int(bin_row['n']):,} forecasts")
    st.caption(
        f"Forecasts in {scope} stated between {float(bin_row['lower']):.0%} and "
        f"{float(bin_row['upper']):.0%} came true {float(bin_row['observed']):.1%} of "
        f"the time. The gap is {float(bin_row['gap']):+.4f}. The full diagram, "
        "filterable by competition and fold, is on the Model page."
    )


def _bin_for(table: pd.DataFrame, stated: float) -> pd.Series | None:
    """The reliability bin a stated probability falls in, if any holds matches.

    ``None`` is the ordinary answer at the ends of the range: a model that
    states 0.93 for a home win in a competition where nothing was ever stated
    above 0.7 has no measured band to be checked against, and saying so is the
    honest result.
    """
    inside = table[(table["lower"] <= stated) & (table["upper"] >= stated) & (table["n"] > 0)]
    return None if inside.empty else inside.iloc[0]


def _history(ctx: context.Context, row: pd.Series) -> None:
    """Form and head-to-head: what the two clubs have actually been doing.

    Read from the match table, never derived. A rolling average computed here
    would be an unprobed second copy of a feature, and the layer that owns
    those is guarded by the leakage suite for a reason.
    """
    home, away = str(row["home_team"]), str(row["away_team"])

    ui.section("Recent form", f"the last {history.FORM_MATCHES} matches, oldest first")
    table = history.all_matches(ctx.matches_path)
    left, right = st.columns(2)
    for column, team in ((left, home), (right, away)):
        column.markdown(f"**{team}**")
        column.markdown(ui.form_string(history.form(table, team)), unsafe_allow_html=True)

    ui.section("Head to head", f"the last {history.HEAD_TO_HEAD_MATCHES} meetings")
    # Over the whole table rather than over one club's most recent matches.
    # Filtering to a club first caps the scan at that club's last 200 fixtures,
    # which would make an old derby report "these two have not met" — a wrong
    # sentence produced by a limit rather than by the football.
    meetings = history.head_to_head(table, home, away)
    home_wins, draws, away_wins = history.record(meetings, home)
    columns = st.columns(3)
    columns[0].metric(f"{home} won", home_wins)
    columns[1].metric("Drawn", draws)
    columns[2].metric(f"{away} won", away_wins)
    st.dataframe(
        meetings[["date", "competition_id", "home_team", "home_goals", "away_goals", "away_team"]],
        width="stretch",
        hide_index=True,
    )


def _future() -> None:
    """The sections with no source, named rather than absent."""
    ui.section("Not on this page yet", "and what each one needs")
    for title, milestone, body in FUTURE_SECTIONS:
        with st.expander(f"{title} — {milestone}"):
            st.markdown(body, unsafe_allow_html=True)
    st.caption(
        "What each feature block is worth to the model is measured in "
        "`docs/EXPLAINABILITY.md`. Per-fixture attribution — this match's "
        "SHAP values — is a serving change rather than a page: the service "
        "would have to return them."
    )
