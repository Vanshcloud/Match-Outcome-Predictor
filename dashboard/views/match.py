"""One fixture: what the model says, and what that is worth.

The page a card links to. It answers the questions a forecast raises:

1. **What does the model say?** Three calibrated probabilities, from the
   service over HTTP. Never computed here — see :mod:`dashboard.client`.
2. **What is that worth?** The rate at which forecasts stated near this one
   actually happened, over this competition, from the same reliability tables
   ``docs/MODEL_CARD.md`` is generated from. A stated probability with no
   measured reliability beside it is the number this project exists to stop
   people quoting.
3. **For a match already played: form and head-to-head**, from the match table.

There is no closing-line panel: the feed's upcoming fixtures carry no odds.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import pandas as pd
import streamlit as st

from dashboard import context, ui
from dashboard.domain import competition as catalogue
from dashboard.domain.match import (
    FEED_ONLY_PREFIX,
    Prediction,
)
from dashboard.services import history, matchday, reports

MATCH_PARAM = "match"
"""The query parameter every card links with. ``match?match=<match_id>``."""

PICKER_FIXTURES = 25


def render() -> None:
    """The prediction detail page."""
    ctx = context.resolve()
    match_id = str(st.query_params.get(MATCH_PARAM, "") or "")
    if not match_id:
        _picker(ctx)
        return

    if match_id.startswith(FEED_ONLY_PREFIX):
        # A live-feed card: open the table's fixture it shows, which the model can price.
        twin = matchday.priced_twin(match_id, fixtures=ctx.fixtures, predictions=ctx.predictions)
        if twin is not None:
            match_id = twin
            st.query_params[MATCH_PARAM] = twin

    row = _lookup(ctx, match_id)
    answer = _price(ctx, match_id)
    prediction = answer if isinstance(answer, Prediction) else None
    if row is None and prediction is None and match_id.startswith(FEED_ONLY_PREFIX):
        st.info(
            "This card comes from the live fixture feed, and no fixture the model "
            "has priced matches it — the competition has no match history here "
            "(the Champions League), or `make fixtures` has not built this round "
            "yet. Find either club in **Search**, or pick a match below."
        )
        _picker(ctx)
        return
    if row is None and prediction is None:
        if ctx.predictions.available:
            st.error(
                f"There is no match `{match_id}`: it is not in the match table, and the "
                "service has no forecast for it. Pick one below."
            )
        else:
            # Not found here, and the service that might know it is not answering.
            st.error(f"There is no match `{match_id}` in the match table.")
            st.caption(str(answer))
        _picker(ctx)
        return
    if prediction is None:
        st.warning(str(answer))

    _header(row, prediction)
    _forecast(ctx, row, prediction)
    if row is not None:
        _history(ctx, row)


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


def _price(ctx: context.Context, match_id: str) -> Prediction | str:
    """The service's answer, or the sentence saying why there is none.

    Advice to start the service only when it is not running: a service that is
    up and answers 404 has been asked about a fixture it does not hold, and
    telling that reader to run `make api` sends them after the wrong problem.
    The reason is read before :attr:`available`, which asks again and resets it.
    """
    prediction = ctx.predictions.predict(match_id)
    if prediction is not None:
        return prediction
    reason = ctx.predictions.error or "no detail given"
    if ctx.predictions.available:
        return (
            f"The prediction service is running but has no forecast for this fixture ({reason}). "
            "A match ingested after the feature table was built is not indexed until "
            "`make ratings features` and an API restart."
        )
    return f"No forecast: {reason}. Start the service with `make api`, or set `DASHBOARD_API_URL`."


def _picker(ctx: context.Context) -> None:
    """What the page shows when it was opened without a fixture."""
    st.title("Pick a match")
    st.caption("Open any fixture the model can price for its forecast.")
    found = ctx.predictions.priceable(limit=PICKER_FIXTURES)
    if not found:
        st.info(
            ctx.predictions.error
            or "The service has no fixtures indexed yet. Run `make model` and `make data`."
        )
        return
    # No probability bars: that would be one HTTP call per card, and the page they open has it.
    ui.card_grid(
        [
            ui.match_card(one, competition_label=catalogue.short_label(one.competition_id))
            for one in matchday.with_crests(found, ctx.fixtures)
        ]
    )


# ---- the page ----------------------------------------------------------------


def _header(row: pd.Series | None, prediction: Prediction | None) -> None:
    """Who played, where and when.

    From the match table, which is the only source here that has a scoreline, a
    season and a kick-off time. A fixture the table does not hold — an upcoming
    one, usually — is titled from the service's own answer, which names the
    clubs and the date it priced.
    """
    if row is None and prediction is not None and prediction.home_team:
        st.title(f"{prediction.home_team} v {prediction.away_team}")
        played = prediction.date is not None and prediction.date < dt.date.today()
        parts = [
            catalogue.label(prediction.competition_id),
            f"{prediction.date:%A %d %B %Y}" if prediction.date else "",
            (
                "played, but its result is not ingested yet — run `make data`"
                if played
                else "upcoming · priced before kick-off"
            ),
        ]
        st.caption(" · ".join(part for part in parts if part))
        return
    if row is None:
        st.title("A fixture the match table does not hold")
        st.caption(
            "The service can price it, but the match table does not hold it: it "
            "is an upcoming fixture, or `make data` has not caught up with its result."
        )
        return

    home, away = str(row["home_team"]), str(row["away_team"])
    competition_id = str(row["competition_id"])
    st.title(f"{home} v {away}")
    when = pd.Timestamp(str(row["date"])).strftime("%A %d %B %Y")
    kickoff = _text(row.get("kickoff"))
    season = _text(row.get("season"))
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


def _text(value: object) -> str:
    """A table cell as stripped text, or ``""`` when it is null.

    ``value or ""`` is not enough: the match table is nullable-typed, and
    ``bool(pd.NA)`` raises. Every match before 2019-07 has no recorded
    kick-off, so the short form crashed the page for most of the history.
    """
    return "" if value is None or pd.isna(value) else str(value).strip()  # type: ignore[call-overload]


def _forecast(ctx: context.Context, row: pd.Series | None, prediction: Prediction | None) -> None:
    """The probabilities, what they were produced by, and what they are worth."""
    ui.section("The forecast", "three calibrated probabilities, answered by the service")
    if prediction is None:
        st.caption("No forecast for this fixture; the note above says why.")
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
        f"{ui.forecaster_label(prediction.model) if prediction.model else 'The model'} "
        f"· version {prediction.model_version or '?'} · "
        "the confidence band above is the largest of the three probabilities, "
        "not a measurement. What it is worth is below."
    )

    if prediction.in_sample:
        st.warning(
            "**This fixture was inside the served model's training window.** "
            "The probabilities above are in-sample and are not what the "
            "walk-forward folds measured — see `docs/MODEL_CARD.md`."
        )

    competition_id = str(row["competition_id"]) if row is not None else prediction.competition_id
    _calibration(ctx, competition_id or None, probabilities)


def _calibration(
    ctx: context.Context, competition_id: str | None, probabilities: Mapping[str, float]
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

    rows = reports.filtered_forecasts(
        loaded.forecasts, competitions=[competition_id] if competition_id else None
    )
    scope = catalogue.label(competition_id) if competition_id else "all competitions"
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
        pd.DataFrame(
            {
                "Date": pd.to_datetime(meetings["date"]).dt.strftime("%d %b %Y"),
                "Competition": meetings["competition_id"].map(catalogue.label),
                "Home": meetings["home_team"],
                "Score": [
                    f"{int(h)}–{int(a)}" if pd.notna(h) and pd.notna(a) else "–"
                    for h, a in zip(meetings["home_goals"], meetings["away_goals"], strict=True)
                ],
                "Away": meetings["away_team"],
            }
        ),
        width="stretch",
        hide_index=True,
    )
