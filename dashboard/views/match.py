"""One fixture, in as much depth as this project can honestly go.

The page a card links to. It answers six questions and is explicit about the
ones it cannot:

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
5. **What did the market think?** The closing line with the overround removed,
   beside the model, and — this is the part that matters — what the *distance*
   between the two was worth over 61,889 out-of-sample forecasts. Milestone 17.
6. **How many goals does the goal model expect?** Dixon-Coles' two Poisson
   rates, labelled as what they are and not as shot-quality xG, which nothing
   here has ever seen.

7. **Who is registered to play?** Both squads, from the same feed the live
   scores come from. Milestone 18, and the panel is careful about what it is:
   a squad is who is *registered*, an upper bound on who is available, and the
   only one of that milestone's three subjects — availability, injuries,
   transfers — that any reachable source answers.

What remains — team sheets, injuries, in-play statistics — has no source in
this repository, and the sections for them say so and name what each would
need rather than being absent.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from dashboard import context, ui
from dashboard.domain import competition as catalogue
from dashboard.domain.match import ExpectedGoals, MarketPrice, Prediction, Squad
from dashboard.services import history, market, matchday, reports

MATCH_PARAM = "match"
"""The query parameter every card links with. ``match?match=<match_id>``."""

PICKER_FIXTURES = 25

FUTURE_SECTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "Shot-quality expected goals",
        "unscheduled",
        "The panel above is a <em>goal-rate</em> model's expectation, which is "
        "not the same thing as xG from a shot map. The ingested feed carries "
        "shots and shots on target and no expected-goals column, so a real xG "
        "figure needs a second provider — and a shot-quality model fitted here "
        "would be a modelling milestone with its own ablation, not a panel.",
    ),
    (
        "Injuries, suspensions and the team sheet",
        "no source",
        "Milestone 18 shipped the squad panel above and found the rest has "
        "nowhere to come from: football-data.org has no injury endpoint at any "
        "tier, and its free plan answers a finished match with an "
        "<em>empty</em> <code>lineup</code> and <code>bench</code> — measured, "
        "not assumed. So a registered squad is the honest ceiling here. The "
        "model has never seen a team sheet either, which "
        "<code>docs/MODEL_CARD.md</code> lists among its limitations.",
    ),
    (
        "In-play statistics on this page",
        "unscheduled",
        "Milestone 15 shipped the live strip on the home page, which is where "
        "a minute and a changing score belong — this page is opened about a "
        "match a reader has already picked. Per-minute statistics (shots, "
        "possession) are a different feed from the fixture one, and no "
        "provider here carries them.",
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
    _market(ctx, match_id, prediction)
    _expected_goals(ctx, match_id, row)
    _availability(ctx, row)
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


def _market(ctx: context.Context, match_id: str, prediction: Prediction | None) -> None:
    """The closing line beside the forecast, and what the distance is worth.

    Milestone 12 kept the odds off this page on the grounds that showing both
    invites the comparison to be made without the folds that make it
    meaningful. Milestone 17 does not reverse that judgement — it satisfies it.
    The comparison is made *with* the folds, in
    :func:`src.pipelines.report.market_comparison`, and what this panel puts on
    the screen is the answer rather than the invitation.

    That answer is not the one a reader expects. A gap between the model and
    the price reads like an edge; over 61,889 out-of-sample forecasts it is the
    opposite — the model's deficit against the closing line grows with the size
    of the gap, and where the two are furthest apart the market gets *sharper*.
    So the caption says what the gap measures, which is this model's likely
    error on this fixture.
    """
    ui.section("What the market said", "the closing line, with the overround removed")
    quoted = market.price(ctx.odds, match_id)
    if quoted is None:
        st.caption(
            "No closing price for this fixture. The feed carries odds for about "
            "81% of the table — effectively everything from 2003 — and for "
            "nothing that has not been played."
            if ctx.odds.available
            else "No match table, so no closing price. Run `make data`."
        )
        return

    st.markdown(ui.probability_bar(quoted.probabilities), unsafe_allow_html=True)
    columns = st.columns(3)
    for column, (label, key) in zip(
        columns, (("Home", "home"), ("Draw", "draw"), ("Away", "away")), strict=True
    ):
        column.metric(label, f"{quoted.probabilities[key]:.1%}", f"{quoted.odds[key]:.2f}")
    st.caption(
        f"Decimal prices below each percentage. The book paid out on "
        f"{1 + quoted.overround:.1%} of the stake — the overround — and it is "
        "removed proportionally, which is the transparent way rather than the "
        "most accurate one: the favourite carries more of the margin than an "
        "equal share. `src/evaluation/market.py` does it once, for this page "
        "and for the benchmark alike."
    )

    if prediction is None:
        return
    _disagreement(ctx, prediction, quoted)


def _disagreement(ctx: context.Context, prediction: Prediction, quoted: MarketPrice) -> None:
    """How far the model is from the price, and what that distance was worth.

    The verdict is a row of a table `make card` wrote, not a computation here.
    A dashboard that recomputed a milestone's measurement to draw one caption
    would be a second number to reconcile with the one in the documents.
    """
    apart = market.gap(prediction, quoted)
    loaded = reports.reports(ctx.reports_dir)
    found = market.verdict(loaded.market, apart)

    ui.section("How far apart, and what that is worth", "measured on the walk-forward folds")
    if found is None:
        st.info(
            f"The model and the market are **{apart:.1%}** apart on this fixture. "
            f"What a gap that size has been worth is measured by `{market.BUILD_COMMAND}`, "
            "which has not been run here."
        )
        return

    deficit = float(found["model_minus_market"])
    columns = st.columns(3)
    columns[0].metric("Apart", f"{apart:.1%}")
    columns[1].metric("Model minus market, in this band", f"{deficit:+.4f}")
    columns[2].metric("Over", f"{int(found['n']):,} forecasts")
    st.caption(
        f"On the {int(found['n']):,} walk-forward forecasts that were "
        f"**{found['band']}** away from the closing line, this model scored "
        f"{float(found['model']):.4f} against the market's {float(found['market']):.4f}, "
        f"and beat it on {float(found['model_better']):.1%} of them. "
        "**A gap is not an edge.** The deficit grows with the distance — 0.0009 "
        "where the two nearly agree, 0.1835 where they are more than twenty "
        "points apart — so the honest reading of a wide gap here is that this "
        "model is more likely to be wrong about this match, not that the price "
        "is. See `docs/EVALUATION.md`."
    )


def _expected_goals(ctx: context.Context, match_id: str, row: pd.Series | None) -> None:
    """What the fitted goal model expects each side to score.

    Not xG. These are the two Poisson rates Milestone 4's Dixon-Coles model
    fits, and the distinction is the whole reason this panel is labelled the
    way it is — nothing in this project has ever seen a shot map. They are
    worth showing because the three-class probability this project reports is a
    sum over a Poisson grid built from exactly these two numbers, so a reader
    asking why a forecast leans one way is looking at its inputs.
    """
    ui.section("Expected goals", "what the fitted goal-rate model expects — not shot-quality xG")
    rates = market.goals(ctx.ratings_path, match_id)
    if rates is None:
        st.caption(
            "No goal rates for this fixture. Dixon-Coles refits per competition "
            "on a rolling window and has nothing to say about a match before its "
            "first fit; a clean checkout has no ratings table at all — "
            "`make ratings` builds one."
        )
        return
    _rates(rates, row)


def _rates(rates: ExpectedGoals, row: pd.Series | None) -> None:
    """The two rates, the total and the supremacy, with the caveat under them."""
    home = str(row["home_team"]) if row is not None else "Home"
    away = str(row["away_team"]) if row is not None else "Away"
    columns = st.columns(4)
    columns[0].metric(home, f"{rates.home:.2f}")
    columns[1].metric(away, f"{rates.away:.2f}")
    columns[2].metric("Total", f"{rates.total:.2f}")
    columns[3].metric("Supremacy", f"{rates.supremacy:+.2f}")
    st.caption(
        "Poisson rates from `dc_home_lambda` and `dc_away_lambda` in the ratings "
        "table, fitted on matches strictly earlier than this one — the property "
        "`src/validation/temporal.py` proves on every build. Expected goals from "
        "a *goal* model, which is a different measurement from expected goals "
        "off a shot map, and this project ingests no shot map."
    )


def _availability(ctx: context.Context, row: pd.Series | None) -> None:
    """Both registered squads, and the three things they are not.

    Milestone 18. The roadmap called it "player availability, injuries,
    transfers"; what a reachable source actually answers is *who is
    registered*, and this panel is named and captioned for that rather than for
    the ask. A squad is an upper bound on availability: it does not know who is
    injured, who is suspended or who is being left out, and no feed this
    project can reach does either.

    Nothing here reaches the model. The forecast above was produced by a model
    fitted on scorelines, and no column of any table in this repository has
    ever held a player's name — so this panel is beside the forecast rather
    than inside it, and saying so is most of what it is for.
    """
    ui.section("Who is registered", "squads, which are not team sheets")
    if row is None:
        st.caption(
            "This fixture is not in the match table, so there is no competition "
            "to look a squad up in."
        )
        return
    if not ctx.squads.available:
        st.caption(matchday.reason(ctx.squads))
        return

    competition_id = str(row["competition_id"])
    # The reason is read immediately after each lookup, not once at the end.
    # A provider carries the *last* failure it had, so asking after both clubs
    # would caption a missing home squad with why the away one failed — or,
    # when the away lookup succeeded, with nothing at all.
    found: list[tuple[str, Squad | None, str]] = []
    for column in ("home_team", "away_team"):
        team = str(row[column])
        squad = ctx.squads.squad(team, competition_id=competition_id)
        found.append((team, squad, "" if squad is not None else matchday.reason(ctx.squads)))

    if all(squad is None for _, squad, _ in found):
        st.caption(found[0][2])
        return
    for side, (team, squad, why) in zip(st.columns(2), found, strict=True):
        with side:
            _squad(team, squad, why)
    st.caption(
        f"Registered squads from {ctx.squads.name}, reread hourly. "
        "**Registered is not available**: a squad list does not know who is "
        "injured, suspended or rested, and it is the whole of what this feed "
        "carries. Nothing on this panel reached the forecast above — no table "
        "in this project holds a player's name."
    )


def _squad(team: str, squad: Squad | None, error: str) -> None:
    """One club's panel, or the sentence saying why there is not one."""
    st.markdown(f"**{team}**")
    if squad is None:
        st.caption(error)
        return
    age = squad.median_age()
    columns = st.columns(2)
    columns[0].metric("Registered", squad.size)
    columns[1].metric("Median age", f"{age:.1f}" if age is not None else "—")
    st.caption(
        ", ".join(f"{count} {position.lower()}" for position, count in squad.positions.items())
        + (f" · listed as {squad.team}" if squad.team != team else "")
    )
    with st.expander(f"The {squad.size} names"):
        st.dataframe(
            pd.DataFrame(
                {
                    "player": [one.name for one in squad.players],
                    "position": [one.position or "" for one in squad.players],
                    "age": [one.age() for one in squad.players],
                    "nationality": [one.nationality or "" for one in squad.players],
                }
            ),
            width="stretch",
            hide_index=True,
        )


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
