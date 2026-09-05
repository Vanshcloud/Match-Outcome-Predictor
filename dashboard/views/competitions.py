"""Competitions: the whole registry to browse, and one league at a time to read.

Every competition in ``configs/leagues.yaml`` has a page here, and it has one
without anybody writing it: the page is a function of the registry, so the
league added in a future milestone appears with no code change. That is the
same property ``tests/unit/test_registry.py`` asserts about ingestion, extended
to the presentation layer.
"""

from __future__ import annotations

import streamlit as st

from dashboard import context, ui
from dashboard.domain import competition as catalogue
from dashboard.domain import favourites
from dashboard.services import history, matchday
from dashboard.views.home import render_section
from src.ingestion.registry import Competition

COMPETITION_PARAM = "competition"
FIXTURE_CARDS = 12

STANDINGS_NOTE = (
    "A league table is a season's results added up, and this project ingests "
    "results — so this is a page that <b>can</b> be built. It is not built "
    "yet because a table that is right needs the competition's own rules for "
    "points, tie-breaks, deductions and play-offs, and thirty-nine "
    "competitions do not share them. A standings table that silently ranked "
    "Argentina by goal difference when Argentina does not would be worse than "
    "no table."
)


def render() -> None:
    """The competition browser, or one competition's page."""
    ctx = context.resolve()
    chosen = str(st.query_params.get(COMPETITION_PARAM, "") or "")
    if chosen and chosen in catalogue.by_id():
        _competition_page(ctx, chosen)
        return
    _browser()


# ---- browsing ----------------------------------------------------------------


def _browser() -> None:
    """Every competition, grouped by country, with what the reader follows first."""
    st.title("Competitions")
    st.caption(
        f"{len(catalogue.competitions())} competitions, from "
        "`configs/leagues.yaml` — the one place a league is added to this "
        "project. Follow the ones you care about and they lead every page."
    )

    followed = st.multiselect(
        "Competitions you follow",
        options=[one.id for one in catalogue.competitions()],
        default=favourites.leagues(),
        format_func=catalogue.label,
        help="Kept for this session. Accounts and saved preferences are Milestone 14.",
    )
    if followed != favourites.leagues():
        favourites.remember_leagues(followed)

    query = st.text_input("Filter", placeholder="England, Bundesliga, BRA_1…").strip().casefold()
    for country, group in catalogue.by_country().items():
        shown = [one for one in group if _matches(one, query)]
        if not shown:
            continue
        ui.section(country, f"{len(shown)} competition{'s' if len(shown) != 1 else ''}")
        lanes = st.columns(min(4, len(shown)))
        for position, one in enumerate(shown):
            with lanes[position % len(lanes)]:
                tier = f"tier {one.tier}" if one.tier else "cup"
                st.markdown(f"**{one.name}**")
                st.caption(f"`{one.id}` · {tier}")
                st.markdown(
                    ui.link("Open", f"competitions?{COMPETITION_PARAM}={one.id}"),
                    unsafe_allow_html=True,
                )


def _matches(one: Competition, query: str) -> bool:
    """Whether a competition answers the filter box. Empty matches everything."""
    return not query or query in f"{one.country} {one.name} {one.id}".casefold()


# ---- one competition ---------------------------------------------------------


def _competition_page(ctx: context.Context, competition_id: str) -> None:
    """One league: what it is, what just happened in it, and what is not here yet."""
    chosen = catalogue.by_id()[competition_id]
    st.title(chosen.name)
    tier = f"tier {chosen.tier}" if chosen.tier else "cup competition"
    st.caption(f"{chosen.country} · {tier} · `{chosen.id}`")

    following = chosen.id in favourites.leagues()
    if st.button("Unfollow" if following else "Follow", type="secondary"):
        favourites.toggle_league(chosen.id)
        st.rerun()

    render_section(matchday.upcoming_section(ctx.fixtures, [chosen.id]))
    _finished(ctx, competition_id, chosen.name)

    ui.section("Standings", "not built yet, and the reason is not laziness")
    ui.placeholder("A league table needs each competition's own rules", STANDINGS_NOTE)


def _finished(ctx: context.Context, competition_id: str, name: str) -> None:
    """This season's completed matches, most recent first."""
    ui.section("Results", "this season, most recent first")
    if not ctx.has_matches:
        st.info("No match table yet. Run `make data` to ingest one.")
        return

    frame = history.for_competition(ctx.matches_path, competition_id)
    if frame.empty:
        st.info(f"No matches ingested for {name} yet.")
        return

    season = str(frame["season"].iloc[0])
    st.caption(f"Season {season} · {len(frame):,} matches ingested")
    ui.card_grid(
        [
            ui.match_card(one, competition_label=name)
            for one in history.as_fixtures(frame, limit=FIXTURE_CARDS)
        ]
    )
    with st.expander(f"Every {season} match ({len(frame):,})"):
        st.dataframe(
            frame[["date", "home_team", "home_goals", "away_goals", "away_team"]],
            width="stretch",
            hide_index=True,
        )
