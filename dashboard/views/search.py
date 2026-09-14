"""One box: a club, its live and upcoming fixtures, and its results.

The entry point into the application. Everything else on this dashboard is a
list somebody scrolls; this is where a reader who already knows what they want
types it. Competitions are browsed on their own page rather than searched here.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from dashboard import context, ui
from dashboard.domain import competition as catalogue
from dashboard.domain import favourites
from dashboard.services import history, matchday
from dashboard.views.home import render_section

MINIMUM_QUERY = 2
"""Shorter than this searches nothing.

A one-character query matches a third of the clubs in thirty-nine
competitions, which is a list nobody reads and a full scan to build it.
"""

TEAM_RESULTS = 12
MATCH_CARDS = 9
QUERY_PARAM = "q"


def render() -> None:
    """The search page."""
    ctx = context.resolve()
    st.title("Search")
    st.caption("Clubs, their fixtures and their results. Type at least two characters.")

    query = st.text_input(
        "Search",
        value=str(st.query_params.get(QUERY_PARAM, "") or ""),
        placeholder="Arsenal · Real Madrid · Bayern",
        label_visibility="collapsed",
    ).strip()
    if len(query) < MINIMUM_QUERY:
        return

    needle = query.casefold()
    if not ctx.has_matches:
        st.info("No match table yet, so clubs and matches cannot be searched. Run `make data`.")
        return
    table = history.all_matches(ctx.matches_path)
    # Only the leagues the fixture feed covers: the rest have no crests, fixtures or forecasts here.
    table = table[table["competition_id"].isin(matchday.COVERED)]
    teams = _teams(ctx, table, needle)
    if teams:
        render_section(matchday.club_fixtures(ctx.fixtures, ctx.predictions, teams))
    _matches(ctx, table, teams, needle)


def _teams(ctx: context.Context, table: pd.DataFrame, needle: str) -> list[str]:
    """Clubs whose name contains the query, each with its crest and a follow control.

    A club is looked up in the club list of the competition of its latest match
    (and that country's other covered division); a club the feed no longer
    lists — dropped out of the covered divisions — is left out. When the feed
    has no list to ask, every club stays rather than none.
    """
    crests = {
        name: _crest(ctx, league, name)
        for name, league in _latest_competitions(table, needle).items()
    }
    names = [name for name, crest in sorted(crests.items()) if crest != MISSING]
    ui.section("Clubs", f"{len(names)} match{'es' if len(names) != 1 else ''}")
    if not names:
        st.caption("No club by that name in the leagues this dashboard covers.")
        return []

    shown = names[:TEAM_RESULTS]
    lanes = st.columns(min(4, len(shown)))
    for position, name in enumerate(shown):
        with lanes[position % len(lanes)]:
            following = name in favourites.teams()
            st.markdown(
                f'<span class="mop-side">{ui.crest_html(name, crests[name] or None)}'
                f"<b>{html.escape(name)}</b></span>",
                unsafe_allow_html=True,
            )
            if st.button(
                "Following" if following else "Follow",
                key=f"follow-{name}",
                type="primary" if following else "secondary",
            ):
                favourites.toggle_team(name)
                st.rerun()
    if len(names) > TEAM_RESULTS:
        st.caption(f"{len(names) - TEAM_RESULTS} more clubs match. Narrow the query.")
    return shown


MISSING = "missing"
"""A club the feed's list for its competition does not include."""


def _latest_competitions(table: pd.DataFrame, needle: str) -> dict[str, str]:
    """Each club whose name contains ``needle``, with the competition of its latest match."""
    sides = pd.concat(
        [
            table[["date", "competition_id", side]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ]
    )
    sides = sides[sides["team"].str.casefold().str.contains(needle, regex=False)]
    latest = sides.sort_values("date").drop_duplicates("team", keep="last")
    return dict(zip(latest["team"], latest["competition_id"].astype(str), strict=True))


def _crest(ctx: context.Context, league: str, name: str) -> str:
    """The club's crest URL, ``""`` when the feed has no list to ask, or :data:`MISSING`."""
    listed = matchday.crests_around(ctx.fixtures, league)
    if not listed:
        return ""
    return matchday.crest_for(name, listed) or MISSING


def _matches(ctx: context.Context, table: pd.DataFrame, teams: list[str], needle: str) -> None:
    """Recent results involving any club the query found."""
    ui.section("Results", "most recent first")
    if not teams:
        st.caption(f"No matches to show for “{needle}”.")
        return
    frame = history.follow(table, teams).sort_values("date", ascending=False).head(MATCH_CARDS)
    ui.card_grid(
        [
            ui.match_card(one, competition_label=catalogue.short_label(one.competition_id))
            for one in matchday.with_crests(history.as_fixtures(frame), ctx.fixtures)
        ],
        empty="No matches in the table for those clubs.",
    )
