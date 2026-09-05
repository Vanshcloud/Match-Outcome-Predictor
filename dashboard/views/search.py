"""One box, three kinds of answer: a competition, a club, or a match.

The entry point into the application. Everything else on this dashboard is a
list somebody scrolls; this is where a reader who already knows what they want
types it.

**One query, searched three ways, ranked by nothing.** No relevance scoring:
the three kinds of result are shown in three sections rather than interleaved,
because a reader typing "Arsenal" wants the club and a reader typing "Serie A"
wants the competition, and a single ranked list would put one of them second
for no reason a person could predict.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import context, ui
from dashboard.domain import competition as catalogue
from dashboard.domain import favourites
from dashboard.services import history

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
    st.caption("Clubs, competitions and matches. Type at least two characters.")

    query = st.text_input(
        "Search",
        value=str(st.query_params.get(QUERY_PARAM, "") or ""),
        placeholder="Arsenal · Bundesliga · Real Madrid",
        label_visibility="collapsed",
    ).strip()
    if len(query) < MINIMUM_QUERY:
        _suggestions()
        return

    needle = query.casefold()
    _competitions(needle)
    if not ctx.has_matches:
        st.info("No match table yet, so clubs and matches cannot be searched. Run `make data`.")
        return
    table = history.all_matches(ctx.matches_path)
    teams = _teams(table, needle)
    _matches(table, teams, needle)


def _suggestions() -> None:
    """What to do before anything has been typed."""
    ui.section("Start here", "the competitions you follow, or the ones most people mean")
    followed = favourites.leagues() or ["ENG_1", "ESP_1", "GER_1", "ITA_1", "FRA_1"]
    st.markdown(
        " ".join(
            ui.link(catalogue.short_label(one), f"competitions?competition={one}")
            for one in followed
            if one in catalogue.by_id()
        ),
        unsafe_allow_html=True,
    )


def _competitions(needle: str) -> None:
    """Competitions whose country, name or id contains the query."""
    found = [
        one
        for one in catalogue.competitions()
        if needle in f"{one.country} {one.name} {one.id}".casefold()
    ]
    ui.section("Competitions", f"{len(found)} match{'es' if len(found) != 1 else ''}")
    if not found:
        st.caption("No competition by that name.")
        return
    st.markdown(
        " ".join(
            ui.link(catalogue.label(one.id), f"competitions?competition={one.id}") for one in found
        ),
        unsafe_allow_html=True,
    )


def _teams(table: pd.DataFrame, needle: str) -> list[str]:
    """Clubs whose name contains the query, with a follow control for each."""
    names = [one for one in history.teams_in(table) if needle in one.casefold()]
    ui.section("Clubs", f"{len(names)} match{'es' if len(names) != 1 else ''}")
    if not names:
        st.caption("No club by that name in the match table.")
        return []

    shown = names[:TEAM_RESULTS]
    lanes = st.columns(min(4, len(shown)))
    for position, name in enumerate(shown):
        with lanes[position % len(lanes)]:
            following = name in favourites.teams()
            st.markdown(f"**{name}**")
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


def _matches(table: pd.DataFrame, teams: list[str], needle: str) -> None:
    """Recent matches involving any club the query found."""
    ui.section("Matches", "most recent first")
    if not teams:
        st.caption(f"No matches to show for “{needle}”.")
        return
    frame = history.follow(table, teams).sort_values("date", ascending=False).head(MATCH_CARDS)
    ui.card_grid(
        [
            ui.match_card(one, competition_label=catalogue.short_label(one.competition_id))
            for one in history.as_fixtures(frame)
        ],
        empty="No matches in the table for those clubs.",
    )
