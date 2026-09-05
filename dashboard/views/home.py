"""The home page, and the live centre beside it.

What is happening in the football this reader follows, in the order they ask:
what is on now, what is on next, what just finished, and what the model can
price.

**Two of those four sections have no data source today and say so.** The
provider behind this project publishes results, so a match that has not been
played is in no table here. Those sections are not special cases in this file —
they are :class:`~dashboard.services.matchday.Section` objects carrying the
provider's own reason, rendered the same way the full ones are. The day a
fixture feed is registered they fill up and nothing in this module changes.
"""

from __future__ import annotations

import streamlit as st

from dashboard import context, ui
from dashboard.domain import competition, favourites
from dashboard.services import history, matchday
from dashboard.services.matchday import Section

LIVE_COLUMNS = 4


def render() -> None:
    """The home dashboard."""
    ctx = context.resolve()
    st.title("Match centre")
    _headline(ctx)

    page = matchday.home_page(
        fixtures=ctx.fixtures,
        results=ctx.results,
        predictions=ctx.predictions,
        competitions=favourites.league_filter(),
        teams=favourites.teams(),
    )
    for section in page.sections:
        render_section(section)


def render_live() -> None:
    """The live centre: the same feed, given the whole screen.

    Its own page rather than a longer home section, because the weekend it is
    for has forty matches running at once and that is a screen, not a strip.
    """
    ctx = context.resolve()
    st.title("Live centre")
    st.caption(
        "Every match in play, across the competitions you follow. Refreshes "
        "when the page does; a feed that pushes updates is Milestone 15."
    )
    render_section(
        matchday.live_section(ctx.fixtures, favourites.league_filter()), columns=LIVE_COLUMNS
    )

    ui.section("What this page is, and what it is not", "with a feed connected")
    st.markdown(
        "- **Scores and minutes** are the feed's, on the same `Fixture` these "
        "cards already rendered when there was no feed at all.\n"
        "- **These matches are not in the match table**, which holds results "
        "this project ingested. A card here opens a page with no history and "
        "no forecast until the match has been played and `make data` has run "
        "— the shipped model is fitted on finished matches.\n"
        "- **In-play probabilities** are a different model from the one this "
        "repository measures, and are not a rendering change. The model card "
        "is explicit that nothing here is fitted on in-play state."
    )


# ---- rendering a section -----------------------------------------------------


def render_section(section: Section, *, columns: int = 3) -> None:
    """One strip: its heading, then its cards or the reason it has none.

    The one place the difference between "nothing happened" and "nothing can
    answer" reaches the screen. Both are an empty list; only one of them is
    about football.
    """
    ui.section(section.title, section.note)
    if section.unavailable is not None:
        ui.placeholder("Nothing to show, and here is why", section.unavailable)
        return
    ui.card_grid(
        [
            ui.match_card(one, competition_label=competition.short_label(one.competition_id))
            for one in section.fixtures
        ],
        columns=columns,
        empty=section.empty,
    )


def _headline(ctx: context.Context) -> None:
    """One line saying what the reader is looking at, and how fresh it is."""
    followed = favourites.leagues()
    scope = (
        f"{len(followed)} competition{'s' if len(followed) != 1 else ''} you follow"
        if followed
        else f"all {len(competition.competitions())} competitions"
    )
    latest = history.latest_date(ctx.matches_path) if ctx.has_matches else None
    freshness = f" · results through {latest:%d %b %Y}" if latest else ""
    st.caption(f"{scope}{freshness}")
