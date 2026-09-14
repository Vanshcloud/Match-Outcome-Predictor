"""The home page, and the live centre beside it.

What is happening in the football this reader follows, in the order they ask:
what is on now, what is on next, and what the model can price. Every section
looks forward; past results live on the competition and match pages.

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
from dashboard.domain.match import EventKind
from dashboard.services import matchday, watch
from dashboard.services.matchday import Section

LIVE_COLUMNS = 4

EVENT_ICONS = {
    EventKind.KICK_OFF: "🟢",
    EventKind.GOAL: "⚽",
    EventKind.FULL_TIME: "🔔",
}
"""One glyph per kind, so a reader who catches a toast out of the corner of an
eye knows whether to look."""


def render() -> None:
    """The home dashboard."""
    ctx = context.resolve()
    st.title("Match centre")

    live_now(ctx)

    page = matchday.home_page(
        fixtures=ctx.fixtures,
        predictions=ctx.predictions,
        competitions=_followed(),
    )
    for section in page.sections:
        # Live is drawn above, by a fragment that repaints on its own clock.
        # The section is still assembled — `home_page` is what a test and one
        # day a second front end call — and rendering it twice is the only
        # thing that would be wrong.
        if section is not page.live:
            render_section(section)


def render_live() -> None:
    """The live centre: the same feed, given the whole screen.

    Its own page rather than a longer home section, because the weekend it is
    for has forty matches running at once and that is a screen, not a strip.
    """
    ctx = context.resolve()
    st.title("Live centre")
    st.caption(
        f"Every match in play, across the competitions you follow. Refreshes "
        f"itself every {watch.REFRESH_SECONDS} seconds."
    )
    live_now(ctx, columns=LIVE_COLUMNS)

    ui.section("What this page is, and what it is not", "with a feed connected")
    st.markdown(
        "- **Scores and minutes** are the feed's, on the same `Fixture` these "
        "cards already rendered when there was no feed at all.\n"
        "- **A card opens the model's pre-match forecast** when `make fixtures` "
        "has priced that fixture; the feed's club names are matched to the "
        "table's. A fixture it has not priced yet, or a competition with no "
        "match history (the Champions League), says so instead.\n"
        "- **In-play probabilities** are a different model from the one this "
        "repository measures, and are not a rendering change. The model card "
        "is explicit that nothing here is fitted on in-play state."
    )


# ---- the live strip, which reruns itself -------------------------------------


@st.fragment(run_every=watch.REFRESH_SECONDS)
def live_now(ctx: context.Context, *, columns: int = 3) -> None:
    """What is in play, repainted without the reader touching anything.

    A fragment rather than a whole-page rerun: everything else here is a file
    read or an HTTP call — the results table, the reliability tables, the
    service's fixture list — and repainting all of it every minute to move one
    score would be the most expensive way to show the cheapest change.

    ``run_every`` is matched to the feed's own memo, so a refresh that finds
    nothing new costs no request at all.
    """
    fixtures, events = watch.since_last_look(
        ctx.fixtures,
        competitions=_followed(),
        teams=favourites.teams(),
    )
    for event in events:
        st.toast(event.message, icon=EVENT_ICONS[event.kind])
    watch.announce(events, ctx.notifier)

    render_section(matchday.live_section(ctx.fixtures, fixtures=fixtures), columns=columns)


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
            ui.match_card(
                one,
                probabilities=section.probabilities.get(one.match_id),
                # None for a feed-only competition, so the card shows the feed's own name.
                competition_label=(
                    competition.short_label(one.competition_id)
                    if one.competition_id in competition.by_id()
                    else None
                ),
            )
            for one in section.fixtures
        ],
        columns=columns,
        empty=section.empty,
    )


def _followed() -> list[str]:
    """What the reader follows, or every league the feed and the model both cover."""
    return favourites.league_filter() or list(matchday.FOLLOWABLE)
