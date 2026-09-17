"""Competitions: the whole registry to browse, and one league at a time to read.

Every competition in ``configs/leagues.yaml`` has a page here, and it has one
without anybody writing it: the page is a function of the registry, so a
league added to the registry appears with no code change. That is the
same property ``tests/unit/test_registry.py`` asserts about ingestion, extended
to the presentation layer.
"""

from __future__ import annotations

import streamlit as st

from dashboard import context, ui
from dashboard.domain import competition as catalogue
from dashboard.domain import favourites
from dashboard.services import matchday
from dashboard.views.home import render_section

COMPETITION_PARAM = "competition"
FIXTURE_CARDS = 12
FEED_ONLY_COUNTRY = "Europe"


def render() -> None:
    """The competition browser, or one competition's page."""
    ctx = context.resolve()
    chosen = str(st.query_params.get(COMPETITION_PARAM, "") or "")
    if chosen and (chosen in catalogue.by_id() or chosen in matchday.FEED_ONLY_NAMES):
        _competition_page(ctx, chosen)
        return
    _browser()


# ---- browsing ----------------------------------------------------------------


def _browser() -> None:
    """Every covered competition as one list of flags and names."""
    st.title("Competitions")

    # Country, then tier, so a country's top flight leads. A row is short, so
    # the Champions League drops its "UEFA" there; its id still says it.
    #
    # Named with `short_label`, the same call every card makes, so the two
    # "Serie A" rows read "Serie A — Italy" and "Serie A — Brazil" here as well.
    # The flag alone distinguished them for anyone who could see it, and left
    # the list contradicting every card that names the country in text.
    tiles = sorted(
        [
            (one.country, one.tier or 99, catalogue.short_label(one.id), one.id)
            for one in catalogue.competitions()
            if one.id in matchday.COVERED
        ]
        + [
            (FEED_ONLY_COUNTRY, 99, name.removeprefix("UEFA "), id_)
            for id_, name in matchday.FEED_ONLY_NAMES.items()
        ]
    )
    rows = [
        ui.competition_row(name, f"competitions?{COMPETITION_PARAM}={id_}", country=country)
        for country, _, name, id_ in tiles
    ]
    st.markdown(f'<span class="mop-leagues">{"".join(rows)}</span>', unsafe_allow_html=True)


# ---- one competition ---------------------------------------------------------


def _competition_page(ctx: context.Context, competition_id: str) -> None:
    """One competition: what it is, and what is coming up in it with the model's forecasts."""
    chosen = catalogue.by_id().get(competition_id)
    if chosen is not None:
        st.title(chosen.name)
        tier = f"tier {chosen.tier}" if chosen.tier else "cup competition"
        st.caption(f"{chosen.country} · {tier}")
    else:
        st.title(matchday.FEED_ONLY_NAMES[competition_id])
        st.caption(
            f"{FEED_ONLY_COUNTRY} · cup competition · no match history "
            "here, so its fixtures show kick-offs and scores without a forecast"
        )

    following = competition_id in favourites.leagues()
    if st.button("Unfollow" if following else "Follow", type="secondary"):
        favourites.toggle_league(competition_id)
        st.rerun()

    # Upcoming only, each card with its forecast: past results are what the
    # model was trained on, not what this page is for.
    upcoming = matchday.upcoming_section(ctx.fixtures, [competition_id])
    render_section(matchday.with_forecasts(upcoming, ctx.predictions))
