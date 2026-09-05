"""What this reader follows, and for how long.

``st.session_state``: per browser tab, lasting as long as it is open. That is
what Milestone 12 asked for, and it is the whole of the Milestone 14 seam — an
account store replaces the four accessors here and nothing else changes,
because no view touches ``st.session_state`` and no view knows where a
favourite is kept.

Kept apart from :mod:`dashboard.domain.competition` deliberately. That module
is a static catalogue loaded from a file in the repository; this one is a
mutable per-person preference. They answer the same question — *which football
matters here* — of two things with nothing else in common.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

LEAGUES_KEY = "favourite_leagues"
TEAMS_KEY = "favourite_teams"


def leagues() -> list[str]:
    """The competition ids this session follows. Empty means "all of them"."""
    return list(st.session_state.get(LEAGUES_KEY, []))


def teams() -> list[str]:
    """The clubs this session follows."""
    return list(st.session_state.get(TEAMS_KEY, []))


def remember_leagues(ids: Sequence[str]) -> None:
    st.session_state[LEAGUES_KEY] = list(ids)


def remember_teams(names: Sequence[str]) -> None:
    st.session_state[TEAMS_KEY] = list(names)


def toggle_league(competition_id: str) -> bool:
    """Follow or unfollow a competition. Returns whether it is now followed."""
    return _toggle(LEAGUES_KEY, competition_id)


def toggle_team(name: str) -> bool:
    """Follow or unfollow a club. Returns whether it is now followed."""
    return _toggle(TEAMS_KEY, name)


def _toggle(key: str, value: str) -> bool:
    current = list(st.session_state.get(key, []))
    if value in current:
        current.remove(value)
        st.session_state[key] = current
        return False
    st.session_state[key] = [*current, value]
    return True


def league_filter(chosen: Sequence[str] | None = None) -> list[str] | None:
    """Favourite leagues as a filter, or ``None`` for "do not filter".

    ``None`` and ``[]`` mean different things to every reader downstream — no
    filter against a filter matching nothing — and a reader who has followed no
    league wants the former. This is the one place that distinction is made.
    """
    selected = list(chosen) if chosen is not None else leagues()
    return selected or None
