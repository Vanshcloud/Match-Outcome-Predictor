"""What this reader follows, and for how long.

**Milestone 14 replaced the "for how long".** These accessors used to read and
write ``st.session_state``, so a favourite lasted as long as a browser tab and
a reader who closed one lost their leagues. They now read and write
:mod:`dashboard.domain.store`, keyed by whatever
:mod:`dashboard.domain.identity` says the reader is.

**Nothing else changed, and that was the claim.** Every signature here is the
one Milestone 12 shipped, no view was touched, and no view has ever named
``st.session_state`` — which is why an account store could replace the storage
under six pages without any of them knowing. CI asserts that property rather
than trusting it.

Kept apart from :mod:`dashboard.domain.competition` deliberately. That module
is a static catalogue loaded from a file in the repository; this one is a
mutable per-person preference. They answer the same question — *which football
matters here* — of two things with nothing else in common.

**Not cached.** Every read is a small JSON file, and the alternative is a cache
that goes stale the moment the same reader changes a profile in a second tab.
If this file ever grows past a few hundred profiles, the fix is the database
the compose file already runs, not a cache here.
"""

from __future__ import annotations

from collections.abc import Sequence

from dashboard.domain import identity, store


def leagues() -> list[str]:
    """The competition ids this reader follows. Empty means "all of them"."""
    return list(store.saved(identity.current())[store.LEAGUES])


def teams() -> list[str]:
    """The clubs this reader follows."""
    return list(store.saved(identity.current())[store.TEAMS])


def remember_leagues(ids: Sequence[str]) -> None:
    store.save(identity.current(), leagues=list(ids), teams=teams())


def remember_teams(names: Sequence[str]) -> None:
    store.save(identity.current(), leagues=leagues(), teams=list(names))


def toggle_league(competition_id: str) -> bool:
    """Follow or unfollow a competition. Returns whether it is now followed."""
    current = leagues()
    following = competition_id not in current
    remember_leagues(
        [*current, competition_id]
        if following
        else [one for one in current if one != competition_id]
    )
    return following


def toggle_team(name: str) -> bool:
    """Follow or unfollow a club. Returns whether it is now followed."""
    current = teams()
    following = name not in current
    remember_teams([*current, name] if following else [one for one in current if one != name])
    return following


def league_filter(chosen: Sequence[str] | None = None) -> list[str] | None:
    """Favourite leagues as a filter, or ``None`` for "do not filter".

    ``None`` and ``[]`` mean different things to every reader downstream — no
    filter against a filter matching nothing — and a reader who has followed no
    league wants the former. This is the one place that distinction is made.
    """
    selected = list(chosen) if chosen is not None else leagues()
    return selected or None
