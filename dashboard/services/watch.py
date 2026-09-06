"""What changed since this page last looked, and who to tell.

Milestone 15. The fixture feed already answers "what is in play"; this asks the
question a person actually has — *what happened while I was not looking* — and
turns the answer into :class:`~dashboard.domain.match.MatchEvent` objects that
a toast and a webhook render the same way.

**The state is one snapshot per browser tab.** For every match being tracked,
its status and its score the last time this tab rendered. Kept in
``st.session_state`` because that is what "since *I* last looked" means: two
tabs are two readers, and a snapshot in the profile store would mean the first
tab to refresh silently consumed the second one's news.

**The first look announces nothing.** With no previous snapshot there is no
"since", and a page that toasted a kick-off for a match already an hour old
would be telling a reader something that is not true. The first render records
and stays quiet.

**An empty answer is not full time.** The feed returns nothing both when no
match is in play and when it could not be reached — a rate limit, an expired
key — and a diff that read absence as "the match ended" would announce eight
full-times because of one HTTP 429. So the feed is asked whether *it* thinks it
answered, and a failed answer produces no events at all rather than a page full
of finals that never happened.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import streamlit as st

from dashboard.domain.match import EventKind, Fixture, MatchEvent, MatchStatus
from dashboard.providers.base import FixtureProvider, Notifier

SNAPSHOT_KEY = "watch_snapshot"
"""Where the last look is kept. The only key this module owns."""

REFRESH_SECONDS = 60
"""How often the live section reruns itself.

Matched to :data:`dashboard.providers.football_data_org.CACHE_SECONDS` rather
than chosen: refreshing faster than the feed is memoised repaints identical
rows, and refreshing slower leaves the memo to expire unused. The free tier
allows ten requests a minute and this is one.
"""


def since_last_look(
    fixtures: FixtureProvider,
    *,
    competitions: Sequence[str] | None = None,
    teams: Sequence[str] | None = None,
) -> tuple[list[Fixture], list[MatchEvent]]:
    """The matches in play now, and what changed since this tab last asked.

    Returns both because the caller renders both from one answer, and asking
    the feed twice for one paint is a request this budget does not have.
    """
    if not fixtures.available:
        return [], []
    live = list(fixtures.live(competitions=competitions))
    if getattr(fixtures, "error", None):
        return [], []
    wanted = [one for one in live if _followed(one, teams)]
    events = diff(st.session_state.get(SNAPSHOT_KEY), wanted)
    st.session_state[SNAPSHOT_KEY] = snapshot(wanted)
    return live, events


def diff(previous: Mapping[str, Fixture] | None, current: Sequence[Fixture]) -> list[MatchEvent]:
    """Events between two looks. Empty when there was no previous look.

    **Full time is a match that has gone**, not one whose status changed:
    :meth:`~dashboard.providers.base.FixtureProvider.live` answers what is in
    play, so a finished match leaves the list rather than appearing in it as
    finished. The event is therefore built from the fixture as it was last
    seen, which is also the only place its final score exists.
    """
    if previous is None:
        return []
    now = snapshot(current)
    events = [
        MatchEvent(kind, fixture)
        for match_id, fixture in now.items()
        if (kind := _kind(previous.get(match_id), fixture)) is not None
    ]
    events.extend(
        MatchEvent(EventKind.FULL_TIME, gone)
        for match_id, gone in previous.items()
        if match_id not in now and gone.status is MatchStatus.LIVE
    )
    return events


def snapshot(fixtures: Sequence[Fixture]) -> dict[str, Fixture]:
    """Each tracked match as it was, by id. What a look consists of.

    The fixtures themselves rather than a status-and-score string, because the
    match that has *gone* from the next look is the one a full-time message has
    to be written from, and by then this is the only record of it.
    """
    return {one.match_id: one for one in fixtures}


def announce(events: Sequence[MatchEvent], notifier: Notifier) -> int:
    """Send each event onward, returning how many were delivered.

    Failures are not retried and not raised: a webhook that is refusing is a
    webhook that will refuse the retry too, and the live section it is attached
    to is about football rather than about a transport.
    """
    return sum(1 for event in events if notifier.send(event))


def _kind(before: Fixture | None, after: Fixture) -> EventKind | None:
    """What the change from one look to the next is called.

    A match this tab has not seen before is a kick-off only if it is in play:
    the alternative is announcing one for a fixture that appeared in the feed
    already finished, which is news about the past.

    There is deliberately no "was scheduled, is now live" branch. A snapshot
    holds what
    :meth:`~dashboard.providers.base.FixtureProvider.live` returned, so
    everything in it was in play when it was recorded, and a branch for a state
    this function cannot be handed is a branch no test can honestly reach. The
    milestone that starts tracking scheduled matches adds it back, with a case.
    """
    if before is None:
        return EventKind.KICK_OFF if after.status is MatchStatus.LIVE else None
    if after.status is MatchStatus.LIVE and _score(after) != _score(before):
        return EventKind.GOAL
    return None


def _score(fixture: Fixture) -> str:
    return f"{fixture.home_goals}-{fixture.away_goals}" if fixture.has_score else ""


def _followed(fixture: Fixture, teams: Sequence[str] | None) -> bool:
    """Whether this match is one the reader asked to be told about.

    No followed clubs means every match in the competitions they follow, which
    is the same rule the home page's other sections use.
    """
    return not teams or any(fixture.involves(one) for one in teams)
