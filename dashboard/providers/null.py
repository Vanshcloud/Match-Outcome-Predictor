"""The honest answer while no fixture feed is connected.

Not a stub that pretends. It returns nothing, reports ``available = False``,
and the views render that as a sentence naming what is missing and which
milestone connects it.

**Nothing here invents a fixture.** Plausible-looking generated matches would
put a game on the screen that is not being played, and that is the one failure
this application cannot recover from: a reader who catches it once stops
believing the real rows too. An empty section with a reason is a boundary; a
fabricated one is a lie with a countdown clock on it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

from dashboard.domain.match import Fixture, MatchEvent, Squad


@dataclass(frozen=True, slots=True)
class NullFixtures:
    """No scheduled matches and no live ones, because there is no feed."""

    name: str = "none"
    available: bool = False

    reason: str = (
        "The data behind this project is a **results** feed — a match that has "
        "not been played is in no table here — so kick-off times, live scores "
        "and minutes need a second provider."
    )
    """Why there is nothing, in the words a view puts on the screen.

    On the provider rather than in the view, because the next provider will
    have a different reason — an expired key, a rate limit, a competition the
    plan does not cover — and a view that hard-coded this one would report it
    wrongly the day a real feed fails.
    """

    def scheduled(
        self,
        *,
        since: dt.date,
        until: dt.date,
        competitions: Sequence[str] | None = None,
    ) -> list[Fixture]:
        return []

    def live(self, *, competitions: Sequence[str] | None = None) -> list[Fixture]:
        return []


@dataclass(frozen=True, slots=True)
class NullNotifier:
    """Nowhere to send an event, which is the default and not a failure.

    A dashboard with no webhook configured still tracks matches and still
    toasts what changed — the toast is in the page and needs no transport. This
    is the *other* half, and it says what configuring one would add rather than
    pretending an event went somewhere.
    """

    name: str = "none"
    available: bool = False

    reason: str = (
        "No transport is configured, so events appear here and nowhere else. "
        "Set `DASHBOARD_WEBHOOK_URL` to post them to Slack, Discord, ntfy or "
        "anything else that accepts a POST."
    )

    def send(self, event: MatchEvent) -> bool:
        """Deliver nothing, and say so by returning ``False``."""
        return False


@dataclass(frozen=True, slots=True)
class NullSquads:
    """No squad lists, because no squad source is configured.

    The default, as with the fixture feed: the results this project ingests are
    scorelines, and no table here has ever held a player's name.
    """

    name: str = "none"
    available: bool = False

    reason: str = (
        "No squad source is configured. The ingested feed is **results** — no "
        "table in this project holds a player's name — so a team sheet needs a "
        "second provider. Set `DASHBOARD_SQUAD_PROVIDER=football-data.org` "
        "with a key to list registered squads; note that *registered* is not "
        "*available*, and no source here answers the second question."
    )

    def squad(self, team: str, *, competition_id: str | None = None) -> Squad | None:
        return None
